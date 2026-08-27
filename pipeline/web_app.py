from __future__ import annotations

import argparse
import html
import mimetypes
import os
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote, unquote, urlparse

from .config import load_base_registry, repository_root
from .dataset_deletion import delete_dataset_items, delete_dataset_source, delete_dataset_workspace
from .dataset_workspace import DatasetWorkspace, list_datasets
from .models import PipelineError
from .service import load_project, project_path
from .state import ProjectState
from .training_config import (
    TrainingConfig,
    create_project_from_training_config,
    list_training_configs,
    make_training_workspace_name,
)


PAGE_SIZE = 48
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _q(value: str) -> str:
    return quote(value, safe="")


def _all_project_states(root: Path | None = None) -> list[ProjectState]:
    base = (root or repository_root()) / "projects"
    if not base.is_dir():
        return []
    states: list[ProjectState] = []
    for path in sorted(base.iterdir(), key=lambda item: item.name.casefold()):
        if (path / "project.yaml").is_file():
            states.append(ProjectState.load(path))
    return states


def _workspace_status(state: ProjectState) -> str:
    next_step = state.next_actionable_step()
    if next_step is None:
        return "complete"
    return f"{state.status(next_step).value}:{next_step}"


def _status_entries(states: Iterable[ProjectState]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for state in states:
        project = state.payload["project"]
        identity = project.get("training_identity", {})
        runs = list(state.payload.get("runs", []))
        if runs:
            for run in runs:
                entries.append(
                    {
                        "project": state.name,
                        "run_id": str(run.get("id") or ""),
                        "status": str(run.get("status", "unknown")),
                        "dataset": identity.get("dataset") or project.get("dataset_snapshot", {}).get("dataset") or "legacy",
                        "config": identity.get("config") or "legacy",
                        "updated": run.get("finished_at") or run.get("interrupted_at") or run.get("started_at"),
                    }
                )
        else:
            entries.append(
                {
                    "project": state.name,
                    "run_id": "",
                    "status": _workspace_status(state),
                    "dataset": identity.get("dataset") or project.get("dataset_snapshot", {}).get("dataset") or "legacy",
                    "config": identity.get("config") or "legacy",
                    "updated": project.get("updated_at"),
                }
            )
    return sorted(entries, key=lambda item: str(item.get("updated") or ""), reverse=True)


def _result_entries(states: Iterable[ProjectState]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for state in states:
        project = state.payload["project"]
        identity = project.get("training_identity", {})
        for run in state.payload.get("runs", []):
            if str(run.get("status")) not in {"trained", "evaluated", "promoted"}:
                continue
            checkpoints = [Path(value) for value in run.get("checkpoints", []) if Path(value).is_file()]
            if not checkpoints:
                continue
            run_dir = Path(str(run.get("path") or ""))
            samples = []
            if run_dir.is_dir():
                sample_dir = run_dir / "samples"
                if sample_dir.is_dir():
                    samples = [path for path in sorted(sample_dir.rglob("*")) if path.suffix.lower() in _IMAGE_SUFFIXES]
            entries.append(
                {
                    "project": state.name,
                    "run_id": str(run.get("id")),
                    "status": str(run.get("status")),
                    "dataset": identity.get("dataset") or project.get("dataset_snapshot", {}).get("dataset") or "legacy",
                    "config": identity.get("config") or "legacy",
                    "checkpoints": len(checkpoints),
                    "samples": len(samples),
                    "promoted": bool(run.get("promotion")),
                    "updated": run.get("finished_at") or run.get("started_at"),
                }
            )
    return sorted(entries, key=lambda item: str(item.get("updated") or ""), reverse=True)


def _find_run(state: ProjectState, run_id: str) -> dict[str, Any] | None:
    for run in state.payload.get("runs", []):
        if str(run.get("id")) == run_id:
            return run
    return None


def _training_command(state: ProjectState) -> list[str]:
    project = state.payload["project"]
    workflow = project.get("interactive_preferences", {})
    snapshot = project.get("training_config_snapshot", {})
    images_seen = int(snapshot.get("images_seen") or project.get("budget", {}).get("value") or 1000)
    caption_mode = str(workflow.get("caption_mode", "generate"))
    command = [
        sys.executable,
        "-m",
        "pipeline.cli",
        "run",
        state.name,
        "--images-seen",
        str(images_seen),
        "--caption-mode",
        caption_mode,
        "--skip-evaluate",
    ]
    if not bool(workflow.get("run_dedup", True)):
        command.append("--skip-dedup")
    elif bool(workflow.get("exclude_exact_duplicates", False)):
        command.append("--exclude-exact")
    if not bool(workflow.get("run_identity", state.concept_type == "character")):
        command.append("--skip-identity")
    if caption_mode == "skip":
        command.append("--skip-caption")
    if not bool(workflow.get("run_review", True)):
        command.append("--skip-review")
    if bool(workflow.get("allow_trigger_only", False)):
        command.append("--allow-trigger-only")
    return command


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def spawn_training_worker(state: ProjectState) -> int:
    pid_path = state.project_dir / "web-worker.pid"
    if pid_path.is_file():
        try:
            existing = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing = 0
        if _pid_alive(existing):
            raise PipelineError(f"Training worker is already running with PID {existing}")
    log_path = state.project_dir / "web-worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            _training_command(state),
            cwd=repository_root(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return int(process.pid)


def _safe_child(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / unquote(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PipelineError("Requested file is outside the allowed workspace") from exc
    return candidate


def _css() -> str:
    return """
:root{color-scheme:dark;--bg:#0b1020;--panel:#121a2d;--panel2:#18223a;--text:#edf2ff;--muted:#96a2bd;--accent:#79a8ff;--good:#75d59b;--warn:#f0c36a;--bad:#ff7b86;--line:#27344f}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#080d19,#0d1425 48%,#101932);color:var(--text);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.shell{max-width:1480px;margin:auto;padding:22px}.top{display:flex;gap:18px;align-items:center;justify-content:space-between;margin-bottom:20px}.brand{font-size:22px;font-weight:800}.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a,.button,button{border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:8px 12px;border-radius:9px;cursor:pointer}.nav a.active{border-color:var(--accent);color:#fff}.button.danger,button.danger{border-color:#733742;background:#3b1d27;color:#ffd9dc}.button.good,button.good{border-color:#356c4b;background:#173728}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.card,.panel{background:rgba(18,26,45,.94);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 12px 30px #0004}.metric{font-size:30px;font-weight:800}.muted{color:var(--muted)}.bad{color:var(--bad)}.good{color:var(--good)}.warn{color:var(--warn)}table{width:100%;border-collapse:collapse;background:rgba(18,26,45,.94);border-radius:12px;overflow:hidden}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}.images{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}.image-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}.image-card img{width:100%;height:220px;object-fit:contain;background:#070b14;display:block}.image-body{padding:10px}.image-title{word-break:break-all;font-size:13px;margin-bottom:6px}textarea,input,select{width:100%;background:#0d1424;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:8px}textarea{min-height:72px;resize:vertical}.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.pill{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:99px;color:var(--muted);font-size:12px}.flash{padding:10px 12px;border:1px solid #2e6845;background:#153524;border-radius:10px;margin-bottom:14px}.error{padding:10px 12px;border:1px solid #743743;background:#3a1c25;border-radius:10px;margin-bottom:14px}.danger-zone{border-color:#69313b}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;word-break:break-word}.pager{display:flex;gap:8px;justify-content:center;margin:16px}.hero{margin:12px 0 20px}.hero h1{margin:0 0 4px;font-size:28px}@media(max-width:720px){.shell{padding:12px}.top{align-items:flex-start;flex-direction:column}.image-card img{height:180px}}
"""


def _page(title: str, body: str, *, active: str = "", flash: str = "", error: str = "") -> str:
    nav = [
        ("datasets", "/datasets", "数据集"),
        ("configs", "/configs", "训练配置"),
        ("status", "/status", "训练状态"),
        ("results", "/results", "训练结果"),
    ]
    links = "".join(
        f'<a class="{"active" if key == active else ""}" href="{url}">{label}</a>'
        for key, url, label in nav
    )
    notice = f'<div class="flash">{_e(flash)}</div>' if flash else ""
    problem = f'<div class="error">{_e(error)}</div>' if error else ""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)} · LoRA Pipeline</title><style>{_css()}</style></head><body><div class="shell"><div class="top"><a class="brand" href="/">LoRA Pipeline</a><nav class="nav">{links}</nav></div>{notice}{problem}{body}</div></body></html>"""


class WebApplication:
    def __init__(self, *, root: Path | None = None):
        self.root = (root or repository_root()).resolve()
        self.csrf = secrets.token_urlsafe(24)

    def datasets(self) -> list[DatasetWorkspace]:
        return list_datasets(root=self.root)

    def configs(self) -> list[TrainingConfig]:
        return list_training_configs(root=self.root)

    def states(self) -> list[ProjectState]:
        return _all_project_states(self.root)


class Handler(BaseHTTPRequestHandler):
    server_version = "LoRAPipelineWeb/1"
    app: WebApplication

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[web] " + fmt % args + "\n")

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._get()
        except (PipelineError, OSError, ValueError, KeyError) as exc:
            self._html(_page("错误", '<div class="hero"><h1>请求失败</h1></div>', error=str(exc)), status=400)

    def do_POST(self) -> None:  # noqa: N802
        try:
            form = self._form()
            if form.get("_csrf", [""])[0] != self.app.csrf:
                raise PipelineError("CSRF validation failed; refresh the page and retry")
            self._post(form)
        except (PipelineError, OSError, ValueError, KeyError) as exc:
            self._html(_page("错误", '<div class="hero"><h1>操作失败</h1></div>', error=str(exc)), status=400)

    def _get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        if path == "/healthz":
            self._text("ok\n")
            return
        if path == "/":
            self._home()
            return
        if path == "/datasets":
            self._datasets()
            return
        if path.startswith("/datasets/"):
            parts = path.split("/")
            if len(parts) == 3:
                self._dataset(unquote(parts[2]))
                return
            if len(parts) >= 5 and parts[3] == "source":
                self._source(unquote(parts[2]), unquote(parts[4]), query)
                return
        if path.startswith("/media/dataset/"):
            parts = path.split("/", 5)
            if len(parts) == 6:
                self._dataset_media(unquote(parts[3]), unquote(parts[4]), parts[5])
                return
        if path == "/configs":
            self._configs()
            return
        if path.startswith("/configs/"):
            self._config(unquote(path.split("/", 2)[2]))
            return
        if path == "/status":
            self._status()
            return
        if path.startswith("/status/"):
            self._status_detail(unquote(path.split("/", 2)[2]))
            return
        if path == "/results":
            self._results()
            return
        if path.startswith("/results/"):
            parts = path.split("/")
            if len(parts) == 4:
                self._result_detail(unquote(parts[2]), unquote(parts[3]))
                return
        if path.startswith("/run-file/"):
            parts = path.split("/", 4)
            if len(parts) == 5:
                project_run = parts[2:4]
                self._run_file(unquote(project_run[0]), unquote(project_run[1]), parts[4])
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _post(self, form: dict[str, list[str]]) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path.startswith("/datasets/"):
            parts = path.split("/")
            name = unquote(parts[2])
            if len(parts) == 4 and parts[3] == "bulk":
                self._dataset_bulk(name, form)
                return
            if len(parts) == 4 and parts[3] == "tag":
                self._dataset_tag(name, form)
                return
            if len(parts) == 4 and parts[3] == "source-action":
                self._source_action(name, form)
                return
            if len(parts) == 4 and parts[3] == "delete":
                self._dataset_delete(name, form)
                return
        if path == "/configs/create":
            self._config_create(form)
            return
        if path.startswith("/configs/") and path.endswith("/save"):
            self._config_save(unquote(path.split("/")[2]), form)
            return
        if path == "/status/start":
            self._status_start(form)
            return
        if path.startswith("/status/") and path.endswith("/continue"):
            self._status_continue(unquote(path.split("/")[2]))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _home(self) -> None:
        datasets = self.app.datasets()
        configs = self.app.configs()
        states = self.app.states()
        status = _status_entries(states)
        results = _result_entries(states)
        active = sum(entry["status"] not in {"trained", "evaluated", "promoted", "complete"} for entry in status)
        body = f"""<div class="hero"><h1>LoRA 工作台</h1><div class="muted">浏览器只是现有 Dataset / Training Config / Run / Result 状态模型的前端。</div></div><div class="grid"><a class="card" href="/datasets"><div class="muted">数据集</div><div class="metric">{len(datasets)}</div><div>来源、图片、Tag、排除与删除</div></a><a class="card" href="/configs"><div class="muted">训练配置</div><div class="metric">{len(configs)}</div><div>底模、Trigger、LoRA 参数、预算</div></a><a class="card" href="/status"><div class="muted">活动 / 待处理训练</div><div class="metric">{active}</div><div>启动、查看状态、恢复</div></a><a class="card" href="/results"><div class="muted">训练结果</div><div class="metric">{len(results)}</div><div>权重、示例图、评测结果</div></a></div>"""
        self._html(_page("工作台", body))

    def _datasets(self) -> None:
        rows = []
        for workspace in self.app.datasets():
            summary = workspace.summary()
            rows.append(f"<tr><td><a href='/datasets/{_q(workspace.name)}'><b>{_e(workspace.name)}</b></a></td><td>{_e(workspace.concept_type)}</td><td>{summary['sources']}</td><td>{summary['active_images']}</td><td>{summary['excluded_images']}</td><td>{summary['captioned_active_images']}</td></tr>")
        body = "<div class='hero'><h1>数据集</h1><div class='muted'>可编辑的数据资产；训练启动时才冻结快照。</div></div>"
        body += "<table><tr><th>名称</th><th>类型</th><th>来源</th><th>可用图片</th><th>已排除</th><th>已有 Tag</th></tr>" + ("".join(rows) or "<tr><td colspan='6' class='muted'>还没有数据集，请先用 CLI 导入或创建。</td></tr>") + "</table>"
        self._html(_page("数据集", body, active="datasets"))

    def _dataset(self, name: str) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        summary = workspace.summary()
        source_rows = []
        for source_id, source in sorted(workspace.sources.items()):
            items = workspace.items(source_id=source_id, include_disabled=True, include_excluded=True)
            source_rows.append(f"<tr><td><a href='/datasets/{_q(name)}/source/{_q(source_id)}'><b>{_e(source.get('label') or source_id)}</b></a><br><span class='muted'>{_e(source_id)}</span></td><td>{_e(source.get('kind'))}</td><td>{'启用' if source.get('enabled', True) else '停用'}</td><td>{len(items)}</td><td><form method='post' action='/datasets/{_q(name)}/source-action'><input type='hidden' name='_csrf' value='{self.app.csrf}'><input type='hidden' name='source_id' value='{_e(source_id)}'><button name='action' value='toggle'>{'停用' if source.get('enabled', True) else '启用'}</button> <button class='danger' name='action' value='delete'>删除来源</button></form></td></tr>")
        body = f"""<div class="hero"><h1>{_e(name)}</h1><div class="muted">{_e(workspace.concept_type)} · {summary['sources']} 个来源 · {summary['active_images']} 张可训练图片 · {summary['excluded_images']} 张已排除</div></div><table><tr><th>来源</th><th>类型</th><th>状态</th><th>图片</th><th>操作</th></tr>{''.join(source_rows) or '<tr><td colspan=5 class=muted>暂无来源</td></tr>'}</table><div class="panel danger-zone" style="margin-top:18px"><h3 class="bad">危险操作</h3><p class="muted">删除 Dataset 只删除 datasets/{_e(name)} 中的副本，不删除原始导入素材，也不影响已经冻结的训练 Run。</p><form method="post" action="/datasets/{_q(name)}/delete"><input type="hidden" name="_csrf" value="{self.app.csrf}"><label>输入数据集名称确认</label><input name="confirm" autocomplete="off"><div class="toolbar"><button class="danger">永久删除整个数据集</button></div></form></div>"""
        self._html(_page(name, body, active="datasets"))

    def _source(self, name: str, source_id: str, query: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        if source_id not in workspace.sources:
            raise PipelineError(f"Unknown dataset source: {source_id}")
        items = workspace.items(source_id=source_id, include_disabled=True, include_excluded=True)
        page = max(1, int(query.get("page", ["1"])[0]))
        page_count = max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(page, page_count)
        start = (page - 1) * PAGE_SIZE
        current = items[start : start + PAGE_SIZE]
        cards: list[str] = []
        for item in current:
            media = f"/media/dataset/{_q(name)}/{_q(source_id)}/{quote(item.relative.as_posix())}"
            state = "已排除" if item.excluded else "保留"
            cards.append(f"""<div class="image-card"><img loading="lazy" src="{media}" alt="{_e(item.relative)}"><div class="image-body"><div class="image-title"><label><input style="width:auto" form="bulk" type="checkbox" name="keys" value="{_e(item.key)}"> {_e(item.relative.as_posix())}</label></div><span class="pill">{state}</span><form method="post" action="/datasets/{_q(name)}/tag" style="margin-top:8px"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="key" value="{_e(item.key)}"><textarea name="caption">{_e(workspace.caption_text(item.key))}</textarea><div class="toolbar"><button>保存 Tag</button></div></form></div></div>""")
        prev_link = f"<a class='button' href='?page={page-1}'>上一页</a>" if page > 1 else ""
        next_link = f"<a class='button' href='?page={page+1}'>下一页</a>" if page < page_count else ""
        source = workspace.sources[source_id]
        body = f"""<div class="hero"><h1>{_e(source.get('label') or source_id)}</h1><div class="muted">{_e(name)} / {_e(source_id)} · {len(items)} 张图片</div></div><form id="bulk" method="post" action="/datasets/{_q(name)}/bulk"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="source_id" value="{_e(source_id)}"><div class="toolbar"><button class="good" name="action" value="restore">恢复所选</button><button name="action" value="exclude">排除所选</button><button class="danger" name="action" value="delete">永久删除所选</button><span class="muted">永久删除会同时删除同名 .txt；日常清洗优先使用“排除”。</span></div></form><div class="images">{''.join(cards) or '<div class=muted>这个来源没有图片。</div>'}</div><div class="pager">{prev_link}<span class="pill">第 {page}/{page_count} 页</span>{next_link}</div>"""
        self._html(_page(f"{name} / {source_id}", body, active="datasets"))

    def _dataset_media(self, name: str, source_id: str, relative: str) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        root = workspace.source_images_dir(source_id)
        path = _safe_child(root, relative)
        self._file(path, inline=True)

    def _dataset_bulk(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        keys = form.get("keys", [])
        if not keys:
            raise PipelineError("请至少选择一张图片")
        action = form.get("action", [""])[0]
        if action == "exclude":
            workspace.exclude(keys, reason="web review")
        elif action == "restore":
            workspace.restore(keys)
        elif action == "delete":
            delete_dataset_items(workspace, keys)
        else:
            raise PipelineError("Unknown bulk action")
        source_id = form.get("source_id", [""])[0]
        self._redirect(f"/datasets/{_q(name)}/source/{_q(source_id)}")

    def _dataset_tag(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        key = form.get("key", [""])[0]
        workspace.replace_caption(key, form.get("caption", [""])[0])
        source_id = key.split("/", 1)[0]
        self._redirect(f"/datasets/{_q(name)}/source/{_q(source_id)}")

    def _source_action(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        source_id = form.get("source_id", [""])[0]
        action = form.get("action", [""])[0]
        if action == "toggle":
            source = workspace.sources[source_id]
            workspace.set_source_enabled(source_id, not bool(source.get("enabled", True)))
        elif action == "delete":
            delete_dataset_source(workspace, source_id)
        else:
            raise PipelineError("Unknown source action")
        self._redirect(f"/datasets/{_q(name)}")

    def _dataset_delete(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        if form.get("confirm", [""])[0] != name:
            raise PipelineError("数据集名称确认不匹配")
        delete_dataset_workspace(workspace)
        self._redirect("/datasets")

    def _configs(self) -> None:
        configs = self.app.configs()
        rows = []
        for config in configs:
            training = config.overrides.get("training", {})
            rows.append(f"<tr><td><a href='/configs/{_q(config.name)}'><b>{_e(config.name)}</b></a></td><td>{_e(config.concept_type)}</td><td>{_e(config.base)}</td><td>{_e(config.strategy)}</td><td>{_e(training.get('network_dim', '默认'))}</td><td>{config.images_seen}</td></tr>")
        bases = [(key, value) for key, value in load_base_registry(self.app.root).items() if value.enabled]
        options = "".join(f"<option value='{_e(key)}'>{_e(key)} · {_e(value.name)}</option>" for key, value in bases)
        body = "<div class='hero'><h1>训练配置</h1><div class='muted'>可复用的训练 recipe；启动 Run 时冻结快照。</div></div>"
        body += "<table><tr><th>名称</th><th>类型</th><th>底模</th><th>策略</th><th>Rank</th><th>images_seen</th></tr>" + ("".join(rows) or "<tr><td colspan=6 class=muted>暂无训练配置</td></tr>") + "</table>"
        body += f"""<div class="panel" style="margin-top:18px"><h3>创建训练配置</h3><form method="post" action="/configs/create"><input type="hidden" name="_csrf" value="{self.app.csrf}"><div class="row"><label>名称<input name="name" required></label><label>类型<select name="concept_type"><option value="character">character</option><option value="style">style</option></select></label><label>底模<select name="base">{options}</select></label><label>Trigger<input name="trigger" required></label><label>策略<select name="strategy"><option>quality</option><option>fast</option><option>cached</option></select></label><label>images_seen<input type="number" min="1" name="images_seen" value="1000"></label><label>Rank（留空=默认）<input type="number" min="1" name="network_dim"></label><label>Alpha（留空=默认）<input type="number" min="1" name="network_alpha"></label><label>UNet LR（留空=默认）<input name="unet_lr"></label></div><div class="toolbar"><button class="good">创建</button></div></form></div>"""
        self._html(_page("训练配置", body, active="configs"))

    def _config(self, name: str) -> None:
        config = TrainingConfig.load(name, root=self.app.root)
        training = config.overrides.get("training", {})
        bases = [(key, value) for key, value in load_base_registry(self.app.root).items() if value.enabled]
        options = "".join(f"<option value='{_e(key)}' {'selected' if key == config.base else ''}>{_e(key)} · {_e(value.name)}</option>" for key, value in bases)
        body = f"""<div class="hero"><h1>{_e(name)}</h1><div class="muted">Config snapshot {_e(config.snapshot()['snapshot_hash'][:16])}</div></div><div class="panel"><form method="post" action="/configs/{_q(name)}/save"><input type="hidden" name="_csrf" value="{self.app.csrf}"><div class="row"><label>底模<select name="base">{options}</select></label><label>Trigger<input name="trigger" value="{_e(config.trigger)}"></label><label>策略<select name="strategy"><option {'selected' if config.strategy == 'quality' else ''}>quality</option><option {'selected' if config.strategy == 'fast' else ''}>fast</option><option {'selected' if config.strategy == 'cached' else ''}>cached</option></select></label><label>images_seen<input type="number" min="1" name="images_seen" value="{config.images_seen}"></label><label>Rank<input type="number" min="1" name="network_dim" value="{_e(training.get('network_dim',''))}"></label><label>Alpha<input type="number" min="1" name="network_alpha" value="{_e(training.get('network_alpha',''))}"></label><label>UNet LR<input name="unet_lr" value="{_e(training.get('unet_lr',''))}"></label></div><div class="toolbar"><button class="good">保存</button></div></form><p class="muted">工作流高级开关仍可在 CLI 调整；Web v1 先覆盖最常改的核心参数。</p></div>"""
        self._html(_page(name, body, active="configs"))

    def _config_create(self, form: dict[str, list[str]]) -> None:
        overrides = self._training_overrides(form)
        TrainingConfig.create(
            form["name"][0].strip(),
            concept_type=form.get("concept_type", ["character"])[0],
            base=form["base"][0],
            trigger=form["trigger"][0].strip(),
            strategy=form.get("strategy", ["quality"])[0],
            images_seen=int(form.get("images_seen", ["1000"])[0]),
            overrides=overrides,
            root=self.app.root,
        )
        self._redirect(f"/configs/{_q(form['name'][0].strip())}")

    def _config_save(self, name: str, form: dict[str, list[str]]) -> None:
        config = TrainingConfig.load(name, root=self.app.root)
        config.data["base"] = form["base"][0]
        config.data["trigger"] = form["trigger"][0].strip()
        config.data["strategy"] = form["strategy"][0]
        config.data["images_seen"] = int(form["images_seen"][0])
        config.data["overrides"] = self._training_overrides(form)
        config.validate(require_enabled_base=True, root=self.app.root)
        config.save()
        self._redirect(f"/configs/{_q(name)}")

    def _training_overrides(self, form: dict[str, list[str]]) -> dict[str, Any]:
        training: dict[str, Any] = {}
        for key in ("network_dim", "network_alpha"):
            raw = form.get(key, [""])[0].strip()
            if raw:
                training[key] = int(raw)
        raw_lr = form.get("unet_lr", [""])[0].strip()
        if raw_lr:
            training["unet_lr"] = float(raw_lr)
        return {"training": training} if training else {}

    def _status(self) -> None:
        entries = _status_entries(self.app.states())
        rows = []
        for entry in entries:
            rows.append(f"<tr><td>{_e(entry['dataset'])}</td><td>{_e(entry['config'])}</td><td><a href='/status/{_q(entry['project'])}'>{_e(entry.get('run_id') or 'pending')}</a></td><td>{_e(entry['status'])}</td><td>{_e(entry.get('updated') or '')}</td></tr>")
        datasets = self.app.datasets()
        configs = self.app.configs()
        dataset_options = "".join(f"<option value='{_e(item.name)}'>{_e(item.name)} · {_e(item.concept_type)}</option>" for item in datasets)
        config_options = "".join(f"<option value='{_e(item.name)}'>{_e(item.name)} · {_e(item.concept_type)} · {_e(item.base)}</option>" for item in configs)
        body = "<div class='hero'><h1>训练状态</h1><div class='muted'>创建 Run 后由独立子进程执行；关闭浏览器不会停止训练。</div></div>"
        body += "<table><tr><th>数据集</th><th>配置</th><th>Run</th><th>状态</th><th>更新时间</th></tr>" + ("".join(rows) or "<tr><td colspan=5 class=muted>暂无训练记录</td></tr>") + "</table>"
        body += f"""<div class="panel" style="margin-top:18px"><h3>开始一次新训练</h3><form method="post" action="/status/start"><input type="hidden" name="_csrf" value="{self.app.csrf}"><div class="row"><label>Dataset<select name="dataset">{dataset_options}</select></label><label>Training Config<select name="config">{config_options}</select></label></div><label style="display:block;margin-top:10px"><input style="width:auto" type="checkbox" name="safe_exclude" value="1" checked> 启动前自动排除损坏文件与完全重复副本</label><div class="toolbar"><button class="good">冻结快照并开始训练</button></div></form></div>"""
        self._html(_page("训练状态", body, active="status"))

    def _status_start(self, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(form["dataset"][0], root=self.app.root)
        config = TrainingConfig.load(form["config"][0], root=self.app.root)
        if workspace.concept_type != config.concept_type:
            raise PipelineError("Dataset 与 Training Config 类型不兼容")
        if form.get("safe_exclude", [""])[0] == "1":
            workspace.apply_safe_audit_exclusions()
            workspace = DatasetWorkspace.load(workspace.name, root=self.app.root)
        config.validate(require_enabled_base=True, root=self.app.root)
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        project_name = make_training_workspace_name(workspace.name, config.name, timestamp=timestamp)
        suffix = 1
        original = project_name
        while project_path(project_name, root=self.app.root).exists():
            tail = f"-{suffix}"
            project_name = (original[: 64 - len(tail)] + tail).rstrip("-._")
            suffix += 1
        state = create_project_from_training_config(workspace, config, project_name=project_name, root=self.app.root)
        spawn_training_worker(state)
        self._redirect(f"/status/{_q(state.name)}")

    def _status_detail(self, project_name: str) -> None:
        state = load_project(project_name, root=self.app.root)
        project = state.payload["project"]
        identity = project.get("training_identity", {})
        steps = "".join(f"<tr><td>{_e(name)}</td><td>{_e(record.get('status'))}</td><td>{_e(record.get('attempts',0))}</td><td>{_e(record.get('last_error',''))}</td></tr>" for name, record in state.payload.get("steps", {}).items())
        log_path = state.project_dir / "web-worker.log"
        log_tail = ""
        if log_path.is_file():
            data = log_path.read_text(encoding="utf-8", errors="replace")
            log_tail = data[-8000:]
        run = state.payload.get("runs", [])[-1] if state.payload.get("runs") else None
        status = str(run.get("status")) if run else _workspace_status(state)
        body = f"""<div class="hero"><h1>{_e(identity.get('dataset') or project_name)} / {_e(identity.get('config') or 'legacy')}</h1><div class="muted">技术工作区 {_e(project_name)} · 当前 {_e(status)}</div></div><div class="grid"><div class="card"><div class="muted">Dataset snapshot</div><div class="mono">{_e(identity.get('dataset_snapshot_hash',''))}</div></div><div class="card"><div class="muted">Config snapshot</div><div class="mono">{_e(identity.get('config_snapshot_hash',''))}</div></div><div class="card"><div class="muted">当前内部步骤</div><div class="metric" style="font-size:20px">{_e(state.next_actionable_step() or 'complete')}</div></div></div><div class="toolbar"><form method="post" action="/status/{_q(project_name)}/continue"><input type="hidden" name="_csrf" value="{self.app.csrf}"><button class="good">继续 / 恢复训练</button></form></div><table><tr><th>步骤</th><th>状态</th><th>尝试</th><th>错误</th></tr>{steps}</table><div class="panel" style="margin-top:18px"><h3>worker log</h3><div class="mono">{_e(log_tail or '暂无日志')}</div></div>"""
        self._html(_page(project_name, body, active="status"))

    def _status_continue(self, project_name: str) -> None:
        state = load_project(project_name, root=self.app.root)
        spawn_training_worker(state)
        self._redirect(f"/status/{_q(project_name)}")

    def _results(self) -> None:
        entries = _result_entries(self.app.states())
        rows = []
        for entry in entries:
            rows.append(f"<tr><td>{_e(entry['dataset'])}</td><td>{_e(entry['config'])}</td><td><a href='/results/{_q(entry['project'])}/{_q(entry['run_id'])}'>{_e(entry['run_id'])}</a></td><td>{_e(entry['status'])}</td><td>{entry['checkpoints']}</td><td>{entry['samples']}</td><td>{'★' if entry['promoted'] else ''}</td></tr>")
        body = "<div class='hero'><h1>训练结果</h1><div class='muted'>权重、示例图片与已经生成的评测产物。</div></div><table><tr><th>数据集</th><th>配置</th><th>Run</th><th>状态</th><th>权重</th><th>示例图</th><th>Best</th></tr>" + ("".join(rows) or "<tr><td colspan=7 class=muted>还没有完成的训练结果。</td></tr>") + "</table>"
        self._html(_page("训练结果", body, active="results"))

    def _result_detail(self, project_name: str, run_id: str) -> None:
        state = load_project(project_name, root=self.app.root)
        run = _find_run(state, run_id)
        if run is None:
            raise PipelineError("Run does not exist")
        run_dir = Path(str(run.get("path") or ""))
        checkpoints = [Path(value) for value in run.get("checkpoints", []) if Path(value).is_file()]
        weight_links = "".join(f"<li><a href='/run-file/{_q(project_name)}/{_q(run_id)}/{quote(path.relative_to(run_dir).as_posix())}'>{_e(path.name)}</a></li>" for path in checkpoints if run_dir in path.parents)
        images: list[Path] = []
        for folder in (run_dir / "samples", run_dir / "contact-sheets"):
            if folder.is_dir():
                images.extend(path for path in sorted(folder.rglob("*")) if path.suffix.lower() in _IMAGE_SUFFIXES)
        image_cards = "".join(f"<div class='image-card'><a href='/run-file/{_q(project_name)}/{_q(run_id)}/{quote(path.relative_to(run_dir).as_posix())}'><img loading='lazy' src='/run-file/{_q(project_name)}/{_q(run_id)}/{quote(path.relative_to(run_dir).as_posix())}'></a><div class='image-body image-title'>{_e(path.relative_to(run_dir))}</div></div>" for path in images[:120])
        promotion = run.get("promotion", {}) if isinstance(run.get("promotion"), dict) else {}
        body = f"""<div class="hero"><h1>Run {_e(run_id)}</h1><div class="muted">{_e(run.get('status'))} · {_e(run_dir)}</div></div><div class="grid"><div class="card"><div class="muted">权重</div><div class="metric">{len(checkpoints)}</div></div><div class="card"><div class="muted">示例 / 对比图</div><div class="metric">{len(images)}</div></div><div class="card"><div class="muted">Best</div><div>{_e(promotion.get('checkpoint') or '—')}</div></div></div><div class="panel" style="margin-top:18px"><h3>权重文件</h3><ul>{weight_links or '<li class=muted>暂无</li>'}</ul></div><div class="images" style="margin-top:18px">{image_cards}</div>"""
        self._html(_page(f"Run {run_id}", body, active="results"))

    def _run_file(self, project_name: str, run_id: str, relative: str) -> None:
        state = load_project(project_name, root=self.app.root)
        run = _find_run(state, run_id)
        if run is None:
            raise PipelineError("Run does not exist")
        root = Path(str(run.get("path") or ""))
        path = _safe_child(root, relative)
        self._file(path, inline=path.suffix.lower() in _IMAGE_SUFFIXES or path.suffix.lower() in {".html", ".txt", ".json"})

    def _form(self) -> dict[str, list[str]]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise PipelineError("Form payload is too large")
        raw = self.rfile.read(length).decode("utf-8", errors="strict")
        return parse_qs(raw, keep_blank_values=True)

    def _html(self, text: str, *, status: int = 200) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def _text(self, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, *, inline: bool) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Disposition", ("inline" if inline else "attachment") + f'; filename="{path.name.replace(chr(34), "")}"')
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


def make_server(host: str = "127.0.0.1", port: int = 7860, *, root: Path | None = None) -> ThreadingHTTPServer:
    app = WebApplication(root=root)
    handler = type("BoundHandler", (Handler,), {"app": app})
    return ThreadingHTTPServer((host, int(port)), handler)


def serve(host: str = "127.0.0.1", port: int = 7860, *, allow_lan: bool = False, root: Path | None = None) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_lan:
        raise PipelineError("Refusing non-loopback bind without --allow-lan; prefer an SSH tunnel")
    server = make_server(host, port, root=root)
    print(f"LoRA Pipeline Web: http://{host}:{port}")
    if host in {"127.0.0.1", "localhost", "::1"}:
        print(f"Remote browser: ssh -L {port}:127.0.0.1:{port} <nas>  then open http://127.0.0.1:{port}")
    else:
        print("WARNING: LAN mode has no login layer yet. Use only on a trusted private network.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LoRA Pipeline lightweight NAS web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--allow-lan", action="store_true")
    args = parser.parse_args(argv)
    serve(args.host, args.port, allow_lan=args.allow_lan)


if __name__ == "__main__":
    main()
