from __future__ import annotations

import argparse
import hmac
import html
import mimetypes
import os
import secrets
import sys
from datetime import UTC, datetime
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote, unquote, urlparse

from .config import load_base_registry, repository_root
from .dataset_deletion import delete_dataset_items, delete_dataset_source, delete_dataset_workspace
from .dataset_workspace import DatasetWorkspace, list_datasets
from .models import PipelineError
from .service import load_project, project_path
from .state import ProjectState, project_lock
from .steps import promote
from .training_config import (
    TrainingConfig,
    create_project_from_training_config,
    list_training_configs,
    make_training_workspace_name,
)
from .web_jobs import (
    active_gpu_jobs,
    job_data_dir,
    jobs_for_project,
    read_job,
    resume_job,
    spawn_job,
    tail_job_log,
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
    result: list[ProjectState] = []
    for path in sorted(base.iterdir(), key=lambda item: item.name.casefold()):
        if (path / "project.yaml").is_file():
            result.append(ProjectState.load(path))
    return result


def _workspace_status(state: ProjectState) -> str:
    next_step = state.next_actionable_step()
    if next_step is None:
        return "complete"
    return f"{state.status(next_step).value}:{next_step}"


def _status_entries(states: Iterable[ProjectState], *, root: Path | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for state in states:
        project = state.payload["project"]
        identity = project.get("training_identity", {})
        jobs = jobs_for_project(state.name, root=root)
        worker = jobs[0] if jobs else None
        runs = list(state.payload.get("runs", []))
        if runs:
            for run in runs:
                status = str(run.get("status", "unknown"))
                if worker and worker.get("status") in {"queued", "starting", "running"}:
                    status = f"web:{worker.get('kind')}"
                entries.append(
                    {
                        "project": state.name,
                        "run_id": str(run.get("id") or ""),
                        "status": status,
                        "dataset": identity.get("dataset") or project.get("dataset_snapshot", {}).get("dataset") or "legacy",
                        "config": identity.get("config") or "legacy",
                        "updated": run.get("finished_at") or run.get("interrupted_at") or run.get("started_at"),
                    }
                )
        else:
            status = _workspace_status(state)
            if worker and worker.get("status") in {"queued", "starting", "running"}:
                status = f"web:{worker.get('kind')}"
            entries.append(
                {
                    "project": state.name,
                    "run_id": "",
                    "status": status,
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
            samples = _images_under(run_dir / "samples")
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
        if str(run.get("id")) == str(run_id):
            return run
    return None


def _images_under(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [
        item
        for item in sorted(path.rglob("*"), key=lambda value: value.as_posix().casefold())
        if item.is_file() and item.suffix.casefold() in _IMAGE_SUFFIXES
    ]


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
*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#080d19,#0d1425 48%,#101932);color:var(--text);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.shell{max-width:1480px;margin:auto;padding:22px}.top{display:flex;gap:18px;align-items:center;justify-content:space-between;margin-bottom:20px}.brand{font-size:22px;font-weight:800}.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a,.button,button{border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:8px 12px;border-radius:9px;cursor:pointer}.nav a.active{border-color:var(--accent);color:#fff}.button.danger,button.danger{border-color:#733742;background:#3b1d27;color:#ffd9dc}.button.good,button.good{border-color:#356c4b;background:#173728}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.card,.panel{background:rgba(18,26,45,.94);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 12px 30px #0004}.metric{font-size:30px;font-weight:800}.muted{color:var(--muted)}.bad{color:var(--bad)}.good{color:var(--good)}.warn{color:var(--warn)}table{width:100%;border-collapse:collapse;background:rgba(18,26,45,.94);border-radius:12px;overflow:hidden}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}.images{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}.image-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}.image-card img{width:100%;height:220px;object-fit:contain;background:#070b14;display:block}.image-body{padding:10px}.image-title{word-break:break-all;font-size:13px;margin-bottom:6px}textarea,input,select{width:100%;background:#0d1424;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:8px}textarea{min-height:72px;resize:vertical}.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.pill{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:99px;color:var(--muted);font-size:12px}.flash{padding:10px 12px;border:1px solid #2e6845;background:#153524;border-radius:10px;margin-bottom:14px}.error{padding:10px 12px;border:1px solid #743743;background:#3a1c25;border-radius:10px;margin-bottom:14px}.danger-zone{border-color:#69313b}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;word-break:break-word}.pager{display:flex;gap:8px;justify-content:center;margin:16px}.hero{margin:12px 0 20px}.hero h1{margin:0 0 4px;font-size:28px}.compact{font-size:13px}.cluster{border:1px solid var(--line);border-radius:13px;padding:12px;background:#0c1324}.cluster-images{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.cluster-images img{width:100%;height:170px;object-fit:contain;background:#050912;border-radius:8px}@media(max-width:720px){.shell{padding:12px}.top{align-items:flex-start;flex-direction:column}.image-card img{height:180px}.cluster-images img{height:120px}}
"""


def _page(
    title: str,
    body: str,
    *,
    active: str = "",
    flash: str = "",
    error: str = "",
    refresh: int | None = None,
) -> str:
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
    refresh_tag = f'<meta http-equiv="refresh" content="{int(refresh)}">' if refresh else ""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh_tag}<title>{_e(title)} · LoRA Pipeline</title><style>{_css()}</style></head><body><div class="shell"><div class="top"><a class="brand" href="/">LoRA Pipeline</a><nav class="nav">{links}</nav></div>{notice}{problem}{body}</div></body></html>"""


class WebApplication:
    def __init__(self, *, root: Path | None = None, auth_token: str | None = None):
        self.root = (root or repository_root()).resolve()
        self.csrf = secrets.token_urlsafe(24)
        self.auth_token = auth_token or None

    def datasets(self) -> list[DatasetWorkspace]:
        return list_datasets(root=self.root)

    def configs(self) -> list[TrainingConfig]:
        return list_training_configs(root=self.root)

    def states(self) -> list[ProjectState]:
        return _all_project_states(self.root)


class Handler(BaseHTTPRequestHandler):
    server_version = "LoRAPipelineWeb/2"
    app: WebApplication

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[web] " + fmt % args + "\n")

    def do_GET(self) -> None:  # noqa: N802
        try:
            if not self._auth_ok() and urlparse(self.path).path not in {"/login", "/healthz"}:
                self._redirect("/login")
                return
            self._get()
        except (PipelineError, OSError, ValueError, KeyError) as exc:
            self._html(_page("错误", '<div class="hero"><h1>请求失败</h1></div>', error=str(exc)), status=400)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"
            form = self._form()
            if path == "/login":
                self._login(form)
                return
            if not self._auth_ok():
                raise PipelineError("authentication required")
            origin = self.headers.get("Origin")
            host = self.headers.get("Host")
            if origin and host and urlparse(origin).netloc != host:
                raise PipelineError("cross-origin state change rejected")
            if form.get("_csrf", [""])[0] != self.app.csrf:
                raise PipelineError("CSRF validation failed; refresh the page and retry")
            self._post(form)
        except (PipelineError, OSError, ValueError, KeyError) as exc:
            self._html(_page("错误", '<div class="hero"><h1>操作失败</h1></div>', error=str(exc)), status=400)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        if path == "/healthz":
            self._text("ok\n")
            return
        if path == "/login":
            self._login_page()
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
                self._run_file(unquote(parts[2]), unquote(parts[3]), parts[4])
                return
        if path.startswith("/jobs/"):
            self._job(unquote(path.split("/", 2)[2]))
            return
        if path.startswith("/media/job/"):
            parts = path.split("/", 4)
            if len(parts) == 5:
                self._job_media(unquote(parts[3]), parts[4])
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _post(self, form: dict[str, list[str]]) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/datasets/create":
            self._dataset_create(form)
            return
        if path.startswith("/datasets/"):
            parts = path.split("/")
            name = unquote(parts[2])
            if len(parts) == 4 and parts[3] == "import-images":
                self._dataset_import_images(name, form)
                return
            if len(parts) == 4 and parts[3] == "import-video":
                self._dataset_import_video(name, form)
                return
            if len(parts) == 4 and parts[3] == "bulk":
                self._dataset_bulk(name, form)
                return
            if len(parts) == 4 and parts[3] == "tag":
                self._dataset_tag(name, form)
                return
            if len(parts) == 4 and parts[3] == "auto-tag":
                self._dataset_auto_tag(name, form)
                return
            if len(parts) == 4 and parts[3] == "audit":
                self._dataset_audit(name, form)
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
        if path.startswith("/configs/") and path.endswith("/delete"):
            self._config_delete(unquote(path.split("/")[2]), form)
            return
        if path == "/status/start":
            self._status_start(form)
            return
        if path.startswith("/status/") and path.endswith("/continue"):
            self._status_continue(unquote(path.split("/")[2]))
            return
        if path.startswith("/results/") and path.endswith("/evaluate"):
            parts = path.split("/")
            self._result_evaluate(unquote(parts[2]), unquote(parts[3]), form)
            return
        if path.startswith("/results/") and path.endswith("/promote"):
            parts = path.split("/")
            self._result_promote(unquote(parts[2]), unquote(parts[3]), form)
            return
        if path.startswith("/jobs/") and path.endswith("/identity"):
            self._job_identity(unquote(path.split("/")[2]), form)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    # ------------------------------------------------------------------
    # Authentication / home
    # ------------------------------------------------------------------
    def _auth_ok(self) -> bool:
        token = self.app.auth_token
        if not token:
            return True
        raw = self.headers.get("Cookie") or ""
        cookie = SimpleCookie()
        try:
            cookie.load(raw)
        except Exception:
            return False
        morsel = cookie.get("lora_web_token")
        supplied = morsel.value if morsel else ""
        return hmac.compare_digest(supplied, token)

    def _login_page(self, *, error: str = "") -> None:
        if not self.app.auth_token:
            self._redirect("/")
            return
        body = f"""<div class="hero"><h1>LoRA Pipeline Web</h1><div class="muted">请输入 Web access token。</div></div>{f'<div class="error">{_e(error)}</div>' if error else ''}<div class="panel" style="max-width:520px"><form method="post" action="/login"><label>Access token<input type="password" name="token" autofocus required></label><div class="toolbar"><button class="good">登录</button></div></form></div>"""
        self._html(_page("登录", body))

    def _login(self, form: dict[str, list[str]]) -> None:
        token = self.app.auth_token or ""
        supplied = form.get("token", [""])[0]
        if not token or not hmac.compare_digest(supplied, token):
            self._login_page(error="Access token 不正确")
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header("Set-Cookie", f"lora_web_token={supplied}; Path=/; HttpOnly; SameSite=Strict")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _home(self) -> None:
        datasets = self.app.datasets()
        configs = self.app.configs()
        states = self.app.states()
        status = _status_entries(states, root=self.app.root)
        results = _result_entries(states)
        active = sum(entry["status"] not in {"trained", "evaluated", "promoted", "complete"} for entry in status)
        gpu = active_gpu_jobs(root=self.app.root)
        gpu_note = ""
        if gpu:
            current = gpu[0]
            gpu_note = f"<div class='panel' style='margin-top:16px'><b>GPU 当前任务</b> · <a href='/jobs/{_e(current['id'])}'>{_e(current['kind'])} / {_e(current['id'])}</a></div>"
        body = f"""<div class="hero"><h1>LoRA 工作台</h1><div class="muted">Dataset / Training Config / Training Status / Results 共用同一份后端状态。</div></div><div class="grid"><a class="card" href="/datasets"><div class="muted">数据集</div><div class="metric">{len(datasets)}</div><div>来源、图片、Tag、排除、视频人物选择</div></a><a class="card" href="/configs"><div class="muted">训练配置</div><div class="metric">{len(configs)}</div><div>底模、Trigger、LoRA 参数、工作流偏好</div></a><a class="card" href="/status"><div class="muted">活动 / 待处理训练</div><div class="metric">{active}</div><div>启动、查看日志、恢复</div></a><a class="card" href="/results"><div class="muted">训练结果</div><div class="metric">{len(results)}</div><div>权重、示例图、评测、Best</div></a></div>{gpu_note}"""
        self._html(_page("工作台", body))

    # ------------------------------------------------------------------
    # Dataset pages / actions
    # ------------------------------------------------------------------
    def _datasets(self) -> None:
        rows = []
        for workspace in self.app.datasets():
            summary = workspace.summary()
            rows.append(
                f"<tr><td><a href='/datasets/{_q(workspace.name)}'><b>{_e(workspace.name)}</b></a></td><td>{_e(workspace.concept_type)}</td><td>{summary['sources']}</td><td>{summary['active_images']}</td><td>{summary['excluded_images']}</td><td>{summary['captioned_active_images']}</td></tr>"
            )
        body = "<div class='hero'><h1>数据集</h1><div class='muted'>长期维护的可编辑数据资产；每次训练只读取冻结快照。</div></div>"
        body += "<table><tr><th>名称</th><th>类型</th><th>来源</th><th>可用图片</th><th>已排除</th><th>已有 Tag</th></tr>" + ("".join(rows) or "<tr><td colspan='6' class='muted'>还没有数据集。</td></tr>") + "</table>"
        body += f"""<div class="panel" style="margin-top:18px"><h3>创建数据集</h3><form method="post" action="/datasets/create"><input type="hidden" name="_csrf" value="{self.app.csrf}"><div class="row"><label>名称<input name="name" required></label><label>类型<select name="concept_type"><option value="character">character</option><option value="style">style</option></select></label></div><div class="toolbar"><button class="good">创建</button></div></form></div>"""
        self._html(_page("数据集", body, active="datasets"))

    def _dataset_create(self, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.create(
            form.get("name", [""])[0].strip(),
            concept_type=form.get("concept_type", ["character"])[0],
            root=self.app.root,
        )
        self._redirect(f"/datasets/{_q(workspace.name)}")

    def _dataset(self, name: str) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        summary = workspace.summary()
        source_rows = []
        for source_id, source in sorted(workspace.sources.items()):
            items = workspace.items(source_id=source_id, include_disabled=True, include_excluded=True)
            source_rows.append(
                f"<tr><td><a href='/datasets/{_q(name)}/source/{_q(source_id)}'><b>{_e(source.get('label') or source_id)}</b></a><br><span class='muted compact'>{_e(source_id)}</span></td><td>{_e(source.get('kind'))}</td><td>{'启用' if source.get('enabled', True) else '停用'}</td><td>{len(items)}</td><td>{sum(not item.excluded for item in items)}</td></tr>"
            )
        video_form = ""
        if workspace.concept_type == "character":
            video_form = f"""<div class="panel"><h3>导入视频 / YouTube</h3><p class="muted">直接填写 NAS 本地视频路径或 URL。4K/HDR、清晰帧择优、DeepGHS 人物裁切和 CCIP 都沿用现有管线；处理后网页会让你选择人物簇。</p><form method="post" action="/datasets/{_q(name)}/import-video"><input type="hidden" name="_csrf" value="{self.app.csrf}"><label>本地路径或 URL<input name="source" required placeholder="/volume1/video/source.mkv 或 https://..."></label><div class="row"><label>来源名称<input name="label" placeholder="video-a"></label><label>每 N 秒采样<input type="number" min="1" name="interval_seconds" value="2"></label><label>最多保留帧<input type="number" min="1" name="max_frames" value="250"></label><label>网络<select name="proxy_mode"><option value="environment">environment</option><option value="direct">direct</option><option value="custom">custom</option></select></label><label>自定义代理（可空）<input name="proxy_url" placeholder="http://127.0.0.1:7890"></label><label>cookies.txt（可空/自动发现）<input name="cookies_path"></label></div><div class="toolbar"><button class="good">开始视频处理</button></div></form></div>"""
        body = f"""<div class="hero"><h1>{_e(name)}</h1><div class="muted">{_e(workspace.concept_type)} · {summary['sources']} 个来源 · {summary['active_images']} 张可训练图片 · {summary['excluded_images']} 张已排除</div></div><table><tr><th>来源</th><th>类型</th><th>状态</th><th>总图片</th><th>可用</th></tr>{''.join(source_rows) or '<tr><td colspan=5 class=muted>暂无来源</td></tr>'}</table><div class="grid" style="margin-top:18px"><div class="panel"><h3>导入图片目录</h3><p class="muted">输入 NAS 上已有图片目录；会复制图片和同名 .txt 到独立 Source。</p><form method="post" action="/datasets/{_q(name)}/import-images"><input type="hidden" name="_csrf" value="{self.app.csrf}"><label>目录路径<input name="directory" required></label><label>来源名称<input name="label"></label><div class="toolbar"><button class="good">导入</button></div></form></div><div class="panel"><h3>自动检查 / Tag</h3><form method="post" action="/datasets/{_q(name)}/audit"><input type="hidden" name="_csrf" value="{self.app.csrf}"><label><input style="width:auto" type="checkbox" name="apply_safe" value="1" checked> 自动排除损坏文件与完全重复副本</label><div class="toolbar"><button>检查整个数据集</button></div></form><form method="post" action="/datasets/{_q(name)}/auto-tag"><input type="hidden" name="_csrf" value="{self.app.csrf}"><div class="row"><label>Tag 阈值<input name="threshold" value="0.35"></label><label><input style="width:auto" type="checkbox" name="overwrite" value="1"> 覆盖已有 Tag</label></div><div class="toolbar"><button>自动打 Tag</button></div></form></div>{video_form}</div><div class="panel danger-zone" style="margin-top:18px"><h3 class="bad">危险操作</h3><p class="muted">只删除 Dataset 工作区副本；原始导入文件、已冻结 Run、权重与结果不会被删除。</p><form method="post" action="/datasets/{_q(name)}/delete"><input type="hidden" name="_csrf" value="{self.app.csrf}"><label>输入数据集名称确认<input name="confirm" autocomplete="off"></label><div class="toolbar"><button class="danger">永久删除整个数据集</button></div></form></div>"""
        self._html(_page(name, body, active="datasets"))

    def _dataset_import_images(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        directory = Path(form.get("directory", [""])[0]).expanduser().resolve()
        label = form.get("label", [""])[0].strip() or directory.name or "images"
        record = workspace.add_source_from_directory(
            directory,
            kind="image_directory",
            label=label,
            origin=str(directory),
        )
        self._redirect(f"/datasets/{_q(name)}/source/{_q(str(record['id']))}")

    def _dataset_import_video(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        if workspace.concept_type != "character":
            raise PipelineError("视频人物导入当前只用于 Character 数据集")
        source = form.get("source", [""])[0].strip()
        if not source:
            raise PipelineError("视频路径或 URL 不能为空")
        payload = {
            "dataset": name,
            "source": source,
            "label": form.get("label", [""])[0].strip() or Path(source).stem or "video",
            "interval_seconds": _positive_int(form.get("interval_seconds", ["2"])[0], "interval_seconds"),
            "max_frames": _positive_int(form.get("max_frames", ["250"])[0], "max_frames"),
            "proxy_mode": form.get("proxy_mode", ["environment"])[0],
            "proxy_url": form.get("proxy_url", [""])[0].strip(),
            "cookies_path": form.get("cookies_path", [""])[0].strip(),
        }
        job = spawn_job("video_prepare", payload, root=self.app.root)
        self._redirect(f"/jobs/{_q(str(job['id']))}")

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
            cards.append(
                f"""<div class="image-card"><img loading="lazy" src="{media}" alt="{_e(item.relative)}"><div class="image-body"><div class="image-title"><label><input style="width:auto" form="bulk" type="checkbox" name="keys" value="{_e(item.key)}"> {_e(item.relative.as_posix())}</label></div><span class="pill">{state}</span><form method="post" action="/datasets/{_q(name)}/tag" style="margin-top:8px"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="key" value="{_e(item.key)}"><textarea name="caption">{_e(workspace.caption_text(item.key))}</textarea><div class="toolbar"><button>保存 Tag</button></div></form></div></div>"""
            )
        prev_link = f"<a class='button' href='?page={page-1}'>上一页</a>" if page > 1 else ""
        next_link = f"<a class='button' href='?page={page+1}'>下一页</a>" if page < page_count else ""
        source = workspace.sources[source_id]
        body = f"""<div class="hero"><h1>{_e(source.get('label') or source_id)}</h1><div class="muted">{_e(name)} / {_e(source_id)} · {len(items)} 张图片 · {'启用' if source.get('enabled', True) else '停用'}</div></div><div class="toolbar"><form method="post" action="/datasets/{_q(name)}/source-action"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="source_id" value="{_e(source_id)}"><button name="action" value="toggle">{'停用来源' if source.get('enabled', True) else '启用来源'}</button></form><form method="post" action="/datasets/{_q(name)}/audit"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="source_id" value="{_e(source_id)}"><input type="hidden" name="apply_safe" value="1"><button>检查来源</button></form><form method="post" action="/datasets/{_q(name)}/auto-tag"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="source_id" value="{_e(source_id)}"><input type="hidden" name="threshold" value="0.35"><button>自动 Tag 此来源</button></form></div><form id="bulk" method="post" action="/datasets/{_q(name)}/bulk"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="source_id" value="{_e(source_id)}"><div class="toolbar"><button class="good" name="action" value="restore">恢复所选</button><button name="action" value="exclude">排除所选</button><button class="danger" name="action" value="delete">永久删除所选</button><label style="max-width:240px">永久删除确认<input name="confirm_delete" placeholder="输入 DELETE"></label></div></form><div class="images">{''.join(cards) or '<div class=muted>这个来源没有图片。</div>'}</div><div class="pager">{prev_link}<span class="pill">第 {page}/{page_count} 页</span>{next_link}</div><div class="panel danger-zone"><h3 class="bad">删除整个来源</h3><p class="muted">派生 Source 不会级联删除；这里只删除当前 Source 在 Dataset 里的副本。</p><form method="post" action="/datasets/{_q(name)}/source-action"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="source_id" value="{_e(source_id)}"><label>输入 Source ID 确认<input name="confirm"></label><div class="toolbar"><button class="danger" name="action" value="delete">永久删除来源</button></div></form></div>"""
        self._html(_page(f"{name} / {source_id}", body, active="datasets"))

    def _dataset_media(self, name: str, source_id: str, relative: str) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        root = workspace.source_images_dir(source_id)
        path = _safe_child(root, relative)
        if path.suffix.casefold() not in _IMAGE_SUFFIXES:
            raise PipelineError("Web media endpoint only serves images")
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
            if form.get("confirm_delete", [""])[0] != "DELETE":
                raise PipelineError("永久删除图片需要输入 DELETE")
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

    def _dataset_auto_tag(self, name: str, form: dict[str, list[str]]) -> None:
        DatasetWorkspace.load(name, root=self.app.root)
        job = spawn_job(
            "dataset_tag",
            {
                "dataset": name,
                "source_id": form.get("source_id", [""])[0],
                "threshold": float(form.get("threshold", ["0.35"])[0]),
                "overwrite": form.get("overwrite", [""])[0] == "1",
            },
            root=self.app.root,
        )
        self._redirect(f"/jobs/{_q(str(job['id']))}")

    def _dataset_audit(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        source_id = form.get("source_id", [""])[0] or None
        audit = workspace.audit(source_id=source_id)
        if form.get("apply_safe", [""])[0] == "1":
            workspace.apply_safe_audit_exclusions(source_id=source_id)
        summary = audit["summary"]
        target = f"/datasets/{_q(name)}/source/{_q(source_id)}" if source_id else f"/datasets/{_q(name)}"
        self._redirect(target + f"?audit={int(summary['flagged'])}")

    def _source_action(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        source_id = form.get("source_id", [""])[0]
        action = form.get("action", [""])[0]
        if action == "toggle":
            source = workspace.sources[source_id]
            workspace.set_source_enabled(source_id, not bool(source.get("enabled", True)))
            self._redirect(f"/datasets/{_q(name)}/source/{_q(source_id)}")
            return
        if action == "delete":
            if form.get("confirm", [""])[0] != source_id:
                raise PipelineError("Source ID 确认不匹配")
            delete_dataset_source(workspace, source_id)
            self._redirect(f"/datasets/{_q(name)}")
            return
        raise PipelineError("Unknown source action")

    def _dataset_delete(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        if form.get("confirm", [""])[0] != name:
            raise PipelineError("数据集名称确认不匹配")
        delete_dataset_workspace(workspace)
        self._redirect("/datasets")

    # ------------------------------------------------------------------
    # Training Configs
    # ------------------------------------------------------------------
    def _configs(self) -> None:
        configs = self.app.configs()
        rows = []
        for config in configs:
            training = config.overrides.get("training", {})
            rows.append(
                f"<tr><td><a href='/configs/{_q(config.name)}'><b>{_e(config.name)}</b></a></td><td>{_e(config.concept_type)}</td><td>{_e(config.base)}</td><td>{_e(config.strategy)}</td><td>{_e(training.get('network_dim', '默认'))}</td><td>{config.images_seen}</td></tr>"
            )
        bases = [(key, value) for key, value in load_base_registry(self.app.root).items() if value.enabled]
        options = "".join(f"<option value='{_e(key)}'>{_e(key)} · {_e(value.name)}</option>" for key, value in bases)
        body = "<div class='hero'><h1>训练配置</h1><div class='muted'>可复用 recipe；Dataset 与 Config 只在启动 Run 时绑定并冻结。</div></div>"
        body += "<table><tr><th>名称</th><th>类型</th><th>底模</th><th>策略</th><th>Rank</th><th>images_seen</th></tr>" + ("".join(rows) or "<tr><td colspan=6 class=muted>暂无训练配置</td></tr>") + "</table>"
        body += f"""<div class="panel" style="margin-top:18px"><h3>创建训练配置</h3><form method="post" action="/configs/create"><input type="hidden" name="_csrf" value="{self.app.csrf}"><div class="row"><label>名称<input name="name" required></label><label>类型<select name="concept_type"><option value="character">character</option><option value="style">style</option></select></label><label>底模<select name="base">{options}</select></label><label>Trigger<input name="trigger" required></label><label>策略<select name="strategy"><option>quality</option><option>fast</option><option>cached</option></select></label><label>images_seen<input type="number" min="1" name="images_seen" value="1000"></label><label>Rank（空=默认）<input type="number" min="1" name="network_dim"></label><label>Alpha（空=默认）<input type="number" min="1" name="network_alpha"></label><label>UNet LR（空=默认）<input name="unet_lr"></label><label>人物评测主体<input name="subject_prompt" value="1girl"></label></div><div class="toolbar"><button class="good">创建</button></div></form></div>"""
        self._html(_page("训练配置", body, active="configs"))

    def _config(self, name: str) -> None:
        config = TrainingConfig.load(name, root=self.app.root)
        training = config.overrides.get("training", {})
        workflow = config.workflow
        bases = [(key, value) for key, value in load_base_registry(self.app.root).items() if value.enabled]
        options = "".join(
            f"<option value='{_e(key)}' {'selected' if key == config.base else ''}>{_e(key)} · {_e(value.name)}</option>"
            for key, value in bases
        )
        caption_modes = ("auto", "generate", "existing_taglist_clean", "existing_passthrough", "hybrid", "skip")
        caption_options = "".join(
            f"<option value='{mode}' {'selected' if workflow.get('caption_mode') == mode else ''}>{mode}</option>"
            for mode in caption_modes
        )
        checkbox = lambda key, default=False: "checked" if bool(workflow.get(key, default)) else ""
        body = f"""<div class="hero"><h1>{_e(name)}</h1><div class="muted">Config snapshot {_e(config.snapshot()['snapshot_hash'][:16])}</div></div><div class="panel"><form method="post" action="/configs/{_q(name)}/save"><input type="hidden" name="_csrf" value="{self.app.csrf}"><h3>核心训练参数</h3><div class="row"><label>底模<select name="base">{options}</select></label><label>Trigger<input name="trigger" value="{_e(config.trigger)}"></label><label>策略<select name="strategy"><option {'selected' if config.strategy == 'quality' else ''}>quality</option><option {'selected' if config.strategy == 'fast' else ''}>fast</option><option {'selected' if config.strategy == 'cached' else ''}>cached</option></select></label><label>images_seen<input type="number" min="1" name="images_seen" value="{config.images_seen}"></label><label>Rank<input type="number" min="1" name="network_dim" value="{_e(training.get('network_dim',''))}"></label><label>Alpha<input type="number" min="1" name="network_alpha" value="{_e(training.get('network_alpha',''))}"></label><label>UNet LR<input name="unet_lr" value="{_e(training.get('unet_lr',''))}"></label></div><h3>工作流</h3><div class="row"><label><input style="width:auto" type="checkbox" name="run_dedup" value="1" {checkbox('run_dedup', True)}> 重复检查</label><label><input style="width:auto" type="checkbox" name="exclude_exact_duplicates" value="1" {checkbox('exclude_exact_duplicates')}> 自动排除完全重复</label><label><input style="width:auto" type="checkbox" name="run_identity" value="1" {checkbox('run_identity', config.concept_type == 'character')}> 人物身份检查</label><label>Caption 模式<select name="caption_mode">{caption_options}</select></label><label><input style="width:auto" type="checkbox" name="allow_trigger_only" value="1" {checkbox('allow_trigger_only')}> 允许 trigger-only</label><label><input style="width:auto" type="checkbox" name="run_review" value="1" {checkbox('run_review', True)}> 训练前审核摘要</label><label>人物评测主体<input name="subject_prompt" value="{_e(config.evaluation.get('subject_prompt','1girl'))}"></label></div><div class="toolbar"><button class="good">保存配置</button></div></form></div><div class="panel danger-zone" style="margin-top:18px"><h3 class="bad">删除训练配置</h3><p class="muted">删除可编辑 Config 不会改变历史 Run 中已冻结的 Config snapshot。</p><form method="post" action="/configs/{_q(name)}/delete"><input type="hidden" name="_csrf" value="{self.app.csrf}"><label>输入配置名称确认<input name="confirm"></label><div class="toolbar"><button class="danger">删除配置</button></div></form></div>"""
        self._html(_page(name, body, active="configs"))

    def _config_create(self, form: dict[str, list[str]]) -> None:
        concept = form.get("concept_type", ["character"])[0]
        evaluation = {}
        subject_prompt = form.get("subject_prompt", [""])[0].strip()
        if concept == "character" and subject_prompt:
            evaluation["subject_prompt"] = subject_prompt
        config = TrainingConfig.create(
            form["name"][0].strip(),
            concept_type=concept,
            base=form["base"][0],
            trigger=form["trigger"][0].strip(),
            strategy=form.get("strategy", ["quality"])[0],
            images_seen=_positive_int(form.get("images_seen", ["1000"])[0], "images_seen"),
            overrides=self._training_overrides(form),
            evaluation=evaluation,
            root=self.app.root,
        )
        self._redirect(f"/configs/{_q(config.name)}")

    def _config_save(self, name: str, form: dict[str, list[str]]) -> None:
        config = TrainingConfig.load(name, root=self.app.root)
        config.data["base"] = form["base"][0]
        config.data["trigger"] = form["trigger"][0].strip()
        config.data["strategy"] = form["strategy"][0]
        config.data["images_seen"] = _positive_int(form["images_seen"][0], "images_seen")
        config.data["overrides"] = self._training_overrides(form)
        workflow = config.workflow
        workflow["run_dedup"] = form.get("run_dedup", [""])[0] == "1"
        workflow["exclude_exact_duplicates"] = workflow["run_dedup"] and form.get("exclude_exact_duplicates", [""])[0] == "1"
        workflow["run_identity"] = config.concept_type == "character" and form.get("run_identity", [""])[0] == "1"
        workflow["caption_mode"] = form.get("caption_mode", ["auto"])[0]
        workflow["allow_trigger_only"] = form.get("allow_trigger_only", [""])[0] == "1"
        workflow["run_review"] = form.get("run_review", [""])[0] == "1"
        workflow["run_screening_evaluation"] = False
        if config.concept_type == "character":
            config.evaluation["subject_prompt"] = form.get("subject_prompt", ["1girl"])[0].strip() or "1girl"
        config.validate(require_enabled_base=True, root=self.app.root)
        config.save()
        self._redirect(f"/configs/{_q(name)}")

    def _config_delete(self, name: str, form: dict[str, list[str]]) -> None:
        config = TrainingConfig.load(name, root=self.app.root)
        if form.get("confirm", [""])[0] != name:
            raise PipelineError("配置名称确认不匹配")
        config.path.unlink()
        self._redirect("/configs")

    def _training_overrides(self, form: dict[str, list[str]]) -> dict[str, Any]:
        training: dict[str, Any] = {}
        for key in ("network_dim", "network_alpha"):
            raw = form.get(key, [""])[0].strip()
            if raw:
                training[key] = _positive_int(raw, key)
        raw_lr = form.get("unet_lr", [""])[0].strip()
        if raw_lr:
            value = float(raw_lr)
            if value <= 0:
                raise PipelineError("unet_lr must be positive")
            training["unet_lr"] = value
        return {"training": training} if training else {}

    # ------------------------------------------------------------------
    # Training status
    # ------------------------------------------------------------------
    def _status(self) -> None:
        entries = _status_entries(self.app.states(), root=self.app.root)
        rows = []
        for entry in entries:
            rows.append(
                f"<tr><td>{_e(entry['dataset'])}</td><td>{_e(entry['config'])}</td><td><a href='/status/{_q(entry['project'])}'>{_e(entry.get('run_id') or 'pending')}</a></td><td>{_e(entry['status'])}</td><td>{_e(entry.get('updated') or '')}</td></tr>"
            )
        datasets = self.app.datasets()
        configs = self.app.configs()
        dataset_options = "".join(
            f"<option value='{_e(item.name)}'>{_e(item.name)} · {_e(item.concept_type)}</option>" for item in datasets
        )
        config_options = "".join(
            f"<option value='{_e(item.name)}'>{_e(item.name)} · {_e(item.concept_type)} · {_e(item.base)}</option>" for item in configs
        )
        gpu = active_gpu_jobs(root=self.app.root)
        gpu_note = f"<div class='panel warn'>GPU 已被 <a href='/jobs/{_q(str(gpu[0]['id']))}'>{_e(gpu[0]['kind'])}</a> 占用；新 GPU 任务会被拒绝。</div>" if gpu else ""
        body = "<div class='hero'><h1>训练状态</h1><div class='muted'>长任务由独立 worker 执行；关闭浏览器或重启 Web 服务不会改变冻结后的 Run 输入。</div></div>"
        body += gpu_note
        body += "<table><tr><th>数据集</th><th>配置</th><th>Run</th><th>状态</th><th>更新时间</th></tr>" + ("".join(rows) or "<tr><td colspan=5 class=muted>暂无训练记录</td></tr>") + "</table>"
        body += f"""<div class="panel" style="margin-top:18px"><h3>开始一次新训练</h3><form method="post" action="/status/start"><input type="hidden" name="_csrf" value="{self.app.csrf}"><div class="row"><label>Dataset<select name="dataset">{dataset_options}</select></label><label>Training Config<select name="config">{config_options}</select></label></div><label style="display:block;margin-top:10px"><input style="width:auto" type="checkbox" name="safe_exclude" value="1" checked> 启动前自动排除损坏文件与完全重复副本</label><div class="toolbar"><button class="good">冻结 Dataset + Config 并开始</button></div></form></div>"""
        self._html(_page("训练状态", body, active="status"))

    def _status_start(self, form: dict[str, list[str]]) -> None:
        if active_gpu_jobs(root=self.app.root):
            raise PipelineError("GPU 当前已有 Web 任务，请等待其完成")
        workspace = DatasetWorkspace.load(form["dataset"][0], root=self.app.root)
        config = TrainingConfig.load(form["config"][0], root=self.app.root)
        if workspace.concept_type != config.concept_type:
            raise PipelineError("Dataset 与 Training Config 类型不兼容")
        if form.get("safe_exclude", [""])[0] == "1":
            workspace.apply_safe_audit_exclusions()
            workspace = DatasetWorkspace.load(workspace.name, root=self.app.root)
        workspace.snapshot()
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
        job = spawn_job("train", {"project": state.name}, root=self.app.root)
        self._redirect(f"/status/{_q(state.name)}?job={_q(str(job['id']))}")

    def _status_detail(self, project_name: str) -> None:
        state = load_project(project_name, root=self.app.root)
        project = state.payload["project"]
        identity = project.get("training_identity", {})
        steps = "".join(
            f"<tr><td>{_e(name)}</td><td>{_e(record.get('status'))}</td><td>{_e(record.get('attempts',0))}</td><td>{_e(record.get('last_error',''))}</td></tr>"
            for name, record in state.payload.get("steps", {}).items()
        )
        jobs = jobs_for_project(project_name, root=self.app.root)
        job = jobs[0] if jobs else None
        log_tail = tail_job_log(str(job["id"]), root=self.app.root) if job else ""
        run = state.payload.get("runs", [])[-1] if state.payload.get("runs") else None
        status = str(run.get("status")) if run else _workspace_status(state)
        if job and job.get("status") in {"queued", "starting", "running"}:
            status = f"web:{job.get('kind')}"
        job_info = ""
        if job:
            job_info = f"<div class='card'><div class='muted'>Web job</div><div><a href='/jobs/{_q(str(job['id']))}'>{_e(job['id'])}</a></div><div class='pill'>{_e(job.get('status'))}</div>{f'<div class="bad compact">{_e(job.get("error"))}</div>' if job.get('error') else ''}</div>"
        body = f"""<div class="hero"><h1>{_e(identity.get('dataset') or project_name)} / {_e(identity.get('config') or 'legacy')}</h1><div class="muted">技术工作区 {_e(project_name)} · 当前 {_e(status)}</div></div><div class="grid"><div class="card"><div class="muted">Dataset snapshot</div><div class="mono">{_e(identity.get('dataset_snapshot_hash',''))}</div></div><div class="card"><div class="muted">Config snapshot</div><div class="mono">{_e(identity.get('config_snapshot_hash',''))}</div></div><div class="card"><div class="muted">当前内部步骤</div><div class="metric" style="font-size:20px">{_e(state.next_actionable_step() or 'complete')}</div></div>{job_info}</div><div class="toolbar"><form method="post" action="/status/{_q(project_name)}/continue"><input type="hidden" name="_csrf" value="{self.app.csrf}"><button class="good">继续 / 恢复训练</button></form></div><table><tr><th>步骤</th><th>状态</th><th>尝试</th><th>错误</th></tr>{steps}</table><div class="panel" style="margin-top:18px"><h3>worker log</h3><div class="mono">{_e(log_tail or '暂无日志')}</div></div>"""
        refresh = 5 if job and job.get("status") in {"queued", "starting", "running"} else None
        self._html(_page(project_name, body, active="status", refresh=refresh))

    def _status_continue(self, project_name: str) -> None:
        load_project(project_name, root=self.app.root)
        job = spawn_job("train", {"project": project_name}, root=self.app.root)
        self._redirect(f"/status/{_q(project_name)}?job={_q(str(job['id']))}")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _results(self) -> None:
        entries = _result_entries(self.app.states())
        rows = []
        for entry in entries:
            rows.append(
                f"<tr><td>{_e(entry['dataset'])}</td><td>{_e(entry['config'])}</td><td><a href='/results/{_q(entry['project'])}/{_q(entry['run_id'])}'>{_e(entry['run_id'])}</a></td><td>{_e(entry['status'])}</td><td>{entry['checkpoints']}</td><td>{entry['samples']}</td><td>{'★' if entry['promoted'] else ''}</td></tr>"
            )
        body = "<div class='hero'><h1>训练结果</h1><div class='muted'>查看权重/示例图，在浏览器触发 Screening / Full 评测并选择 Best。</div></div><table><tr><th>数据集</th><th>配置</th><th>Run</th><th>状态</th><th>权重</th><th>示例图</th><th>Best</th></tr>" + ("".join(rows) or "<tr><td colspan=7 class=muted>还没有完成的训练结果。</td></tr>") + "</table>"
        self._html(_page("训练结果", body, active="results"))

    def _result_detail(self, project_name: str, run_id: str) -> None:
        state = load_project(project_name, root=self.app.root)
        run = _find_run(state, run_id)
        if run is None:
            raise PipelineError("Run does not exist")
        run_dir = Path(str(run.get("path") or ""))
        checkpoints = [Path(value) for value in run.get("checkpoints", []) if Path(value).is_file()]
        weight_links = "".join(
            f"<li><a href='/run-file/{_q(project_name)}/{_q(run_id)}/{quote(path.relative_to(run_dir).as_posix())}'>{_e(path.name)}</a></li>"
            for path in checkpoints
            if run_dir in path.parents
        )
        images: list[Path] = []
        for folder in (run_dir / "samples", run_dir / "contact-sheets"):
            images.extend(_images_under(folder))
        image_cards = "".join(
            f"<div class='image-card'><a href='/run-file/{_q(project_name)}/{_q(run_id)}/{quote(path.relative_to(run_dir).as_posix())}'><img loading='lazy' src='/run-file/{_q(project_name)}/{_q(run_id)}/{quote(path.relative_to(run_dir).as_posix())}'></a><div class='image-body image-title'>{_e(path.relative_to(run_dir))}</div></div>"
            for path in images[:120]
        )
        evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
        evaluated = {
            str(value)
            for record in evidence.values()
            if isinstance(record, dict)
            for value in record.get("checkpoints", [])
        }
        promotion = run.get("promotion", {}) if isinstance(run.get("promotion"), dict) else {}
        checkboxes = "".join(
            f"<label><input style='width:auto' type='checkbox' name='checkpoints' value='{_e(path.name)}'> {_e(path.name)}</label>"
            for path in checkpoints
        )
        evaluated_options = "".join(
            f"<option value='{_e(path.name)}'>{_e(path.name)}</option>"
            for path in checkpoints
            if path.name in evaluated or path.stem in evaluated
        )
        eval_jobs = [job for job in jobs_for_project(project_name, root=self.app.root) if job.get("kind") == "evaluate"]
        current_job = eval_jobs[0] if eval_jobs else None
        job_note = ""
        if current_job and current_job.get("status") in {"queued", "starting", "running"}:
            job_note = f"<div class='panel warn'><a href='/jobs/{_q(str(current_job['id']))}'>评测任务 {_e(current_job['id'])}</a> 正在运行。</div>"
        report_link = ""
        report = run_dir / "report.html"
        if report.is_file():
            report_link = f"<a class='button' href='/run-file/{_q(project_name)}/{_q(run_id)}/report.html'>打开评测报告</a>"
        body = f"""<div class="hero"><h1>Run {_e(run_id)}</h1><div class="muted">{_e(run.get('status'))} · {_e(run_dir)}</div></div>{job_note}<div class="grid"><div class="card"><div class="muted">权重</div><div class="metric">{len(checkpoints)}</div></div><div class="card"><div class="muted">示例 / 对比图</div><div class="metric">{len(images)}</div></div><div class="card"><div class="muted">评测阶段</div><div>{_e(', '.join(sorted(evidence)) or '未评测')}</div></div><div class="card"><div class="muted">Best</div><div>{_e(promotion.get('checkpoint') or '—')}</div></div></div><div class="grid" style="margin-top:18px"><div class="panel"><h3>权重文件</h3><ul>{weight_links or '<li class=muted>暂无</li>'}</ul>{report_link}</div><div class="panel"><h3>Screening</h3><form method="post" action="/results/{_q(project_name)}/{_q(run_id)}/evaluate"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="stage" value="screening"><label><input style="width:auto" type="checkbox" name="force" value="1"> 已存在时重跑</label><div class="toolbar"><button class="good">运行 Screening</button></div></form><h3>Full</h3><p class="muted">选择 1–2 个 finalist。</p><form method="post" action="/results/{_q(project_name)}/{_q(run_id)}/evaluate"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="stage" value="full"><div>{checkboxes}</div><label><input style="width:auto" type="checkbox" name="force" value="1"> 已存在时重跑</label><div class="toolbar"><button>运行 Full</button></div></form></div><div class="panel"><h3>选择最佳权重</h3><form method="post" action="/results/{_q(project_name)}/{_q(run_id)}/promote"><input type="hidden" name="_csrf" value="{self.app.csrf}"><label>已评测 checkpoint<select name="checkpoint">{evaluated_options}</select></label><label>推荐 LoRA 强度<input name="strength" value="0.8"></label><div class="toolbar"><button class="good">生成 best.safetensors</button></div></form></div></div><div class="images" style="margin-top:18px">{image_cards}</div>"""
        self._html(_page(f"Run {run_id}", body, active="results", refresh=5 if current_job and current_job.get("status") in {"queued", "starting", "running"} else None))

    def _result_evaluate(self, project_name: str, run_id: str, form: dict[str, list[str]]) -> None:
        state = load_project(project_name, root=self.app.root)
        run = _find_run(state, run_id)
        if run is None:
            raise PipelineError("Run does not exist")
        stage = form.get("stage", ["screening"])[0]
        checkpoints = form.get("checkpoints", [])
        if stage not in {"screening", "full"}:
            raise PipelineError("Unknown evaluation stage")
        if stage == "full" and not 1 <= len(checkpoints) <= 2:
            raise PipelineError("Full 评测必须选择 1–2 个 checkpoint")
        available = {Path(value).name for value in run.get("checkpoints", []) if Path(value).is_file()}
        if any(value not in available for value in checkpoints):
            raise PipelineError("选择了不属于该 Run 的 checkpoint")
        job = spawn_job(
            "evaluate",
            {
                "project": project_name,
                "run_id": run_id,
                "stage": stage,
                "checkpoints": checkpoints,
                "force": form.get("force", [""])[0] == "1",
            },
            root=self.app.root,
        )
        self._redirect(f"/jobs/{_q(str(job['id']))}")

    def _result_promote(self, project_name: str, run_id: str, form: dict[str, list[str]]) -> None:
        state = load_project(project_name, root=self.app.root)
        run = _find_run(state, run_id)
        if run is None:
            raise PipelineError("Run does not exist")
        checkpoint = form.get("checkpoint", [""])[0]
        strength = float(form.get("strength", ["0.8"])[0])
        if strength <= 0:
            raise PipelineError("LoRA strength must be positive")
        evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
        evaluated = {
            str(value)
            for record in evidence.values()
            if isinstance(record, dict)
            for value in record.get("checkpoints", [])
        }
        path = next((Path(value) for value in run.get("checkpoints", []) if Path(value).name == checkpoint), None)
        if path is None or not path.is_file():
            raise PipelineError("Checkpoint 不存在")
        if checkpoint not in evaluated and path.stem not in evaluated:
            raise PipelineError("只能提升已经经过评测的 checkpoint")
        fresh = load_project(project_name, root=self.app.root)
        with project_lock(fresh.project_dir):
            promote.run(
                ProjectState.load(fresh.project_dir),
                run_id=run_id,
                checkpoint_name=checkpoint,
                strength=strength,
            )
        self._redirect(f"/results/{_q(project_name)}/{_q(run_id)}")

    def _run_file(self, project_name: str, run_id: str, relative: str) -> None:
        state = load_project(project_name, root=self.app.root)
        run = _find_run(state, run_id)
        if run is None:
            raise PipelineError("Run does not exist")
        root = Path(str(run.get("path") or ""))
        path = _safe_child(root, relative)
        inline = path.suffix.casefold() in _IMAGE_SUFFIXES or path.suffix.casefold() in {".html", ".txt", ".json"}
        self._file(path, inline=inline)

    # ------------------------------------------------------------------
    # Persistent jobs / video identity selection
    # ------------------------------------------------------------------
    def _job(self, job_id: str) -> None:
        job = read_job(job_id, root=self.app.root)
        log = tail_job_log(job_id, root=self.app.root)
        status = str(job.get("status") or "")
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        content = ""
        if status == "awaiting_identity":
            identity = result.get("identity") if isinstance(result.get("identity"), dict) else {}
            clusters = []
            for cluster in identity.get("clusters", []):
                reps = "".join(
                    f"<img src='/media/job/{_q(job_id)}/{quote(str(path))}' alt='representative'>"
                    for path in cluster.get("representatives", [])
                )
                clusters.append(
                    f"""<div class="cluster"><div><b>人物簇 {int(cluster.get('cluster_id', -1))}</b> · {int(cluster.get('size', 0))} 个候选</div><div class="cluster-images">{reps}</div><form method="post" action="/jobs/{_q(job_id)}/identity"><input type="hidden" name="_csrf" value="{self.app.csrf}"><input type="hidden" name="cluster_id" value="{int(cluster.get('cluster_id', -1))}"><div class="toolbar"><button class="good">选择这个人物</button></div></form></div>"""
                )
            content = f"<div class='panel'><h3>选择目标人物</h3><p class='muted'>CCIP 已按人物 crop 聚类。选择后才会生成构图平衡的最终 Dataset Source。</p><div class='grid'>{''.join(clusters)}</div></div>"
        elif status == "completed":
            if job.get("kind") == "video_finalize" and result.get("dataset"):
                content = f"<div class='panel good'>视频来源已写入 <a href='/datasets/{_q(str(result['dataset']))}/source/{_q(str(result['source_id']))}'>{_e(result.get('label') or result['source_id'])}</a></div>"
            elif job.get("kind") == "train" and (job.get("payload") or {}).get("project"):
                project = str((job.get("payload") or {})["project"])
                content = f"<div class='panel good'><a href='/status/{_q(project)}'>打开训练状态</a></div>"
            elif job.get("kind") == "evaluate" and (job.get("payload") or {}).get("project"):
                payload = job.get("payload") or {}
                content = f"<div class='panel good'><a href='/results/{_q(str(payload['project']))}/{_q(str(payload['run_id']))}'>打开评测结果</a></div>"
            elif job.get("kind") == "dataset_tag" and result.get("dataset"):
                content = f"<div class='panel good'><a href='/datasets/{_q(str(result['dataset']))}'>返回数据集</a></div>"
        error = f"<div class='error'>{_e(job.get('error'))}</div>" if job.get("error") else ""
        body = f"""<div class="hero"><h1>Web Job</h1><div class="muted">{_e(job_id)} · {_e(job.get('kind'))}</div></div><div class="grid"><div class="card"><div class="muted">状态</div><div class="metric" style="font-size:20px">{_e(status)}</div></div><div class="card"><div class="muted">PID</div><div>{_e(job.get('pid') or '—')}</div></div><div class="card"><div class="muted">更新时间</div><div>{_e(job.get('updated_at') or '')}</div></div></div>{error}{content}<div class="panel" style="margin-top:18px"><h3>日志</h3><div class="mono">{_e(log or '暂无日志')}</div></div>"""
        refresh = 4 if status in {"queued", "starting", "running"} else None
        self._html(_page(f"Job {job_id}", body, refresh=refresh))

    def _job_identity(self, job_id: str, form: dict[str, list[str]]) -> None:
        job = read_job(job_id, root=self.app.root)
        if job.get("status") != "awaiting_identity":
            raise PipelineError("这个视频任务当前不等待人物选择")
        selected = int(form.get("cluster_id", ["-1"])[0])
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        identity = result.get("identity") if isinstance(result.get("identity"), dict) else {}
        valid = {int(value.get("cluster_id")) for value in identity.get("clusters", [])}
        if selected not in valid:
            raise PipelineError("无效的人物簇")
        resume_job(
            job_id,
            kind="video_finalize",
            payload_updates={"selected_cluster": selected},
            root=self.app.root,
        )
        self._redirect(f"/jobs/{_q(job_id)}")

    def _job_media(self, job_id: str, relative: str) -> None:
        read_job(job_id, root=self.app.root)
        path = _safe_child(job_data_dir(job_id, root=self.app.root), relative)
        if path.suffix.casefold() not in _IMAGE_SUFFIXES:
            raise PipelineError("Web job media endpoint only serves images")
        self._file(path, inline=True)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
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
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'",
        )
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
        disposition = "inline" if inline else "attachment"
        self.send_header("Content-Disposition", f'{disposition}; filename="{path.name.replace(chr(34), "")}"')
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _positive_int(raw: str, label: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise PipelineError(f"{label} must be an integer") from exc
    if value < 1:
        raise PipelineError(f"{label} must be at least 1")
    return value


def make_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    root: Path | None = None,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    app = WebApplication(root=root, auth_token=auth_token)
    handler = type("BoundHandler", (Handler,), {"app": app})
    return ThreadingHTTPServer((host, int(port)), handler)


def serve(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    allow_lan: bool = False,
    auth_token: str | None = None,
    unsafe_no_auth: bool = False,
    root: Path | None = None,
) -> None:
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    if not loopback and not allow_lan:
        raise PipelineError("Refusing non-loopback bind without --allow-lan; prefer an SSH tunnel")
    if not loopback and not auth_token and not unsafe_no_auth:
        raise PipelineError(
            "LAN mode requires LORA_WEB_TOKEN/--token. Use --unsafe-no-auth only on an explicitly trusted network."
        )
    server = make_server(host, port, root=root, auth_token=auth_token)
    print(f"LoRA Pipeline Web: http://{host}:{port}")
    if loopback:
        print(f"Remote browser: ssh -L {port}:127.0.0.1:{port} <nas>  then open http://127.0.0.1:{port}")
    elif auth_token:
        print("LAN mode enabled with access-token authentication. Prefer HTTPS/reverse proxy on untrusted networks.")
    else:
        print("WARNING: unauthenticated LAN mode enabled explicitly; never expose this service to the public Internet.")
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
    parser.add_argument("--token", default=os.environ.get("LORA_WEB_TOKEN"))
    parser.add_argument("--unsafe-no-auth", action="store_true")
    args = parser.parse_args(argv)
    serve(
        args.host,
        args.port,
        allow_lan=args.allow_lan,
        auth_token=args.token,
        unsafe_no_auth=args.unsafe_no_auth,
    )


if __name__ == "__main__":
    main()
