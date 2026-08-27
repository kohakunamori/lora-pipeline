from __future__ import annotations

import argparse
import mimetypes
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from .dataset_workspace import DatasetWorkspace
from .models import PipelineError
from .service import load_project, project_path
from .training_config import TrainingConfig, create_project_from_training_config, make_training_workspace_name
from .web_app import Handler, WebApplication, _e, _page, _q, _safe_child
from .web_jobs import job_data_dir, jobs_for_project, list_jobs, read_job, resume_job, spawn_job, tail_job_log

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _source_options(workspace: DatasetWorkspace) -> str:
    options = ["<option value=''>整个数据集</option>"]
    for source_id, source in sorted(workspace.sources.items()):
        label = source.get("label") or source_id
        options.append(f"<option value='{_e(source_id)}'>{_e(label)}</option>")
    return "".join(options)


def _recent_dataset_jobs(name: str, *, root: Path) -> list[dict[str, Any]]:
    return [
        job
        for job in list_jobs(root=root, limit=60)
        if str((job.get("payload") or {}).get("dataset") or "") == name
    ][:8]


class FullHandler(Handler):
    """Browser routes layered over the existing four-part domain model."""

    def _get(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path.startswith("/jobs/"):
            parts = path.split("/")
            if len(parts) == 3:
                self._job(unquote(parts[2]))
                return
        if path.startswith("/job-file/"):
            parts = path.split("/", 3)
            if len(parts) == 4:
                self._job_file(unquote(parts[2]), parts[3])
                return
        if path.startswith("/dataset-tools/"):
            self._dataset_tools(unquote(path.split("/", 2)[2]))
            return
        if path.startswith("/result-tools/"):
            parts = path.split("/")
            if len(parts) == 4:
                self._result_tools(unquote(parts[2]), unquote(parts[3]))
                return
        super()._get()

    def _post(self, form: dict[str, list[str]]) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path == "/datasets/create":
            self._dataset_create(form)
            return
        if path.startswith("/dataset-tools/"):
            parts = path.split("/")
            if len(parts) == 4 and parts[3] == "import-dir":
                self._dataset_import_dir(unquote(parts[2]), form)
                return
            if len(parts) == 4 and parts[3] == "auto-tag":
                self._dataset_auto_tag(unquote(parts[2]), form)
                return
            if len(parts) == 4 and parts[3] == "video-prepare":
                self._dataset_video_prepare(unquote(parts[2]), form)
                return
        if path.startswith("/jobs/") and path.endswith("/video-finalize"):
            self._video_finalize(unquote(path.split("/")[2]), form)
            return
        if path == "/status/start":
            self._status_start_job(form)
            return
        if path.startswith("/status/") and path.endswith("/continue"):
            self._status_continue_job(unquote(path.split("/")[2]))
            return
        if path.startswith("/result-tools/") and path.endswith("/evaluate"):
            parts = path.split("/")
            if len(parts) == 5:
                self._result_evaluate(unquote(parts[2]), unquote(parts[3]), form)
                return
        super()._post(form)

    def _datasets(self) -> None:
        rows: list[str] = []
        for workspace in self.app.datasets():
            summary = workspace.summary()
            rows.append(
                "<tr>"
                f"<td><a href='/datasets/{_q(workspace.name)}'><b>{_e(workspace.name)}</b></a></td>"
                f"<td>{_e(workspace.concept_type)}</td>"
                f"<td>{summary['sources']}</td><td>{summary['active_images']}</td>"
                f"<td>{summary['excluded_images']}</td><td>{summary['captioned_active_images']}</td>"
                f"<td><a class='button' href='/dataset-tools/{_q(workspace.name)}'>导入 / 自动处理</a></td>"
                "</tr>"
            )
        table_rows = "".join(rows) or "<tr><td colspan='7' class='muted'>还没有数据集。</td></tr>"
        body = (
            "<div class='hero'><h1>数据集</h1>"
            "<div class='muted'>一个 Dataset 可以持续追加多个独立 Source，并在浏览器里审核图片和 Tag。</div></div>"
            "<table><tr><th>名称</th><th>类型</th><th>来源</th><th>可用图片</th><th>已排除</th><th>已有 Tag</th><th></th></tr>"
            f"{table_rows}</table>"
            "<div class='panel' style='margin-top:18px'><h3>创建数据集</h3>"
            f"<form method='post' action='/datasets/create'><input type='hidden' name='_csrf' value='{self.app.csrf}'>"
            "<div class='row'><label>名称<input name='name' required pattern='[A-Za-z0-9][A-Za-z0-9._-]{0,63}'></label>"
            "<label>类型<select name='concept_type'><option value='character'>人物 character</option>"
            "<option value='style'>风格 style</option></select></label></div>"
            "<div class='toolbar'><button class='good'>创建</button></div></form></div>"
        )
        self._html(_page("数据集", body, active="datasets"))

    def _dataset_create(self, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.create(
            form.get("name", [""])[0].strip(),
            concept_type=form.get("concept_type", ["character"])[0],
            root=self.app.root,
        )
        self._redirect(f"/datasets/{_q(workspace.name)}")

    def _dataset_tools(self, name: str) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        jobs = _recent_dataset_jobs(name, root=self.app.root)
        job_rows = "".join(
            "<tr>"
            f"<td><a href='/jobs/{_q(str(job['id']))}'>{_e(job['id'])}</a></td>"
            f"<td>{_e(job.get('kind'))}</td><td>{_e(job.get('status'))}</td>"
            f"<td>{_e(job.get('updated_at', ''))}</td></tr>"
            for job in jobs
        )
        image_import = (
            "<div class='panel'><h3>导入图片目录</h3>"
            f"<form method='post' action='/dataset-tools/{_q(name)}/import-dir'>"
            f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
            "<label>NAS 上的目录路径<input name='directory' required placeholder='/volume1/dataset/images'></label>"
            "<label>来源名称<input name='label' placeholder='official-images'></label>"
            "<div class='toolbar'><button class='good'>导入为新 Source</button></div></form></div>"
        )
        tag_form = (
            "<div class='panel'><h3>自动 Tag</h3>"
            f"<form method='post' action='/dataset-tools/{_q(name)}/auto-tag'>"
            f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
            f"<div class='row'><label>范围<select name='source_id'>{_source_options(workspace)}</select></label>"
            "<label>阈值<input name='threshold' value='0.35'></label></div>"
            "<label><input style='width:auto' type='checkbox' name='overwrite' value='1'> 覆盖已有 Tag</label>"
            "<div class='toolbar'><button>后台自动打 Tag</button></div></form></div>"
        )
        video_form = ""
        if workspace.concept_type == "character":
            video_form = (
                "<div class='panel'><h3>导入视频</h3>"
                f"<form method='post' action='/dataset-tools/{_q(name)}/video-prepare'>"
                f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
                "<label>本地视频路径或 URL<input name='source' required "
                "placeholder='/volume1/video/source.mkv 或 https://...'></label>"
                "<div class='row'><label>来源名称<input name='label' placeholder='video'></label>"
                "<label>每 N 秒采样<input type='number' min='1' name='interval_seconds' value='2'></label>"
                "<label>最多候选帧<input type='number' min='1' name='max_frames' value='250'></label></div>"
                "<details><summary>网络 / YouTube 高级设置</summary><div class='row'>"
                "<label>代理模式<select name='proxy_mode'><option value='environment'>environment</option>"
                "<option value='direct'>direct</option><option value='custom'>custom</option></select></label>"
                "<label>自定义代理<input name='proxy_url' placeholder='http://192.168.2.31:7890'></label>"
                "<label>cookies.txt 路径<input name='cookies_path' "
                "placeholder='~/.config/lora-pipeline/youtube-cookies.txt'></label></div></details>"
                "<div class='toolbar'><button class='good'>后台抽帧并检测人物</button></div>"
                "<p class='muted'>完成后网页显示 CCIP 人物簇缩略图，由你选择目标人物，不自动选择最大簇。</p>"
                "</form></div>"
            )
        jobs_panel = ""
        if jobs:
            jobs_panel = (
                "<div class='panel' style='margin-top:18px'><h3>最近 Web Jobs</h3>"
                "<table><tr><th>Job</th><th>类型</th><th>状态</th><th>更新时间</th></tr>"
                f"{job_rows}</table></div>"
            )
        body = (
            f"<div class='hero'><h1>{_e(name)} · 导入 / 自动处理</h1>"
            f"<div class='muted'><a href='/datasets/{_q(name)}'>返回数据集详情</a></div></div>"
            f"<div class='grid'>{image_import}{video_form}{tag_form}</div>{jobs_panel}"
        )
        self._html(_page(f"{name} · 工具", body, active="datasets"))

    def _dataset_import_dir(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        directory = Path(form.get("directory", [""])[0]).expanduser().resolve()
        label = form.get("label", [""])[0].strip() or directory.name or "images"
        workspace.add_source_from_directory(
            directory,
            kind="image_directory",
            label=label,
            origin=str(directory),
        )
        self._redirect(f"/datasets/{_q(name)}")

    def _dataset_auto_tag(self, name: str, form: dict[str, list[str]]) -> None:
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

    def _dataset_video_prepare(self, name: str, form: dict[str, list[str]]) -> None:
        source = form.get("source", [""])[0].strip()
        if not source:
            raise PipelineError("视频来源不能为空")
        if not (source.startswith("http://") or source.startswith("https://")):
            video_path = Path(source).expanduser().resolve()
            if not video_path.is_file():
                raise PipelineError(f"本地视频不存在：{video_path}")
            source = str(video_path)
        cookies = form.get("cookies_path", [""])[0].strip()
        if cookies:
            cookies = str(Path(cookies).expanduser().resolve())
        label = form.get("label", [""])[0].strip()
        if not label:
            label = Path(source).stem if not source.startswith("http") else "online-video"
        job = spawn_job(
            "video_prepare",
            {
                "dataset": name,
                "source": source,
                "label": label,
                "interval_seconds": int(form.get("interval_seconds", ["2"])[0]),
                "max_frames": int(form.get("max_frames", ["250"])[0]),
                "proxy_mode": form.get("proxy_mode", ["environment"])[0],
                "proxy_url": form.get("proxy_url", [""])[0].strip(),
                "cookies_path": cookies,
            },
            root=self.app.root,
        )
        self._redirect(f"/jobs/{_q(str(job['id']))}")

    def _job(self, job_id: str) -> None:
        jobs = list_jobs(root=self.app.root, limit=300)
        job = next((item for item in jobs if str(item.get("id")) == job_id), None)
        if job is None:
            job = read_job(job_id, root=self.app.root)
        status = str(job.get("status"))
        refresh = "<meta http-equiv='refresh' content='3'>" if status in {"queued", "running"} else ""
        payload = dict(job.get("payload") or {})
        result = dict(job.get("result") or {})
        body = (
            f"{refresh}<div class='hero'><h1>Web Job {_e(job_id)}</h1>"
            f"<div class='muted'>{_e(job.get('kind'))} · {_e(status)} · PID {_e(job.get('pid') or '—')}</div></div>"
        )
        if job.get("error"):
            body += f"<div class='error'>{_e(job['error'])}</div>"
        if status == "awaiting_identity":
            clusters = list((result.get("identity") or {}).get("clusters") or [])
            cards: list[str] = []
            for cluster in clusters:
                representatives = list(cluster.get("representatives") or [])
                images = "".join(
                    f"<img loading='lazy' src='/job-file/{_q(job_id)}/{quote(str(relative))}' alt='cluster representative'>"
                    for relative in representatives[:4]
                )
                cluster_id = int(cluster["cluster_id"])
                size = int(cluster.get("size", 0))
                cards.append(
                    "<label class='image-card' style='cursor:pointer'>"
                    f"<div style='display:grid;grid-template-columns:repeat(2,1fr)'>{images}</div>"
                    "<div class='image-body'>"
                    f"<input style='width:auto' form='identity' type='radio' name='selected_cluster' value='{cluster_id}' required> "
                    f"<b>人物簇 {cluster_id}</b> · {size} 个候选</div></label>"
                )
            body += (
                "<div class='panel'><h3>选择目标人物</h3>"
                "<p class='muted'>CCIP 已完成。请选择真正要训练的人物簇。</p>"
                f"<form id='identity' method='post' action='/jobs/{_q(job_id)}/video-finalize'>"
                f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
                f"<div class='images'>{''.join(cards) or '<div class=muted>没有可选人物簇</div>'}</div>"
                "<div class='toolbar'><button class='good'>确认人物并写入 Dataset</button></div></form></div>"
            )
        if status == "completed":
            dataset = str(payload.get("dataset") or result.get("dataset") or "")
            if dataset:
                body += f"<div class='toolbar'><a class='button good' href='/datasets/{_q(dataset)}'>返回数据集</a></div>"
        log = tail_job_log(job_id, root=self.app.root)
        body += f"<div class='panel' style='margin-top:18px'><h3>Job Log</h3><div class='mono'>{_e(log or '暂无日志')}</div></div>"
        self._html(_page(f"Job {job_id}", body))

    def _job_file(self, job_id: str, relative: str) -> None:
        path = _safe_child(job_data_dir(job_id, root=self.app.root), relative)
        self._file(path, inline=True)

    def _video_finalize(self, job_id: str, form: dict[str, list[str]]) -> None:
        selected = form.get("selected_cluster", [""])[0]
        if not selected:
            raise PipelineError("请选择目标人物簇")
        record = read_job(job_id, root=self.app.root)
        if str(record.get("status")) != "awaiting_identity":
            raise PipelineError("这个视频 Job 当前不在人物选择阶段")
        resume_job(
            job_id,
            kind="video_finalize",
            payload_updates={"selected_cluster": int(selected)},
            root=self.app.root,
        )
        self._redirect(f"/jobs/{_q(job_id)}")

    def _status_start_job(self, form: dict[str, list[str]]) -> None:
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
        state = create_project_from_training_config(
            workspace,
            config,
            project_name=project_name,
            root=self.app.root,
        )
        spawn_job("train", {"project": state.name}, root=self.app.root)
        self._redirect(f"/status/{_q(state.name)}")

    def _status_continue_job(self, project_name: str) -> None:
        state = load_project(project_name, root=self.app.root)
        spawn_job("train", {"project": state.name}, root=self.app.root)
        self._redirect(f"/status/{_q(project_name)}")

    def _status_detail(self, project_name: str) -> None:
        state = load_project(project_name, root=self.app.root)
        project = state.payload["project"]
        identity = project.get("training_identity", {})
        next_step = state.next_actionable_step()
        run = state.payload.get("runs", [])[-1] if state.payload.get("runs") else None
        status = str(run.get("status")) if run else (f"{state.status(next_step).value}:{next_step}" if next_step else "complete")
        step_rows = "".join(
            "<tr>"
            f"<td>{_e(step)}</td><td>{_e(record.get('status'))}</td><td>{_e(record.get('attempts', 0))}</td>"
            f"<td>{_e(record.get('last_error', ''))}</td></tr>"
            for step, record in state.payload.get("steps", {}).items()
        )
        jobs = jobs_for_project(project_name, root=self.app.root)[:10]
        job_rows = "".join(
            "<tr>"
            f"<td><a href='/jobs/{_q(str(job['id']))}'>{_e(job['id'])}</a></td>"
            f"<td>{_e(job.get('kind'))}</td><td>{_e(job.get('status'))}</td><td>{_e(job.get('updated_at', ''))}</td></tr>"
            for job in jobs
        )
        body = (
            f"<div class='hero'><h1>{_e(identity.get('dataset') or project_name)} / {_e(identity.get('config') or 'legacy')}</h1>"
            f"<div class='muted'>技术工作区 {_e(project_name)} · 当前 {_e(status)}</div></div>"
            "<div class='grid'>"
            f"<div class='card'><div class='muted'>Dataset snapshot</div><div class='mono'>{_e(identity.get('dataset_snapshot_hash', ''))}</div></div>"
            f"<div class='card'><div class='muted'>Config snapshot</div><div class='mono'>{_e(identity.get('config_snapshot_hash', ''))}</div></div>"
            f"<div class='card'><div class='muted'>当前内部步骤</div><div class='metric' style='font-size:20px'>{_e(next_step or 'complete')}</div></div>"
            "</div>"
            f"<div class='toolbar'><form method='post' action='/status/{_q(project_name)}/continue'>"
            f"<input type='hidden' name='_csrf' value='{self.app.csrf}'><button class='good'>继续 / 恢复训练</button></form></div>"
            f"<table><tr><th>步骤</th><th>状态</th><th>尝试</th><th>错误</th></tr>{step_rows}</table>"
        )
        if jobs:
            body += (
                "<div class='panel' style='margin-top:18px'><h3>Web Jobs</h3>"
                "<table><tr><th>Job</th><th>类型</th><th>状态</th><th>更新时间</th></tr>"
                f"{job_rows}</table></div>"
            )
        if run and str(run.get("status")) in {"trained", "evaluated", "promoted"}:
            body += f"<div class='toolbar'><a class='button' href='/results/{_q(project_name)}/{_q(str(run.get('id')))}'>查看训练结果</a></div>"
        self._html(_page(project_name, body, active="status"))

    def _result_detail(self, project_name: str, run_id: str) -> None:
        super()._result_detail(project_name, run_id)

    def _results(self) -> None:
        entries = []
        from .web_app import _result_entries

        entries = _result_entries(self.app.states())
        rows: list[str] = []
        for entry in entries:
            rows.append(
                "<tr>"
                f"<td>{_e(entry['dataset'])}</td><td>{_e(entry['config'])}</td>"
                f"<td><a href='/results/{_q(entry['project'])}/{_q(entry['run_id'])}'>{_e(entry['run_id'])}</a></td>"
                f"<td>{_e(entry['status'])}</td><td>{entry['checkpoints']}</td><td>{entry['samples']}</td>"
                f"<td>{'★' if entry['promoted'] else ''}</td>"
                f"<td><a class='button' href='/result-tools/{_q(entry['project'])}/{_q(entry['run_id'])}'>评测</a></td>"
                "</tr>"
            )
        table_rows = "".join(rows) or "<tr><td colspan='8' class='muted'>还没有完成的训练结果。</td></tr>"
        body = (
            "<div class='hero'><h1>训练结果</h1><div class='muted'>权重、示例图片、对比图和评测。</div></div>"
            "<table><tr><th>数据集</th><th>配置</th><th>Run</th><th>状态</th><th>权重</th><th>示例图</th><th>Best</th><th></th></tr>"
            f"{table_rows}</table>"
        )
        self._html(_page("训练结果", body, active="results"))

    def _result_tools(self, project_name: str, run_id: str) -> None:
        state = load_project(project_name, root=self.app.root)
        run = next((item for item in state.payload.get("runs", []) if str(item.get("id")) == run_id), None)
        if run is None:
            raise PipelineError("Run does not exist")
        checkpoints = [Path(value) for value in run.get("checkpoints", []) if Path(value).is_file()]
        choices = "".join(
            f"<label style='display:block'><input style='width:auto' type='checkbox' name='checkpoints' value='{_e(path.name)}'> {_e(path.name)}</label>"
            for path in checkpoints
        )
        evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
        body = (
            f"<div class='hero'><h1>Run {_e(run_id)} · 评测</h1>"
            f"<div class='muted'><a href='/results/{_q(project_name)}/{_q(run_id)}'>返回结果图片 / 权重</a></div></div>"
            "<div class='panel'><form method='post' "
            f"action='/result-tools/{_q(project_name)}/{_q(run_id)}/evaluate'>"
            f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
            "<label>阶段<select name='stage'><option value='screening'>Screening</option>"
            "<option value='full'>Full</option></select></label>"
            f"<div style='margin-top:10px'>{choices or '<span class=muted>暂无 checkpoint</span>'}</div>"
            "<p class='muted'>Screening 按 profile 选择；Full 必须勾选 1–2 个 finalist。</p>"
            f"<p class='muted'>已有评测：{_e(', '.join(sorted(evidence)) or '无')}</p>"
            "<div class='toolbar'><button class='good'>后台运行评测</button></div></form></div>"
        )
        self._html(_page(f"Run {run_id} · 评测", body, active="results"))

    def _result_evaluate(self, project_name: str, run_id: str, form: dict[str, list[str]]) -> None:
        state = load_project(project_name, root=self.app.root)
        run = next((item for item in state.payload.get("runs", []) if str(item.get("id")) == run_id), None)
        if run is None:
            raise PipelineError("Run does not exist")
        stage = form.get("stage", ["screening"])[0]
        if stage not in {"screening", "full"}:
            raise PipelineError("Unknown evaluation stage")
        checkpoints = form.get("checkpoints", [])
        if stage == "full" and not 1 <= len(checkpoints) <= 2:
            raise PipelineError("Full 评测必须选择 1–2 个 finalist checkpoint")
        evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
        job = spawn_job(
            "evaluate",
            {
                "project": project_name,
                "run_id": run_id,
                "stage": stage,
                "checkpoints": checkpoints,
                "force": stage in evidence,
            },
            root=self.app.root,
        )
        self._redirect(f"/jobs/{_q(str(job['id']))}")

    def _file(self, path: Path, *, inline: bool) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        filename = path.name.replace('"', "")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("X-Content-Type-Options", "nosniff")
        disposition = "inline" if inline else "attachment"
        self.send_header("Content-Disposition", f'{disposition}; filename="{filename}"')
        self.end_headers()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)


def make_server(host: str = "127.0.0.1", port: int = 7860, *, root: Path | None = None) -> ThreadingHTTPServer:
    app = WebApplication(root=root)
    handler = type("BoundFullHandler", (FullHandler,), {"app": app})
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
    parser = argparse.ArgumentParser(description="LoRA Pipeline NAS web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--allow-lan", action="store_true")
    args = parser.parse_args(argv)
    serve(args.host, args.port, allow_lan=args.allow_lan)
