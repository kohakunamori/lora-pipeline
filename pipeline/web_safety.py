from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .dataset_deletion import delete_dataset_source
from .dataset_workspace import DatasetWorkspace
from .models import PipelineError
from .web_app import WebApplication, _e, _page, _q
from .web_routes import FullHandler as BaseFullHandler


class FullHandler(BaseFullHandler):
    """Final Web handler with explicit confirmation for destructive Source deletion."""

    def _post(self, form: dict[str, list[str]]) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path.startswith("/dataset-tools/") and path.endswith("/delete-source"):
            parts = path.split("/")
            if len(parts) == 4:
                self._delete_source_confirmed(unquote(parts[2]), form)
                return
        super()._post(form)

    def _source_action(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        source_id = form.get("source_id", [""])[0]
        if source_id not in workspace.sources:
            raise PipelineError(f"Unknown dataset source: {source_id}")
        action = form.get("action", [""])[0]
        if action == "toggle":
            source = workspace.sources[source_id]
            workspace.set_source_enabled(source_id, not bool(source.get("enabled", True)))
            self._redirect(f"/datasets/{_q(name)}")
            return
        if action != "delete":
            raise PipelineError("Unknown source action")

        source = workspace.sources[source_id]
        items = workspace.items(
            source_id=source_id,
            include_disabled=True,
            include_excluded=True,
        )
        children = [
            child_id
            for child_id, child in workspace.sources.items()
            if child.get("parent_source") == source_id
        ]
        child_note = ""
        if children:
            child_note = (
                "<p class='warn'>派生来源不会级联删除："
                + _e(", ".join(children))
                + "</p>"
            )
        body = (
            "<div class='hero'><h1 class='bad'>确认删除整个 Source</h1>"
            f"<div class='muted'><a href='/datasets/{_q(name)}'>取消并返回数据集</a></div></div>"
            "<div class='panel danger-zone'>"
            f"<p><b>{_e(source.get('label') or source_id)}</b></p>"
            f"<p>ID：<span class='mono'>{_e(source_id)}</span><br>Dataset 图片副本：{len(items)}</p>"
            f"{child_note}"
            "<p class='muted'>只删除 Dataset 工作区中的这个 Source；不会删除原始目录/视频，也不会修改已冻结的 Run。</p>"
            f"<form method='post' action='/dataset-tools/{_q(name)}/delete-source'>"
            f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
            f"<input type='hidden' name='source_id' value='{_e(source_id)}'>"
            f"<label>输入 Source ID <b>{_e(source_id)}</b> 确认<input name='confirm' autocomplete='off'></label>"
            "<div class='toolbar'><button class='danger'>永久删除这个 Source</button></div></form></div>"
        )
        self._html(_page("确认删除 Source", body, active="datasets"))

    def _delete_source_confirmed(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        source_id = form.get("source_id", [""])[0]
        if form.get("confirm", [""])[0] != source_id:
            raise PipelineError("Source ID 确认不匹配")
        delete_dataset_source(workspace, source_id)
        self._redirect(f"/datasets/{_q(name)}")


def make_server(host: str = "127.0.0.1", port: int = 7860, *, root: Path | None = None) -> ThreadingHTTPServer:
    app = WebApplication(root=root)
    handler = type("BoundSafeFullHandler", (FullHandler,), {"app": app})
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
