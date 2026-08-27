from __future__ import annotations

import argparse
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from .dataset_tag_editor import batch_edit_tags, parse_tag_input
from .dataset_workspace import DatasetWorkspace
from .models import PipelineError
from .web_app import PAGE_SIZE, WebApplication, _e, _page, _q
from .web_entry import FinalHandler as BaseFinalHandler


class FinalHandler(BaseFinalHandler):
    """Final Web handler with multi-select tag prepend/append/remove controls."""

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
        body = (
            f"<div class='hero'><h1>{_e(source.get('label') or source_id)}</h1>"
            f"<div class='muted'>{_e(name)} / {_e(source_id)} · {len(items)} 张图片</div></div>"
            f"<form id='bulk' method='post' action='/datasets/{_q(name)}/bulk'>"
            f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
            f"<input type='hidden' name='source_id' value='{_e(source_id)}'>"
            "<div class='panel' style='margin-bottom:18px'><h3>批量操作</h3>"
            "<label>批量 Tag（逗号或换行分隔）<textarea name='tags' placeholder='trigger, 1girl'></textarea></label>"
            "<div class='toolbar'>"
            "<button class='good' name='action' value='tag-prepend'>Tag 添加到首部</button>"
            "<button class='good' name='action' value='tag-append'>Tag 添加到尾部</button>"
            "<button name='action' value='tag-remove'>删除指定 Tag</button>"
            "</div><div class='toolbar'>"
            "<button class='good' name='action' value='restore'>恢复所选</button>"
            "<button name='action' value='exclude'>排除所选</button>"
            "<button class='danger' name='action' value='delete'>永久删除所选</button>"
            "<span class='muted'>Tag 操作只修改 Dataset caption；永久删除会同时删除同名 .txt。</span>"
            "</div></div></form>"
            f"<div class='images'>{''.join(cards) or '<div class=muted>这个来源没有图片。</div>'}</div>"
            f"<div class='pager'>{prev_link}<span class='pill'>第 {page}/{page_count} 页</span>{next_link}</div>"
        )
        self._html(_page(f"{name} / {source_id}", body, active="datasets"))

    def _dataset_bulk(self, name: str, form: dict[str, list[str]]) -> None:
        action = form.get("action", [""])[0]
        action_map = {
            "tag-prepend": "prepend",
            "tag-append": "append",
            "tag-remove": "remove",
        }
        if action not in action_map:
            super()._dataset_bulk(name, form)
            return

        keys = form.get("keys", [])
        if not keys:
            raise PipelineError("请至少选择一张图片")
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        batch_edit_tags(
            workspace,
            keys,
            parse_tag_input(form.get("tags", [""])[0]),
            action=action_map[action],
        )
        source_id = form.get("source_id", [""])[0]
        self._redirect(f"/datasets/{_q(name)}/source/{_q(source_id)}")


def make_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    root: Path | None = None,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    app = WebApplication(root=root)
    app.auth_token = auth_token or None
    handler = type("BoundBatchTagHandler", (FinalHandler,), {"app": app})
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
            "LAN mode requires LORA_WEB_TOKEN/--token. "
            "Use --unsafe-no-auth only on an explicitly trusted private network."
        )
    server = make_server(host, port, root=root, auth_token=auth_token)
    print(f"LoRA Pipeline Web: http://{host}:{port}")
    if loopback:
        print(
            f"Remote browser: ssh -L {port}:127.0.0.1:{port} <nas>  then open http://127.0.0.1:{port}"
        )
    elif auth_token:
        print("LAN mode enabled with access-token authentication.")
        print("Prefer HTTPS/reverse proxy if the LAN itself is not trusted.")
    else:
        print("WARNING: unauthenticated LAN mode was explicitly enabled.")
        print("Never expose this service directly to the public Internet.")
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
    parser.add_argument("--token", default=os.environ.get("LORA_WEB_TOKEN"))
    parser.add_argument(
        "--unsafe-no-auth",
        action="store_true",
        help="Allow explicit non-loopback binding without Web authentication.",
    )
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
