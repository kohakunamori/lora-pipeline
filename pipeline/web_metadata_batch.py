from __future__ import annotations

import argparse
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from .dataset_metadata import COMPOSITION_TYPES, composition_summary, item_metadata, prune_source_metadata
from .dataset_tag_editor import batch_edit_tags, parse_tag_input
from .dataset_workspace import DatasetWorkspace
from .models import PipelineError
from .web_app import PAGE_SIZE, WebApplication, _e, _page, _q
from .web_metadata import (
    MetadataHandler,
    _COMPOSITION_LABELS,
    _VARIANT_LABELS,
    _composition_badges,
    _percent,
    _variant_badges,
)


class MetadataBatchHandler(MetadataHandler):
    """Composition-aware image wall plus the existing batch tag operations."""

    def _source(self, name: str, source_id: str, query: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        if source_id not in workspace.sources:
            raise PipelineError(f"Unknown dataset source: {source_id}")
        all_items = workspace.items(
            source_id=source_id,
            include_disabled=True,
            include_excluded=True,
        )
        composition_filter = query.get("composition", ["all"])[0]
        variant_filter = query.get("variant", ["all"])[0]
        state_filter = query.get("state", ["all"])[0]
        enriched: list[tuple[Any, dict[str, Any]]] = []
        for item in all_items:
            metadata = item_metadata(workspace, item)
            if composition_filter != "all" and metadata.get("composition_type") != composition_filter:
                continue
            if variant_filter != "all" and metadata.get("variant_kind") != variant_filter:
                continue
            if state_filter == "active" and item.excluded:
                continue
            if state_filter == "excluded" and not item.excluded:
                continue
            enriched.append((item, metadata))

        page = max(1, int(query.get("page", ["1"])[0]))
        page_count = max(1, (len(enriched) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(page, page_count)
        start = (page - 1) * PAGE_SIZE
        current = enriched[start : start + PAGE_SIZE]
        cards: list[str] = []
        for item, metadata in current:
            media = f"/media/dataset/{_q(name)}/{_q(source_id)}/{quote(item.relative.as_posix())}"
            state = "已排除" if item.excluded else "保留"
            composition = str(metadata.get("composition_type") or "unknown")
            variant = str(metadata.get("variant_kind") or "original")
            resolution = metadata.get("resolution") or {}
            analysis = metadata.get("analysis") or {}
            quality = metadata.get("quality") or {}
            identity = metadata.get("identity") or {}
            coverage = _percent(analysis.get("subject_coverage") or analysis.get("subject_area_ratio"))
            head_ratio = _percent(analysis.get("head_to_person_ratio"))
            cluster = identity.get("ccip_cluster")
            cluster_badge = f"<span class='pill'>CCIP {int(cluster)}</span>" if cluster is not None else ""
            details = [
                f"{int(resolution.get('width') or 0)}×{int(resolution.get('height') or 0)}",
                str(quality.get("tier") or resolution.get("tier") or "unknown"),
            ]
            if coverage:
                details.append(f"人物占比 {coverage}")
            if head_ratio:
                details.append(f"头/人物 {head_ratio}")
            group_id = str(metadata.get("source_group_id") or "")
            cards.append(
                f"<div class='image-card'><img loading='lazy' src='{media}' alt='{_e(item.relative)}'>"
                "<div class='image-body'>"
                f"<div class='image-title'><label><input style='width:auto' form='bulk' type='checkbox' name='keys' value='{_e(item.key)}'> {_e(item.relative.as_posix())}</label></div>"
                f"<div class='toolbar compact'><span class='pill'>{_e(_COMPOSITION_LABELS.get(composition, composition))}</span>"
                f"<span class='pill'>{_e(_VARIANT_LABELS.get(variant, variant))}</span>"
                f"<span class='pill'>{state}</span>{cluster_badge}</div>"
                f"<div class='compact muted'>{_e(' · '.join(details))}</div>"
                f"<div class='compact muted'>组：{_e(group_id or '—')} · 分析：{_e(analysis.get('status') or 'not_analyzed')}</div>"
                f"<form method='post' action='/datasets/{_q(name)}/tag' style='margin-top:8px'>"
                f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
                f"<input type='hidden' name='key' value='{_e(item.key)}'>"
                f"<textarea name='caption'>{_e(workspace.caption_text(item.key))}</textarea>"
                "<div class='toolbar'><button>保存 Tag</button></div></form></div></div>"
            )

        source = workspace.sources[source_id]
        summary = composition_summary(workspace, source_id=source_id)
        filters = _filter_form(
            composition_filter=composition_filter,
            variant_filter=variant_filter,
            state_filter=state_filter,
        )
        base_query = {
            "composition": composition_filter,
            "variant": variant_filter,
            "state": state_filter,
        }
        prev_link = _page_link(page - 1, base_query, "上一页") if page > 1 else ""
        next_link = _page_link(page + 1, base_query, "下一页") if page < page_count else ""
        body = (
            f"<div class='hero'><h1>{_e(source.get('label') or source_id)}</h1>"
            f"<div class='muted'>{_e(name)} / {_e(source_id)} · {len(all_items)} 张图片 · 当前筛选 {len(enriched)} 张</div></div>"
            f"<div class='grid'><div class='card'><div class='muted'>构图</div>{_composition_badges(summary['composition_counts'])}</div>"
            f"<div class='card'><div class='muted'>变体</div>{_variant_badges(summary['variant_counts'])}</div>"
            f"<div class='card'><div class='muted'>分析进度</div><div class='metric'>{summary['analyzed']}/{summary['total']}</div></div></div>"
            f"<div class='panel' style='margin-top:14px'><form method='get'>{filters}<div class='toolbar'><button>筛选</button>"
            f"<a class='button' href='/datasets/{_q(name)}/source/{_q(source_id)}'>重置</a></div></form>"
            f"<form method='post' action='/metadata-tools/{_q(name)}/analyze'><input type='hidden' name='_csrf' value='{self.app.csrf}'>"
            f"<input type='hidden' name='source_id' value='{_e(source_id)}'><button>后台分析这个 Source</button></form></div>"
            f"<form id='bulk' method='post' action='/datasets/{_q(name)}/bulk'>"
            f"<input type='hidden' name='_csrf' value='{self.app.csrf}'><input type='hidden' name='source_id' value='{_e(source_id)}'>"
            "<div class='panel' style='margin-top:14px'><h3>批量操作</h3>"
            "<label>批量 Tag（逗号或换行分隔）<textarea name='tags' placeholder='trigger, 1girl'></textarea></label>"
            "<div class='toolbar'><button class='good' name='action' value='tag-prepend'>Tag 添加到首部</button>"
            "<button class='good' name='action' value='tag-append'>Tag 添加到尾部</button>"
            "<button name='action' value='tag-remove'>删除指定 Tag</button></div>"
            "<div class='toolbar'><button class='good' name='action' value='restore'>恢复所选</button>"
            "<button name='action' value='exclude'>排除所选</button>"
            "<button class='danger' name='action' value='delete'>永久删除所选</button>"
            "<span class='muted'>Tag 操作只修改 Dataset caption；永久删除会同时删除同名 .txt 和对应 metadata。</span></div></div></form>"
            f"<div class='images'>{''.join(cards) or '<div class=muted>当前筛选没有图片。</div>'}</div>"
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
        if action in action_map:
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
            return
        super()._dataset_bulk(name, form)


def _filter_form(*, composition_filter: str, variant_filter: str, state_filter: str) -> str:
    compositions = ["all", *COMPOSITION_TYPES]
    variants = ["all", "original", "original_full", "smart_crop", "derived_crop"]
    states = [("all", "全部状态"), ("active", "仅保留"), ("excluded", "仅已排除")]
    composition_options = "".join(
        f"<option value='{_e(value)}' {'selected' if value == composition_filter else ''}>{_e('全部构图' if value == 'all' else _COMPOSITION_LABELS.get(value, value))}</option>"
        for value in compositions
    )
    variant_options = "".join(
        f"<option value='{_e(value)}' {'selected' if value == variant_filter else ''}>{_e('全部变体' if value == 'all' else _VARIANT_LABELS.get(value, value))}</option>"
        for value in variants
    )
    state_options = "".join(
        f"<option value='{_e(value)}' {'selected' if value == state_filter else ''}>{_e(label)}</option>"
        for value, label in states
    )
    return (
        "<div class='row'>"
        f"<label>构图<select name='composition'>{composition_options}</select></label>"
        f"<label>变体<select name='variant'>{variant_options}</select></label>"
        f"<label>状态<select name='state'>{state_options}</select></label>"
        "</div>"
    )


def _page_link(page: int, query: dict[str, str], label: str) -> str:
    values = dict(query)
    values["page"] = str(page)
    return f"<a class='button' href='?{urlencode(values)}'>{label}</a>"


def make_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    root: Path | None = None,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    app = WebApplication(root=root)
    app.auth_token = auth_token or None
    handler = type("BoundMetadataBatchHandler", (MetadataBatchHandler,), {"app": app})
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
            "LAN mode requires LORA_WEB_TOKEN/--token. Use --unsafe-no-auth only on an explicitly trusted private network."
        )
    server = make_server(host, port, root=root, auth_token=auth_token)
    print(f"LoRA Pipeline Web: http://{host}:{port}")
    if loopback:
        print(f"Remote browser: ssh -L {port}:127.0.0.1:{port} <nas>  then open http://127.0.0.1:{port}")
    elif auth_token:
        print("LAN mode enabled with access-token authentication.")
    else:
        print("WARNING: unauthenticated LAN mode was explicitly enabled.")
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
