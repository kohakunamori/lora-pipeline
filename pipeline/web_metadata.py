from __future__ import annotations

import argparse
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse

from . import web_routes as _web_routes
from .dataset_deletion import delete_dataset_items
from .dataset_metadata import (
    COMPOSITION_TYPES,
    composition_summary,
    item_metadata,
    prune_source_metadata,
)
from .dataset_workspace import DatasetWorkspace
from .models import PipelineError
from .web_app import PAGE_SIZE, WebApplication, _e, _page, _q
from .web_entry import FinalHandler
from .web_jobs_enriched import resume_job as enriched_resume_job
from .web_jobs_enriched import spawn_job as enriched_spawn_job


# Existing FullHandler methods resolve these names from web_routes at call time. Point
# them at the enriched worker layer so video finalization and all future long jobs use
# the same persistent job format without duplicating the route implementation.
_web_routes.spawn_job = enriched_spawn_job
_web_routes.resume_job = enriched_resume_job


_COMPOSITION_LABELS = {
    "portrait": "Portrait / 头肩",
    "upper_body": "Upper body / 上半身",
    "three_quarter": "3/4 body",
    "full_body": "Full body / 全身",
    "context": "Context / 环境",
    "unknown": "Unknown / 未分析",
}
_VARIANT_LABELS = {
    "original": "原始图",
    "original_full": "保留全图",
    "smart_crop": "智能裁切",
    "derived_crop": "派生裁切",
}


class MetadataHandler(FinalHandler):
    """Final Web handler with source/image composition provenance and filtering."""

    def _post(self, form: dict[str, list[str]]) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path.startswith("/metadata-tools/") and path.endswith("/analyze"):
            parts = path.split("/")
            if len(parts) == 4:
                self._metadata_analyze(parts[2], form)
                return
        super()._post(form)

    def _dataset(self, name: str) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        summary = workspace.summary()
        compositions = composition_summary(workspace)
        source_rows: list[str] = []
        for source_id, source in sorted(workspace.sources.items()):
            items = workspace.items(
                source_id=source_id,
                include_disabled=True,
                include_excluded=True,
            )
            source_compositions = composition_summary(workspace, source_id=source_id)
            source_rows.append(
                "<tr>"
                f"<td><a href='/datasets/{_q(name)}/source/{_q(source_id)}'><b>{_e(source.get('label') or source_id)}</b></a>"
                f"<br><span class='muted'>{_e(source_id)}</span></td>"
                f"<td>{_e(source.get('kind'))}</td>"
                f"<td>{'启用' if source.get('enabled', True) else '停用'}</td>"
                f"<td>{len(items)}</td>"
                f"<td>{_composition_badges(source_compositions['active_composition_counts'])}</td>"
                f"<td>{source_compositions['analyzed']}/{source_compositions['total']}</td>"
                f"<td><form method='post' action='/datasets/{_q(name)}/source-action'>"
                f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
                f"<input type='hidden' name='source_id' value='{_e(source_id)}'>"
                f"<button name='action' value='toggle'>{'停用' if source.get('enabled', True) else '启用'}</button> "
                "<button class='danger' name='action' value='delete'>删除来源</button></form></td></tr>"
            )
        body = (
            f"<div class='hero'><h1>{_e(name)}</h1>"
            f"<div class='muted'>{_e(workspace.concept_type)} · {summary['sources']} 个来源 · "
            f"{summary['active_images']} 张可训练图片 · {summary['excluded_images']} 张已排除</div></div>"
            "<div class='grid'>"
            f"<div class='card'><div class='muted'>构图分布（可训练）</div><div style='margin-top:8px'>{_composition_badges(compositions['active_composition_counts'])}</div></div>"
            f"<div class='card'><div class='muted'>已完成构图分析</div><div class='metric'>{compositions['analyzed']}/{compositions['total']}</div>"
            "<div class='compact muted'>视频智能裁切会直接带 metadata；普通图片可后台分析。</div></div>"
            f"<div class='card'><div class='muted'>变体</div><div style='margin-top:8px'>{_variant_badges(compositions['variant_counts'])}</div></div>"
            "</div>"
            f"<div class='toolbar'><a class='button' href='/dataset-tools/{_q(name)}'>导入 / 自动处理</a>"
            f"<form method='post' action='/metadata-tools/{_q(name)}/analyze'>"
            f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
            "<button>后台分析全部图片构图</button></form></div>"
            "<table><tr><th>来源</th><th>类型</th><th>状态</th><th>图片</th><th>构图</th><th>已分析</th><th>操作</th></tr>"
            f"{''.join(source_rows) or '<tr><td colspan=7 class=muted>暂无来源</td></tr>'}</table>"
            f"<div class='panel danger-zone' style='margin-top:18px'><h3 class='bad'>危险操作</h3>"
            f"<p class='muted'>删除 Dataset 只删除 datasets/{_e(name)} 中的副本，不删除原始导入素材，也不影响已经冻结的训练 Run。</p>"
            f"<form method='post' action='/datasets/{_q(name)}/delete'>"
            f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>"
            "<label>输入数据集名称确认</label><input name='confirm' autocomplete='off'>"
            "<div class='toolbar'><button class='danger'>永久删除整个数据集</button></div></form></div>"
        )
        self._html(_page(name, body, active="datasets"))

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
            "<div class='toolbar'><button class='good' name='action' value='restore'>恢复所选</button>"
            "<button name='action' value='exclude'>排除所选</button>"
            "<button class='danger' name='action' value='delete'>永久删除所选</button>"
            "<span class='muted'>永久删除同时删除 Tag；日常清洗优先使用排除。</span></div></form>"
            f"<div class='images'>{''.join(cards) or '<div class=muted>当前筛选没有图片。</div>'}</div>"
            f"<div class='pager'>{prev_link}<span class='pill'>第 {page}/{page_count} 页</span>{next_link}</div>"
        )
        self._html(_page(f"{name} / {source_id}", body, active="datasets"))

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
            source_ids = {key.split("/", 1)[0] for key in keys if "/" in key}
            workspace = DatasetWorkspace.load(name, root=self.app.root)
            for source_id in source_ids:
                if source_id in workspace.sources:
                    prune_source_metadata(workspace, source_id)
        else:
            raise PipelineError("Unknown bulk action")
        source_id = form.get("source_id", [""])[0]
        self._redirect(f"/datasets/{_q(name)}/source/{_q(source_id)}")

    def _metadata_analyze(self, name: str, form: dict[str, list[str]]) -> None:
        workspace = DatasetWorkspace.load(name, root=self.app.root)
        if workspace.concept_type != "character":
            raise PipelineError("只有人物 Dataset 需要人物构图分析")
        source_id = form.get("source_id", [""])[0].strip()
        if source_id and source_id not in workspace.sources:
            raise PipelineError(f"Unknown dataset source: {source_id}")
        job = enriched_spawn_job(
            "dataset_analyze",
            {
                "dataset": name,
                "source_id": source_id,
                "detection_proxy_long_edge": 1280,
            },
            root=self.app.root,
        )
        self._redirect(f"/jobs/{_q(str(job['id']))}")


def _composition_badges(counts: dict[str, Any]) -> str:
    values = []
    for key in COMPOSITION_TYPES:
        count = int(counts.get(key, 0) or 0)
        if count:
            values.append(f"<span class='pill'>{_e(_COMPOSITION_LABELS.get(key, key))} {count}</span>")
    return " ".join(values) or "<span class='muted'>暂无</span>"


def _variant_badges(counts: dict[str, Any]) -> str:
    values = []
    for key, count in sorted(counts.items()):
        if int(count or 0):
            values.append(f"<span class='pill'>{_e(_VARIANT_LABELS.get(key, key))} {int(count)}</span>")
    return " ".join(values) or "<span class='muted'>暂无</span>"


def _percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number * 100:.0f}%"


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
    handler = type("BoundMetadataHandler", (MetadataHandler,), {"app": app})
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
