from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from ..models import PipelineError

PREVIEW_FILENAME = "preview.html"


def write_generation_preview(
    project_dir: Path,
    generation_root: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Write a self-contained local HTML review page for one prepared generation.

    The preview is deliberately outside the generation hash contract: it is a
    deterministic review artifact derived from the immutable manifest and files.
    Training never reads it.
    """

    project_dir = project_dir.resolve()
    generation_root = generation_root.resolve()
    payload = dict(manifest or _load_manifest(generation_root / "manifest.json"))
    records = payload.get("images", [])
    if not isinstance(records, list):
        raise PipelineError("Prepared manifest images must be a list")

    cards = [
        _render_record(project_dir, generation_root, record, index=index)
        for index, record in enumerate(records, start=1)
        if isinstance(record, Mapping)
    ]
    target_type = html.escape(str(payload.get("training_target_type", "unknown")))
    generation_id = html.escape(str(payload.get("generation_id", generation_root.name)))
    manifest_hash = html.escape(str(payload.get("manifest_hash", "")))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoRA materialization preview - {generation_id}</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0; padding: 24px; background: Canvas; color: CanvasText; }}
header {{ margin-bottom: 24px; }}
.meta {{ opacity: .72; font-size: .9rem; overflow-wrap: anywhere; }}
.card {{ border: 1px solid color-mix(in srgb, CanvasText 24%, transparent); border-radius: 12px; padding: 16px; margin: 0 0 20px; }}
.images {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
figure {{ margin: 0; }}
figcaption {{ margin-bottom: 8px; font-weight: 600; }}
img {{ width: 100%; max-height: 720px; object-fit: contain; background: color-mix(in srgb, CanvasText 5%, Canvas); border-radius: 8px; }}
.caption {{ margin-top: 14px; padding: 12px; border-radius: 8px; background: color-mix(in srgb, CanvasText 7%, Canvas); white-space: pre-wrap; overflow-wrap: anywhere; }}
.details {{ margin-top: 10px; font-size: .9rem; opacity: .78; overflow-wrap: anywhere; }}
@media (max-width: 760px) {{ .images {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
<h1>Materialization Preview</h1>
<div>Target: <strong>{target_type}</strong> · Images: <strong>{len(cards)}</strong></div>
<div class="meta">Generation: {generation_id}<br>Manifest: {manifest_hash}</div>
</header>
{''.join(cards)}
</body>
</html>
"""
    path = generation_root / PREVIEW_FILENAME
    path.write_text(document, encoding="utf-8")
    return path


def _render_record(
    project_dir: Path,
    generation_root: Path,
    record: Mapping[str, Any],
    *,
    index: int,
) -> str:
    source_rel = str(record.get("source", ""))
    prepared_rel = str(record.get("image", ""))
    caption_rel = str(record.get("caption", ""))
    raw_path = project_dir / "raw" / source_rel
    prepared_path = generation_root / prepared_rel
    caption_path = generation_root / caption_rel

    raw_href = _local_href(generation_root, raw_path)
    prepared_href = _local_href(generation_root, prepared_path)
    caption = (
        caption_path.read_text(encoding="utf-8", errors="replace").strip()
        if caption_path.is_file()
        else "[caption missing]"
    )
    crop = record.get("crop", {})
    crop_reason = str(crop.get("reason", "unknown")) if isinstance(crop, Mapping) else "unknown"
    source_size = _size_text(record.get("source_size"))
    crop_size = _size_text(record.get("crop_size"))
    prepared_size = _size_text(record.get("prepared_size"))
    label = html.escape(source_rel or f"image-{index}")

    return f"""<section class="card">
<h2>{index}. {label}</h2>
<div class="images">
<figure><figcaption>RAW</figcaption><img loading="lazy" src="{raw_href}" alt="raw {label}"></figure>
<figure><figcaption>TRAINING PIXELS</figcaption><img loading="lazy" src="{prepared_href}" alt="prepared {label}"></figure>
</div>
<div class="caption"><strong>Caption</strong><br>{html.escape(caption)}</div>
<div class="details">crop={html.escape(crop_reason)} · source={source_size} · crop={crop_size} · prepared={prepared_size} · downscaled={bool(record.get('downscaled'))}</div>
</section>
"""


def _local_href(base: Path, target: Path) -> str:
    relative = Path(os.path.relpath(target.resolve(), start=base.resolve())).as_posix()
    return quote(relative, safe="/._-")


def _size_text(value: object) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return html.escape(f"{value[0]}×{value[1]}")
    return "?"


def _load_manifest(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise PipelineError(f"Prepared manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Prepared manifest is invalid: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"Prepared manifest must be a JSON object: {path}")
    return payload
