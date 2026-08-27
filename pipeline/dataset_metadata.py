from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, TYPE_CHECKING

from PIL import Image

from .config import write_json_atomic
from .models import OptionalBackendUnavailable, PipelineError
from .state import utc_now

if TYPE_CHECKING:
    from .dataset_workspace import DatasetItem, DatasetWorkspace


COMPOSITION_TYPES = (
    "portrait",
    "upper_body",
    "three_quarter",
    "full_body",
    "context",
    "unknown",
)
VARIANT_KINDS = ("original", "original_full", "smart_crop", "derived_crop")


def source_metadata_path(workspace: "DatasetWorkspace", source_id: str) -> Path:
    return workspace.source_dir(source_id) / "metadata.json"


def load_source_metadata(workspace: "DatasetWorkspace", source_id: str) -> dict[str, dict[str, Any]]:
    path = source_metadata_path(workspace, source_id)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Invalid dataset source metadata: {path}") from exc
    raw = payload.get("items", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        raise PipelineError(f"Invalid dataset source metadata items: {path}")
    return {
        str(relative): dict(value or {})
        for relative, value in raw.items()
        if isinstance(value, dict)
    }


def save_source_metadata(
    workspace: "DatasetWorkspace",
    source_id: str,
    items: Mapping[str, Mapping[str, Any]],
) -> None:
    write_json_atomic(
        source_metadata_path(workspace, source_id),
        {
            "schema_version": 1,
            "dataset": workspace.name,
            "source_id": source_id,
            "updated_at": utc_now(),
            "items": {
                str(key): dict(value)
                for key, value in sorted(items.items(), key=lambda pair: pair[0].casefold())
            },
        },
    )


def default_item_metadata(item: "DatasetItem") -> dict[str, Any]:
    width = height = 0
    try:
        with Image.open(item.image) as opened:
            width, height = opened.size
    except OSError:
        pass
    short_edge = min(width, height) if width and height else 0
    long_edge = max(width, height) if width and height else 0
    return {
        "schema_version": 1,
        "source_group_id": item.relative.with_suffix("").as_posix(),
        "variant_kind": "original",
        "composition_type": "unknown",
        "resolution": {
            "width": width,
            "height": height,
            "short_edge": short_edge,
            "long_edge": long_edge,
            "tier": _resolution_tier(width, height),
        },
        "analysis": {
            "status": "not_analyzed",
            "person_count": None,
            "person_bbox": None,
            "head_bbox": None,
            "subject_height_ratio": None,
            "subject_area_ratio": None,
            "head_height_ratio": None,
            "head_to_person_ratio": None,
            "full_keep_score": None,
        },
        "quality": {
            "tier": _resolution_tier(width, height),
        },
    }


def item_metadata(
    workspace: "DatasetWorkspace",
    item: "DatasetItem",
    *,
    persist_default: bool = False,
) -> dict[str, Any]:
    items = load_source_metadata(workspace, item.source_id)
    relative = item.relative.as_posix()
    base = default_item_metadata(item)
    stored = items.get(relative)
    if stored:
        return _deep_merge(base, stored)
    if persist_default:
        items[relative] = base
        save_source_metadata(workspace, item.source_id, items)
    return base


def set_item_metadata(
    workspace: "DatasetWorkspace",
    item: "DatasetItem",
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    items = load_source_metadata(workspace, item.source_id)
    merged = _deep_merge(default_item_metadata(item), dict(metadata))
    merged["updated_at"] = utc_now()
    items[item.relative.as_posix()] = merged
    save_source_metadata(workspace, item.source_id, items)
    return merged


def seed_source_defaults(workspace: "DatasetWorkspace", source_id: str) -> int:
    stored = load_source_metadata(workspace, source_id)
    changed = 0
    for item in workspace.items(
        source_id=source_id,
        include_disabled=True,
        include_excluded=True,
    ):
        key = item.relative.as_posix()
        if key in stored:
            continue
        stored[key] = default_item_metadata(item)
        changed += 1
    if changed:
        save_source_metadata(workspace, source_id, stored)
    return changed


def import_composition_records(
    workspace: "DatasetWorkspace",
    source_id: str,
    records: Iterable[Mapping[str, Any]],
    *,
    selected_cluster: int | None = None,
) -> int:
    by_name = {
        item.relative.as_posix(): item
        for item in workspace.items(
            source_id=source_id,
            include_disabled=True,
            include_excluded=True,
        )
    }
    stored = load_source_metadata(workspace, source_id)
    changed = 0
    for raw in records:
        relative = str(raw.get("path") or "")
        item = by_name.get(relative)
        if item is None:
            item = next(
                (candidate for key, candidate in by_name.items() if Path(key).name == Path(relative).name),
                None,
            )
        if item is None:
            continue
        composition = str(raw.get("composition_type") or raw.get("crop_type") or "unknown")
        if composition not in COMPOSITION_TYPES:
            composition = "unknown"
        variant = str(raw.get("variant_kind") or "smart_crop")
        if variant not in VARIANT_KINDS:
            variant = "derived_crop"
        saved = raw.get("saved_resolution") or []
        width = int(saved[0]) if isinstance(saved, (list, tuple)) and len(saved) >= 2 else 0
        height = int(saved[1]) if isinstance(saved, (list, tuple)) and len(saved) >= 2 else 0
        metadata = {
            "schema_version": 1,
            "source_group_id": str(raw.get("source_group_id") or raw.get("source_frame") or item.relative.stem),
            "variant_kind": variant,
            "composition_type": composition,
            "derived_from": raw.get("source_frame"),
            "subject_id": raw.get("subject_id"),
            "resolution": {
                "width": width,
                "height": height,
                "short_edge": min(width, height) if width and height else 0,
                "long_edge": max(width, height) if width and height else 0,
                "tier": _resolution_tier(width, height),
                "native": list(raw.get("native_resolution") or []),
                "downscaled": bool(raw.get("downscaled", False)),
            },
            "analysis": {
                "status": "derived_from_video_detection",
                "person_count": raw.get("frame_subject_count"),
                "person_bbox": raw.get("person_bbox"),
                "head_bbox": raw.get("head_bbox"),
                "crop_box": raw.get("crop_box"),
                "source_resolution": raw.get("source_resolution"),
                "subject_height_ratio": raw.get("subject_height_ratio"),
                "subject_area_ratio": raw.get("subject_area_ratio"),
                "head_height_ratio": raw.get("head_height_ratio"),
                "head_to_person_ratio": raw.get("head_to_person_ratio"),
                "subject_coverage": raw.get("subject_coverage"),
                "full_keep_score": raw.get("full_keep_score"),
            },
            "quality": {
                "tier": str(raw.get("quality_tier") or _resolution_tier(width, height)),
            },
            "identity": {
                "ccip_cluster": selected_cluster,
                "target_cluster": selected_cluster is not None,
            },
        }
        stored[item.relative.as_posix()] = _deep_merge(default_item_metadata(item), metadata)
        changed += 1
    if changed:
        save_source_metadata(workspace, source_id, stored)
    return changed


def analyze_workspace_compositions(
    workspace: "DatasetWorkspace",
    *,
    source_id: str | None = None,
    detection_proxy_long_edge: int = 1280,
) -> dict[str, Any]:
    if workspace.concept_type != "character":
        raise PipelineError("Composition analysis is currently available for character datasets only")
    items = workspace.items(
        source_id=source_id,
        include_disabled=True,
        include_excluded=True,
    )
    changed = 0
    failures: list[dict[str, str]] = []
    source_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for index, item in enumerate(items, start=1):
        try:
            analysis = analyze_character_image(
                item.image,
                detection_proxy_long_edge=detection_proxy_long_edge,
            )
        except OptionalBackendUnavailable:
            raise
        except Exception as exc:
            failures.append({"key": item.key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        stored = source_cache.setdefault(item.source_id, load_source_metadata(workspace, item.source_id))
        current = _deep_merge(default_item_metadata(item), stored.get(item.relative.as_posix(), {}))
        current = _deep_merge(current, analysis)
        current["updated_at"] = utc_now()
        stored[item.relative.as_posix()] = current
        changed += 1
        if index % 25 == 0:
            for current_source, payload in source_cache.items():
                save_source_metadata(workspace, current_source, payload)
    for current_source, payload in source_cache.items():
        save_source_metadata(workspace, current_source, payload)
    return {
        "dataset": workspace.name,
        "source_id": source_id,
        "analyzed": changed,
        "failed": len(failures),
        "failures": failures[:20],
        "composition": composition_summary(workspace, source_id=source_id),
    }


def analyze_character_image(
    path: Path,
    *,
    detection_proxy_long_edge: int = 1280,
) -> dict[str, Any]:
    try:
        from imgutils.detect import detect_heads, detect_person
    except ImportError as exc:
        raise OptionalBackendUnavailable("DeepGHS imgutils anime character detectors are unavailable") from exc

    with Image.open(path) as opened:
        source = opened.convert("RGB")
    proxy, scale_x, scale_y = _make_proxy(source, detection_proxy_long_edge)
    try:
        persons = detect_person(
            proxy,
            level="m",
            version="v1.1",
            conf_threshold=0.3,
            iou_threshold=0.5,
        )
        heads = detect_heads(proxy, conf_threshold=0.4, iou_threshold=0.7)
    except Exception as exc:
        raise OptionalBackendUnavailable(f"DeepGHS composition analysis failed: {exc}") from exc

    width, height = source.size
    resolution = {
        "width": width,
        "height": height,
        "short_edge": min(width, height),
        "long_edge": max(width, height),
        "tier": _resolution_tier(width, height),
    }
    if not persons:
        return {
            "composition_type": "context" if heads else "unknown",
            "resolution": resolution,
            "analysis": {
                "status": "no_person_detection",
                "person_count": 0,
                "person_bbox": None,
                "head_bbox": None,
                "subject_height_ratio": None,
                "subject_area_ratio": None,
                "head_height_ratio": None,
                "head_to_person_ratio": None,
                "full_keep_score": 0.0,
            },
            "quality": {"tier": _resolution_tier(width, height)},
        }

    person_proxy, _label, person_score = max(
        persons,
        key=lambda item: (_box_area(_clip_box(item[0], proxy.width, proxy.height)), float(item[2])),
    )
    person_proxy = _clip_box(person_proxy, proxy.width, proxy.height)
    person_box = _map_box(person_proxy, scale_x, scale_y, width, height)
    head_box = None
    head_score = None
    candidates = []
    for raw_box, _head_label, score in heads:
        clipped = _clip_box(raw_box, proxy.width, proxy.height)
        cx = (clipped[0] + clipped[2]) / 2
        cy = (clipped[1] + clipped[3]) / 2
        if person_proxy[0] <= cx <= person_proxy[2] and person_proxy[1] <= cy <= person_proxy[3]:
            candidates.append((float(score), clipped))
    if candidates:
        head_score, best = max(candidates, key=lambda item: item[0])
        head_box = _map_box(best, scale_x, scale_y, width, height)

    ratios = _subject_ratios(person_box, head_box, width, height)
    composition = classify_composition(**ratios)
    keep_score = full_keep_score(
        quality_tier=_native_quality(person_box, head_box),
        person_count=len(persons),
        width=width,
        height=height,
        subject_area_ratio=ratios["subject_area_ratio"],
        head_present=head_box is not None,
    )
    return {
        "composition_type": composition,
        "resolution": resolution,
        "analysis": {
            "status": "analyzed",
            "person_count": len(persons),
            "person_bbox": list(person_box),
            "head_bbox": list(head_box) if head_box else None,
            **{key: round(float(value), 4) for key, value in ratios.items()},
            "person_score": round(float(person_score), 4),
            "head_score": round(float(head_score), 4) if head_score is not None else None,
            "full_keep_score": round(keep_score, 4),
        },
        "quality": {"tier": _native_quality(person_box, head_box)},
    }


def classify_composition(
    *,
    subject_height_ratio: float,
    subject_area_ratio: float,
    head_height_ratio: float,
    head_to_person_ratio: float,
) -> str:
    if subject_height_ratio < 0.50 or subject_area_ratio < 0.16:
        return "context"
    if head_to_person_ratio >= 0.38 or head_height_ratio >= 0.28:
        return "portrait"
    if head_to_person_ratio >= 0.26:
        return "upper_body"
    if head_to_person_ratio >= 0.18:
        if subject_height_ratio >= 0.78:
            return "full_body"
        return "three_quarter"
    if subject_height_ratio >= 0.68:
        return "full_body"
    return "three_quarter"


def full_keep_score(
    *,
    quality_tier: str,
    person_count: int,
    width: int,
    height: int,
    subject_area_ratio: float,
    head_present: bool,
) -> float:
    if person_count != 1 or width <= 0 or height <= 0:
        return 0.0
    score = 0.25 if quality_tier == "high" else 0.12 if quality_tier == "medium" else 0.0
    if 0.18 <= subject_area_ratio <= 0.58:
        score += 0.30
    elif 0.10 <= subject_area_ratio <= 0.72:
        score += 0.18
    short_edge = min(width, height)
    if short_edge >= 1024:
        score += 0.20
    elif short_edge >= 768:
        score += 0.15
    elif short_edge >= 512:
        score += 0.08
    if head_present:
        score += 0.10
    aspect = width / max(1, height)
    if 0.45 <= aspect <= 2.20:
        score += 0.05
    score += 0.10
    return min(1.0, score)


def composition_summary(
    workspace: "DatasetWorkspace",
    *,
    source_id: str | None = None,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    active: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    analyzed = 0
    total = 0
    for item in workspace.items(
        source_id=source_id,
        include_disabled=True,
        include_excluded=True,
    ):
        total += 1
        metadata = item_metadata(workspace, item)
        composition = str(metadata.get("composition_type") or "unknown")
        if composition not in COMPOSITION_TYPES:
            composition = "unknown"
        counts[composition] += 1
        variant = str(metadata.get("variant_kind") or "original")
        variants[variant] += 1
        analysis = metadata.get("analysis") or {}
        if analysis.get("status") not in {None, "not_analyzed"}:
            analyzed += 1
        if item.source_enabled and not item.excluded:
            active[composition] += 1
    return {
        "total": total,
        "analyzed": analyzed,
        "composition_counts": dict(sorted(counts.items())),
        "active_composition_counts": dict(sorted(active.items())),
        "variant_counts": dict(sorted(variants.items())),
    }


def prune_source_metadata(workspace: "DatasetWorkspace", source_id: str) -> int:
    existing = {
        item.relative.as_posix()
        for item in workspace.items(
            source_id=source_id,
            include_disabled=True,
            include_excluded=True,
        )
    }
    stored = load_source_metadata(workspace, source_id)
    removed = len([key for key in stored if key not in existing])
    if removed:
        save_source_metadata(
            workspace,
            source_id,
            {key: value for key, value in stored.items() if key in existing},
        )
    return removed


def _subject_ratios(person_box: tuple[int, int, int, int], head_box: tuple[int, int, int, int] | None, width: int, height: int) -> dict[str, float]:
    person_width = max(1, person_box[2] - person_box[0])
    person_height = max(1, person_box[3] - person_box[1])
    head_height = max(0, head_box[3] - head_box[1]) if head_box else 0
    return {
        "subject_height_ratio": person_height / max(1, height),
        "subject_area_ratio": (person_width * person_height) / max(1, width * height),
        "head_height_ratio": head_height / max(1, height),
        "head_to_person_ratio": head_height / max(1, person_height),
    }


def _native_quality(person_box: tuple[int, int, int, int], head_box: tuple[int, int, int, int] | None) -> str:
    person_height = max(1, person_box[3] - person_box[1])
    head_size = max(head_box[2] - head_box[0], head_box[3] - head_box[1]) if head_box else 0
    if head_size >= 256 or person_height >= 800:
        return "high"
    if head_size >= 160 or person_height >= 512:
        return "medium"
    return "low"


def _resolution_tier(width: int, height: int) -> str:
    short_edge = min(width, height) if width and height else 0
    if short_edge >= 1024:
        return "high"
    if short_edge >= 768:
        return "medium"
    if short_edge >= 512:
        return "usable"
    return "low"


def _make_proxy(image: Image.Image, long_edge: int) -> tuple[Image.Image, float, float]:
    width, height = image.size
    scale = min(1.0, long_edge / max(width, height))
    if scale < 1.0:
        proxy = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    else:
        proxy = image.copy()
    return proxy, width / proxy.width, height / proxy.height


def _map_box(box: tuple[int, int, int, int], scale_x: float, scale_y: float, width: int, height: int) -> tuple[int, int, int, int]:
    return _clip_box(
        (
            round(box[0] * scale_x),
            round(box[1] * scale_y),
            round(box[2] * scale_x),
            round(box[3] * scale_y),
        ),
        width,
        height,
    )


def _clip_box(box: Any, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in box)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result
