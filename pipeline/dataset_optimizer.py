from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Protocol, Sequence

from PIL import Image, ImageFilter, ImageOps, ImageStat

from .config import stable_hash, write_json_atomic
from .dataset.caption_cleaner import normalize_tag, parse_caption
from .dataset.image_info import inspect_image
from .dataset_workspace import DatasetItem, DatasetWorkspace
from .models import OptionalBackendUnavailable
from .state import utc_now


@dataclass(frozen=True)
class FastQualityPolicy:
    """Conservative technical-quality policy for anime LoRA datasets.

    Absolute thresholds are intentionally reserved for extreme technical failures.
    Sharpness is assessed relative to the current dataset with a robust MAD score,
    because line art, screenshots, illustrations, and soft-painted styles have very
    different absolute edge statistics.
    """

    proxy_long_edge: int = 512
    relative_min_samples: int = 8
    blur_mad_z: float = 3.5
    clipped_fraction: float = 0.90
    low_information_entropy: float = 1.0
    low_information_stddev: float = 5.0


@dataclass(frozen=True)
class OptimizeOptions:
    apply_safe: bool = False
    deep: bool = False
    auto_tag: bool = False
    tag_threshold: float = 0.35
    phash_distance: int = 6
    lpips_threshold: float = 0.45
    identity_min_samples: int = 2


class DeepAnimeBackend(Protocol):
    def inspect(self, image: Path) -> Mapping[str, Any]:
        ...

    def lpips_clusters(self, images: Sequence[Path], *, threshold: float) -> Sequence[int]:
        ...


class ImgutilsDeepAnimeBackend:
    """Optional anime-specific quality backend backed by dghs-imgutils.

    Model-backed outputs are advisory except for the deterministic truncated-file
    validator. Model downloads/inference are deliberately opt-in through ``--deep``.
    """

    def inspect(self, image: Path) -> Mapping[str, Any]:
        try:
            from imgutils.detect import detect_heads
            from imgutils.metrics import anime_dbaesthetic
            from imgutils.validate import (
                anime_classify,
                anime_portrait,
                is_monochrome,
                is_truncated_file,
            )
        except ImportError as exc:
            raise OptionalBackendUnavailable(
                "dghs-imgutils deep anime quality backend is unavailable"
            ) from exc

        image_type, image_type_score = anime_classify(str(image))
        aesthetic_label, aesthetic_percentile = anime_dbaesthetic(
            str(image), fmt=("label", "percentile")
        )
        portrait_type, portrait_score = anime_portrait(str(image))
        heads = detect_heads(str(image))
        return {
            "truncated": bool(is_truncated_file(str(image))),
            "monochrome": bool(is_monochrome(str(image))),
            "image_type": str(image_type),
            "image_type_score": float(image_type_score),
            "aesthetic_label": str(aesthetic_label),
            "aesthetic_percentile": float(aesthetic_percentile),
            "portrait_type": str(portrait_type),
            "portrait_score": float(portrait_score),
            "head_count": len(heads),
            "heads": [
                {
                    "bbox": [int(value) for value in bbox],
                    "confidence": float(confidence),
                }
                for bbox, _label, confidence in heads
            ],
        }

    def lpips_clusters(
        self, images: Sequence[Path], *, threshold: float
    ) -> Sequence[int]:
        try:
            from imgutils.metrics import lpips_clustering
        except ImportError as exc:
            raise OptionalBackendUnavailable(
                "dghs-imgutils LPIPS backend is unavailable"
            ) from exc
        return list(lpips_clustering([str(path) for path in images], threshold=threshold))


_CAPTION_RISK_GROUPS: dict[str, frozenset[str]] = {
    "text_overlay": frozenset(
        {
            "watermark",
            "signature",
            "artist name",
            "username",
            "twitter username",
            "patreon username",
            "web address",
            "logo",
            "text",
            "english text",
            "japanese text",
            "chinese text",
            "korean text",
            "speech bubble",
        }
    ),
    "technical_quality": frozenset(
        {
            "blurry",
            "motion blur",
            "jpeg artifacts",
            "lowres",
            "scan artifacts",
        }
    ),
    "layout_contamination": frozenset(
        {
            "comic",
            "manga",
            "4koma",
            "multiple views",
            "reference sheet",
            "character sheet",
            "split screen",
        }
    ),
    "monochrome": frozenset({"monochrome", "greyscale", "grayscale"}),
}


def optimize_dataset(
    workspace: DatasetWorkspace,
    *,
    options: OptimizeOptions | None = None,
    deep_backend: DeepAnimeBackend | None = None,
) -> dict[str, Any]:
    """Build and optionally apply a conservative automatic curation plan.

    ``apply_safe`` only records reversible Dataset exclusions for deterministic
    failures: corrupt/truncated files and redundant exact copies. Perceptual/LPIPS
    similarity, aesthetic quality, framing, monochrome, caption contamination, and
    relative blur remain review/ranking signals.
    """

    options = options or OptimizeOptions()
    if not 0.0 <= options.tag_threshold <= 1.0:
        raise ValueError("tag_threshold must be between 0 and 1")
    if options.phash_distance < 0:
        raise ValueError("phash_distance must be non-negative")
    if options.lpips_threshold <= 0:
        raise ValueError("lpips_threshold must be positive")

    before = workspace.summary()
    fast = analyze_fast_quality(workspace)
    safe_exclusions: list[dict[str, str]] = []
    if options.apply_safe:
        safe_exclusions.extend(_apply_safe_records(workspace, fast["records"]))

    deep: dict[str, Any] | None = None
    if options.deep:
        backend = deep_backend or ImgutilsDeepAnimeBackend()
        try:
            deep = analyze_deep_quality(workspace, backend=backend)
        except Exception as exc:
            deep = {
                "schema_version": 1,
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "records": [],
                "summary": {},
            }
        if options.apply_safe and deep.get("status") == "ok":
            safe_exclusions.extend(_apply_safe_records(workspace, deep["records"]))

    tagging: dict[str, Any] | None = None
    if options.auto_tag:
        tagging = workspace.auto_tag(threshold=options.tag_threshold)

    captions = analyze_caption_risks(workspace)

    duplicates: dict[str, Any] | None
    try:
        duplicates = workspace.analyze_duplicates(
            phash_distance=options.phash_distance
        )
    except Exception as exc:
        duplicates = {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "summary": {},
            "near_groups": [],
        }

    lpips: dict[str, Any] | None = None
    if options.deep and deep is not None and deep.get("status") == "ok":
        backend = deep_backend or ImgutilsDeepAnimeBackend()
        try:
            lpips = analyze_lpips_duplicates(
                workspace,
                backend=backend,
                threshold=options.lpips_threshold,
                quality_records=deep.get("records", []),
            )
        except Exception as exc:
            lpips = {
                "schema_version": 1,
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "groups": [],
                "summary": {},
            }

    identity: dict[str, Any] | None = None
    if workspace.concept_type == "character":
        try:
            identity = workspace.analyze_identity(
                min_samples=options.identity_min_samples
            )
        except Exception as exc:
            identity = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "summary": {},
            }

    after = workspace.summary()
    report = {
        "schema_version": 1,
        "dataset": workspace.name,
        "concept_type": workspace.concept_type,
        "generated_at": utc_now(),
        "options": asdict(options),
        "before": before,
        "after": after,
        "safe_exclusions_applied": safe_exclusions,
        "fast": _manifest_pointer(fast),
        "deep": _manifest_pointer(deep) if deep is not None else None,
        "caption_risks": _manifest_pointer(captions),
        "dedup": _compact_analysis(duplicates),
        "lpips": _manifest_pointer(lpips) if lpips is not None else None,
        "identity": _compact_analysis(identity),
        "tagging": _compact_analysis(tagging),
    }
    report["recommendations"] = _recommendations(
        fast=fast,
        deep=deep,
        captions=captions,
        duplicates=duplicates,
        lpips=lpips,
        identity=identity,
    )
    report["optimization_hash"] = stable_hash(
        {
            "dataset": workspace.name,
            "options": asdict(options),
            "after": after,
            "fast": fast.get("input_hash"),
            "deep": deep.get("input_hash") if isinstance(deep, Mapping) else None,
            "caption_risks": captions.get("input_hash"),
        }
    )
    path = _optimization_dir(workspace) / "report.json"
    write_json_atomic(path, report)
    report["manifest"] = str(path)
    return report


def analyze_fast_quality(
    workspace: DatasetWorkspace,
    *,
    policy: FastQualityPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or FastQualityPolicy()
    items = workspace.items(include_disabled=False, include_excluded=False)
    records: list[dict[str, Any]] = []
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in items:
        root = workspace.source_images_dir(item.source_id)
        inspected = inspect_image(item.image, root)
        record: dict[str, Any] = {
            **inspected,
            "path": item.key,
            "key": item.key,
            "source_id": item.source_id,
            "caption": item.caption.is_file(),
            "flags": [],
            "safe_exclude": False,
        }
        if inspected.get("corrupt"):
            _flag(record, "corrupt", "reject", safe=True)
        else:
            try:
                record["technical"] = _technical_metrics(
                    item.image, proxy_long_edge=policy.proxy_long_edge
                )
                _append_absolute_technical_flags(record, policy)
            except Exception as exc:
                _flag(
                    record,
                    "technical_metrics_unavailable",
                    "review",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            if item.caption.is_file():
                _append_caption_flags(
                    record,
                    item.caption.read_text(encoding="utf-8", errors="replace"),
                )
        records.append(record)
        digest = str(record.get("sha256") or "")
        if digest:
            by_sha[digest].append(record)

    _append_relative_blur_flags(records, policy)

    exact_groups: list[dict[str, Any]] = []
    for digest, group in sorted(by_sha.items()):
        if len(group) < 2:
            continue
        canonical = _choose_canonical(group)
        candidates: list[str] = []
        for record in group:
            if record is canonical:
                continue
            _flag(
                record,
                "exact_duplicate",
                "reject",
                safe=True,
                canonical=str(canonical["key"]),
            )
            candidates.append(str(record["key"]))
        exact_groups.append(
            {
                "sha256": digest,
                "canonical": str(canonical["key"]),
                "members": sorted(str(record["key"]) for record in group),
                "safe_exclude_candidates": sorted(candidates),
            }
        )

    payload = {
        "schema_version": 1,
        "dataset": workspace.name,
        "generated_at": utc_now(),
        "policy": asdict(policy),
        "input_hash": stable_hash(
            [
                {
                    "key": record["key"],
                    "sha256": record.get("sha256"),
                    "caption": record.get("caption"),
                }
                for record in records
            ]
        ),
        "records": records,
        "exact_groups": exact_groups,
        "summary": {
            "images": len(records),
            "flagged": sum(bool(record["flags"]) for record in records),
            "safe_exclude_suggestions": sum(
                bool(record.get("safe_exclude")) for record in records
            ),
            "corrupt": _count_flag(records, "corrupt"),
            "exact_duplicate_images": _count_flag(records, "exact_duplicate"),
            "relative_blur_outliers": _count_flag(records, "relative_blur_outlier"),
            "extreme_exposure": _count_flag(records, "near_black_frame")
            + _count_flag(records, "near_white_frame"),
            "caption_risk_images": sum(
                any(str(flag.get("code", "")).startswith("caption_") for flag in record["flags"])
                for record in records
            ),
        },
    }
    path = _optimization_dir(workspace) / "fast-audit.json"
    write_json_atomic(path, payload)
    payload["manifest"] = str(path)
    return payload


def analyze_caption_risks(workspace: DatasetWorkspace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for item in workspace.items(include_disabled=False, include_excluded=False):
        text = (
            item.caption.read_text(encoding="utf-8", errors="replace")
            if item.caption.is_file()
            else ""
        )
        flags = _caption_risk_flags(text)
        records.append(
            {
                "key": item.key,
                "caption": item.caption.is_file(),
                "flags": flags,
            }
        )
    payload = {
        "schema_version": 1,
        "dataset": workspace.name,
        "generated_at": utc_now(),
        "input_hash": stable_hash(records),
        "records": records,
        "summary": {
            "images": len(records),
            "captioned": sum(bool(record["caption"]) for record in records),
            "flagged": sum(bool(record["flags"]) for record in records),
            "by_group": {
                group: sum(
                    any(flag.get("group") == group for flag in record["flags"])
                    for record in records
                )
                for group in _CAPTION_RISK_GROUPS
            },
        },
    }
    path = _optimization_dir(workspace) / "caption-risks.json"
    write_json_atomic(path, payload)
    payload["manifest"] = str(path)
    return payload


def analyze_deep_quality(
    workspace: DatasetWorkspace,
    *,
    backend: DeepAnimeBackend,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for item in workspace.items(include_disabled=False, include_excluded=False):
        result = dict(backend.inspect(item.image))
        record: dict[str, Any] = {
            "key": item.key,
            "source_id": item.source_id,
            "flags": [],
            "safe_exclude": False,
            **result,
        }
        if bool(result.get("truncated")):
            _flag(record, "truncated_file", "reject", safe=True)
        if bool(result.get("monochrome")):
            _flag(record, "monochrome", "review")
        image_type = str(result.get("image_type") or "")
        image_type_score = float(result.get("image_type_score") or 0.0)
        if image_type in {"3d", "comic", "not_painting"} and image_type_score >= 0.85:
            _flag(
                record,
                "non_target_image_type",
                "review",
                image_type=image_type,
                confidence=round(image_type_score, 6),
            )
        aesthetic_label = str(result.get("aesthetic_label") or "")
        aesthetic_percentile = float(result.get("aesthetic_percentile") or 0.0)
        if aesthetic_label in {"low", "worst"} or aesthetic_percentile < 0.10:
            _flag(
                record,
                "low_anime_aesthetic",
                "review",
                label=aesthetic_label,
                percentile=round(aesthetic_percentile, 6),
            )
        if workspace.concept_type == "character":
            head_count = int(result.get("head_count") or 0)
            if head_count == 0:
                _flag(record, "no_anime_head", "review")
            elif head_count > 1:
                _flag(
                    record,
                    "multiple_anime_heads",
                    "review",
                    head_count=head_count,
                )
        records.append(record)

    payload = {
        "schema_version": 1,
        "status": "ok",
        "dataset": workspace.name,
        "generated_at": utc_now(),
        "input_hash": stable_hash(
            [
                {
                    "key": record["key"],
                    "truncated": record.get("truncated"),
                    "monochrome": record.get("monochrome"),
                    "image_type": record.get("image_type"),
                    "aesthetic_percentile": record.get("aesthetic_percentile"),
                    "head_count": record.get("head_count"),
                }
                for record in records
            ]
        ),
        "records": records,
        "summary": {
            "images": len(records),
            "flagged": sum(bool(record["flags"]) for record in records),
            "safe_exclude_suggestions": sum(
                bool(record.get("safe_exclude")) for record in records
            ),
            "truncated": _count_flag(records, "truncated_file"),
            "monochrome": _count_flag(records, "monochrome"),
            "non_target_image_type": _count_flag(records, "non_target_image_type"),
            "low_aesthetic": _count_flag(records, "low_anime_aesthetic"),
            "head_count_issues": _count_flag(records, "no_anime_head")
            + _count_flag(records, "multiple_anime_heads"),
        },
    }
    path = _optimization_dir(workspace) / "deep-audit.json"
    write_json_atomic(path, payload)
    payload["manifest"] = str(path)
    return payload


def analyze_lpips_duplicates(
    workspace: DatasetWorkspace,
    *,
    backend: DeepAnimeBackend,
    threshold: float = 0.45,
    quality_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    items = workspace.items(include_disabled=False, include_excluded=False)
    labels = list(
        backend.lpips_clusters([item.image for item in items], threshold=threshold)
    )
    if len(labels) != len(items):
        raise ValueError(
            f"LPIPS backend returned {len(labels)} labels for {len(items)} images"
        )
    quality = {
        str(record.get("key")): dict(record)
        for record in quality_records
        if record.get("key")
    }
    grouped: dict[int, list[DatasetItem]] = defaultdict(list)
    for item, label in zip(items, labels):
        label = int(label)
        if label >= 0:
            grouped[label].append(item)

    groups: list[dict[str, Any]] = []
    for label, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        canonical = _choose_lpips_canonical(members, quality)
        groups.append(
            {
                "cluster": label,
                "recommended_keep": canonical.key,
                "members": [item.key for item in members],
                "review_exclude_candidates": [
                    item.key for item in members if item.key != canonical.key
                ],
            }
        )

    payload = {
        "schema_version": 1,
        "status": "ok",
        "dataset": workspace.name,
        "generated_at": utc_now(),
        "threshold": threshold,
        "groups": groups,
        "summary": {
            "groups": len(groups),
            "images_in_groups": sum(len(group["members"]) for group in groups),
            "review_exclude_candidates": sum(
                len(group["review_exclude_candidates"]) for group in groups
            ),
        },
        "note": (
            "LPIPS groups are review suggestions only. Similar anime frames can carry "
            "useful pose/expression/outfit variation and are never auto-excluded."
        ),
    }
    path = _optimization_dir(workspace) / "lpips-duplicates.json"
    write_json_atomic(path, payload)
    payload["manifest"] = str(path)
    return payload


def _technical_metrics(path: Path, *, proxy_long_edge: int) -> dict[str, Any]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if max(image.size) > proxy_long_edge:
            image.thumbnail((proxy_long_edge, proxy_long_edge), Image.Resampling.LANCZOS)

        transparent_fraction = 0.0
        if "A" in image.getbands() or "transparency" in image.info:
            alpha = image.convert("RGBA").getchannel("A")
            alpha_hist = alpha.histogram()
            alpha_pixels = sum(alpha_hist)
            transparent_fraction = (
                sum(alpha_hist[:9]) / alpha_pixels if alpha_pixels else 0.0
            )

        gray = ImageOps.grayscale(image.convert("RGB"))
        histogram = gray.histogram()
        pixels = sum(histogram)
        stat = ImageStat.Stat(gray)
        mean = float(stat.mean[0])
        stddev = float(stat.stddev[0])
        entropy = float(gray.entropy())
        dark_fraction = sum(histogram[:5]) / pixels if pixels else 0.0
        bright_fraction = sum(histogram[251:]) / pixels if pixels else 0.0

        edges = gray.filter(ImageFilter.FIND_EDGES)
        if edges.width > 4 and edges.height > 4:
            edges = edges.crop((2, 2, edges.width - 2, edges.height - 2))
        sharpness = float(ImageStat.Stat(edges).var[0])

    return {
        "proxy_size": [int(gray.width), int(gray.height)],
        "luminance_mean": round(mean, 6),
        "luminance_stddev": round(stddev, 6),
        "entropy": round(entropy, 6),
        "dark_clip_fraction": round(dark_fraction, 6),
        "bright_clip_fraction": round(bright_fraction, 6),
        "transparent_fraction": round(transparent_fraction, 6),
        "edge_variance": round(sharpness, 6),
    }


def _append_absolute_technical_flags(
    record: dict[str, Any], policy: FastQualityPolicy
) -> None:
    metrics = record.get("technical", {})
    if float(metrics.get("dark_clip_fraction") or 0.0) >= policy.clipped_fraction:
        _flag(record, "near_black_frame", "review")
    if float(metrics.get("bright_clip_fraction") or 0.0) >= policy.clipped_fraction:
        _flag(record, "near_white_frame", "review")
    if (
        float(metrics.get("entropy") or 0.0) <= policy.low_information_entropy
        and float(metrics.get("luminance_stddev") or 0.0)
        <= policy.low_information_stddev
    ):
        _flag(record, "very_low_information", "review")


def _append_relative_blur_flags(
    records: Sequence[dict[str, Any]], policy: FastQualityPolicy
) -> None:
    valid = [
        record
        for record in records
        if not record.get("corrupt")
        and isinstance(record.get("technical"), Mapping)
        and float(record["technical"].get("edge_variance") or 0.0) >= 0.0
    ]
    if len(valid) < policy.relative_min_samples:
        return
    values = [math.log1p(float(record["technical"]["edge_variance"])) for record in valid]
    center = median(values)
    mad = median(abs(value - center) for value in values)
    if mad <= 1e-9:
        return
    scale = 1.4826 * mad
    for record, value in zip(valid, values):
        robust_z = (value - center) / scale
        record["technical"]["sharpness_robust_z"] = round(robust_z, 6)
        if robust_z <= -abs(policy.blur_mad_z):
            _flag(
                record,
                "relative_blur_outlier",
                "review",
                robust_z=round(robust_z, 6),
            )


def _append_caption_flags(record: dict[str, Any], text: str) -> None:
    for flag in _caption_risk_flags(text):
        record["flags"].append(flag)


def _caption_risk_flags(text: str) -> list[dict[str, Any]]:
    tags = {normalize_tag(tag) for tag in parse_caption(text)}
    flags: list[dict[str, Any]] = []
    for group, risks in _CAPTION_RISK_GROUPS.items():
        matches = sorted(tags & risks)
        if matches:
            flags.append(
                {
                    "code": f"caption_{group}",
                    "severity": "review",
                    "group": group,
                    "tags": matches,
                }
            )
    return flags


def _choose_canonical(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def rank(record: Mapping[str, Any]) -> tuple[int, int, int, str]:
        pixels = int(record.get("width") or 0) * int(record.get("height") or 0)
        return (
            -int(bool(record.get("caption"))),
            -pixels,
            -int(record.get("bytes") or 0),
            str(record.get("key") or "").casefold(),
        )

    return sorted(records, key=rank)[0]


def _choose_lpips_canonical(
    items: Sequence[DatasetItem], quality: Mapping[str, Mapping[str, Any]]
) -> DatasetItem:
    def rank(item: DatasetItem) -> tuple[int, float, int, str]:
        record = quality.get(item.key, {})
        head_count = int(record.get("head_count") or 0)
        good_head = int(head_count == 1)
        aesthetic = float(record.get("aesthetic_percentile") or 0.0)
        pixels = 0
        try:
            with Image.open(item.image) as image:
                pixels = int(image.width) * int(image.height)
        except OSError:
            pass
        return (
            -good_head,
            -aesthetic,
            -pixels,
            item.key.casefold(),
        )

    return sorted(items, key=rank)[0]


def _apply_safe_records(
    workspace: DatasetWorkspace, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    applied: list[dict[str, str]] = []
    for record in records:
        if not record.get("safe_exclude"):
            continue
        key = str(record.get("key") or "")
        if not key:
            continue
        codes = [
            str(flag.get("code"))
            for flag in record.get("flags", [])
            if flag.get("severity") == "reject"
        ]
        reason = "automatic safe optimization: " + ", ".join(codes or ["deterministic reject"])
        changed = workspace.exclude([key], reason=reason, mode="automatic_safe")
        if changed:
            applied.append({"key": key, "reason": reason})
    return applied


def _flag(
    record: dict[str, Any],
    code: str,
    severity: str,
    *,
    safe: bool = False,
    **details: Any,
) -> None:
    record.setdefault("flags", []).append(
        {"code": code, "severity": severity, **details}
    )
    if safe:
        record["safe_exclude"] = True


def _count_flag(records: Sequence[Mapping[str, Any]], code: str) -> int:
    return sum(
        any(flag.get("code") == code for flag in record.get("flags", []))
        for record in records
    )


def _optimization_dir(workspace: DatasetWorkspace) -> Path:
    path = workspace.dataset_dir / "review" / "optimization"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_pointer(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "status": payload.get("status", "ok"),
        "manifest": payload.get("manifest"),
        "summary": dict(payload.get("summary", {})),
        "error": payload.get("error"),
    }


def _compact_analysis(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "status": payload.get("status", "ok"),
        "summary": dict(payload.get("summary", {})),
        "error": payload.get("error"),
    }


def _recommendations(
    *,
    fast: Mapping[str, Any],
    deep: Mapping[str, Any] | None,
    captions: Mapping[str, Any],
    duplicates: Mapping[str, Any] | None,
    lpips: Mapping[str, Any] | None,
    identity: Mapping[str, Any] | None,
) -> list[str]:
    recommendations: list[str] = []
    fast_summary = fast.get("summary", {})
    if int(fast_summary.get("relative_blur_outliers", 0)):
        recommendations.append(
            "Review relative blur outliers; sharpness is ranked against this dataset, not a global photo threshold."
        )
    if int(fast_summary.get("extreme_exposure", 0)):
        recommendations.append(
            "Review near-black/near-white frames for fades, blank screens, or accidental captures."
        )
    if int(captions.get("summary", {}).get("flagged", 0)):
        recommendations.append(
            "Review captions that indicate text/watermarks, technical defects, comics/reference sheets, or monochrome content."
        )
    if duplicates and int(duplicates.get("summary", {}).get("near_groups", 0)):
        recommendations.append(
            "Review pHash near-duplicate groups; retain pose/expression/outfit variants that add information."
        )
    if lpips and int(lpips.get("summary", {}).get("groups", 0)):
        recommendations.append(
            "Review LPIPS semantic/variant groups; recommended_keep is a ranking hint, never an automatic deletion."
        )
    if deep and deep.get("status") == "ok":
        summary = deep.get("summary", {})
        if int(summary.get("head_count_issues", 0)):
            recommendations.append(
                "Review character images with zero or multiple detected anime heads before identity analysis."
            )
        if int(summary.get("non_target_image_type", 0)):
            recommendations.append(
                "Review high-confidence comic/3D/not-painting classifications for source-domain contamination."
            )
        if int(summary.get("low_aesthetic", 0)):
            recommendations.append(
                "Treat low anime-aesthetic scores as ranking evidence only; identity coverage and rare poses can be more valuable than aesthetics."
            )
    if identity:
        summary = identity.get("summary", {})
        if int(summary.get("possible_outliers", 0)) or int(
            summary.get("possible_mixed_characters", 0)
        ):
            recommendations.append(
                "Review CCIP identity outliers/mixed-character candidates; do not auto-delete without visual confirmation."
            )
    return recommendations


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Conservative automatic DatasetWorkspace curation"
    )
    parser.add_argument("dataset")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--apply-safe", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--auto-tag", action="store_true")
    parser.add_argument("--tag-threshold", type=float, default=0.35)
    parser.add_argument("--phash-distance", type=int, default=6)
    parser.add_argument("--lpips-threshold", type=float, default=0.45)
    parser.add_argument("--identity-min-samples", type=int, default=2)
    args = parser.parse_args(argv)

    workspace = DatasetWorkspace.load(args.dataset, root=args.root)
    report = optimize_dataset(
        workspace,
        options=OptimizeOptions(
            apply_safe=args.apply_safe,
            deep=args.deep,
            auto_tag=args.auto_tag,
            tag_threshold=args.tag_threshold,
            phash_distance=args.phash_distance,
            lpips_threshold=args.lpips_threshold,
            identity_min_samples=args.identity_min_samples,
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
