from __future__ import annotations

import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imagehash
from PIL import Image

from .config import write_json_atomic
from .dataset_metadata import classify_composition, full_keep_score
from .models import PipelineError
from .video_character import VideoSubject, VideoSubjectReport


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class EnrichedVideoCompositionRecord:
    path: Path
    subject_id: str
    source_frame: str
    source_group_id: str
    composition_type: str
    variant_kind: str
    crop_box: Box
    person_bbox: Box
    head_bbox: Box | None
    source_resolution: tuple[int, int]
    frame_subject_count: int
    subject_coverage: float
    subject_height_ratio: float
    subject_area_ratio: float
    head_height_ratio: float
    head_to_person_ratio: float
    full_keep_score: float
    native_resolution: tuple[int, int]
    saved_resolution: tuple[int, int]
    downscaled: bool
    quality_tier: str

    @property
    def crop_type(self) -> str:
        return self.composition_type

    def as_dict(self, *, root: Path | None = None) -> dict[str, Any]:
        path = self.path.name
        if root is not None:
            try:
                path = self.path.relative_to(root).as_posix()
            except ValueError:
                pass
        return {
            "path": path,
            "subject_id": self.subject_id,
            "source_frame": self.source_frame,
            "source_group_id": self.source_group_id,
            "composition_type": self.composition_type,
            "crop_type": self.composition_type,
            "variant_kind": self.variant_kind,
            "crop_box": list(self.crop_box),
            "person_bbox": list(self.person_bbox),
            "head_bbox": list(self.head_bbox) if self.head_bbox else None,
            "source_resolution": list(self.source_resolution),
            "frame_subject_count": self.frame_subject_count,
            "subject_coverage": round(self.subject_coverage, 4),
            "subject_height_ratio": round(self.subject_height_ratio, 4),
            "subject_area_ratio": round(self.subject_area_ratio, 4),
            "head_height_ratio": round(self.head_height_ratio, 4),
            "head_to_person_ratio": round(self.head_to_person_ratio, 4),
            "full_keep_score": round(self.full_keep_score, 4),
            "native_resolution": list(self.native_resolution),
            "saved_resolution": list(self.saved_resolution),
            "downscaled": self.downscaled,
            "quality_tier": self.quality_tier,
        }


@dataclass(frozen=True)
class EnrichedVideoCompositionReport:
    output_dir: Path
    records: tuple[EnrichedVideoCompositionRecord, ...]
    selected_subjects: int
    rejected_near_duplicate: int
    rejected_too_small: int
    max_saved_long_edge: int
    max_saved_pixels: int
    minimum_saved_short_edge: int
    full_keep_threshold: float
    full_variant_target: int

    def as_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        counts = Counter(record.composition_type for record in self.records)
        variants = Counter(record.variant_kind for record in self.records)
        resolutions = [record.saved_resolution for record in self.records]
        short_edges = [min(size) for size in resolutions]
        long_edges = [max(size) for size in resolutions]
        payload: dict[str, Any] = {
            "method": "balanced-character-compositions-v2",
            "selected_subjects": self.selected_subjects,
            "training_images": len(self.records),
            "composition_counts": dict(sorted(counts.items())),
            "variant_counts": dict(sorted(variants.items())),
            "rejected_near_duplicate": self.rejected_near_duplicate,
            "rejected_too_small": self.rejected_too_small,
            "downscaled_images": sum(record.downscaled for record in self.records),
            "max_saved_long_edge": self.max_saved_long_edge,
            "max_saved_pixels": self.max_saved_pixels,
            "minimum_saved_short_edge": self.minimum_saved_short_edge,
            "saved_short_edge_range": [min(short_edges), max(short_edges)] if short_edges else None,
            "saved_long_edge_range": [min(long_edges), max(long_edges)] if long_edges else None,
            "upscale_generated": False,
            "full_keep_threshold": self.full_keep_threshold,
            "full_variant_target": self.full_variant_target,
            "full_variants_kept": variants.get("original_full", 0),
            "max_variants_per_subject": 2,
            "target_distribution": {
                "portrait": 0.20,
                "upper_body": 0.25,
                "three_quarter": 0.20,
                "full_body": 0.20,
                "context": 0.15,
            },
        }
        if include_records:
            payload["images"] = [record.as_dict(root=self.output_dir) for record in self.records]
        return payload


def build_enriched_character_dataset(
    subject_report: VideoSubjectReport,
    selected_identity_paths: Iterable[Path],
    output_dir: Path,
    *,
    maximum_saved_long_edge: int = 2048,
    maximum_saved_pixels: int = 4_194_304,
    minimum_saved_short_edge: int = 512,
    phash_distance: int = 5,
    full_keep_threshold: float = 0.72,
    full_variant_ratio: float = 0.12,
) -> EnrichedVideoCompositionReport:
    """Build balanced crops plus a small number of high-value original full frames.

    Each detected subject receives at most one primary composition. A second image is
    allowed only when the original full frame scores highly for composition value and
    the primary view is a tighter crop. This preserves useful full-body/context images
    without mechanically doubling the dataset.
    """

    subjects = list(subject_report.subjects_for_identity_paths(selected_identity_paths))
    if not subjects:
        raise PipelineError("No detected subjects correspond to the selected CCIP cluster")
    subjects.sort(key=_subject_sort_key)

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    desired = _desired_counts(len(subjects))
    assigned: Counter[str] = Counter()
    hashes: list[imagehash.ImageHash] = []
    records: list[EnrichedVideoCompositionRecord] = []
    primary_by_subject: dict[str, str] = {}
    rejected_duplicates = 0
    rejected_too_small = 0

    for subject in subjects:
        with Image.open(subject.source_frame) as opened:
            source = opened.convert("RGB")
        saved = False
        for composition in _composition_priority(subject, desired, assigned):
            crop_box = _composition_box(subject, composition)
            image = source.crop(crop_box)
            if min(image.size) < minimum_saved_short_edge:
                continue
            prepared, downscaled = _capped_image(
                image,
                max_long_edge=maximum_saved_long_edge,
                max_pixels=maximum_saved_pixels,
            )
            fingerprint = imagehash.phash(prepared)
            if any(fingerprint - previous <= phash_distance for previous in hashes):
                continue
            record = _record_for(
                subject,
                path=output_dir / f"train-{len(records) + 1:05d}-{composition}.jpg",
                composition_type=composition,
                variant_kind="smart_crop",
                crop_box=crop_box,
                native_resolution=image.size,
                saved_resolution=prepared.size,
                downscaled=downscaled,
            )
            prepared.save(record.path, quality=95, subsampling=0)
            hashes.append(fingerprint)
            records.append(record)
            assigned[composition] += 1
            primary_by_subject[subject.subject_id] = composition
            saved = True
            break
        if not saved:
            broad = source.crop(_composition_box(subject, "full_body"))
            if min(broad.size) < minimum_saved_short_edge:
                rejected_too_small += 1
            else:
                rejected_duplicates += 1

    if not records:
        raise PipelineError(
            "All selected character crops were too small or near-duplicates after composition balancing"
        )

    full_target = 0
    if full_variant_ratio > 0:
        full_target = max(1, round(len(subjects) * full_variant_ratio))
        full_target = min(full_target, max(1, math.ceil(len(subjects) / 4)))
    full_candidates: list[tuple[float, VideoSubject]] = []
    for subject in subjects:
        primary = primary_by_subject.get(subject.subject_id)
        if primary not in {"portrait", "upper_body", "three_quarter"}:
            continue
        score = _full_score(subject)
        if score >= full_keep_threshold:
            full_candidates.append((score, subject))
    full_candidates.sort(key=lambda item: (-item[0], *_subject_sort_key(item[1])))

    kept_full = 0
    for score, subject in full_candidates:
        if kept_full >= full_target:
            break
        with Image.open(subject.source_frame) as opened:
            source = opened.convert("RGB")
        if min(source.size) < minimum_saved_short_edge:
            rejected_too_small += 1
            continue
        prepared, downscaled = _capped_image(
            source,
            max_long_edge=maximum_saved_long_edge,
            max_pixels=maximum_saved_pixels,
        )
        fingerprint = imagehash.phash(prepared)
        if any(fingerprint - previous <= phash_distance for previous in hashes):
            rejected_duplicates += 1
            continue
        ratios = _subject_ratios(subject)
        composition = classify_composition(**ratios)
        output_path = output_dir / f"train-{len(records) + 1:05d}-{composition}-full.jpg"
        record = _record_for(
            subject,
            path=output_path,
            composition_type=composition,
            variant_kind="original_full",
            crop_box=(0, 0, source.width, source.height),
            native_resolution=source.size,
            saved_resolution=prepared.size,
            downscaled=downscaled,
            score_override=score,
        )
        prepared.save(output_path, quality=95, subsampling=0)
        hashes.append(fingerprint)
        records.append(record)
        kept_full += 1

    report = EnrichedVideoCompositionReport(
        output_dir=output_dir,
        records=tuple(records),
        selected_subjects=len(subjects),
        rejected_near_duplicate=rejected_duplicates,
        rejected_too_small=rejected_too_small,
        max_saved_long_edge=maximum_saved_long_edge,
        max_saved_pixels=maximum_saved_pixels,
        minimum_saved_short_edge=minimum_saved_short_edge,
        full_keep_threshold=full_keep_threshold,
        full_variant_target=full_target,
    )
    write_json_atomic(
        output_dir / "composition-manifest.json",
        report.as_dict(include_records=True),
    )
    return report


def _record_for(
    subject: VideoSubject,
    *,
    path: Path,
    composition_type: str,
    variant_kind: str,
    crop_box: Box,
    native_resolution: tuple[int, int],
    saved_resolution: tuple[int, int],
    downscaled: bool,
    score_override: float | None = None,
) -> EnrichedVideoCompositionRecord:
    ratios = _subject_ratios(subject)
    coverage = _box_area(subject.person_bbox) / max(1, _box_area(crop_box))
    return EnrichedVideoCompositionRecord(
        path=path,
        subject_id=subject.subject_id,
        source_frame=subject.source_frame.name,
        source_group_id=f"{subject.source_frame.stem}:{subject.subject_id}",
        composition_type=composition_type,
        variant_kind=variant_kind,
        crop_box=crop_box,
        person_bbox=subject.person_bbox,
        head_bbox=subject.head_bbox,
        source_resolution=subject.source_resolution,
        frame_subject_count=subject.frame_subject_count,
        subject_coverage=min(1.0, coverage),
        subject_height_ratio=ratios["subject_height_ratio"],
        subject_area_ratio=ratios["subject_area_ratio"],
        head_height_ratio=ratios["head_height_ratio"],
        head_to_person_ratio=ratios["head_to_person_ratio"],
        full_keep_score=score_override if score_override is not None else _full_score(subject),
        native_resolution=native_resolution,
        saved_resolution=saved_resolution,
        downscaled=downscaled,
        quality_tier=subject.quality_tier,
    )


def _subject_sort_key(subject: VideoSubject) -> tuple[Any, ...]:
    return (
        subject.source_timestamp_seconds is None,
        subject.source_timestamp_seconds or 0.0,
        subject.source_frame.name,
        subject.subject_id,
    )


def _available(subject: VideoSubject) -> tuple[str, ...]:
    values: list[str] = []
    if subject.head_bbox is not None:
        values.append("portrait")
    values.extend(["upper_body", "three_quarter", "full_body"])
    if subject.frame_subject_count == 1:
        values.append("context")
    return tuple(values)


def _desired_counts(count: int) -> dict[str, int]:
    weights = {
        "portrait": 0.20,
        "upper_body": 0.25,
        "three_quarter": 0.20,
        "full_body": 0.20,
        "context": 0.15,
    }
    exact = {name: count * weight for name, weight in weights.items()}
    desired = {name: math.floor(value) for name, value in exact.items()}
    remaining = count - sum(desired.values())
    for name in sorted(
        weights,
        key=lambda key: (exact[key] - desired[key], weights[key], key),
        reverse=True,
    ):
        if remaining <= 0:
            break
        desired[name] += 1
        remaining -= 1
    return desired


def _composition_priority(
    subject: VideoSubject,
    desired: dict[str, int],
    assigned: Counter[str],
) -> list[str]:
    return sorted(
        _available(subject),
        key=lambda name: (
            desired.get(name, 0) - assigned.get(name, 0),
            desired.get(name, 0),
            name,
        ),
        reverse=True,
    )


def _composition_box(subject: VideoSubject, composition: str) -> Box:
    width, height = subject.source_resolution
    if composition == "portrait":
        if subject.head_bbox is None:
            return _upper_body_box(subject)
        head_width, head_height = _box_size(subject.head_bbox)
        size = max(head_width, head_height)
        cx, cy = _box_center(subject.head_bbox)
        target_width = size * 2.8
        target_height = size * 3.3
        cy += head_height * 0.50
        return _clip_box(
            (
                round(cx - target_width / 2),
                round(cy - target_height / 2),
                round(cx + target_width / 2),
                round(cy + target_height / 2),
            ),
            width,
            height,
        )
    if composition == "upper_body":
        return _upper_body_box(subject)
    if composition == "three_quarter":
        x0, y0, x1, y1 = subject.person_bbox
        partial = (x0, y0, x1, round(y0 + (y1 - y0) * 0.84))
        return _expand_box(partial, 1.16, 1.12, width, height)
    if composition == "full_body":
        return _expand_box(subject.person_bbox, 1.18, 1.14, width, height)
    if composition == "context":
        return _expand_box(subject.person_bbox, 1.90, 1.65, width, height)
    raise PipelineError(f"Unknown character composition: {composition}")


def _upper_body_box(subject: VideoSubject) -> Box:
    width, height = subject.source_resolution
    if subject.halfbody_bbox is not None:
        return _expand_box(subject.halfbody_bbox, 1.20, 1.14, width, height)
    x0, y0, x1, y1 = subject.person_bbox
    fallback = (x0, y0, x1, round(y0 + (y1 - y0) * 0.70))
    return _expand_box(fallback, 1.20, 1.14, width, height)


def _full_score(subject: VideoSubject) -> float:
    ratios = _subject_ratios(subject)
    return full_keep_score(
        quality_tier=subject.quality_tier,
        person_count=subject.frame_subject_count,
        width=subject.source_resolution[0],
        height=subject.source_resolution[1],
        subject_area_ratio=ratios["subject_area_ratio"],
        head_present=subject.head_bbox is not None,
    )


def _subject_ratios(subject: VideoSubject) -> dict[str, float]:
    width, height = subject.source_resolution
    person_width, person_height = _box_size(subject.person_bbox)
    head_height = _box_size(subject.head_bbox)[1] if subject.head_bbox else 0
    return {
        "subject_height_ratio": person_height / max(1, height),
        "subject_area_ratio": (person_width * person_height) / max(1, width * height),
        "head_height_ratio": head_height / max(1, height),
        "head_to_person_ratio": head_height / max(1, person_height),
    }


def _capped_image(
    image: Image.Image,
    *,
    max_long_edge: int,
    max_pixels: int,
) -> tuple[Image.Image, bool]:
    width, height = image.size
    scale = min(
        1.0,
        max_long_edge / max(width, height),
        math.sqrt(max_pixels / max(1, width * height)),
    )
    if scale >= 1.0:
        return image.copy(), False
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(target, Image.Resampling.LANCZOS), True


def _expand_box(box: Box, x_scale: float, y_scale: float, width: int, height: int) -> Box:
    box_width, box_height = _box_size(box)
    cx, cy = _box_center(box)
    target_width = box_width * x_scale
    target_height = box_height * y_scale
    return _clip_box(
        (
            round(cx - target_width / 2),
            round(cy - target_height / 2),
            round(cx + target_width / 2),
            round(cy + target_height / 2),
        ),
        width,
        height,
    )


def _clip_box(box: Box, width: int, height: int) -> Box:
    x0, y0, x1, y1 = (int(value) for value in box)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def _box_size(box: Box | None) -> tuple[int, int]:
    if box is None:
        return 0, 0
    return box[2] - box[0], box[3] - box[1]


def _box_area(box: Box) -> int:
    width, height = _box_size(box)
    return max(0, width) * max(0, height)


def _box_center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
