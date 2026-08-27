from __future__ import annotations

import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import imagehash
from PIL import Image

from .config import write_json_atomic
from .dataset.image_info import discover_images
from .models import OptionalBackendUnavailable, PipelineError


Box = tuple[int, int, int, int]
Detection = tuple[Box, str, float]
Detector = Callable[..., list[Detection]]


@dataclass(frozen=True)
class VideoSubject:
    subject_id: str
    identity_path: Path
    source_frame: Path
    source_timestamp_seconds: float | None
    source_resolution: tuple[int, int]
    person_bbox: Box
    head_bbox: Box | None
    halfbody_bbox: Box | None
    person_score: float | None
    head_score: float | None
    halfbody_score: float | None
    detection_kind: str
    quality_tier: str
    frame_subject_count: int
    native_identity_resolution: tuple[int, int]
    saved_identity_resolution: tuple[int, int]

    def as_dict(self, *, root: Path | None = None) -> dict[str, Any]:
        def display(path: Path) -> str:
            if root is not None:
                try:
                    return path.relative_to(root).as_posix()
                except ValueError:
                    pass
            return path.name

        return {
            "subject_id": self.subject_id,
            "identity_image": display(self.identity_path),
            "source_frame": self.source_frame.name,
            "source_timestamp_seconds": self.source_timestamp_seconds,
            "source_resolution": list(self.source_resolution),
            "person_bbox": list(self.person_bbox),
            "head_bbox": list(self.head_bbox) if self.head_bbox else None,
            "halfbody_bbox": list(self.halfbody_bbox) if self.halfbody_bbox else None,
            "person_score": self.person_score,
            "head_score": self.head_score,
            "halfbody_score": self.halfbody_score,
            "detection_kind": self.detection_kind,
            "quality_tier": self.quality_tier,
            "frame_subject_count": self.frame_subject_count,
            "native_identity_resolution": list(self.native_identity_resolution),
            "saved_identity_resolution": list(self.saved_identity_resolution),
        }


@dataclass(frozen=True)
class VideoSubjectReport:
    identity_dir: Path
    subjects: tuple[VideoSubject, ...]
    total_frames: int
    frames_with_subjects: int
    detected_persons: int
    head_fallbacks: int
    rejected_low_resolution: int
    detection_proxy_long_edge: int
    minimum_person_height: int
    minimum_head_size: int
    maximum_saved_long_edge: int
    maximum_saved_pixels: int

    def as_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        quality = Counter(subject.quality_tier for subject in self.subjects)
        payload: dict[str, Any] = {
            "method": "deepghs-imgutils",
            "total_frames": self.total_frames,
            "frames_with_subjects": self.frames_with_subjects,
            "usable_subjects": len(self.subjects),
            "detected_persons": self.detected_persons,
            "head_fallbacks": self.head_fallbacks,
            "rejected_low_resolution": self.rejected_low_resolution,
            "quality_tiers": dict(sorted(quality.items())),
            "detection_proxy_long_edge": self.detection_proxy_long_edge,
            "minimum_person_height": self.minimum_person_height,
            "minimum_head_size": self.minimum_head_size,
            "maximum_saved_long_edge": self.maximum_saved_long_edge,
            "maximum_saved_pixels": self.maximum_saved_pixels,
            "upscale_generated": False,
        }
        if include_records:
            payload["subjects"] = [
                subject.as_dict(root=self.identity_dir) for subject in self.subjects
            ]
        return payload

    def subjects_for_identity_paths(self, paths: Iterable[Path]) -> tuple[VideoSubject, ...]:
        wanted = {path.resolve() for path in paths}
        return tuple(
            subject for subject in self.subjects if subject.identity_path.resolve() in wanted
        )


@dataclass(frozen=True)
class VideoCompositionRecord:
    path: Path
    subject_id: str
    source_frame: str
    crop_type: str
    crop_box: Box
    subject_coverage: float
    native_resolution: tuple[int, int]
    saved_resolution: tuple[int, int]
    downscaled: bool
    quality_tier: str

    def as_dict(self, *, root: Path | None = None) -> dict[str, Any]:
        if root is not None:
            try:
                path = self.path.relative_to(root).as_posix()
            except ValueError:
                path = self.path.name
        else:
            path = self.path.name
        return {
            "path": path,
            "subject_id": self.subject_id,
            "source_frame": self.source_frame,
            "crop_type": self.crop_type,
            "crop_box": list(self.crop_box),
            "subject_coverage": round(self.subject_coverage, 4),
            "native_resolution": list(self.native_resolution),
            "saved_resolution": list(self.saved_resolution),
            "downscaled": self.downscaled,
            "quality_tier": self.quality_tier,
        }


@dataclass(frozen=True)
class VideoCompositionReport:
    output_dir: Path
    records: tuple[VideoCompositionRecord, ...]
    selected_subjects: int
    rejected_near_duplicate: int
    rejected_too_small: int
    max_saved_long_edge: int
    max_saved_pixels: int
    minimum_saved_short_edge: int

    def as_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        counts = Counter(record.crop_type for record in self.records)
        resolutions = [record.saved_resolution for record in self.records]
        short_edges = [min(size) for size in resolutions]
        long_edges = [max(size) for size in resolutions]
        payload: dict[str, Any] = {
            "method": "balanced-character-compositions",
            "selected_subjects": self.selected_subjects,
            "training_images": len(self.records),
            "composition_counts": dict(sorted(counts.items())),
            "rejected_near_duplicate": self.rejected_near_duplicate,
            "rejected_too_small": self.rejected_too_small,
            "downscaled_images": sum(record.downscaled for record in self.records),
            "max_saved_long_edge": self.max_saved_long_edge,
            "max_saved_pixels": self.max_saved_pixels,
            "minimum_saved_short_edge": self.minimum_saved_short_edge,
            "saved_short_edge_range": [min(short_edges), max(short_edges)] if short_edges else None,
            "saved_long_edge_range": [min(long_edges), max(long_edges)] if long_edges else None,
            "upscale_generated": False,
            "target_distribution": {
                "portrait": 0.25,
                "upper_body": 0.30,
                "full_body": 0.30,
                "context": 0.15,
            },
        }
        if include_records:
            payload["images"] = [record.as_dict(root=self.output_dir) for record in self.records]
        return payload


def _load_detectors() -> tuple[Detector, Detector, Detector]:
    try:
        from imgutils.detect import detect_halfbody, detect_heads, detect_person
    except ImportError as exc:
        raise OptionalBackendUnavailable(
            "DeepGHS imgutils anime character detectors are unavailable"
        ) from exc
    return detect_person, detect_heads, detect_halfbody


def detect_video_subjects(
    frame_dir: Path,
    output_dir: Path,
    *,
    interval_seconds: int = 2,
    detection_proxy_long_edge: int = 1600,
    minimum_person_height: int = 512,
    minimum_head_size: int = 160,
    maximum_saved_long_edge: int = 2048,
    maximum_saved_pixels: int = 4_194_304,
) -> VideoSubjectReport:
    """Detect anime characters on lightweight proxies and crop identities from source frames.

    Detection intentionally runs on a reduced proxy image. Bounding boxes are mapped back to
    the original sampled frame before cropping, so small characters in 4K sources retain the
    native detail that would be lost by downscaling the whole frame first.
    """

    if detection_proxy_long_edge < 512:
        raise PipelineError("Detection proxy long edge must be at least 512 pixels")
    if minimum_person_height < 1 or minimum_head_size < 1:
        raise PipelineError("Character resolution thresholds must be positive")
    if maximum_saved_long_edge < 512 or maximum_saved_pixels < 512 * 512:
        raise PipelineError("Saved character crop limits are too small")

    frames = discover_images(frame_dir)
    if not frames:
        raise PipelineError(f"No filtered video frames were found under {frame_dir}")

    detect_person, detect_heads, detect_halfbody = _load_detectors()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    identity_dir = output_dir / "identity"
    identity_dir.mkdir(parents=True, exist_ok=True)

    subjects: list[VideoSubject] = []
    detected_persons = 0
    head_fallbacks = 0
    rejected_low_resolution = 0
    frames_with_subjects = 0
    next_id = 1

    for frame in frames:
        with Image.open(frame) as opened:
            source = opened.convert("RGB")
        proxy, scale_x, scale_y = _make_detection_proxy(source, detection_proxy_long_edge)
        try:
            person_detections = detect_person(
                proxy,
                level="m",
                version="v1.1",
                conf_threshold=0.3,
                iou_threshold=0.5,
            )
            head_detections = detect_heads(
                proxy,
                conf_threshold=0.4,
                iou_threshold=0.7,
            )
        except Exception as exc:  # model/cache/runtime failures should fall back cleanly in the UI
            raise OptionalBackendUnavailable(
                f"DeepGHS anime character detection failed: {exc}"
            ) from exc

        detected_persons += len(person_detections)
        matched_heads: set[int] = set()
        drafts: list[dict[str, Any]] = []
        source_size = source.size

        for person_box_proxy, _label, person_score in person_detections:
            person_box_proxy = _clip_box(person_box_proxy, proxy.width, proxy.height)
            person_box = _map_box(person_box_proxy, scale_x, scale_y, *source_size)
            head_index = _best_head_for_person(head_detections, person_box_proxy)
            head_box: Box | None = None
            head_score: float | None = None
            if head_index is not None:
                matched_heads.add(head_index)
                head_box_proxy, _head_label, head_score = head_detections[head_index]
                head_box = _map_box(
                    _clip_box(head_box_proxy, proxy.width, proxy.height),
                    scale_x,
                    scale_y,
                    *source_size,
                )

            halfbody_box: Box | None = None
            halfbody_score: float | None = None
            px0, py0, px1, py1 = person_box_proxy
            person_proxy = proxy.crop((px0, py0, px1, py1))
            if person_proxy.width >= 96 and person_proxy.height >= 96:
                try:
                    halfbody = detect_halfbody(
                        person_proxy,
                        level="s",
                        version="v1.0",
                        conf_threshold=0.5,
                        iou_threshold=0.7,
                    )
                except Exception as exc:
                    raise OptionalBackendUnavailable(
                        f"DeepGHS anime half-body detection failed: {exc}"
                    ) from exc
                if halfbody:
                    local_box, _half_label, halfbody_score = max(
                        halfbody, key=lambda item: float(item[2])
                    )
                    hx0, hy0, hx1, hy1 = _clip_box(
                        local_box, person_proxy.width, person_proxy.height
                    )
                    global_proxy_box = (px0 + hx0, py0 + hy0, px0 + hx1, py0 + hy1)
                    halfbody_box = _map_box(
                        global_proxy_box, scale_x, scale_y, *source_size
                    )

            quality = _quality_tier(
                person_box,
                head_box,
                minimum_person_height=minimum_person_height,
                minimum_head_size=minimum_head_size,
            )
            if quality == "low":
                rejected_low_resolution += 1
                continue
            drafts.append(
                {
                    "person_bbox": person_box,
                    "head_bbox": head_box,
                    "halfbody_bbox": halfbody_box,
                    "person_score": float(person_score),
                    "head_score": float(head_score) if head_score is not None else None,
                    "halfbody_score": float(halfbody_score) if halfbody_score is not None else None,
                    "detection_kind": "person",
                    "quality_tier": quality,
                }
            )

        for index, (head_box_proxy, _label, head_score) in enumerate(head_detections):
            if index in matched_heads:
                continue
            head_box = _map_box(
                _clip_box(head_box_proxy, proxy.width, proxy.height),
                scale_x,
                scale_y,
                *source_size,
            )
            if max(_box_size(head_box)) < minimum_head_size:
                rejected_low_resolution += 1
                continue
            person_box = _infer_person_from_head(head_box, *source_size)
            quality = _quality_tier(
                person_box,
                head_box,
                minimum_person_height=minimum_person_height,
                minimum_head_size=minimum_head_size,
            )
            if quality == "low":
                rejected_low_resolution += 1
                continue
            head_fallbacks += 1
            drafts.append(
                {
                    "person_bbox": person_box,
                    "head_bbox": head_box,
                    "halfbody_bbox": None,
                    "person_score": None,
                    "head_score": float(head_score),
                    "halfbody_score": None,
                    "detection_kind": "head_fallback",
                    "quality_tier": quality,
                }
            )

        if drafts:
            frames_with_subjects += 1
        frame_subject_count = len(drafts)
        for draft in drafts:
            person_box = draft["person_bbox"]
            identity_box = _expand_box(person_box, 1.12, 1.10, *source_size)
            identity_image = source.crop(identity_box)
            native_resolution = identity_image.size
            subject_id = f"subject-{next_id:05d}"
            identity_path = identity_dir / f"{subject_id}.jpg"
            saved_resolution, _downscaled = _save_capped_image(
                identity_image,
                identity_path,
                max_long_edge=maximum_saved_long_edge,
                max_pixels=maximum_saved_pixels,
            )
            subjects.append(
                VideoSubject(
                    subject_id=subject_id,
                    identity_path=identity_path,
                    source_frame=frame,
                    source_timestamp_seconds=_timestamp_for_frame(frame, interval_seconds),
                    source_resolution=source_size,
                    person_bbox=person_box,
                    head_bbox=draft["head_bbox"],
                    halfbody_bbox=draft["halfbody_bbox"],
                    person_score=draft["person_score"],
                    head_score=draft["head_score"],
                    halfbody_score=draft["halfbody_score"],
                    detection_kind=draft["detection_kind"],
                    quality_tier=draft["quality_tier"],
                    frame_subject_count=frame_subject_count,
                    native_identity_resolution=native_resolution,
                    saved_identity_resolution=saved_resolution,
                )
            )
            next_id += 1

    if not subjects:
        raise PipelineError(
            "DeepGHS detected no character crops above the minimum native-resolution thresholds"
        )

    report = VideoSubjectReport(
        identity_dir=identity_dir,
        subjects=tuple(subjects),
        total_frames=len(frames),
        frames_with_subjects=frames_with_subjects,
        detected_persons=detected_persons,
        head_fallbacks=head_fallbacks,
        rejected_low_resolution=rejected_low_resolution,
        detection_proxy_long_edge=detection_proxy_long_edge,
        minimum_person_height=minimum_person_height,
        minimum_head_size=minimum_head_size,
        maximum_saved_long_edge=maximum_saved_long_edge,
        maximum_saved_pixels=maximum_saved_pixels,
    )
    write_json_atomic(output_dir / "subjects.json", report.as_dict(include_records=True))
    return report


def build_balanced_character_dataset(
    subject_report: VideoSubjectReport,
    selected_identity_paths: Iterable[Path],
    output_dir: Path,
    *,
    maximum_saved_long_edge: int = 2048,
    maximum_saved_pixels: int = 4_194_304,
    minimum_saved_short_edge: int = 512,
    phash_distance: int = 5,
) -> VideoCompositionReport:
    """Create one useful composition per selected subject without synthetic resolution copies."""

    subjects = list(subject_report.subjects_for_identity_paths(selected_identity_paths))
    if not subjects:
        raise PipelineError("No detected subjects correspond to the selected CCIP cluster")
    subjects.sort(
        key=lambda subject: (
            subject.source_timestamp_seconds is None,
            subject.source_timestamp_seconds or 0.0,
            subject.source_frame.name,
            subject.subject_id,
        )
    )

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    desired = _desired_composition_counts(len(subjects))
    assigned = Counter()
    hashes: list[imagehash.ImageHash] = []
    records: list[VideoCompositionRecord] = []
    rejected_duplicates = 0
    rejected_too_small = 0

    for subject in subjects:
        candidate_types = _composition_priority(subject, desired, assigned)
        saved = False
        with Image.open(subject.source_frame) as opened:
            source = opened.convert("RGB")
        for crop_type in candidate_types:
            crop_box = _composition_box(subject, crop_type)
            image = source.crop(crop_box)
            native_resolution = image.size
            if min(native_resolution) < minimum_saved_short_edge:
                continue
            prepared, downscaled = _capped_image(
                image,
                max_long_edge=maximum_saved_long_edge,
                max_pixels=maximum_saved_pixels,
            )
            fingerprint = imagehash.phash(prepared)
            if any(fingerprint - previous <= phash_distance for previous in hashes):
                continue
            output_path = output_dir / f"train-{len(records) + 1:05d}-{crop_type}.jpg"
            prepared.save(output_path, quality=95, subsampling=0)
            hashes.append(fingerprint)
            assigned[crop_type] += 1
            coverage = _box_area(subject.person_bbox) / max(1, _box_area(crop_box))
            records.append(
                VideoCompositionRecord(
                    path=output_path,
                    subject_id=subject.subject_id,
                    source_frame=subject.source_frame.name,
                    crop_type=crop_type,
                    crop_box=crop_box,
                    subject_coverage=min(1.0, coverage),
                    native_resolution=native_resolution,
                    saved_resolution=prepared.size,
                    downscaled=downscaled,
                    quality_tier=subject.quality_tier,
                )
            )
            saved = True
            break
        if not saved:
            # Distinguish the common reasons for project provenance. Re-test at the
            # broadest full-person composition rather than inventing an upscaled image.
            broad = source.crop(_composition_box(subject, "full_body"))
            if min(broad.size) < minimum_saved_short_edge:
                rejected_too_small += 1
            else:
                rejected_duplicates += 1

    if not records:
        raise PipelineError(
            "All selected character crops were too small or near-duplicates after composition balancing"
        )

    report = VideoCompositionReport(
        output_dir=output_dir,
        records=tuple(records),
        selected_subjects=len(subjects),
        rejected_near_duplicate=rejected_duplicates,
        rejected_too_small=rejected_too_small,
        max_saved_long_edge=maximum_saved_long_edge,
        max_saved_pixels=maximum_saved_pixels,
        minimum_saved_short_edge=minimum_saved_short_edge,
    )
    write_json_atomic(
        output_dir / "composition-manifest.json",
        report.as_dict(include_records=True),
    )
    return report


def _make_detection_proxy(image: Image.Image, long_edge: int) -> tuple[Image.Image, float, float]:
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


def _map_box(box: Box, scale_x: float, scale_y: float, width: int, height: int) -> Box:
    x0, y0, x1, y1 = box
    mapped = (
        round(x0 * scale_x),
        round(y0 * scale_y),
        round(x1 * scale_x),
        round(y1 * scale_y),
    )
    return _clip_box(mapped, width, height)


def _clip_box(box: Box, width: int, height: int) -> Box:
    x0, y0, x1, y1 = (int(value) for value in box)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def _box_size(box: Box) -> tuple[int, int]:
    return box[2] - box[0], box[3] - box[1]


def _box_area(box: Box) -> int:
    width, height = _box_size(box)
    return max(0, width) * max(0, height)


def _box_center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _best_head_for_person(heads: list[Detection], person_box: Box) -> int | None:
    matches: list[tuple[float, int]] = []
    for index, (head_box, _label, score) in enumerate(heads):
        cx, cy = _box_center(head_box)
        if person_box[0] <= cx <= person_box[2] and person_box[1] <= cy <= person_box[3]:
            matches.append((float(score), index))
    return max(matches)[1] if matches else None


def _quality_tier(
    person_box: Box,
    head_box: Box | None,
    *,
    minimum_person_height: int,
    minimum_head_size: int,
) -> str:
    person_height = _box_size(person_box)[1]
    head_size = max(_box_size(head_box)) if head_box else 0
    if head_size >= 256 or person_height >= 800:
        return "high"
    if head_size >= minimum_head_size or person_height >= minimum_person_height:
        return "medium"
    return "low"


def _infer_person_from_head(head_box: Box, width: int, height: int) -> Box:
    hx0, hy0, hx1, hy1 = head_box
    head_width, head_height = _box_size(head_box)
    cx = (hx0 + hx1) / 2
    inferred_height = max(head_height * 4.5, head_width * 4.0)
    inferred_width = max(head_width * 2.5, inferred_height * 0.45)
    top = hy0 - head_height * 0.35
    return _clip_box(
        (
            round(cx - inferred_width / 2),
            round(top),
            round(cx + inferred_width / 2),
            round(top + inferred_height),
        ),
        width,
        height,
    )


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


def _portrait_box(subject: VideoSubject) -> Box | None:
    if subject.head_bbox is None:
        return None
    head_width, head_height = _box_size(subject.head_bbox)
    cx, cy = _box_center(subject.head_bbox)
    target_width = max(head_width, head_height) * 2.5
    target_height = max(head_width, head_height) * 3.0
    cy += head_height * 0.45
    return _clip_box(
        (
            round(cx - target_width / 2),
            round(cy - target_height / 2),
            round(cx + target_width / 2),
            round(cy + target_height / 2),
        ),
        *subject.source_resolution,
    )


def _upper_body_box(subject: VideoSubject) -> Box:
    width, height = subject.source_resolution
    if subject.halfbody_bbox is not None:
        return _expand_box(subject.halfbody_bbox, 1.12, 1.10, width, height)
    x0, y0, x1, y1 = subject.person_bbox
    fallback = (x0, y0, x1, round(y0 + (y1 - y0) * 0.68))
    return _expand_box(fallback, 1.12, 1.10, width, height)


def _composition_box(subject: VideoSubject, crop_type: str) -> Box:
    width, height = subject.source_resolution
    if crop_type == "portrait":
        portrait = _portrait_box(subject)
        if portrait is None:
            return _upper_body_box(subject)
        return portrait
    if crop_type == "upper_body":
        return _upper_body_box(subject)
    if crop_type == "context":
        return _expand_box(subject.person_bbox, 1.80, 1.55, width, height)
    if crop_type == "full_body":
        return _expand_box(subject.person_bbox, 1.14, 1.12, width, height)
    raise PipelineError(f"Unknown video character crop type: {crop_type}")


def _available_compositions(subject: VideoSubject) -> tuple[str, ...]:
    values: list[str] = []
    if subject.head_bbox is not None:
        values.append("portrait")
    values.extend(["upper_body", "full_body"])
    # Context images are valuable for scale/generalization but can re-introduce other
    # characters. Only allow them when DeepGHS found a single usable subject in the frame.
    if subject.frame_subject_count == 1:
        values.append("context")
    return tuple(values)


def _desired_composition_counts(count: int) -> dict[str, int]:
    weights = {
        "portrait": 0.25,
        "upper_body": 0.30,
        "full_body": 0.30,
        "context": 0.15,
    }
    exact = {name: count * weight for name, weight in weights.items()}
    desired = {name: math.floor(value) for name, value in exact.items()}
    remaining = count - sum(desired.values())
    for name in sorted(weights, key=lambda item: (exact[item] - desired[item], weights[item]), reverse=True):
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
    available = _available_compositions(subject)
    return sorted(
        available,
        key=lambda name: (
            desired.get(name, 0) - assigned.get(name, 0),
            desired.get(name, 0),
            name,
        ),
        reverse=True,
    )


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
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    return resized, True


def _save_capped_image(
    image: Image.Image,
    path: Path,
    *,
    max_long_edge: int,
    max_pixels: int,
) -> tuple[tuple[int, int], bool]:
    prepared, downscaled = _capped_image(
        image,
        max_long_edge=max_long_edge,
        max_pixels=max_pixels,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared.save(path, quality=95, subsampling=0)
    return prepared.size, downscaled


def _timestamp_for_frame(path: Path, interval_seconds: int) -> float | None:
    stem = path.stem
    try:
        index = int(stem.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None
    if index < 1:
        return None
    return float((index - 1) * interval_seconds)
