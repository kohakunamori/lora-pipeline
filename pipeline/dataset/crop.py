from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from ..models import OptionalBackendUnavailable, PipelineError
from .subject import Box, SubjectDetector, SubjectObservation

CROP_POLICY_VERSION = 1
MINIMUM_CROP_SHORT_EDGE = 512


@dataclass(frozen=True)
class CropPlan:
    target_type: str
    source_size: tuple[int, int]
    crop_box: Box | None
    mode: str
    reason: str
    subject: SubjectObservation | None
    minimum_crop_short_edge: int = MINIMUM_CROP_SHORT_EDGE
    policy_version: int = CROP_POLICY_VERSION

    @property
    def cropped(self) -> bool:
        return self.crop_box is not None

    @property
    def crop_size(self) -> tuple[int, int]:
        if self.crop_box is None:
            return self.source_size
        return (
            self.crop_box[2] - self.crop_box[0],
            self.crop_box[3] - self.crop_box[1],
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "target_type": self.target_type,
            "mode": self.mode,
            "reason": self.reason,
            "source_size": list(self.source_size),
            "crop_box": list(self.crop_box) if self.crop_box else None,
            "crop_size": list(self.crop_size),
            "cropped": self.cropped,
            "minimum_crop_short_edge": self.minimum_crop_short_edge,
            "policy_version": self.policy_version,
            "subject": self.subject.as_dict() if self.subject else None,
        }


def plan_target_crop(
    source: Path,
    *,
    target_type: str,
    detector: SubjectDetector | None = None,
    minimum_crop_short_edge: int = MINIMUM_CROP_SHORT_EDGE,
) -> CropPlan:
    """Plan a conservative target-aware crop from original pixels.

    Subject detection is an optimization, not a validity requirement. If the
    optional DeepGHS backend is unavailable, preserve the original composition
    and record the reason rather than blocking materialization.
    """

    if target_type not in {"character", "character_outfit", "style"}:
        raise PipelineError(f"Unsupported crop target: {target_type}")
    if minimum_crop_short_edge < 1:
        raise ValueError("minimum_crop_short_edge must be positive")

    with Image.open(source) as opened:
        source_size = ImageOps.exif_transpose(opened).size

    if target_type == "style":
        return CropPlan(
            target_type=target_type,
            source_size=source_size,
            crop_box=None,
            mode="composition_preserving",
            reason="style_preserves_composition",
            subject=None,
            minimum_crop_short_edge=minimum_crop_short_edge,
        )

    # Cropping cannot create detail. Avoid detector/model work when the source is
    # already below the no-upscale floor.
    if min(source_size) < minimum_crop_short_edge:
        return CropPlan(
            target_type=target_type,
            source_size=source_size,
            crop_box=None,
            mode="keep",
            reason="source_too_small_for_safe_crop",
            subject=None,
            minimum_crop_short_edge=minimum_crop_short_edge,
        )

    try:
        observation = (detector or SubjectDetector()).detect_path(source)
    except OptionalBackendUnavailable:
        return CropPlan(
            target_type=target_type,
            source_size=source_size,
            crop_box=None,
            mode="keep",
            reason="subject_detector_unavailable",
            subject=None,
            minimum_crop_short_edge=minimum_crop_short_edge,
        )
    if observation is None:
        return CropPlan(
            target_type=target_type,
            source_size=source_size,
            crop_box=None,
            mode="keep",
            reason="no_subject_detected",
            subject=None,
            minimum_crop_short_edge=minimum_crop_short_edge,
        )
    if observation.ambiguous:
        return _keep(
            observation,
            target_type=target_type,
            reason="multiple_subjects_ambiguous",
            minimum_crop_short_edge=minimum_crop_short_edge,
        )
    if target_type == "character_outfit":
        return _plan_character_outfit(
            observation,
            minimum_crop_short_edge=minimum_crop_short_edge,
        )
    return _plan_character(
        observation,
        minimum_crop_short_edge=minimum_crop_short_edge,
    )


def _plan_character(
    subject: SubjectObservation,
    *,
    minimum_crop_short_edge: int,
) -> CropPlan:
    if subject.person_height_fraction >= 0.72 or subject.person_area_fraction >= 0.45:
        return _keep(
            subject,
            target_type="character",
            reason="subject_already_prominent",
            minimum_crop_short_edge=minimum_crop_short_edge,
        )

    if subject.detection_kind == "head_fallback":
        candidate = _expand_box(
            subject.person_bbox,
            x_scale=1.45,
            y_scale=1.30,
            source_size=subject.source_size,
        )
        reason = "head_fallback_subject_crop"
    else:
        candidate = _expand_box(
            subject.person_bbox,
            x_scale=1.28,
            y_scale=1.18,
            source_size=subject.source_size,
        )
        reason = "subject_too_small"

    return _crop_or_keep(
        subject,
        candidate,
        target_type="character",
        reason=reason,
        minimum_crop_short_edge=minimum_crop_short_edge,
    )


def _plan_character_outfit(
    subject: SubjectObservation,
    *,
    minimum_crop_short_edge: int,
) -> CropPlan:
    # Head-only inference cannot prove garment completeness.
    if subject.detection_kind == "head_fallback":
        return _keep(
            subject,
            target_type="character_outfit",
            reason="head_fallback_cannot_preserve_outfit",
            minimum_crop_short_edge=minimum_crop_short_edge,
        )

    # Outfit training values full-person context more than tight face framing.
    if subject.person_height_fraction >= 0.78 or subject.person_area_fraction >= 0.50:
        return _keep(
            subject,
            target_type="character_outfit",
            reason="outfit_subject_already_prominent",
            minimum_crop_short_edge=minimum_crop_short_edge,
        )

    candidate = _expand_box(
        subject.person_bbox,
        x_scale=1.38,
        y_scale=1.26,
        source_size=subject.source_size,
    )
    return _crop_or_keep(
        subject,
        candidate,
        target_type="character_outfit",
        reason="outfit_preserving_subject_crop",
        minimum_crop_short_edge=minimum_crop_short_edge,
    )


def _crop_or_keep(
    subject: SubjectObservation,
    candidate: Box,
    *,
    target_type: str,
    reason: str,
    minimum_crop_short_edge: int,
) -> CropPlan:
    width = candidate[2] - candidate[0]
    height = candidate[3] - candidate[1]
    if min(width, height) < minimum_crop_short_edge:
        return _keep(
            subject,
            target_type=target_type,
            reason="candidate_crop_too_small_no_upscale",
            minimum_crop_short_edge=minimum_crop_short_edge,
        )

    source_area = max(1, subject.source_size[0] * subject.source_size[1])
    if (width * height) / source_area >= 0.88:
        return _keep(
            subject,
            target_type=target_type,
            reason="crop_would_not_materially_change_composition",
            minimum_crop_short_edge=minimum_crop_short_edge,
        )

    return CropPlan(
        target_type=target_type,
        source_size=subject.source_size,
        crop_box=candidate,
        mode="subject_crop" if target_type == "character" else "outfit_preserving",
        reason=reason,
        subject=subject,
        minimum_crop_short_edge=minimum_crop_short_edge,
    )


def _keep(
    subject: SubjectObservation,
    *,
    target_type: str,
    reason: str,
    minimum_crop_short_edge: int,
) -> CropPlan:
    return CropPlan(
        target_type=target_type,
        source_size=subject.source_size,
        crop_box=None,
        mode="keep",
        reason=reason,
        subject=subject,
        minimum_crop_short_edge=minimum_crop_short_edge,
    )


def _expand_box(
    box: Box,
    *,
    x_scale: float,
    y_scale: float,
    source_size: tuple[int, int],
) -> Box:
    width, height = source_size
    x0, y0, x1, y1 = box
    box_width = x1 - x0
    box_height = y1 - y0
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
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
