from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

Box = tuple[int, int, int, int]

DEFAULT_MAX_PIXELS = 1_048_576
DEFAULT_BUCKET_STEP = 32


@dataclass(frozen=True)
class ImageNormalizationPlan:
    source_size: tuple[int, int]
    input_size: tuple[int, int]
    target_size: tuple[int, int]
    downscaled: bool
    max_pixels: int
    bucket_step: int
    crop_box: Box | None = None

    @property
    def cropped(self) -> bool:
        return self.crop_box is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "source_size": list(self.source_size),
            "input_size": list(self.input_size),
            "target_size": list(self.target_size),
            "downscaled": self.downscaled,
            "cropped": self.cropped,
            "crop_box": list(self.crop_box) if self.crop_box else None,
            "max_pixels": self.max_pixels,
            "bucket_step": self.bucket_step,
            "no_upscale": True,
        }


def plan_training_size(
    width: int,
    height: int,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    bucket_step: int = DEFAULT_BUCKET_STEP,
) -> ImageNormalizationPlan:
    """Plan a downscale-only SDXL/Illustrious-friendly image size."""

    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    if bucket_step < 1:
        raise ValueError("bucket_step must be positive")

    source = (int(width), int(height))
    target = _target_size(
        width,
        height,
        max_pixels=max_pixels,
        bucket_step=bucket_step,
    )
    return ImageNormalizationPlan(
        source_size=source,
        input_size=source,
        target_size=target,
        downscaled=target != source,
        max_pixels=max_pixels,
        bucket_step=bucket_step,
    )


def plan_training_image(
    source: Path,
    *,
    crop_box: Box | None = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    bucket_step: int = DEFAULT_BUCKET_STEP,
) -> ImageNormalizationPlan:
    """Plan crop -> downscale materialization without ever upscaling."""

    with Image.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened)
        source_size = oriented.size

    normalized_crop = _validate_crop_box(crop_box, source_size) if crop_box else None
    if normalized_crop is None:
        input_size = source_size
    else:
        input_size = (
            normalized_crop[2] - normalized_crop[0],
            normalized_crop[3] - normalized_crop[1],
        )
    target = _target_size(
        *input_size,
        max_pixels=max_pixels,
        bucket_step=bucket_step,
    )
    return ImageNormalizationPlan(
        source_size=source_size,
        input_size=input_size,
        target_size=target,
        downscaled=target != input_size,
        max_pixels=max_pixels,
        bucket_step=bucket_step,
        crop_box=normalized_crop,
    )


def normalize_training_image(
    source: Path,
    destination: Path,
    *,
    crop_box: Box | None = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    bucket_step: int = DEFAULT_BUCKET_STEP,
) -> ImageNormalizationPlan:
    """Materialize one training image as crop -> downscale.

    No-crop images at or below the target area are copied byte-for-byte. Cropped
    images are decoded from the original source, cropped before any resize, and
    only then downscaled if the crop still exceeds the configured pixel budget.
    """

    plan = plan_training_image(
        source,
        crop_box=crop_box,
        max_pixels=max_pixels,
        bucket_step=bucket_step,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not plan.cropped and not plan.downscaled:
        shutil.copy2(source, destination)
        return plan

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        if plan.crop_box is not None:
            image = image.crop(plan.crop_box)
        if plan.downscaled:
            image = image.resize(plan.target_size, Image.Resampling.LANCZOS)
        prepared = image
        suffix = destination.suffix.casefold()
        if suffix in {".jpg", ".jpeg"} and prepared.mode not in {"RGB", "L"}:
            prepared = prepared.convert("RGB")
        save_kwargs: dict[str, object] = {}
        if suffix in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": 95, "subsampling": 0})
        elif suffix == ".webp":
            save_kwargs.update({"quality": 95, "method": 4})
        prepared.save(destination, **save_kwargs)
    return plan


def _target_size(
    width: int,
    height: int,
    *,
    max_pixels: int,
    bucket_step: int,
) -> tuple[int, int]:
    area = width * height
    if area <= max_pixels:
        return int(width), int(height)

    scale = math.sqrt(max_pixels / area)
    raw_width = max(1, math.floor(width * scale))
    raw_height = max(1, math.floor(height * scale))

    if raw_width >= bucket_step:
        target_width = max(bucket_step, (raw_width // bucket_step) * bucket_step)
    else:
        target_width = raw_width
    if raw_height >= bucket_step:
        target_height = max(bucket_step, (raw_height // bucket_step) * bucket_step)
    else:
        target_height = raw_height

    while target_width * target_height > max_pixels:
        if target_width >= target_height and target_width > bucket_step:
            target_width -= bucket_step
        elif target_height > bucket_step:
            target_height -= bucket_step
        else:
            break

    return max(1, target_width), max(1, target_height)


def _validate_crop_box(box: Box, source_size: tuple[int, int]) -> Box:
    width, height = source_size
    x0, y0, x1, y1 = (int(value) for value in box)
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(
            f"crop box {box!r} falls outside oriented source size {source_size!r}"
        )
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"crop box must have positive area: {box!r}")
    return x0, y0, x1, y1
