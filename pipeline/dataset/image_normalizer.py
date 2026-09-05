from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_MAX_PIXELS = 1_048_576
DEFAULT_BUCKET_STEP = 32


@dataclass(frozen=True)
class ImageNormalizationPlan:
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    downscaled: bool
    max_pixels: int
    bucket_step: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source_size": list(self.source_size),
            "target_size": list(self.target_size),
            "downscaled": self.downscaled,
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
    """Plan a downscale-only SDXL/Illustrious-friendly image size.

    Images at or below the target area are left byte-for-byte eligible for copying.
    Oversized images preserve aspect ratio approximately and are rounded down to the
    configured bucket step so the prepared files align naturally with sd-scripts
    bucket geometry. The result never exceeds ``max_pixels`` and is never upscaled.
    """

    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    if bucket_step < 1:
        raise ValueError("bucket_step must be positive")

    source = (int(width), int(height))
    area = width * height
    if area <= max_pixels:
        return ImageNormalizationPlan(source, source, False, max_pixels, bucket_step)

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

    # Rounding down should already satisfy the area cap, but keep this invariant
    # explicit in case very small bucket steps or future rounding changes are used.
    while target_width * target_height > max_pixels:
        if target_width >= target_height and target_width > bucket_step:
            target_width -= bucket_step
        elif target_height > bucket_step:
            target_height -= bucket_step
        else:
            break

    target = (max(1, target_width), max(1, target_height))
    return ImageNormalizationPlan(source, target, target != source, max_pixels, bucket_step)


def plan_training_image(
    source: Path,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    bucket_step: int = DEFAULT_BUCKET_STEP,
) -> ImageNormalizationPlan:
    with Image.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened)
        width, height = oriented.size
    return plan_training_size(
        width,
        height,
        max_pixels=max_pixels,
        bucket_step=bucket_step,
    )


def normalize_training_image(
    source: Path,
    destination: Path,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    bucket_step: int = DEFAULT_BUCKET_STEP,
) -> ImageNormalizationPlan:
    """Copy or downscale one image into a prepared training generation.

    Files already at or below the target area are copied unchanged. Oversized files
    are decoded once, EXIF orientation is applied, and the resulting pixels are
    downscaled with Lanczos. No path ever performs an upscale.
    """

    plan = plan_training_image(
        source,
        max_pixels=max_pixels,
        bucket_step=bucket_step,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not plan.downscaled:
        shutil.copy2(source, destination)
        return plan

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        prepared = image.resize(plan.target_size, Image.Resampling.LANCZOS)
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
