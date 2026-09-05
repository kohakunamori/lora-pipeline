from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import sha256_file, stable_hash
from ..dataset.crop import CropPlan, plan_target_crop
from ..dataset.image_normalizer import (
    DEFAULT_BUCKET_STEP,
    DEFAULT_MAX_PIXELS,
    ImageNormalizationPlan,
    normalize_training_image,
    plan_training_image,
)
from ..dataset.subject import SubjectDetector


@dataclass(frozen=True)
class MaterializedVisual:
    source: Path
    relative: Path
    source_sha256: str
    cache_path: Path
    visual_hash: str
    crop: CropPlan
    normalization: ImageNormalizationPlan

    def as_manifest_record(self) -> dict[str, object]:
        return {
            "source": self.relative.as_posix(),
            "source_image_sha256": self.source_sha256,
            "visual_hash": self.visual_hash,
            "source_size": list(self.normalization.source_size),
            "crop_size": list(self.normalization.input_size),
            "prepared_size": list(self.normalization.target_size),
            "cropped": self.normalization.cropped,
            "downscaled": self.normalization.downscaled,
            "crop": self.crop.as_dict(),
        }


def materialize_visual(
    source: Path,
    relative: Path,
    cache_root: Path,
    *,
    target_type: str,
    detector: SubjectDetector | None = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    bucket_step: int = DEFAULT_BUCKET_STEP,
) -> MaterializedVisual:
    """Compile one source image into the exact pixels used by tagging/training.

    The cache is content addressed by raw bytes plus crop/normalization policy.
    Caption generation can therefore tag the same transformed pixels that are later
    copied into the immutable prepared generation.
    """

    source_sha = sha256_file(source)
    crop = plan_target_crop(
        source,
        target_type=target_type,
        detector=detector,
    )
    normalization = plan_training_image(
        source,
        crop_box=crop.crop_box,
        max_pixels=max_pixels,
        bucket_step=bucket_step,
    )
    basis = {
        "schema_version": 1,
        "source_sha256": source_sha,
        "target_type": target_type,
        "crop": crop.as_dict(),
        "normalization": normalization.as_dict(),
    }
    visual_hash = stable_hash(basis)
    cache_path = cache_root / visual_hash / relative
    if not cache_path.is_file():
        normalize_training_image(
            source,
            cache_path,
            crop_box=crop.crop_box,
            max_pixels=max_pixels,
            bucket_step=bucket_step,
        )
    return MaterializedVisual(
        source=source,
        relative=relative,
        source_sha256=source_sha,
        cache_path=cache_path,
        visual_hash=visual_hash,
        crop=crop,
        normalization=normalization,
    )


def copy_visual_to_generation(visual: MaterializedVisual, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(visual.cache_path, destination)
