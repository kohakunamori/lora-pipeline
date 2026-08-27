from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from PIL import ExifTags, Image, UnidentifiedImageError

from ..config import sha256_file, stable_hash


SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"})
ORIENTATION_TAG = next((key for key, value in ExifTags.TAGS.items() if value == "Orientation"), 274)


def discover_images(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def unique_caption_relative(image_relative: Path) -> Path:
    """Avoid sidecar collisions when different image formats share one stem."""
    suffix = image_relative.suffix.lower().lstrip(".") or "image"
    return image_relative.parent / f"{image_relative.stem}__{suffix}.txt"


def inspect_image(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    record: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "caption": path.with_suffix(".txt").exists(),
        "corrupt": False,
    }
    try:
        # Pillow requires verify() to run immediately after open(); metadata and
        # EXIF access may seek or load data first, so validation gets its own handle.
        with Image.open(path) as verification:
            verification.verify()
        with Image.open(path) as image:
            width, height = image.size
            record.update(
                {
                    "format": image.format,
                    "mode": image.mode,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 6) if height else None,
                    "megapixels": round(width * height / 1_000_000, 4),
                    "alpha": "A" in image.getbands() or "transparency" in image.info,
                    "animated": bool(getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) > 1),
                    "frames": int(getattr(image, "n_frames", 1)),
                    "exif_orientation": image.getexif().get(ORIENTATION_TAG),
                    "very_small": min(width, height) < 512,
                }
            )
    except (UnidentifiedImageError, OSError, RuntimeError, ValueError) as exc:
        record.update({"corrupt": True, "error": f"{type(exc).__name__}: {exc}"})
    return record


def inspect_dataset(raw_dir: Path) -> dict[str, Any]:
    images = discover_images(raw_dir)
    records = [inspect_image(path, raw_dir) for path in images]
    valid = [record for record in records if not record["corrupt"]]
    widths = [int(record["width"]) for record in valid]
    heights = [int(record["height"]) for record in valid]
    megapixels = [float(record["megapixels"]) for record in valid]
    formats = Counter(str(record.get("format") or "unknown") for record in records)
    summary = {
        "image_count": len(records),
        "valid_images": len(valid),
        "corrupt_images": sum(bool(record["corrupt"]) for record in records),
        "formats": dict(sorted(formats.items())),
        "caption_count": sum(bool(record["caption"]) for record in records),
        "very_small_images": sum(bool(record.get("very_small")) for record in records),
        "alpha_images": sum(bool(record.get("alpha")) for record in records),
        "animated_images": sum(bool(record.get("animated")) for record in records),
        "exif_oriented_images": sum(record.get("exif_orientation") not in (None, 1) for record in records),
        "width": _range_summary(widths),
        "height": _range_summary(heights),
        "megapixels": _range_summary(megapixels),
    }
    return {
        "schema_version": 1,
        "root": str(raw_dir),
        "input_hash": stable_hash(
            [{"path": record["path"], "bytes": record["bytes"], "sha256": record["sha256"]} for record in records]
        ),
        "summary": summary,
        "images": records,
    }


def _range_summary(values: list[int] | list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": median(values), "max": max(values)}
