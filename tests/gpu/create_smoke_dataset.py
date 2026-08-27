"""Create a deterministic, synthetic SDXL bucket smoke-test dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


SIZES = (
    (1024, 1024),
    (896, 1152),
    (1152, 896),
    (768, 1344),
    (1344, 768),
    (832, 1216),
    (1216, 832),
    (1024, 896),
)
PAIRED_SIZES = (
    (1024, 1024),
    (1024, 1024),
    (896, 1152),
    (896, 1152),
    (1152, 896),
    (1152, 896),
    (768, 1344),
    (768, 1344),
)


def create_dataset(output_dir: Path, sizes: tuple[tuple[int, int], ...] = SIZES) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (width, height) in enumerate(sizes, start=1):
        base = (
            (37 * index) % 256,
            (83 * index) % 256,
            (149 * index) % 256,
        )
        image = Image.new("RGB", (width, height), base)
        draw = ImageDraw.Draw(image)
        margin = max(32, min(width, height) // 10)
        accent = tuple(255 - channel for channel in base)
        draw.rectangle(
            (margin, margin, width - margin, height - margin),
            outline=accent,
            width=max(8, margin // 8),
        )
        radius = max(24, min(width, height) // 7)
        center_x = width * index // (len(sizes) + 1)
        center_y = height * (len(sizes) + 1 - index) // (len(sizes) + 1)
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=accent,
        )
        stem = f"smoke_{index:02d}"
        image.save(output_dir / f"{stem}.png", compress_level=3)
        caption = (
            "zz_smoke, abstract geometric composition, colored rectangle, "
            f"single circle, synthetic test image, aspect variant {index}"
        )
        (output_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--paired-buckets",
        action="store_true",
        help="Create two images in each of four buckets for a real batch-2 test.",
    )
    args = parser.parse_args()
    create_dataset(args.output_dir, PAIRED_SIZES if args.paired_buckets else SIZES)


if __name__ == "__main__":
    main()
