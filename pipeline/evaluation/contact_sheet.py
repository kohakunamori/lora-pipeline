from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..models import GeneratedImage


def create_contact_sheet(
    generated: Sequence[GeneratedImage],
    output: Path,
    *,
    prompt_id: str | None = None,
    tile_size: tuple[int, int] = (256, 256),
) -> Path:
    positive = [item for item in generated if item.case.contains_trigger]
    if not positive:
        raise ValueError("No positive generated images are available for a contact sheet")
    selected_prompt = prompt_id or positive[0].case.prompt_id
    selected = [item for item in positive if item.case.prompt_id == selected_prompt]
    checkpoints = list(dict.fromkeys(item.case.checkpoint_label for item in selected))
    strengths = sorted({item.case.strength for item in selected})
    lookup = {(item.case.checkpoint_label, item.case.strength): item for item in selected}
    tile_w, tile_h = tile_size
    row_header, column_header, footer = 180, 44, 42
    sheet = Image.new(
        "RGB",
        (row_header + tile_w * len(strengths), column_header + tile_h * len(checkpoints) + footer),
        "#202124",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, strength in enumerate(strengths):
        draw.text((row_header + column * tile_w + 8, 14), f"strength {strength:g}", fill="white", font=font)
    for row, checkpoint in enumerate(checkpoints):
        y = column_header + row * tile_h
        draw.multiline_text((8, y + 10), _wrap(checkpoint, 24), fill="white", font=font, spacing=3)
        for column, strength in enumerate(strengths):
            item = lookup.get((checkpoint, strength))
            if item is None:
                tile = Image.new("RGB", tile_size, "#5f6368")
                ImageDraw.Draw(tile).text((10, 10), "missing", fill="white", font=font)
            else:
                with Image.open(item.path) as image:
                    tile = ImageOps.fit(image.convert("RGB"), tile_size, method=Image.Resampling.LANCZOS)
            sheet.paste(tile, (row_header + column * tile_w, y))
    draw.text(
        (8, column_header + tile_h * len(checkpoints) + 12),
        f"Aligned prompt: {selected_prompt} · automatic scores are auxiliary; inspect samples manually",
        fill="#e8eaed",
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=92, optimize=True)
    return output


def _wrap(text: str, width: int) -> str:
    return "\n".join(text[index : index + width] for index in range(0, len(text), width))
