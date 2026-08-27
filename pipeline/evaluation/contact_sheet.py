from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..models import GeneratedImage


def create_contact_sheet(
    generated: Sequence[GeneratedImage],
    output: Path,
    *,
    prompt_id: str | None = None,
    tile_size: tuple[int, int] = (256, 256),
) -> Path:
    """Checkpoint x strength sheet for one aligned positive prompt."""

    positive = [item for item in generated if item.case.contains_trigger]
    if not positive:
        raise ValueError("No positive generated images are available for a contact sheet")
    selected_prompt = prompt_id or positive[0].case.prompt_id
    selected = [item for item in positive if item.case.prompt_id == selected_prompt]
    checkpoints = list(dict.fromkeys(item.case.checkpoint_label for item in selected))
    strengths = sorted({item.case.strength for item in selected})
    lookup = {(item.case.checkpoint_label, item.case.strength): item for item in selected}
    return _matrix_sheet(
        output,
        rows=checkpoints,
        columns=[f"strength {value:g}" for value in strengths],
        lookup=lambda row, column_index: lookup.get((row, strengths[column_index])),
        footer=f"Aligned prompt: {selected_prompt} · automatic scores are auxiliary",
        tile_size=tile_size,
    )


def create_prompt_checkpoint_sheet(
    generated: Sequence[GeneratedImage],
    output: Path,
    *,
    strength: float | None = None,
    tile_size: tuple[int, int] = (256, 256),
) -> Path:
    positive = [item for item in generated if item.case.contains_trigger]
    if not positive:
        raise ValueError("No positive generated images are available")
    strengths = sorted({item.case.strength for item in positive})
    selected_strength = strength if strength is not None else min(
        strengths, key=lambda value: abs(value - 0.8)
    )
    selected = [item for item in positive if item.case.strength == selected_strength]
    prompts = list(dict.fromkeys(item.case.prompt_id for item in selected))
    checkpoints = list(dict.fromkeys(item.case.checkpoint_label for item in selected))
    lookup = {(item.case.prompt_id, item.case.checkpoint_label): item for item in selected}
    return _matrix_sheet(
        output,
        rows=prompts,
        columns=checkpoints,
        lookup=lambda row, column_index: lookup.get((row, checkpoints[column_index])),
        footer=f"Prompt x checkpoint at strength {selected_strength:g}",
        tile_size=tile_size,
    )


def create_leakage_sheet(
    generated: Sequence[GeneratedImage],
    output: Path,
    *,
    checkpoint_label: str | None = None,
    strength: float | None = None,
    tile_size: tuple[int, int] = (256, 256),
) -> Path:
    if not generated:
        raise ValueError("No generated images are available")
    checkpoint = checkpoint_label or generated[0].case.checkpoint_label
    available_strengths = sorted(
        {
            item.case.strength
            for item in generated
            if item.case.checkpoint_label == checkpoint
        }
    )
    if not available_strengths:
        raise ValueError(f"No generated images exist for checkpoint {checkpoint}")
    selected_strength = strength if strength is not None else min(
        available_strengths, key=lambda value: abs(value - 0.8)
    )
    selected = [
        item
        for item in generated
        if item.case.checkpoint_label == checkpoint
        and item.case.strength == selected_strength
    ]
    prompts = list(dict.fromkeys(item.case.prompt_id for item in selected))
    lookup = {
        (item.case.prompt_id, item.case.contains_trigger): item for item in selected
    }
    return _matrix_sheet(
        output,
        rows=prompts,
        columns=["trigger on", "trigger off"],
        lookup=lambda row, column_index: lookup.get((row, column_index == 0)),
        footer=f"Leakage pairs · {checkpoint} · strength {selected_strength:g}",
        tile_size=tile_size,
    )


def _matrix_sheet(
    output: Path,
    *,
    rows: Sequence[str],
    columns: Sequence[str],
    lookup: Callable[[str, int], GeneratedImage | None],
    footer: str,
    tile_size: tuple[int, int],
) -> Path:
    if not rows or not columns:
        raise ValueError("Contact sheet matrix cannot be empty")
    tile_w, tile_h = tile_size
    row_header, column_header, footer_height = 190, 50, 42
    sheet = Image.new(
        "RGB",
        (
            row_header + tile_w * len(columns),
            column_header + tile_h * len(rows) + footer_height,
        ),
        "#202124",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, label in enumerate(columns):
        draw.multiline_text(
            (row_header + column * tile_w + 8, 12),
            _wrap(str(label), 28),
            fill="white",
            font=font,
            spacing=2,
        )
    for row_index, row in enumerate(rows):
        y = column_header + row_index * tile_h
        draw.multiline_text((8, y + 10), _wrap(str(row), 26), fill="white", font=font, spacing=3)
        for column_index in range(len(columns)):
            item = lookup(row, column_index)
            if item is None:
                tile = Image.new("RGB", tile_size, "#5f6368")
                ImageDraw.Draw(tile).text((10, 10), "missing", fill="white", font=font)
            else:
                with Image.open(item.path) as image:
                    tile = ImageOps.fit(
                        image.convert("RGB"), tile_size, method=Image.Resampling.LANCZOS
                    )
            sheet.paste(tile, (row_header + column_index * tile_w, y))
    draw.text(
        (8, column_header + tile_h * len(rows) + 12),
        footer,
        fill="#e8eaed",
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=92, optimize=True)
    return output


def _wrap(text: str, width: int) -> str:
    return "\n".join(text[index : index + width] for index in range(0, len(text), width))
