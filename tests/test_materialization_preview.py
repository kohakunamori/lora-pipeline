from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pipeline.materialization import run as materialize
from pipeline.materialization.preview import preview_path
from pipeline.prepared import load_current_generation
from pipeline.state import ProjectState


def test_materialization_writes_side_by_side_preview(tmp_path: Path) -> None:
    state = ProjectState.create(
        tmp_path / "project",
        name="preview",
        concept_type="style",
        base="base",
        trigger="zz_preview",
        strategy="quality",
    )
    image = state.project_dir / "raw" / "sample.png"
    Image.new("RGB", (900, 1200), "blue").save(image)
    image.with_suffix(".txt").write_text(
        "zz_preview, 1girl, blue dress, outdoors\n",
        encoding="utf-8",
    )

    materialize(state)
    generation = load_current_generation(state.project_dir)
    preview = preview_path(state.project_dir, generation.generation_id)
    pointer = json.loads(
        (state.project_dir / "prepared" / "current.json").read_text(encoding="utf-8")
    )

    assert preview.is_file()
    assert generation.root not in preview.parents
    text = preview.read_text(encoding="utf-8")
    assert "Materialization Preview" in text
    assert "sample.png" in text
    assert "zz_preview, 1girl, blue dress, outdoors" in text
    assert "style_preserves_composition" in text
    assert "../../raw/sample.png" in text
    assert f"../generations/{generation.generation_id}/images/sample.png" in text
    assert pointer["preview"] == preview.relative_to(state.project_dir).as_posix()


def test_preview_does_not_change_generation_identity_or_bytes(tmp_path: Path) -> None:
    state = ProjectState.create(
        tmp_path / "project",
        name="preview-stable",
        concept_type="style",
        base="base",
        trigger="zz_preview",
        strategy="quality",
    )
    image = state.project_dir / "raw" / "sample.png"
    Image.new("RGB", (768, 1024), "red").save(image)
    image.with_suffix(".txt").write_text("zz_preview, portrait\n", encoding="utf-8")

    first = materialize(state)
    generation = load_current_generation(state.project_dir)
    preview = preview_path(state.project_dir, generation.generation_id)
    generation_bytes = {
        path.relative_to(generation.root).as_posix(): path.read_bytes()
        for path in generation.root.rglob("*")
        if path.is_file()
    }
    preview.write_text("locally edited review artifact", encoding="utf-8")

    second = materialize(state)
    assert second.details["generation_id"] == first.details["generation_id"]
    assert second.details["reused_generation"] is True
    assert "Materialization Preview" in preview.read_text(encoding="utf-8")
    assert {
        path.relative_to(generation.root).as_posix(): path.read_bytes()
        for path in generation.root.rglob("*")
        if path.is_file()
    } == generation_bytes
