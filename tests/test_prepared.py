from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pipeline.materialization import run as materialize
from pipeline.models import PipelineError
from pipeline.prepared import load_current_generation
from pipeline.state import ProjectState


def _state(tmp_path: Path) -> ProjectState:
    state = ProjectState.create(
        tmp_path / "project",
        name="prepared",
        concept_type="character",
        base="base",
        trigger="zz_prepared",
        strategy="quality",
    )
    image = state.project_dir / "raw" / "sample.png"
    Image.new("RGB", (64, 96), "red").save(image)
    return state


def test_materialize_requires_captions_by_default(tmp_path) -> None:
    state = _state(tmp_path)
    with pytest.raises(PipelineError, match="no usable caption"):
        materialize(state)


def test_trigger_only_requires_explicit_opt_in(tmp_path) -> None:
    state = _state(tmp_path)
    result = materialize(state, allow_trigger_only=True)
    generation = load_current_generation(state.project_dir)
    record = generation.manifest["images"][0]
    caption = generation.root / record["caption"]
    assert caption.read_text(encoding="utf-8").strip() == "zz_prepared"
    assert record["caption_source"] == "explicit-trigger-only"
    assert result.details["trigger_only_captions"] == 1


def test_materialized_generations_are_content_addressed_and_immutable(tmp_path) -> None:
    state = _state(tmp_path)
    caption = state.project_dir / "raw" / "sample.txt"
    caption.write_text("zz_prepared, red dress\n", encoding="utf-8")
    first = materialize(state)
    first_generation = load_current_generation(state.project_dir)
    first_caption = first_generation.root / first_generation.manifest["images"][0]["caption"]
    first_bytes = first_caption.read_bytes()
    assert first.details["reused_generation"] is False
    assert not any(path.is_symlink() for path in first_generation.root.rglob("*"))

    repeated = materialize(state)
    assert repeated.details["generation_id"] == first.details["generation_id"]
    assert repeated.details["reused_generation"] is True

    caption.write_text("zz_prepared, blue dress\n", encoding="utf-8")
    second = materialize(state)
    second_generation = load_current_generation(state.project_dir)
    assert second.details["generation_id"] != first.details["generation_id"]
    assert first_generation.root.is_dir()
    assert first_caption.read_bytes() == first_bytes
    assert (second_generation.root / second_generation.manifest["images"][0]["caption"]).read_text(
        encoding="utf-8"
    ).strip() == "zz_prepared, blue dress"


def test_validation_images_are_never_added_to_training_generation(tmp_path) -> None:
    state = _state(tmp_path)
    (state.project_dir / "raw" / "sample.txt").write_text(
        "zz_prepared, portrait\n", encoding="utf-8"
    )
    validation = state.project_dir / "validation" / "holdout.png"
    Image.new("RGB", (64, 64), "blue").save(validation)
    materialize(state)
    generation = load_current_generation(state.project_dir)
    assert [record["source"] for record in generation.manifest["images"]] == ["sample.png"]
    assert all("holdout" not in path.as_posix() for path in generation.root.rglob("*"))
