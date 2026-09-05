from pathlib import Path

from PIL import Image

import pipeline.materialization.service as materialization_service
from pipeline.dataset.crop import CropPlan
from pipeline.materialization import run as materialize
from pipeline.prepared import load_current_generation
from pipeline.state import ProjectState


def test_materialization_applies_crop_before_one_mp_normalization(
    tmp_path: Path, monkeypatch
) -> None:
    state = ProjectState.create(
        tmp_path / "project",
        name="crop-materialize",
        concept_type="character",
        base="base",
        trigger="zz_character",
        strategy="quality",
    )
    state.payload["project"]["training_target_type"] = "character"
    state.save()

    image = state.project_dir / "raw" / "sample.png"
    Image.new("RGB", (3000, 2000), "green").save(image)
    image.with_suffix(".txt").write_text(
        "zz_character, 1girl, standing\n",
        encoding="utf-8",
    )

    def fake_crop(source, *, target_type, detector=None, minimum_crop_short_edge=512):
        del source, detector
        assert target_type == "character"
        return CropPlan(
            target_type="character",
            source_size=(3000, 2000),
            crop_box=(1050, 500, 1950, 1500),
            mode="subject_crop",
            reason="test_subject_crop",
            subject=None,
            minimum_crop_short_edge=minimum_crop_short_edge,
        )

    monkeypatch.setattr(materialization_service, "plan_target_crop", fake_crop)

    result = materialize(state)
    generation = load_current_generation(state.project_dir)
    record = generation.manifest["images"][0]
    prepared = generation.root / record["image"]

    assert record["cropped"] is True
    assert record["crop_size"] == [900, 1000]
    assert record["prepared_size"] == [900, 1000]
    assert record["downscaled"] is False
    assert result.details["cropped_images"] == 1
    with Image.open(prepared) as opened:
        assert opened.size == (900, 1000)
