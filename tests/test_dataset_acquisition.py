from __future__ import annotations

from pathlib import Path

from PIL import Image

from pipeline.dataset_acquisition import analyze_acquisition_gaps
from pipeline.dataset_workspace import DatasetWorkspace


def _image(path: Path, *, size: tuple[int, int] = (768, 1024)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "red").save(path)


def test_character_acquisition_reports_concentrated_pose_and_missing_outfit_diversity(tmp_path) -> None:
    source = tmp_path / "source"
    for index in range(12):
        image = source / f"{index}.png"
        _image(image)
        image.with_suffix(".txt").write_text(
            "1girl, standing, smile, portrait, indoors, day\n",
            encoding="utf-8",
        )

    workspace = DatasetWorkspace.create("demo", concept_type="character", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    report = analyze_acquisition_gaps(workspace, target_type="character")

    actions = {action["dimension"]: action for action in report["actions"]}
    assert report["status"] == "needs_data"
    assert actions["pose"]["priority"] == "acquire"
    assert "standing" in actions["pose"]["reason"]
    assert actions["outfit"]["priority"] == "acquire"


def test_character_outfit_acquisition_requests_full_body_coverage(tmp_path) -> None:
    source = tmp_path / "source"
    for index in range(12):
        image = source / f"{index}.png"
        _image(image)
        image.with_suffix(".txt").write_text(
            "1girl, standing, smile, portrait, outdoors, day\n",
            encoding="utf-8",
        )

    workspace = DatasetWorkspace.create("demo", concept_type="character", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    report = analyze_acquisition_gaps(workspace, target_type="character_outfit")

    actions = {action["dimension"]: action for action in report["actions"]}
    assert actions["full_body"]["priority"] == "acquire"
    assert "Full-body coverage" in actions["full_body"]["reason"]


def test_style_acquisition_turns_bias_warnings_into_specific_capture_goals(tmp_path) -> None:
    source = tmp_path / "source"
    for index in range(20):
        image = source / f"{index}.png"
        _image(image, size=(768, 1024))
        image.with_suffix(".txt").write_text(
            "1girl, portrait, simple_background, indoors, day\n",
            encoding="utf-8",
        )

    workspace = DatasetWorkspace.create("demo", concept_type="style", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    report = analyze_acquisition_gaps(workspace)

    actions = {action["dimension"]: action for action in report["actions"]}
    assert report["status"] == "needs_data"
    assert actions["style_entanglement"]["priority"] == "blocking_acquire"
    assert "composition" in actions
    assert "background" in actions
    assert "subject_count" in actions
    assert "aspect_ratio" in actions


def test_missing_captions_are_metadata_gap_not_fake_diversity_gap(tmp_path) -> None:
    source = tmp_path / "source"
    for index in range(3):
        _image(source / f"{index}.png")

    workspace = DatasetWorkspace.create("demo", concept_type="character", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    report = analyze_acquisition_gaps(workspace)

    assert report["missing_captions"] == 3
    assert report["actions"][0]["priority"] == "metadata"
    assert report["actions"][0]["dimension"] == "caption_coverage"
