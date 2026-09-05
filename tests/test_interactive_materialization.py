from io import StringIO
from pathlib import Path

from PIL import Image
from rich.console import Console

from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.interactive_materialization import InteractiveWizard


def _console() -> Console:
    return Console(file=StringIO(), force_terminal=False, width=140)


def _image(path: Path, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (768, 1024), color).save(path)


def test_materialization_smart_crop_activates_derived_and_disables_original(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    source_dir = tmp_path / "source"
    _image(source_dir / "a.png")
    workspace = DatasetWorkspace.create("demo", concept_type="character", root=tmp_path)
    source = workspace.add_source_from_directory(source_dir, kind="image_directory", label="original")
    wizard = InteractiveWizard(console=_console())

    def fake_materialize(frame_dir: Path):
        output = frame_dir.parent / "selected-character"
        _image(output / "crop-a.jpg", "blue")
        return output, {"status": "identity_assumed_valid", "selected_subjects": 1}

    monkeypatch.setattr(wizard, "_select_video_identity", fake_materialize)
    wizard._smart_crop_source(workspace, str(source["id"]))

    reloaded = DatasetWorkspace.load("demo", root=tmp_path)
    assert reloaded.sources[str(source["id"])]["enabled"] is False
    derived = [row for row in reloaded.sources.values() if row["kind"] == "smart_crop"]
    assert len(derived) == 1
    assert derived[0]["enabled"] is True
    assert derived[0]["parent_source"] == source["id"]
    assert derived[0]["label"] == "original-crop"
    assert derived[0]["processing"]["subject_materialization"]["identity_assumed_valid"] is True


def test_materialization_smart_crop_keeps_original_when_detector_returns_originals(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    source_dir = tmp_path / "source"
    _image(source_dir / "a.png")
    workspace = DatasetWorkspace.create("demo", concept_type="character", root=tmp_path)
    source = workspace.add_source_from_directory(source_dir, kind="image_directory", label="original")
    wizard = InteractiveWizard(console=_console())

    monkeypatch.setattr(
        wizard,
        "_select_video_identity",
        lambda frame_dir: (
            frame_dir,
            {"status": "subject_detection_unavailable_keep_originals", "identity_assumed_valid": True},
        ),
    )
    wizard._smart_crop_source(workspace, str(source["id"]))

    reloaded = DatasetWorkspace.load("demo", root=tmp_path)
    assert reloaded.sources[str(source["id"])]["enabled"] is True
    assert not [row for row in reloaded.sources.values() if row["kind"] == "smart_crop"]
