from __future__ import annotations

from io import StringIO
from pathlib import Path

from PIL import Image
from rich.console import Console

from pipeline import i18n
from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.interactive_datasets import InteractiveWizard, parse_csv_tags


def _console() -> tuple[Console, StringIO]:
    stream = StringIO()
    return Console(file=stream, force_terminal=False, width=140), stream


def _image(path: Path, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (768, 1024), color).save(path)


def test_home_makes_dataset_management_a_first_class_action(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    i18n.set_language("en")
    console, _ = _console()
    wizard = InteractiveWizard(console=console)
    seen: list[str] = []

    def menu(_title, items, **_kwargs):
        seen.extend(item.value for item in items)
        return "quit"

    monkeypatch.setattr(wizard, "_menu", menu)
    wizard.home()

    assert "datasets" in seen
    assert "new" in seen
    assert "bases" in seen
    assert "doctor" in seen


def test_create_dataset_is_separate_from_project_creation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    i18n.set_language("en")
    console, _ = _console()
    wizard = InteractiveWizard(console=console)
    monkeypatch.setattr(wizard, "_ask_text", lambda *args, **kwargs: "character-data")
    monkeypatch.setattr(wizard, "_menu", lambda *args, **kwargs: "character")

    workspace = wizard._create_dataset()

    assert workspace is not None
    assert workspace.name == "character-data"
    assert workspace.concept_type == "character"
    assert (tmp_path / "datasets" / "character-data" / "dataset.yaml").is_file()
    assert not (tmp_path / "projects").exists()


def test_smart_crop_creates_a_derived_source_and_can_disable_original(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    i18n.set_language("en")
    original_dir = tmp_path / "original"
    _image(original_dir / "a.png", "red")
    workspace = DatasetWorkspace.create("demo", concept_type="character", root=tmp_path)
    original = workspace.add_source_from_directory(original_dir, kind="image_directory", label="original")

    console, _ = _console()
    wizard = InteractiveWizard(console=console)

    def fake_select(frame_dir: Path):
        output = frame_dir.parent / "selected-character"
        output.mkdir(parents=True, exist_ok=True)
        _image(output / "crop-a.jpg", "blue")
        return output, {"status": "selected_crop_cluster", "selected_cluster": 0}

    monkeypatch.setattr(wizard, "_select_video_identity", fake_select)
    monkeypatch.setattr(wizard, "_ask_text", lambda *args, **kwargs: "cropped")
    monkeypatch.setattr(wizard, "_confirm", lambda *args, **kwargs: True)

    wizard._smart_crop_source(workspace, str(original["id"]))

    reloaded = DatasetWorkspace.load("demo", root=tmp_path)
    assert reloaded.sources[str(original["id"])]["enabled"] is False
    derived = [source for source in reloaded.sources.values() if source["kind"] == "smart_crop"]
    assert len(derived) == 1
    assert derived[0]["parent_source"] == original["id"]
    assert derived[0]["label"] == "cropped"
    assert reloaded.summary()["active_images"] == 1


def test_parse_csv_tags_accepts_chinese_and_ascii_commas() -> None:
    assert parse_csv_tags("1girl，blue hair, smile\nupper body") == [
        "1girl",
        "blue hair",
        "smile",
        "upper body",
    ]
