from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pipeline.dataset_tag_editor import batch_edit_tags
from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.models import PipelineError


def _image(path: Path, color: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 128), color).save(path)


def _workspace(tmp_path: Path) -> tuple[DatasetWorkspace, str, dict[str, str]]:
    source = tmp_path / "source"
    _image(source / "a.png", "red")
    _image(source / "b.png", "blue")
    _image(source / "c.png", "green")
    (source / "a.txt").write_text("1girl, blue_hair, smile\n", encoding="utf-8")
    (source / "b.txt").write_text("solo, Blue Hair, outdoors\n", encoding="utf-8")
    (source / "c.txt").write_text("portrait\n", encoding="utf-8")
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    record = workspace.add_source_from_directory(source, kind="image_directory", label="images")
    items = {item.relative.name: item.key for item in workspace.items()}
    return workspace, str(record["id"]), items


def test_batch_prepend_moves_existing_semantic_duplicates_to_front(tmp_path: Path) -> None:
    workspace, _, items = _workspace(tmp_path)
    result = batch_edit_tags(
        workspace,
        [items["a.png"], items["b.png"]],
        ["blue hair", "trigger", "Trigger"],
        action="prepend",
    )
    assert result["changed"] == 2
    assert result["tags"] == ["blue hair", "trigger"]
    assert workspace.caption_text(items["a.png"]) == "blue hair, trigger, 1girl, smile"
    assert workspace.caption_text(items["b.png"]) == "blue hair, trigger, solo, outdoors"


def test_batch_append_moves_existing_tags_to_tail_and_remove_normalizes_names(tmp_path: Path) -> None:
    workspace, _, items = _workspace(tmp_path)
    batch_edit_tags(
        workspace,
        [items["a.png"], items["c.png"]],
        ["smile", "masterpiece"],
        action="append",
    )
    assert workspace.caption_text(items["a.png"]) == "1girl, blue_hair, smile, masterpiece"
    assert workspace.caption_text(items["c.png"]) == "portrait, smile, masterpiece"
    result = batch_edit_tags(
        workspace,
        [items["a.png"], items["b.png"]],
        ["BLUE HAIR", "smile"],
        action="remove",
    )
    assert result["changed"] == 2
    assert workspace.caption_text(items["a.png"]) == "1girl, masterpiece"
    assert workspace.caption_text(items["b.png"]) == "solo, outdoors"


def test_batch_edit_validates_every_key_before_writing(tmp_path: Path) -> None:
    workspace, _, items = _workspace(tmp_path)
    before = workspace.caption_text(items["a.png"])
    with pytest.raises(PipelineError, match="Unknown dataset item"):
        batch_edit_tags(
            workspace,
            [items["a.png"], "image-directory-999/missing.png"],
            ["trigger"],
            action="prepend",
        )
    assert workspace.caption_text(items["a.png"]) == before


def test_interactive_entrypoint_combines_batch_tags_and_composition_metadata() -> None:
    from pipeline.interactive import InteractiveWizard
    from pipeline.interactive_batch_tags import InteractiveWizard as BatchTagWizard
    from pipeline.interactive_composition import InteractiveWizard as CompositionWizard

    assert issubclass(InteractiveWizard, BatchTagWizard)
    assert issubclass(InteractiveWizard, CompositionWizard)
