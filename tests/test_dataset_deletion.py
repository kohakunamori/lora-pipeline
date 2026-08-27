from __future__ import annotations

from pathlib import Path

from PIL import Image

from pipeline.dataset_deletion import (
    delete_dataset_items,
    delete_dataset_source,
    delete_dataset_workspace,
)
from pipeline.dataset_workspace import DatasetWorkspace


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (value, value, value)).save(path)


def _source_dir(root: Path, name: str, values: list[int]) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for index, value in enumerate(values, start=1):
        image = directory / f"{index:03d}.png"
        _image(image, value)
        image.with_suffix(".txt").write_text(f"tag_{index}\n", encoding="utf-8")
    return directory


def test_delete_selected_items_removes_image_caption_and_exclusion(tmp_path) -> None:
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    record = workspace.add_source_from_directory(
        _source_dir(tmp_path, "input", [20, 80, 140]),
        kind="image_directory",
    )
    source_id = str(record["id"])
    items = workspace.items(source_id=source_id)
    target = items[1]
    workspace.exclude([target.key], reason="test")

    result = delete_dataset_items(workspace, [target.key])

    assert result["deleted_images"] == 1
    assert result["deleted_captions"] == 1
    assert not target.image.exists()
    assert not target.caption.exists()
    assert target.key not in workspace._load_exclusions()
    remaining = workspace.items(source_id=source_id, include_excluded=True)
    assert [item.relative.name for item in remaining] == ["001.png", "003.png"]


def test_delete_source_removes_only_that_source_and_keeps_derived_or_other_sources(tmp_path) -> None:
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    first = workspace.add_source_from_directory(
        _source_dir(tmp_path, "first", [10, 30]),
        kind="image_directory",
        label="first",
    )
    second = workspace.add_source_from_directory(
        _source_dir(tmp_path, "second", [90]),
        kind="smart_crop",
        label="derived",
        parent_source=str(first["id"]),
    )
    first_id = str(first["id"])
    second_id = str(second["id"])
    first_dir = workspace.source_dir(first_id)
    workspace.exclude([workspace.items(source_id=first_id)[0].key], reason="test")

    result = delete_dataset_source(workspace, first_id)

    assert result["deleted_images"] == 2
    assert not first_dir.exists()
    assert first_id not in workspace.sources
    assert second_id in workspace.sources
    assert workspace.sources[second_id]["parent_source"] == first_id
    assert all(not key.startswith(f"{first_id}/") for key in workspace._load_exclusions())
    assert len(workspace.items(source_id=second_id)) == 1


def test_delete_dataset_workspace_does_not_touch_sibling_project_like_data(tmp_path) -> None:
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    workspace.add_source_from_directory(
        _source_dir(tmp_path, "input", [50]),
        kind="image_directory",
    )
    sibling = tmp_path / "projects" / "old-run" / "raw"
    sibling.mkdir(parents=True)
    marker = sibling / "keep.txt"
    marker.write_text("immutable snapshot", encoding="utf-8")
    dataset_dir = workspace.dataset_dir

    result = delete_dataset_workspace(workspace)

    assert result["dataset"] == "demo"
    assert result["sources"] == 1
    assert result["images"] == 1
    assert not dataset_dir.exists()
    assert marker.read_text(encoding="utf-8") == "immutable snapshot"
