from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from pipeline.dataset.tagger import TagResult, TaggerBackend
from pipeline.dataset_workspace import (
    DatasetWorkspace,
    create_project_from_dataset,
    parse_number_selection,
)
from pipeline.models import PipelineError


class FakeTagger(TaggerBackend):
    def tag(self, image: Path) -> TagResult:
        del image
        return TagResult(
            ratings={"general": 1.0},
            tags={"1girl": 0.99, "blue_hair": 0.91, "smile": 0.7, "low": 0.1},
            characters={"some_character": 0.9},
            backend="fake",
            metadata={"cache_hit": False},
        )


def _image(path: Path, color: str = "red", size: tuple[int, int] = (768, 1024)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _base_registry(root: Path) -> None:
    (root / "bases").mkdir(parents=True, exist_ok=True)
    checkpoint = root / "base.safetensors"
    checkpoint.write_bytes(b"base")
    (root / "bases" / "registry.yaml").write_text(
        yaml.safe_dump(
            {
                "bases": {
                    "base": {
                        "name": "Base",
                        "path": str(checkpoint),
                        "family": "illustrious_sdxl",
                        "prediction_type": "epsilon",
                        "sha256": None,
                        "enabled": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_dataset_keeps_same_named_images_separate_by_source(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _image(first / "same.png", "red")
    _image(second / "same.png", "blue")

    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    one = workspace.add_source_from_directory(first, kind="image_directory", label="first")
    two = workspace.add_source_from_directory(second, kind="image_directory", label="second")

    assert one["id"] == "image-directory-001"
    assert two["id"] == "image-directory-002"
    keys = [item.key for item in workspace.items()]
    assert keys == [
        "image-directory-001/same.png",
        "image-directory-002/same.png",
    ]
    assert workspace.summary()["active_images"] == 2


def test_audit_only_auto_excludes_safe_corrupt_and_exact_duplicate_items(tmp_path) -> None:
    source = tmp_path / "source"
    _image(source / "a.png", "red")
    (source / "b.png").write_bytes((source / "a.png").read_bytes())
    (source / "broken.jpg").write_bytes(b"not-an-image")
    _image(source / "tiny.png", "blue", (320, 480))

    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    audit = workspace.audit()

    records = {record["key"]: record for record in audit["records"]}
    assert audit["summary"]["safe_exclude_suggestions"] == 2
    assert records["image-directory-001/b.png"]["safe_exclude"] is True
    assert records["image-directory-001/broken.jpg"]["safe_exclude"] is True
    assert records["image-directory-001/tiny.png"]["safe_exclude"] is False
    assert any(flag["code"] == "very_small" for flag in records["image-directory-001/tiny.png"]["flags"])

    result = workspace.apply_safe_audit_exclusions()
    assert result["excluded"] == 2
    assert workspace.summary()["excluded_images"] == 2
    assert workspace.summary()["active_images"] == 2


def test_manual_exclusion_is_reversible_and_number_ranges_are_convenient(tmp_path) -> None:
    source = tmp_path / "source"
    for index in range(5):
        _image(source / f"{index}.png", "red")
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    items = workspace.items()

    assert parse_number_selection("1,3-5", maximum=5) == [1, 3, 4, 5]
    assert parse_number_selection("5-3", maximum=5) == [3, 4, 5]
    with pytest.raises(PipelineError, match="out of range"):
        parse_number_selection("6", maximum=5)

    keys = [items[index - 1].key for index in parse_number_selection("2-4", maximum=5)]
    assert workspace.exclude(keys, reason="bad pose") == 3
    assert workspace.summary()["active_images"] == 2
    assert workspace.restore([keys[1]]) == 1
    assert workspace.summary()["active_images"] == 3


def test_auto_tag_preserves_existing_manual_tags_and_supports_manual_edits(tmp_path) -> None:
    source = tmp_path / "source"
    _image(source / "manual.png", "red")
    _image(source / "auto.png", "blue")
    (source / "manual.txt").write_text("hand edited, portrait\n", encoding="utf-8")

    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    record = workspace.add_source_from_directory(source, kind="image_directory")
    result = workspace.auto_tag(source_id=record["id"], tagger=FakeTagger(), threshold=0.35)

    assert result["tagged"] == 1
    assert result["skipped_existing"] == 1
    items = {item.relative.name: item for item in workspace.items()}
    assert workspace.caption_text(items["manual.png"].key) == "hand edited, portrait"
    assert workspace.caption_text(items["auto.png"].key) == "1girl, blue_hair, smile"

    key = items["auto.png"].key
    workspace.add_tags(key, ["upper body", "smile"])
    assert workspace.caption_text(key) == "1girl, blue_hair, smile, upper body"
    workspace.remove_tags(key, ["blue hair", "smile"])
    assert workspace.caption_text(key) == "1girl, upper body"
    workspace.replace_caption(key, "full body, outdoors")
    assert workspace.caption_text(key) == "full body, outdoors"


def test_source_enablement_controls_dataset_snapshot_without_deleting_files(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _image(first / "a.png", "red")
    _image(second / "b.png", "blue")
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    one = workspace.add_source_from_directory(first, kind="image_directory")
    two = workspace.add_source_from_directory(second, kind="image_directory")
    workspace.set_source_enabled(one["id"], False)

    snapshot = workspace.snapshot()
    assert snapshot["image_count"] == 1
    assert snapshot["sources"] == [
        {
            "id": two["id"],
            "kind": "image_directory",
            "label": two["id"],
            "parent_source": None,
        }
    ]
    assert workspace.source_images_dir(one["id"]).joinpath("a.png").is_file()


def test_project_is_an_immutable_snapshot_of_mutable_dataset(tmp_path) -> None:
    _base_registry(tmp_path)
    source = tmp_path / "source"
    _image(source / "a.png", "red")
    (source / "a.txt").write_text("portrait, smile\n", encoding="utf-8")

    workspace = DatasetWorkspace.create("demo", concept_type="character", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory", label="photos")
    state = create_project_from_dataset(
        workspace,
        name="train-one",
        base="base",
        trigger="zz_demo",
        strategy="quality",
        images_seen=1000,
        root=tmp_path,
    )

    snapshot = state.payload["project"]["dataset_snapshot"]
    assert snapshot["dataset"] == "demo"
    assert snapshot["image_count"] == 1
    assert snapshot["caption_count"] == 1
    assert state.payload["project"]["interactive_preferences"]["caption_mode"] == "existing_taglist_clean"
    inspection_path = state.project_dir / "dataset-manifest.json"
    assert inspection_path.is_file()
    inspection = yaml.safe_load(inspection_path.read_text(encoding="utf-8"))
    assert inspection["summary"]["image_count"] == 1
    assert inspection["summary"]["corrupt_images"] == 0
    assert inspection["images"][0]["path"] == "image-directory-001/a.png"
    raw_caption = state.project_dir / "raw" / "image-directory-001" / "a.txt"
    assert raw_caption.read_text(encoding="utf-8").strip() == "portrait, smile"

    item = workspace.items()[0]
    workspace.replace_caption(item.key, "full body, outdoors")
    workspace.exclude([item.key], reason="changed mind")

    reloaded = type(state).load(state.project_dir)
    assert reloaded.payload["project"]["dataset_snapshot"]["snapshot_hash"] == snapshot["snapshot_hash"]
    assert raw_caption.read_text(encoding="utf-8").strip() == "portrait, smile"
    assert (state.project_dir / "raw" / "image-directory-001" / "a.png").is_file()


def test_dataset_project_creation_rejects_corrupt_active_images_at_dataset_boundary(
    tmp_path,
) -> None:
    _base_registry(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "broken.jpg").write_bytes(b"not-an-image")

    workspace = DatasetWorkspace.create("demo", concept_type="character", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")

    with pytest.raises(PipelineError, match="corrupt active image"):
        create_project_from_dataset(
            workspace,
            name="train-one",
            base="base",
            trigger="zz_demo",
            strategy="quality",
            images_seen=1000,
            root=tmp_path,
        )


def test_dataset_curation_freshness_tracks_only_active_image_set(tmp_path) -> None:
    source = tmp_path / "source"
    _image(source / "a.png", "red")
    _image(source / "b.png", "blue")
    workspace = DatasetWorkspace.create("demo", concept_type="style", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")

    workspace.analyze_duplicates()
    fresh = workspace.curation_status()
    assert fresh["analyses"]["dedup"]["fresh"] is True

    item = workspace.items()[0]
    workspace.replace_caption(item.key, "portrait, outdoors")
    assert workspace.curation_status()["analyses"]["dedup"]["fresh"] is True

    workspace.exclude([item.key], reason="test freshness")
    stale = workspace.curation_status()
    assert stale["analyses"]["dedup"]["fresh"] is False


def test_dataset_project_freezes_fresh_duplicate_analysis_as_snapshot_evidence(tmp_path) -> None:
    _base_registry(tmp_path)
    source = tmp_path / "source"
    _image(source / "a.png", "red")
    _image(source / "b.png", "blue")
    workspace = DatasetWorkspace.create("demo", concept_type="style", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    workspace.analyze_duplicates()

    state = create_project_from_dataset(
        workspace,
        name="train-style",
        base="base",
        trigger="zz_style",
        strategy="quality",
        images_seen=1000,
        root=tmp_path,
    )

    record = state.payload["project"]["dataset_curation"]["analyses"]["dedup"]
    assert record["fresh"] is True
    assert record["summary"]
    assert (state.project_dir / "review" / "duplicates" / "manifest.json").is_file()


def test_dataset_project_marks_stale_duplicate_analysis_as_non_reusable_evidence(tmp_path) -> None:
    _base_registry(tmp_path)
    source = tmp_path / "source"
    _image(source / "a.png", "red")
    _image(source / "b.png", "blue")
    workspace = DatasetWorkspace.create("demo", concept_type="style", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    workspace.analyze_duplicates()
    workspace.exclude([workspace.items()[0].key], reason="make analysis stale")

    state = create_project_from_dataset(
        workspace,
        name="train-stale",
        base="base",
        trigger="zz_style",
        strategy="quality",
        images_seen=1000,
        root=tmp_path,
    )

    assert state.payload["project"]["dataset_curation"]["analyses"]["dedup"]["fresh"] is False
    assert not (state.project_dir / "review" / "duplicates" / "manifest.json").exists()


def test_dataset_project_freezes_fresh_character_identity_analysis(
    tmp_path,
    monkeypatch,
) -> None:
    _base_registry(tmp_path)
    source = tmp_path / "source"
    _image(source / "a.png", "red")
    _image(source / "b.png", "blue")
    workspace = DatasetWorkspace.create("demo", concept_type="character", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")

    def fake_identity(paths, *, min_samples):
        assert min_samples == 2
        paths = list(paths)
        return {
            "method": "ccip",
            "labels": [0 for _ in paths],
            "clusters": {0: len(paths)},
            "main_cluster": [str(path) for path in paths],
            "possible_outliers": [],
            "possible_mixed_characters": [],
        }

    monkeypatch.setattr(
        "pipeline.dataset_workspace.analyze_character_identity",
        fake_identity,
    )
    workspace.analyze_duplicates()
    workspace.analyze_identity()

    state = create_project_from_dataset(
        workspace,
        name="train-character",
        base="base",
        trigger="zz_character",
        strategy="quality",
        images_seen=1000,
        root=tmp_path,
    )

    record = state.payload["project"]["dataset_curation"]["analyses"]["identity"]
    assert record["fresh"] is True
    assert record["summary"]["possible_outliers"] == 0
    assert (state.project_dir / "review" / "outliers" / "manifest.json").is_file()
