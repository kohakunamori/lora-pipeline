from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.steps.train import _freeze_run_snapshot
from pipeline.training_config import (
    TrainingConfig,
    create_project_from_training_config,
)


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


def _workspace(root: Path) -> DatasetWorkspace:
    source = root / "source"
    source.mkdir()
    image = source / "a.png"
    Image.new("RGB", (768, 1024), "red").save(image)
    image.with_suffix(".txt").write_text("portrait, smile\n", encoding="utf-8")
    workspace = DatasetWorkspace.create("demo", concept_type="character", root=root)
    workspace.add_source_from_directory(source, kind="image_directory", label="source")
    return workspace


def test_training_config_is_reusable_and_has_content_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    _base_registry(tmp_path)
    config = TrainingConfig.create(
        "quality-char",
        concept_type="character",
        base="base",
        trigger="zz_demo",
        strategy="quality",
        images_seen=1600,
        overrides={"training": {"network_dim": 32, "network_alpha": 16}},
        root=tmp_path,
    )

    first = config.snapshot()
    assert first["name"] == "quality-char"
    assert first["images_seen"] == 1600
    assert first["overrides"]["training"]["network_dim"] == 32
    assert first["workflow"] == {
        "caption_mode": "auto",
        "allow_trigger_only": False,
    }

    loaded = TrainingConfig.load("quality-char", root=tmp_path)
    assert loaded.snapshot()["snapshot_hash"] == first["snapshot_hash"]
    loaded.data["images_seen"] = 2200
    loaded.save()
    assert loaded.snapshot()["snapshot_hash"] != first["snapshot_hash"]


def test_dataset_and_training_config_are_frozen_together(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    _base_registry(tmp_path)
    workspace = _workspace(tmp_path)
    config = TrainingConfig.create(
        "quality-char",
        concept_type="character",
        base="base",
        trigger="zz_demo",
        strategy="quality",
        images_seen=1600,
        overrides={"training": {"network_dim": 32, "network_alpha": 16, "unet_lr": 0.00008}},
        evaluation={"subject_prompt": "1girl"},
        root=tmp_path,
    )

    state = create_project_from_training_config(
        workspace,
        config,
        project_name="run-demo",
        root=tmp_path,
    )
    project = state.payload["project"]
    dataset_hash = project["dataset_snapshot"]["snapshot_hash"]
    config_hash = project["training_config_snapshot"]["snapshot_hash"]
    assert project["workspace_role"] == "training_run"
    assert project["training_identity"]["dataset"] == "demo"
    assert project["training_identity"]["config"] == "quality-char"
    assert project["overrides"]["training"]["network_dim"] == 32
    assert project["interactive_preferences"] == {
        "caption_mode": "existing_taglist_clean",
        "allow_trigger_only": False,
    }

    item = workspace.items()[0]
    workspace.replace_caption(item.key, "full body")
    config.data["images_seen"] = 3000
    config.save()

    reloaded = type(state).load(state.project_dir)
    assert reloaded.payload["project"]["dataset_snapshot"]["snapshot_hash"] == dataset_hash
    assert reloaded.payload["project"]["training_config_snapshot"]["snapshot_hash"] == config_hash
    assert reloaded.payload["project"]["training_config_snapshot"]["images_seen"] == 1600


def test_run_snapshot_manifest_is_not_rewritten_on_resume(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    _base_registry(tmp_path)
    workspace = _workspace(tmp_path)
    config = TrainingConfig.create(
        "quality-char",
        concept_type="character",
        base="base",
        trigger="zz_demo",
        root=tmp_path,
    )
    state = create_project_from_training_config(
        workspace,
        config,
        project_name="run-demo",
        root=tmp_path,
    )
    run_dir = state.project_dir / "runs" / "run-001"
    run_dir.mkdir(parents=True)
    run_record: dict[str, object] = {"id": "run-001", "path": str(run_dir), "status": "running"}

    _freeze_run_snapshot(state, run_record, run_dir)
    manifest = run_dir / "config" / "run-snapshot.yaml"
    first = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    first_hash = first["training_config_snapshot"]["snapshot_hash"]

    state.payload["project"]["training_config_snapshot"]["images_seen"] = 999999
    state.save()
    _freeze_run_snapshot(state, run_record, run_dir)
    second = yaml.safe_load(manifest.read_text(encoding="utf-8"))

    assert second["training_config_snapshot"]["snapshot_hash"] == first_hash
    assert second["training_config_snapshot"]["images_seen"] != 999999
