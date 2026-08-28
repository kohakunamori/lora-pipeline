from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pipeline.config import write_yaml_atomic
from pipeline.dataset_deletion import delete_dataset_workspace
from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.models import PipelineError
from pipeline.resource_deletion import (
    delete_training_config,
    delete_training_project,
    delete_training_run,
    guarded_create_project_from_training_config,
)
from pipeline.state import ProjectState
from pipeline.training_config import TrainingConfig
from pipeline.web_jobs import create_job, update_job


def _registry(root: Path) -> None:
    write_yaml_atomic(
        root / "bases" / "registry.yaml",
        {
            "bases": {
                "base": {
                    "name": "Test base",
                    "path": str(root / "models" / "base.safetensors"),
                    "family": "sdxl",
                    "prediction_type": "epsilon",
                    "enabled": True,
                }
            }
        },
    )


def _config(root: Path, name: str = "cfg") -> TrainingConfig:
    _registry(root)
    return TrainingConfig.create(
        name,
        concept_type="character",
        base="base",
        trigger="test_trigger",
        root=root,
    )


def _project(
    root: Path,
    *,
    name: str = "run",
    dataset: str = "ds",
    config: str = "cfg",
    run_status: str | None = None,
) -> ProjectState:
    state = ProjectState.create(
        root / "projects" / name,
        name=name,
        concept_type="character",
        base="base",
        trigger="test_trigger",
        strategy="quality",
    )
    state.payload["project"].update(
        {
            "workspace_role": "training_run",
            "training_identity": {"dataset": dataset, "config": config},
        }
    )
    if run_status is not None:
        state.payload["runs"] = [
            {
                "id": "r1",
                "path": str(state.project_dir / "runs" / "r1"),
                "status": run_status,
            }
        ]
        (state.project_dir / "runs" / "r1").mkdir(parents=True, exist_ok=True)
    state.save()
    return state


def test_pending_training_workspace_blocks_config_deletion(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _project(tmp_path, config=config.name, run_status=None)

    with pytest.raises(PipelineError, match="active lifecycle references"):
        delete_training_config(config.name, root=tmp_path)

    assert config.path.is_file()


def test_historical_trained_project_does_not_block_config_deletion(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _project(tmp_path, config=config.name, run_status="trained")

    result = delete_training_config(config.name, root=tmp_path)

    assert result["config"] == config.name
    assert not config.path.exists()
    assert (tmp_path / "projects" / "run" / "project.yaml").is_file()


def test_queued_job_blocks_dataset_deletion(tmp_path: Path) -> None:
    workspace = DatasetWorkspace.create("ds", root=tmp_path)
    _project(tmp_path, dataset=workspace.name, run_status="trained")
    create_job("evaluate", {"project": "run"}, root=tmp_path)

    with pytest.raises(PipelineError, match="active lifecycle references"):
        delete_dataset_workspace(workspace)

    assert workspace.dataset_dir.is_dir()


def test_completed_job_and_historical_project_allow_dataset_deletion(tmp_path: Path) -> None:
    workspace = DatasetWorkspace.create("ds", root=tmp_path)
    _project(tmp_path, dataset=workspace.name, run_status="trained")
    job = create_job("evaluate", {"project": "run"}, root=tmp_path)
    update_job(str(job["id"]), root=tmp_path, status="completed")

    result = delete_dataset_workspace(workspace)

    assert result["dataset"] == "ds"
    assert not workspace.dataset_dir.exists()
    assert (tmp_path / "projects" / "run" / "project.yaml").is_file()


def test_active_project_cannot_be_deleted(tmp_path: Path) -> None:
    _project(tmp_path, run_status="running")

    with pytest.raises(PipelineError, match="active lifecycle references"):
        delete_training_project("run", root=tmp_path)

    assert (tmp_path / "projects" / "run").is_dir()


def test_completed_project_can_be_deleted_without_touching_dataset_or_config(tmp_path: Path) -> None:
    workspace = DatasetWorkspace.create("ds", root=tmp_path)
    config = _config(tmp_path)
    _project(tmp_path, dataset=workspace.name, config=config.name, run_status="trained")

    result = delete_training_project("run", root=tmp_path)

    assert result["project"] == "run"
    assert not (tmp_path / "projects" / "run").exists()
    assert workspace.dataset_dir.is_dir()
    assert config.path.is_file()


def test_completed_run_can_be_deleted_without_removing_project(tmp_path: Path) -> None:
    state = _project(tmp_path, run_status="trained")
    run_dir = state.project_dir / "runs" / "r1"
    (run_dir / "weights.safetensors").write_bytes(b"weights")

    result = delete_training_run("run", "r1", root=tmp_path)

    assert result["run_id"] == "r1"
    assert not run_dir.exists()
    fresh = ProjectState.load(tmp_path / "projects" / "run")
    assert fresh.payload["runs"] == []
    assert fresh.project_dir.is_dir()


def test_running_run_cannot_be_deleted(tmp_path: Path) -> None:
    state = _project(tmp_path, run_status="running")

    with pytest.raises(PipelineError, match="active lifecycle references"):
        delete_training_run("run", "r1", root=tmp_path)

    assert (state.project_dir / "runs" / "r1").is_dir()


def test_guarded_snapshot_creation_makes_pending_dependency_visible(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    Image.new("RGB", (64, 64), (120, 120, 120)).save(source / "001.png")
    workspace = DatasetWorkspace.create("ds", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    config = _config(tmp_path)

    state = guarded_create_project_from_training_config(
        workspace,
        config,
        project_name="run",
        root=tmp_path,
    )

    assert state.payload["project"]["training_identity"]["dataset"] == "ds"
    with pytest.raises(PipelineError, match="active lifecycle references"):
        delete_training_config(config.name, root=tmp_path)
