from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import repository_root
from .dataset_workspace import DatasetWorkspace
from .lifecycle_guard import assert_deletable, lifecycle_lock
from .models import StateError, StepStatus
from .service import project_path
from .state import ProjectState, utc_now
from .training_config import (
    TrainingConfig,
    create_project_from_training_config,
    training_configs_root,
)


def guarded_create_project_from_training_config(
    workspace: DatasetWorkspace,
    config: TrainingConfig,
    *,
    project_name: str,
    root: Path | None = None,
):
    """Compatibility name for the canonical atomic Project factory."""

    return create_project_from_training_config(
        workspace,
        config,
        project_name=project_name,
        root=root,
    )


def delete_training_config(name: str, *, root: Path | None = None) -> dict[str, Any]:
    resolved = (root or repository_root()).resolve()
    with lifecycle_lock(resolved):
        config = TrainingConfig.load(name, root=resolved)
        assert_deletable("training_config", config.name, root=resolved)
        config_root = training_configs_root(resolved).resolve()
        path = config.path.resolve()
        try:
            path.relative_to(config_root)
        except ValueError as exc:
            raise StateError(f"Refusing to delete training config outside {config_root}: {path}") from exc
        if path.parent != config_root or path.suffix != ".yaml":
            raise StateError(f"Refusing to delete invalid training config path: {path}")
        size = path.stat().st_size
        path.unlink()
        return {"config": config.name, "path": str(path), "deleted_bytes": size}


def delete_training_project(project_name: str, *, root: Path | None = None) -> dict[str, Any]:
    resolved = (root or repository_root()).resolve()
    with lifecycle_lock(resolved):
        assert_deletable("project", project_name, root=resolved)
        path = project_path(project_name, root=resolved).resolve()
        projects_root = (resolved / "projects").resolve()
        try:
            path.relative_to(projects_root)
        except ValueError as exc:
            raise StateError(f"Refusing to delete project outside {projects_root}: {path}") from exc
        if path.parent != projects_root or not (path / "project.yaml").is_file():
            raise StateError(f"Refusing to delete invalid project workspace: {path}")
        deleted_bytes = _directory_size(path)
        run_count = len([item for item in (path / "runs").iterdir()]) if (path / "runs").is_dir() else 0
        shutil.rmtree(path)
        return {
            "project": project_name,
            "path": str(path),
            "runs": run_count,
            "deleted_bytes": deleted_bytes,
        }


def delete_training_run(
    project_name: str,
    run_id: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Delete one finished Run and all of its run-scoped Results artifacts."""

    resolved = (root or repository_root()).resolve()
    with lifecycle_lock(resolved):
        assert_deletable("run", f"{project_name}/{run_id}", root=resolved)
        project_dir = project_path(project_name, root=resolved).resolve()
        state = ProjectState.load(project_dir)
        runs = list(state.payload.get("runs", []))
        run = next((item for item in runs if str(item.get("id")) == run_id), None)
        if run is None:
            raise StateError(f"Run does not exist: {project_name}/{run_id}")

        run_dir = Path(str(run.get("path") or project_dir / "runs" / run_id)).resolve()
        runs_root = (project_dir / "runs").resolve()
        try:
            run_dir.relative_to(runs_root)
        except ValueError as exc:
            raise StateError(f"Refusing to delete run outside {runs_root}: {run_dir}") from exc
        if run_dir.parent != runs_root or run_dir.name != run_id:
            raise StateError(f"Refusing to delete invalid run directory: {run_dir}")

        deleted_bytes = _directory_size(run_dir) if run_dir.exists() else 0
        tombstone = runs_root / f".{run_id}.deleting"
        if tombstone.exists():
            shutil.rmtree(tombstone)
        if run_dir.exists():
            run_dir.rename(tombstone)

        original_runs = state.payload.get("runs", [])
        state.payload["runs"] = [item for item in runs if str(item.get("id")) != run_id]
        try:
            _invalidate_train_pointer(state, run_id, run_dir)
            state.save()
        except BaseException:
            state.payload["runs"] = original_runs
            if tombstone.exists() and not run_dir.exists():
                tombstone.rename(run_dir)
            raise

        if tombstone.exists():
            shutil.rmtree(tombstone)
        return {
            "project": project_name,
            "run_id": run_id,
            "path": str(run_dir),
            "deleted_bytes": deleted_bytes,
        }


def _invalidate_train_pointer(state: ProjectState, run_id: str, run_dir: Path) -> None:
    """Reset the train step only when it points at the deleted Run.

    Evaluation and promotion are stored inside the Run itself, so deleting the Run
    removes those Results without touching the Project step namespace.
    """

    record = state.step("train")
    reference = json.dumps(
        {
            "output_manifest": record.get("output_manifest"),
            "details": record.get("details"),
        },
        ensure_ascii=False,
        default=str,
    )
    if run_id not in reference and str(run_dir) not in reference:
        return
    attempts = int(record.get("attempts", 0))
    record.clear()
    record.update(
        {
            "status": StepStatus.PENDING.value,
            "attempts": attempts,
            "invalidated_at": utc_now(),
            "invalidation_reason": f"run {run_id} was permanently deleted",
        }
    )


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total
