from __future__ import annotations

from pathlib import Path

from .config import repository_root
from .dataset_metadata_snapshot import attach_dataset_metadata_snapshot
from .dataset_workspace import DatasetWorkspace
from .lifecycle_guard import lifecycle_lock
from .state import ProjectState
from .target_policy import attach_target_aware_dataset_semantics_snapshot
from .training_config import TrainingConfig, create_project_from_training_config


def create_training_project(
    workspace: DatasetWorkspace,
    config: TrainingConfig,
    *,
    project_name: str,
    root: Path | None = None,
) -> ProjectState:
    """Freeze one complete training workspace through a single explicit path."""

    resolved = (root or repository_root()).resolve()
    with lifecycle_lock(resolved):
        # Re-read mutable inputs while holding the lifecycle lock so deletion or
        # concurrent edits cannot race snapshot creation.
        fresh_workspace = DatasetWorkspace.load(workspace.name, root=resolved)
        fresh_config = TrainingConfig.load(config.name, root=resolved)
        state = create_project_from_training_config(
            fresh_workspace,
            fresh_config,
            project_name=project_name,
            root=resolved,
        )
        state = attach_dataset_metadata_snapshot(state, fresh_workspace)
        state = attach_target_aware_dataset_semantics_snapshot(state, fresh_workspace)
        return state
