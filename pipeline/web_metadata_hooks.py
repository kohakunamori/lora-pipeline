from __future__ import annotations

from pathlib import Path

from . import web_routes
from .dataset_metadata_snapshot import attach_dataset_metadata_snapshot
from .dataset_workspace import DatasetWorkspace
from .state import ProjectState
from .training_config import TrainingConfig
from .training_config import create_project_from_training_config as _create_project_from_training_config


def _create_with_metadata(
    workspace: DatasetWorkspace,
    config: TrainingConfig,
    *,
    project_name: str,
    root: Path | None = None,
) -> ProjectState:
    state = _create_project_from_training_config(
        workspace,
        config,
        project_name=project_name,
        root=root,
    )
    return attach_dataset_metadata_snapshot(state, workspace)


web_routes.create_project_from_training_config = _create_with_metadata
