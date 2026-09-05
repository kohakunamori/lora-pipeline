from __future__ import annotations

from pathlib import Path

from .dataset_workspace import DatasetWorkspace
from .state import ProjectState
from .training_config import TrainingConfig, create_project_from_training_config


def create_training_project(
    workspace: DatasetWorkspace,
    config: TrainingConfig,
    *,
    project_name: str,
    root: Path | None = None,
) -> ProjectState:
    """Public name for the canonical atomic training Project factory."""

    return create_project_from_training_config(
        workspace,
        config,
        project_name=project_name,
        root=root,
    )
