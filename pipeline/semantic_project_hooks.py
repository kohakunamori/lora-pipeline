from __future__ import annotations

from . import resource_deletion as _resource_deletion
from .dataset_metadata_snapshot import attach_dataset_metadata_snapshot
from .dataset_semantics import attach_dataset_semantics_snapshot
from .semantic_runtime import install_semantic_runtime_hooks


_ORIGINAL_CREATE = _resource_deletion._create_project_from_training_config


def _create_with_dataset_snapshots(workspace, config, *, project_name, root=None):
    state = _ORIGINAL_CREATE(workspace, config, project_name=project_name, root=root)
    state = attach_dataset_metadata_snapshot(state, workspace)
    return attach_dataset_semantics_snapshot(state, workspace)


def install_semantic_project_hooks() -> None:
    current = _resource_deletion._create_project_from_training_config
    if not getattr(current, "_dataset_semantics_wrapped", False):
        _create_with_dataset_snapshots._dataset_semantics_wrapped = True
        _resource_deletion._create_project_from_training_config = _create_with_dataset_snapshots
    install_semantic_runtime_hooks()


install_semantic_project_hooks()
