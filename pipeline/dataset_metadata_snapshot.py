from __future__ import annotations

from typing import Any

from .config import stable_hash
from .dataset_metadata import composition_summary, item_metadata
from .dataset_workspace import DatasetWorkspace
from .state import ProjectState, utc_now


def attach_dataset_metadata_snapshot(
    state: ProjectState,
    workspace: DatasetWorkspace,
) -> ProjectState:
    records: list[dict[str, Any]] = []
    for item in workspace.items(include_disabled=False, include_excluded=False):
        metadata = item_metadata(workspace, item)
        records.append(
            {
                "key": item.key,
                "source_id": item.source_id,
                "relative": item.relative.as_posix(),
                "source_group_id": metadata.get("source_group_id"),
                "variant_kind": metadata.get("variant_kind"),
                "composition_type": metadata.get("composition_type"),
                "resolution": metadata.get("resolution"),
                "analysis": metadata.get("analysis"),
                "quality": metadata.get("quality"),
                "identity": metadata.get("identity"),
            }
        )
    basis = {
        "schema_version": 1,
        "dataset": workspace.name,
        "composition": composition_summary(workspace),
        "images": records,
    }
    snapshot = {
        **basis,
        "snapshot_hash": stable_hash(basis),
        "created_at": utc_now(),
    }
    project = state.payload["project"]
    project["dataset_metadata_snapshot"] = snapshot
    identity = dict(project.get("training_identity") or {})
    identity["dataset_metadata_snapshot_hash"] = snapshot["snapshot_hash"]
    project["training_identity"] = identity
    state.save()
    return state
