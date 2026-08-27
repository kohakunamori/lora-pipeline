from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

from .dataset_workspace import DatasetWorkspace
from .models import PipelineError, StateError


def delete_dataset_items(workspace: DatasetWorkspace, keys: Sequence[str]) -> dict[str, Any]:
    """Permanently delete selected dataset copies and their caption sidecars.

    This only touches files owned by the mutable Dataset workspace. Original import
    directories/videos and immutable Project snapshots are outside this tree and are
    never modified here.
    """

    unique_keys = list(dict.fromkeys(str(key) for key in keys))
    if not unique_keys:
        raise PipelineError("No dataset items were selected for deletion")

    by_key = {
        item.key: item
        for item in workspace.items(include_disabled=True, include_excluded=True)
    }
    unknown = [key for key in unique_keys if key not in by_key]
    if unknown:
        raise PipelineError(
            "Cannot delete unknown dataset item(s): " + ", ".join(unknown[:5])
        )

    deleted_captions = 0
    touched_sources: set[str] = set()
    for key in unique_keys:
        item = by_key[key]
        touched_sources.add(item.source_id)
        if item.caption.is_file():
            item.caption.unlink()
            deleted_captions += 1
        item.image.unlink()
        _prune_empty_parents(item.image.parent, workspace.source_images_dir(item.source_id))

    exclusions = workspace._load_exclusions()
    for key in unique_keys:
        exclusions.pop(key, None)
    workspace._save_exclusions(exclusions)
    workspace.save()

    return {
        "dataset": workspace.name,
        "deleted_images": len(unique_keys),
        "deleted_captions": deleted_captions,
        "sources": sorted(touched_sources),
    }


def delete_dataset_source(workspace: DatasetWorkspace, source_id: str) -> dict[str, Any]:
    """Permanently remove one imported/derived source from a Dataset workspace."""

    if source_id not in workspace.sources:
        raise StateError(f"Unknown dataset source: {source_id}")

    source = dict(workspace.sources[source_id])
    items = workspace.items(
        source_id=source_id,
        include_disabled=True,
        include_excluded=True,
    )
    source_dir = workspace.source_dir(source_id)

    # Remove metadata only after we know the source directory can be addressed.
    if source_dir.exists():
        shutil.rmtree(source_dir)
    workspace.sources.pop(source_id, None)

    exclusions = workspace._load_exclusions()
    prefix = f"{source_id}/"
    exclusions = {
        key: value for key, value in exclusions.items() if not key.startswith(prefix)
    }
    workspace._save_exclusions(exclusions)

    # Per-source audits are disposable derived metadata.
    (workspace.dataset_dir / "review" / f"audit-{source_id}.json").unlink(missing_ok=True)
    workspace.save()

    return {
        "dataset": workspace.name,
        "source_id": source_id,
        "label": str(source.get("label") or source_id),
        "deleted_images": len(items),
    }


def delete_dataset_workspace(workspace: DatasetWorkspace) -> dict[str, Any]:
    """Permanently remove one Dataset workspace, without touching Projects/Runs."""

    dataset_dir = workspace.dataset_dir
    if dataset_dir.name != workspace.name or not (dataset_dir / "dataset.yaml").is_file():
        raise StateError(f"Refusing to delete invalid dataset workspace: {dataset_dir}")

    summary = workspace.summary()
    result = {
        "dataset": workspace.name,
        "sources": int(summary["sources"]),
        "images": int(summary["images"]),
        "path": str(dataset_dir),
    }
    shutil.rmtree(dataset_dir)
    return result


def _prune_empty_parents(path: Path, stop: Path) -> None:
    """Remove now-empty nested image directories, never the source images root."""

    stop = stop.resolve()
    current = path.resolve()
    while current != stop:
        try:
            current.relative_to(stop)
        except ValueError:
            return
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent
