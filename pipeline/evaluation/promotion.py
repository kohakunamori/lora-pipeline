from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import load_base_registry, sha256_file, write_yaml_atomic
from ..models import PipelineError
from ..state import ProjectState


def run(
    state: ProjectState,
    *,
    run_id: str,
    checkpoint_name: str,
    strength: float,
    allow_unreviewed: bool = False,
) -> dict[str, Any]:
    if strength <= 0:
        raise PipelineError("Recommended LoRA strength must be greater than zero")
    run_record = _find_run(state, run_id)
    checkpoints = [Path(value) for value in run_record.get("checkpoints", [])]
    matches = [
        path
        for path in checkpoints
        if path.is_file() and checkpoint_name in {path.name, path.stem}
    ]
    if not matches:
        raise PipelineError(f"Checkpoint is not part of run {run_id}: {checkpoint_name}")
    if len(matches) > 1:
        raise PipelineError(f"Checkpoint selection is ambiguous: {checkpoint_name}")
    checkpoint = matches[0]
    evidence = dict(run_record.get("evaluation", {}))
    evaluated_names = {
        value
        for record in evidence.values()
        if isinstance(record, dict)
        for value in record.get("checkpoints", [])
    }
    checkpoint_was_evaluated = checkpoint.name in evaluated_names or checkpoint.stem in evaluated_names
    if (not evidence or not checkpoint_was_evaluated) and not allow_unreviewed:
        if not evidence:
            detail = "Run has not been evaluated"
        else:
            detail = f"Checkpoint {checkpoint.name} has not been included in an evaluation stage"
        raise PipelineError(
            f"{detail}. Review screening/full contact sheets first, "
            "or explicitly use --allow-unreviewed."
        )

    run_dir = Path(run_record["path"])
    destination = run_dir / "best.safetensors"
    destination.unlink(missing_ok=True)
    try:
        os.link(checkpoint, destination)
    except OSError:
        shutil.copy2(checkpoint, destination)

    project = state.payload["project"]
    base = load_base_registry()[str(project["base"])]
    accounting = dict(run_record.get("accounting", {}))
    payload = {
        "schema_version": 2,
        "project": state.name,
        "type": state.concept_type,
        "trigger": project["trigger"],
        "base": {
            "id": base.id,
            "filename": base.path.name,
            "sha256": base.sha256,
        },
        "selection": {
            "method": "manual",
            "selected_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "recommended_strength": strength,
            "allow_unreviewed": allow_unreviewed,
            "checkpoint_was_evaluated": checkpoint_was_evaluated,
        },
        "training": {
            "physical_batch": accounting.get("physical_batch"),
            "effective_batch": accounting.get("effective_batch"),
            "optimizer_steps": accounting.get("optimizer_steps"),
            "target_images_seen": accounting.get("target_images_seen"),
            "images_seen": accounting.get("images_seen"),
            "epochs": accounting.get("epochs"),
        },
        "evidence": evidence,
        "artifacts": {
            "source_checkpoint": str(checkpoint),
            "promoted_lora": str(destination),
        },
    }
    metadata = run_dir / "best.yaml"
    write_yaml_atomic(metadata, payload)
    run_record.update(
        {
            "status": "promoted",
            "promotion": {
                "checkpoint": checkpoint.name,
                "strength": strength,
                "best": str(destination),
                "metadata": str(metadata),
                "selected_at": payload["selection"]["selected_at"],
            },
        }
    )
    state.save()
    return payload


def _find_run(state: ProjectState, run_id: str) -> dict[str, Any]:
    for record in state.payload.get("runs", []):
        if record.get("id") == run_id:
            return record
    raise PipelineError(f"Unknown run id for project {state.name}: {run_id}")
