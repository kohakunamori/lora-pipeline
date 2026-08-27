from __future__ import annotations

from ..config import write_json_atomic
from ..dataset.image_info import inspect_dataset
from ..models import StepResult
from ..state import ProjectState


def run(state: ProjectState) -> StepResult:
    manifest = inspect_dataset(state.project_dir / "raw")
    path = state.project_dir / "dataset-manifest.json"
    write_json_atomic(path, manifest)
    return StepResult(
        input_hash=manifest["input_hash"],
        output_manifest=str(path),
        details=manifest["summary"],
    )
