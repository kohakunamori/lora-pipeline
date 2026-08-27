from __future__ import annotations

from pathlib import Path

from ..config import stable_hash, write_json_atomic
from ..dataset.character import analyze_identity
from ..dataset.image_info import discover_images
from ..models import StepResult, StepStatus
from ..state import ProjectState


def run(state: ProjectState, *, min_samples: int = 2) -> StepResult:
    if state.concept_type == "style":
        return StepResult(status=StepStatus.SKIPPED, details={"reason": "N/A for style concepts"})
    raw = state.project_dir / "raw"
    images = discover_images(raw)
    result = analyze_identity(images, min_samples=min_samples)
    for key in ("main_cluster", "possible_outliers", "possible_mixed_characters"):
        result[key] = [Path(path).relative_to(raw).as_posix() for path in result[key]]
    manifest_path = state.project_dir / "review" / "outliers" / "manifest.json"
    write_json_atomic(manifest_path, result)
    details = {
        "main_cluster": len(result["main_cluster"]),
        "possible_outliers": len(result["possible_outliers"]),
        "possible_mixed_characters": len(result["possible_mixed_characters"]),
    }
    input_hash = stable_hash(
        {"dataset": state.step("inspect").get("input_hash"), "images": result["labels"], "min_samples": min_samples}
    )
    return StepResult(input_hash=input_hash, output_manifest=str(manifest_path), details=details)
