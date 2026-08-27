from __future__ import annotations

from ..config import read_yaml, write_json_atomic, write_yaml_atomic
from ..dataset.duplicates import find_duplicates, suggested_exact_exclusions
from ..models import StepResult
from ..state import ProjectState


def run(state: ProjectState, *, phash_distance: int = 6, exclude_exact: bool = False) -> StepResult:
    manifest = find_duplicates(state.project_dir / "raw", phash_distance=phash_distance)
    manifest_path = state.project_dir / "review" / "duplicates" / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    exclusions_path = state.project_dir / "review" / "exclusions.yaml"
    exclusions = read_yaml(exclusions_path) if exclusions_path.exists() else {"excluded": [], "reasons": {}}
    if exclude_exact:
        for path in suggested_exact_exclusions(manifest):
            if path not in exclusions["excluded"]:
                exclusions["excluded"].append(path)
                exclusions["reasons"][path] = "exact duplicate"
    exclusions["excluded"] = sorted(set(exclusions.get("excluded", [])))
    write_yaml_atomic(exclusions_path, exclusions)
    details = dict(manifest["summary"])
    details["exact_auto_excluded"] = len(suggested_exact_exclusions(manifest)) if exclude_exact else 0
    return StepResult(input_hash=manifest["input_hash"], output_manifest=str(manifest_path), details=details)
