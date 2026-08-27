from __future__ import annotations

import json
from pathlib import Path

from ..config import read_yaml, stable_hash, write_json_atomic, write_yaml_atomic
from ..models import StepResult
from ..state import ProjectState


def run(state: ProjectState, *, exclude: list[str] | None = None) -> StepResult:
    project_dir = state.project_dir
    exclusions_path = project_dir / "review" / "exclusions.yaml"
    exclusions = read_yaml(exclusions_path) if exclusions_path.exists() else {"excluded": [], "reasons": {}}
    for relative in exclude or []:
        path = project_dir / "raw" / relative
        if not path.is_file():
            raise FileNotFoundError(f"Cannot exclude missing raw image: {relative}")
        if relative not in exclusions["excluded"]:
            exclusions["excluded"].append(relative)
            exclusions["reasons"][relative] = "manual review"
    exclusions["excluded"] = sorted(set(exclusions.get("excluded", [])))
    write_yaml_atomic(exclusions_path, exclusions)
    summary: dict[str, object] = {"excluded": len(exclusions["excluded"])}
    for label, relative in (
        ("duplicates", Path("review/duplicates/manifest.json")),
        ("outliers", Path("review/outliers/manifest.json")),
        ("captions", Path("review/captions/manifest.json")),
    ):
        path = project_dir / relative
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary[label] = payload.get("summary", len(payload.get("records", [])))
    summary_path = project_dir / "review" / "summary.json"
    write_json_atomic(summary_path, summary)
    input_hash = stable_hash(
        {
            "excluded": exclusions["excluded"],
            "dedup": state.step("dedup").get("input_hash"),
            "identity": state.step("identity").get("input_hash"),
            "caption": state.step("caption").get("input_hash"),
        }
    )
    return StepResult(input_hash=input_hash, output_manifest=str(summary_path), details=summary)
