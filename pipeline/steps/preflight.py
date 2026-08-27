from __future__ import annotations

import json
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

from ..config import load_base_registry, resolve_profiles, sha256_file, stable_hash, write_json_atomic
from ..dataset.caption_cleaner import estimate_tokens, parse_caption
from ..models import PipelineError, StepResult
from ..state import ProjectState


def run(state: ProjectState, *, minimum_free_gib: float = 10.0) -> StepResult:
    project = state.payload["project"]
    registry = load_base_registry()
    base_id = str(project["base"])
    if base_id not in registry:
        raise PipelineError(f"Base model is not registered: {base_id}")
    base = registry[base_id]
    profiles = resolve_profiles(
        str(project.get("hardware", "v100_16gb")),
        str(project["type"]),
        str(project.get("strategy", "quality")),
        project_overrides=project.get("overrides", {}),
    )
    blocking: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    base_exists = base.path.is_file()
    actual_sha: str | None = None
    if not base_exists:
        blocking.append(f"Base model does not exist: {base.path}")
    else:
        actual_sha = sha256_file(base.path)
        if base.sha256 and actual_sha != base.sha256:
            blocking.append(f"Base SHA256 mismatch: expected {base.sha256}, got {actual_sha}")
    checks["base"] = {
        "id": base.id,
        "name": base.name,
        "path": str(base.path),
        "exists": base_exists,
        "expected_sha256": base.sha256,
        "actual_sha256": actual_sha,
        "sha256_ok": bool(base_exists and (not base.sha256 or actual_sha == base.sha256)),
    }

    inspection_path = state.project_dir / "dataset-manifest.json"
    prepared_path = state.project_dir / "prepared" / "manifest.json"
    if not inspection_path.exists():
        blocking.append("Dataset inspection manifest is missing")
        inspection: dict[str, Any] = {"summary": {}, "images": []}
    else:
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    if not prepared_path.exists():
        blocking.append("Prepared dataset manifest is missing")
        prepared: dict[str, Any] = {"images": []}
    else:
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    selected = prepared.get("images", [])
    selected_sources = {record["source"] for record in selected}
    corrupt_selected = [
        record["path"] for record in inspection.get("images", []) if record.get("corrupt") and record["path"] in selected_sources
    ]
    if not selected:
        blocking.append("Prepared dataset contains no images")
    if corrupt_selected:
        blocking.append(f"Prepared dataset contains {len(corrupt_selected)} corrupt image(s)")
    checks["dataset"] = {
        "raw_images": inspection.get("summary", {}).get("image_count", 0),
        "prepared_images": len(selected),
        "corrupt_selected": corrupt_selected,
        "source_width": inspection.get("summary", {}).get("width", {}),
        "source_height": inspection.get("summary", {}).get("height", {}),
        "source_megapixels": inspection.get("summary", {}).get("megapixels", {}),
    }

    token_counts: list[int] = []
    missing_captions: list[str] = []
    max_tokens = int(
        profiles.merged.get("caption", {}).get(
            "max_token_length", profiles.hardware.get("caption", {}).get("default_max_token_length", 75)
        )
    )
    for record in selected:
        caption_path = state.project_dir / "prepared" / record["caption"]
        if not caption_path.is_file():
            missing_captions.append(record["source"])
            continue
        tags = parse_caption(caption_path.read_text(encoding="utf-8", errors="replace"))
        token_counts.append(estimate_tokens(tags))
    if missing_captions:
        blocking.append(f"{len(missing_captions)} prepared image(s) have no caption")
    truncated = sum(value > max_tokens for value in token_counts)
    if truncated:
        blocking.append(f"{truncated} caption(s) exceed the configured {max_tokens}-token budget")
    checks["captions"] = {
        "present": len(token_counts),
        "missing": missing_captions,
        "median_tokens": statistics.median(token_counts) if token_counts else None,
        "p95_tokens": _percentile(token_counts, 0.95),
        "max_tokens": max(token_counts, default=None),
        "budget": max_tokens,
        "truncated_captions": truncated,
    }

    exclusions_path = state.project_dir / "review" / "exclusions.yaml"
    excluded: list[str] = []
    if exclusions_path.exists():
        from ..config import read_yaml

        excluded = list(read_yaml(exclusions_path).get("excluded", []))
    duplicate_manifest = state.project_dir / "review" / "duplicates" / "manifest.json"
    duplicate_summary: dict[str, Any] = {}
    if duplicate_manifest.exists():
        duplicate_summary = json.loads(duplicate_manifest.read_text(encoding="utf-8")).get("summary", {})
    checks["duplicates"] = {"excluded": len(excluded), **duplicate_summary}

    identity_manifest = state.project_dir / "review" / "outliers" / "manifest.json"
    if state.concept_type == "character" and identity_manifest.exists():
        identity = json.loads(identity_manifest.read_text(encoding="utf-8"))
        checks["identity"] = {
            "possible_outliers": len(identity.get("possible_outliers", [])),
            "possible_mixed_characters": len(identity.get("possible_mixed_characters", [])),
        }
        if checks["identity"]["possible_outliers"] or checks["identity"]["possible_mixed_characters"]:
            warnings.append("Character identity review contains unresolved flagged images")
    elif state.concept_type == "style":
        checks["identity"] = {"status": "N/A for style concepts"}

    resolution = profiles.merged.get("resolution", {})
    area = int(resolution.get("max_bucket_area", 0))
    hardware_area = int(profiles.hardware.get("resolution", {}).get("max_bucket_area", 0))
    checks["resolution"] = {
        "target": int(resolution.get("default", 1024)),
        "max_bucket_area": area,
        "hardware_max_bucket_area": hardware_area,
        "pass": area <= hardware_area,
    }

    usage = shutil.disk_usage(state.project_dir)
    free_gib = round(usage.free / 1024**3, 3)
    if free_gib < minimum_free_gib:
        blocking.append(f"Only {free_gib:.2f} GiB is free; at least {minimum_free_gib:.2f} GiB is required")
    writable = _is_writable(state.project_dir / "runs")
    if not writable:
        blocking.append("Run output directory is not writable")
    checks["storage"] = {"free_gib": free_gib, "minimum_free_gib": minimum_free_gib, "output_writable": writable}
    checks["training"] = {
        "hardware": profiles.hardware.get("id"),
        "concept": profiles.concept.get("id"),
        "strategy": profiles.training.get("id"),
        "physical_batch": profiles.merged.get("training", {}).get("batch_size"),
        "gradient_accumulation": profiles.merged.get("training", {}).get("gradient_accumulation_steps"),
    }
    report = {
        "schema_version": 1,
        "status": "READY" if not blocking else "BLOCKED",
        "checks": checks,
        "blocking": blocking,
        "warnings": warnings,
    }
    report["input_hash"] = stable_hash(report["checks"])
    path = state.project_dir / "preflight.json"
    write_json_atomic(path, report)
    if blocking:
        raise PipelineError("Preflight BLOCKED: " + "; ".join(blocking))
    return StepResult(
        input_hash=report["input_hash"],
        output_manifest=str(path),
        details={"status": "READY", "warnings": warnings, "checks": checks},
    )


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def _is_writable(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(prefix=".write-test-", dir=directory, delete=True):
            return True
    except OSError:
        return False
