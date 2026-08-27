from __future__ import annotations

import json
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

from ..bases import resolve_base_sha256
from ..budget import resolve_budget
from ..config import (
    load_base_registry,
    read_yaml,
    repository_root,
    resolve_profiles,
    sha256_file,
    stable_hash,
    write_json_atomic,
)
from ..dataset.image_info import inspect_dataset
from ..models import PipelineError, StepResult
from ..prepared import load_current_generation
from ..state import ProjectState
from ..tokenizers import count_sdxl_tokens


def run(state: ProjectState, *, minimum_free_gib: float = 10.0) -> StepResult:
    project = state.payload["project"]
    root = repository_root()
    registry = load_base_registry(root)
    base_id = str(project["base"])
    if base_id not in registry:
        raise PipelineError(f"Base model is not registered: {base_id}")
    base = registry[base_id]
    profiles = resolve_profiles(
        str(project.get("hardware", "v100_16gb")),
        str(project["type"]),
        str(project.get("strategy", "quality")),
        project_overrides=project.get("overrides", {}),
        root=root,
    )
    blocking: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    base_exists = base.path.is_file()
    actual_sha: str | None = None
    cache_reused = False
    stat_signature: dict[str, int] | None = None
    if not base_exists:
        blocking.append(f"Base model does not exist: {base.path}")
    else:
        actual_sha, cache_reused, stat_signature = resolve_base_sha256(base_id, root=root)
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
        "sha256_cache_reused": cache_reused,
        "stat_signature": stat_signature,
    }

    inspection_path = state.project_dir / "dataset-manifest.json"
    if not inspection_path.exists():
        blocking.append("Dataset inspection manifest is missing")
        inspection: dict[str, Any] = {"summary": {}, "images": []}
    else:
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    try:
        generation = load_current_generation(state.project_dir)
        prepared = dict(generation.manifest)
    except PipelineError as exc:
        blocking.append(str(exc))
        generation = None
        prepared = {"images": []}
    selected = list(prepared.get("images", []))
    selected_sources = {str(record["source"]) for record in selected}
    corrupt_selected = [
        str(record["path"])
        for record in inspection.get("images", [])
        if record.get("corrupt") and record["path"] in selected_sources
    ]
    if not selected:
        blocking.append("Prepared dataset contains no images")
    if corrupt_selected:
        blocking.append(f"Prepared dataset contains {len(corrupt_selected)} corrupt image(s)")
    checks["dataset"] = {
        "raw_images": inspection.get("summary", {}).get("image_count", 0),
        "prepared_images": len(selected),
        "prepared_generation": generation.generation_id if generation else None,
        "prepared_generation_path": str(generation.root) if generation else None,
        "legacy_prepared_layout": bool(generation and generation.legacy),
        "corrupt_selected": corrupt_selected,
        "source_width": inspection.get("summary", {}).get("width", {}),
        "source_height": inspection.get("summary", {}).get("height", {}),
        "source_megapixels": inspection.get("summary", {}).get("megapixels", {}),
    }

    validation = inspect_dataset(state.project_dir / "validation")
    validation_records = list(validation.get("images", []))
    corrupt_validation = [
        str(record["path"]) for record in validation_records if record.get("corrupt")
    ]
    if corrupt_validation:
        blocking.append(
            f"Validation split contains {len(corrupt_validation)} corrupt image(s)"
        )
    training_hashes = _prepared_image_hashes(generation, selected)
    validation_hashes = {
        str(record["sha256"]): str(record["path"])
        for record in validation_records
        if not record.get("corrupt") and record.get("sha256")
    }
    holdout_overlap = sorted(
        {
            validation_path
            for image_hash, validation_path in validation_hashes.items()
            if image_hash in training_hashes
        }
    )
    if holdout_overlap:
        blocking.append(
            f"Validation split contains {len(holdout_overlap)} exact training duplicate(s)"
        )
    validation_summary = dict(validation.get("summary", {}))
    if validation_summary.get("very_small_images"):
        warnings.append(
            f"Validation split contains {validation_summary['very_small_images']} very small image(s)"
        )
    if validation_summary.get("animated_images"):
        warnings.append(
            f"Validation split contains {validation_summary['animated_images']} animated image(s)"
        )
    checks["validation"] = {
        "root": validation.get("root"),
        "input_hash": validation.get("input_hash"),
        "summary": validation_summary,
        "corrupt_images": corrupt_validation,
        "exact_training_overlap": holdout_overlap,
        "excluded_from_training": True,
    }

    clip_l_counts: list[int] = []
    clip_g_counts: list[int] = []
    missing_captions: list[str] = []
    fallback_errors: set[str] = set()
    max_tokens = int(
        profiles.merged.get("caption", {}).get(
            "max_token_length",
            profiles.hardware.get("caption", {}).get("default_max_token_length", 75),
        )
    )
    trigger_only = 0
    if generation:
        for record in selected:
            caption_path = generation.root / str(record["caption"])
            if not caption_path.is_file():
                missing_captions.append(str(record["source"]))
                continue
            text = caption_path.read_text(encoding="utf-8", errors="replace").strip()
            counts = count_sdxl_tokens(text)
            clip_l_counts.append(counts.clip_l)
            clip_g_counts.append(counts.clip_g)
            if not counts.exact and counts.error:
                fallback_errors.add(counts.error)
            if record.get("caption_source") == "explicit-trigger-only":
                trigger_only += 1
    if missing_captions:
        blocking.append(f"{len(missing_captions)} prepared image(s) have no caption")
    truncated_l = sum(value > max_tokens for value in clip_l_counts)
    truncated_g = sum(value > max_tokens for value in clip_g_counts)
    if truncated_l or truncated_g:
        blocking.append(
            f"Captions exceed the configured {max_tokens}-token budget "
            f"(CLIP-L={truncated_l}, CLIP-G={truncated_g})"
        )
    if fallback_errors:
        warnings.append(
            "Exact SDXL tokenizer assets were unavailable; token counts use a heuristic. "
            "Cache both SDXL tokenizers and rerun preflight for exact truncation checks."
        )
    if trigger_only:
        warnings.append(f"{trigger_only} caption(s) explicitly use trigger-only training")
    checks["captions"] = {
        "present": len(clip_l_counts),
        "missing": missing_captions,
        "tokenizer_exact": not fallback_errors,
        "tokenizer_fallback_errors": sorted(fallback_errors),
        "clip_l": _stats(clip_l_counts),
        "clip_g": _stats(clip_g_counts),
        "budget": max_tokens,
        "truncated_clip_l": truncated_l,
        "truncated_clip_g": truncated_g,
        "trigger_only": trigger_only,
    }

    exclusions_path = state.project_dir / "review" / "exclusions.yaml"
    excluded: list[str] = []
    if exclusions_path.exists():
        excluded = list(read_yaml(exclusions_path).get("excluded", []))
    duplicate_manifest = state.project_dir / "review" / "duplicates" / "manifest.json"
    duplicate_summary: dict[str, Any] = {}
    if duplicate_manifest.exists():
        duplicate_summary = json.loads(
            duplicate_manifest.read_text(encoding="utf-8")
        ).get("summary", {})
    checks["duplicates"] = {"excluded": len(excluded), **duplicate_summary}

    identity_manifest = state.project_dir / "review" / "outliers" / "manifest.json"
    if state.concept_type == "character" and identity_manifest.exists():
        identity = json.loads(identity_manifest.read_text(encoding="utf-8"))
        checks["identity"] = {
            "possible_outliers": len(identity.get("possible_outliers", [])),
            "possible_mixed_characters": len(identity.get("possible_mixed_characters", [])),
        }
        if checks["identity"]["possible_outliers"] or checks["identity"][
            "possible_mixed_characters"
        ]:
            warnings.append("Character identity review contains unresolved flagged images")
    elif state.concept_type == "style":
        checks["identity"] = {"status": "N/A for style concepts"}

    resolution = profiles.merged.get("resolution", {})
    area = int(resolution.get("max_bucket_area", 0))
    hardware_area = int(profiles.hardware.get("resolution", {}).get("max_bucket_area", 0))
    if area > hardware_area:
        blocking.append(
            f"Configured bucket area {area} exceeds hardware envelope {hardware_area}"
        )
    checks["resolution"] = {
        "target": int(resolution.get("default", 1024)),
        "max_bucket_area": area,
        "hardware_max_bucket_area": hardware_area,
        "pass": area <= hardware_area,
    }

    if selected:
        resolved_budget = resolve_budget(project, profiles, image_count=len(selected))
        checks["training_budget"] = resolved_budget.as_dict()
    else:
        checks["training_budget"] = None

    persistent_usage = shutil.disk_usage(state.project_dir)
    persistent_free_gib = round(persistent_usage.free / 1024**3, 3)
    if persistent_free_gib < minimum_free_gib:
        blocking.append(
            f"Only {persistent_free_gib:.2f} GiB is free; at least "
            f"{minimum_free_gib:.2f} GiB is required"
        )
    writable = _is_writable(state.project_dir / "runs")
    if not writable:
        blocking.append("Run output directory is not writable")
    scratch_value = profiles.merged.get("storage", {}).get("scratch_root")
    scratch: dict[str, Any] | None = None
    if scratch_value:
        scratch_path = Path(str(scratch_value)).expanduser()
        scratch_writable = _is_writable(scratch_path)
        scratch_usage = shutil.disk_usage(scratch_path) if scratch_writable else None
        scratch = {
            "path": str(scratch_path),
            "writable": scratch_writable,
            "free_gib": round(scratch_usage.free / 1024**3, 3)
            if scratch_usage
            else None,
        }
        if not scratch_writable:
            blocking.append(f"Configured scratch_root is not writable: {scratch_path}")
    checks["storage"] = {
        "persistent_free_gib": persistent_free_gib,
        "minimum_free_gib": minimum_free_gib,
        "output_writable": writable,
        "scratch": scratch,
    }
    checks["training"] = {
        "hardware": profiles.hardware.get("id"),
        "concept": profiles.concept.get("id"),
        "strategy": profiles.training.get("id"),
        "physical_batch": profiles.merged.get("training", {}).get("batch_size"),
        "gradient_accumulation": profiles.merged.get("training", {}).get(
            "gradient_accumulation_steps"
        ),
    }
    report = {
        "schema_version": 3,
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


def _prepared_image_hashes(
    generation: Any, selected: list[dict[str, Any]]
) -> set[str]:
    hashes: set[str] = set()
    if generation is None:
        return hashes
    for record in selected:
        image_hash = record.get("source_image_sha256")
        if image_hash:
            hashes.add(str(image_hash))
            continue
        path = generation.root / str(record["image"])
        if path.is_file():
            hashes.add(sha256_file(path))
    return hashes


def _stats(values: list[int]) -> dict[str, float | int | None]:
    return {
        "median": statistics.median(values) if values else None,
        "p95": _percentile(values, 0.95),
        "max": max(values, default=None),
    }


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
