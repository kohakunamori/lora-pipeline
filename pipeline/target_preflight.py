from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from .config import resolve_profiles, stable_hash, write_json_atomic
from .models import PipelineError, StepResult
from .prepared import load_current_generation
from .target_training_advisor import target_training_advice


_DEFAULT_STYLE_LIMITS = {
    "minimum_images": 12,
    "extreme_fraction": 0.95,
}


def install_target_preflight_hook(preflight_module: ModuleType) -> None:
    """Add target-aware advisory/guardrails without replacing core preflight logic."""

    original = preflight_module.run
    if getattr(original, "_target_preflight_wrapped", False):
        return

    def run(state, *args, **kwargs):
        original_error: PipelineError | None = None
        result: StepResult | None = None
        try:
            result = original(state, *args, **kwargs)
        except PipelineError as exc:
            original_error = exc

        report_path = state.project_dir / "preflight.json"
        if not report_path.is_file():
            if original_error is not None:
                raise original_error
            assert result is not None
            return result

        report = json.loads(report_path.read_text(encoding="utf-8"))
        _augment_preflight_report(state, report)
        report["status"] = "READY" if not report.get("blocking") else "BLOCKED"
        report["input_hash"] = stable_hash(report.get("checks", {}))
        write_json_atomic(report_path, report)

        if report.get("blocking"):
            raise PipelineError("Preflight BLOCKED: " + "; ".join(report["blocking"])) from original_error

        assert result is not None
        details = dict(result.details)
        details.update(
            {
                "status": "READY",
                "warnings": list(report.get("warnings", [])),
                "checks": dict(report.get("checks", {})),
            }
        )
        return StepResult(
            status=result.status,
            input_hash=report["input_hash"],
            output_manifest=str(report_path),
            details=details,
        )

    run._target_preflight_wrapped = True
    run._target_preflight_original = original
    preflight_module.run = run


def _augment_preflight_report(state, report: dict[str, Any]) -> None:
    project = state.payload["project"]
    target_type = str(project.get("training_target_type", project.get("type", "character")))
    profiles = resolve_profiles(
        str(project.get("hardware", "v100_16gb")),
        str(project["type"]),
        str(project.get("strategy", "quality")),
        project_overrides=project.get("overrides", {}),
    )

    image_count = _prepared_image_count(state)
    style_distribution = _style_distribution(state) if target_type == "style" else None
    current_images_seen = _current_images_seen(project)
    advice = target_training_advice(
        target_type,
        image_count=image_count,
        current_training=profiles.merged.get("training", {}),
        current_images_seen=current_images_seen,
        style_distribution=style_distribution,
    )

    checks = report.setdefault("checks", {})
    checks["target_training_advisor"] = advice
    warnings = report.setdefault("warnings", [])
    for warning in advice.get("warnings", []):
        _append_unique(warnings, "Target training advisory: " + str(warning))

    if target_type != "style":
        return

    style_assessment = assess_style_distribution(
        style_distribution or {},
        limits=profiles.merged.get("limits", {}).get("style_bias", {}),
    )
    checks["style_bias_guardrail"] = style_assessment
    for warning in style_assessment["warnings"]:
        _append_unique(warnings, warning)
    blocking = report.setdefault("blocking", [])
    for reason in style_assessment["blocking"]:
        _append_unique(blocking, reason)


def assess_style_distribution(
    style_distribution: Mapping[str, Any],
    *,
    limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess style-only subject/composition entanglement risk.

    A single concentrated axis is a warning.  Blocking requires both an extremely
    concentrated subject and an extremely concentrated portrait/background axis,
    which is much stronger evidence that the requested target is not separable style.
    """

    configured = {**_DEFAULT_STYLE_LIMITS, **dict(limits or {})}
    minimum_images = max(1, int(configured["minimum_images"]))
    extreme = float(configured["extreme_fraction"])
    count = int(style_distribution.get("image_count", 0) or 0)
    dominant = float(style_distribution.get("dominant_subject", {}).get("fraction", 0.0) or 0.0)
    buckets = style_distribution.get("distribution", {})
    portrait = float(buckets.get("portrait", {}).get("fraction", 0.0) or 0.0)
    simple = float(buckets.get("simple_background", {}).get("fraction", 0.0) or 0.0)

    warnings: list[str] = []
    for item in style_distribution.get("warnings", []) or []:
        code = str(item.get("code", "style_bias"))
        value = item.get("value")
        message = str(item.get("message", code))
        suffix = f" ({value})" if value is not None else ""
        _append_unique(warnings, f"Style dataset warning [{code}]: {message}{suffix}")

    blocking: list[str] = []
    joint_portrait = count >= minimum_images and dominant >= extreme and portrait >= extreme
    joint_background = count >= minimum_images and dominant >= extreme and simple >= extreme
    if joint_portrait or joint_background:
        concentrated_axes = []
        if joint_portrait:
            concentrated_axes.append(f"portrait={portrait:.3f}")
        if joint_background:
            concentrated_axes.append(f"simple_background={simple:.3f}")
        blocking.append(
            "Style dataset is pathologically entangled: "
            f"dominant_subject={dominant:.3f} and "
            + ", ".join(concentrated_axes)
            + f" across {count} images. Diversify subject/composition before training a style target."
        )

    if count < minimum_images:
        _append_unique(
            warnings,
            f"Style bias guardrail has only {count} image(s); automatic blocking requires at least {minimum_images} samples.",
        )

    risk = "blocked" if blocking else ("warning" if warnings else "low")
    return {
        "status": risk,
        "image_count": count,
        "minimum_images_for_blocking": minimum_images,
        "extreme_fraction": extreme,
        "dominant_subject_fraction": dominant,
        "portrait_fraction": portrait,
        "simple_background_fraction": simple,
        "joint_extreme_subject_portrait": joint_portrait,
        "joint_extreme_subject_simple_background": joint_background,
        "warnings": warnings,
        "blocking": blocking,
        "ground_truth": False,
        "note": (
            "This guardrail detects severe style/content entanglement risk. It does not score artistic quality."
        ),
    }


def _prepared_image_count(state) -> int:
    try:
        generation = load_current_generation(state.project_dir)
    except PipelineError:
        snapshot = state.payload.get("project", {}).get("dataset_snapshot", {})
        return max(1, int(snapshot.get("image_count", 1) or 1))
    return max(1, len(generation.manifest.get("images", [])))


def _style_distribution(state) -> dict[str, Any] | None:
    path = state.project_dir / "review" / "captions" / "manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("style_distribution")
    return dict(value) if isinstance(value, Mapping) else None


def _current_images_seen(project: Mapping[str, Any]) -> int | None:
    budget = project.get("budget", {})
    if isinstance(budget, Mapping) and budget.get("unit") == "images_seen" and budget.get("value") is not None:
        return int(budget["value"])
    snapshot = project.get("training_config_snapshot", {})
    if isinstance(snapshot, Mapping) and snapshot.get("images_seen") is not None:
        return int(snapshot["images_seen"])
    return None


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
