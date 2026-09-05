from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .config import resolve_profiles, stable_hash, write_json_atomic
from .dataset.caption_cleaner import normalize_tag, parse_caption
from .dataset.tag_categories import is_identity_tag, is_outfit_tag
from .models import PipelineError, StepResult, StepStatus
from .tokenizers import count_sdxl_tokens

CHARACTER_CAPTION_POLICIES = ("strong_identity", "balanced", "controllable")
STYLE_CAPTION_POLICY = "content_rich"

_POLICY_THRESHOLDS: dict[str, tuple[float, float]] = {
    # Strong identity deliberately lets protected triggers absorb stable visual
    # identity. Outfit tags remain explicit for generic character training, while
    # character_outfit uses the outfit trigger to absorb stable garment features.
    "strong_identity": (0.80, 0.80),
    # Balanced only removes extremely stable identity/garment features. This is the
    # default because it leaves more prompt control while still strengthening the
    # protected trigger namespace.
    "balanced": (0.95, 0.90),
    "controllable": (1.01, 1.01),
}
_MIN_INFERENCE_SAMPLES = 3


def default_caption_policy(target_type: str) -> str:
    if target_type == "style":
        return STYLE_CAPTION_POLICY
    if target_type in {"character", "character_outfit"}:
        return "balanced"
    raise PipelineError(f"Unsupported caption target: {target_type}")


def resolve_caption_policy(state) -> str:
    project = state.payload.get("project", {})
    target_type = str(project.get("training_target_type", project.get("type", "")))
    profiles = resolve_profiles(
        str(project.get("hardware", "v100_16gb")),
        str(project.get("type", "character")),
        str(project.get("strategy", "quality")),
        project_overrides=project.get("overrides", {}),
    )
    requested = str(
        profiles.merged.get("caption", {}).get("policy")
        or default_caption_policy(target_type)
    )
    validate_caption_policy(target_type, requested)
    return requested


def validate_caption_policy(target_type: str, policy: str) -> None:
    if target_type == "style":
        if policy != STYLE_CAPTION_POLICY:
            raise PipelineError(
                f"Style caption policy must be {STYLE_CAPTION_POLICY!r}, got {policy!r}"
            )
        return
    if target_type in {"character", "character_outfit"}:
        if policy not in CHARACTER_CAPTION_POLICIES:
            raise PipelineError(
                "Character caption policy must be one of: "
                + ", ".join(CHARACTER_CAPTION_POLICIES)
            )
        return
    raise PipelineError(f"Unsupported caption target: {target_type}")


def apply_target_caption_policy(state, result: StepResult) -> StepResult:
    """Apply one small, target-aware caption policy after raw tag cleaning.

    The implementation intentionally does not depend on Dataset identity metadata.
    It only uses tag frequency in the actual training captions. This matches the
    input contract that the user has already selected identity-correct images.
    """

    project = state.payload.get("project", {})
    target_type = str(project.get("training_target_type", project.get("type", "")))
    policy = resolve_caption_policy(state)
    if target_type == "style":
        return _annotate_style_policy(result, policy)
    if result.status not in {StepStatus.DONE, StepStatus.SKIPPED} or not result.output_manifest:
        return result

    manifest_path = Path(result.output_manifest)
    if not manifest_path.is_file():
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Passthrough is an explicit request to preserve user caption bytes.
    if manifest.get("mode") == "existing_passthrough":
        return result
    records = manifest.get("records", [])
    if not isinstance(records, list):
        return result

    fixed = [
        str(project.get("trigger") or "").strip(),
        *[
            str(value).strip()
            for value in project.get("caption_anchor_tags", [])
            if str(value).strip()
        ],
    ]
    protected = {normalize_tag(tag) for tag in fixed if tag}
    identity_min, outfit_min = _POLICY_THRESHOLDS[policy]
    identity_tags = _invariant_tags(
        records,
        minimum_coverage=identity_min,
        predicate=is_identity_tag,
    )
    # Generic character training keeps clothing explicit so a single/default outfit
    # does not silently bind itself to the character trigger. Outfit targets do the
    # opposite: stable garment features belong to the protected outfit trigger.
    outfit_tags = (
        _invariant_tags(
            records,
            minimum_coverage=outfit_min,
            predicate=is_outfit_tag,
        )
        if target_type == "character_outfit"
        else set()
    )
    suppressed = (identity_tags | outfit_tags) - protected

    changed = 0
    total_suppressed = 0
    for record in records:
        tags = parse_caption(str(record.get("text") or ""))
        retained: list[str] = []
        removed: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            normalized = normalize_tag(tag)
            if not normalized or normalized in seen:
                continue
            if normalized in suppressed:
                removed.append(tag)
                continue
            seen.add(normalized)
            retained.append(tag)
        text = ", ".join(retained)
        if text != str(record.get("text") or ""):
            changed += 1
        total_suppressed += len(removed)
        if record.get("caption"):
            Path(str(record["caption"])).write_text(text + "\n", encoding="utf-8")
        counts = count_sdxl_tokens(text)
        record["text"] = text
        record["token_counts"] = {
            "clip_l": counts.clip_l,
            "clip_g": counts.clip_g,
            "exact": counts.exact,
            "backend": counts.backend,
            "error": counts.error,
        }
        record["caption_policy"] = {
            "policy": policy,
            "suppressed": removed,
        }

    manifest.pop("input_hash", None)
    manifest["target_caption_policy"] = {
        "target_type": target_type,
        "policy": policy,
        "minimum_samples": _MIN_INFERENCE_SAMPLES,
        "identity_min_coverage": identity_min,
        "outfit_min_coverage": outfit_min if target_type == "character_outfit" else None,
        "suppressed_identity_tags": sorted(identity_tags),
        "suppressed_outfit_tags": sorted(outfit_tags),
    }
    manifest.setdefault("summary", {})["target_caption_policy_updates"] = changed
    manifest["summary"]["target_caption_policy_suppressions"] = total_suppressed
    manifest["input_hash"] = stable_hash(manifest)
    write_json_atomic(manifest_path, manifest)

    details = dict(result.details)
    details.update(
        {
            "caption_policy": policy,
            "target_caption_policy_updates": changed,
            "target_caption_policy_suppressions": total_suppressed,
            "suppressed_identity_tags": sorted(identity_tags),
            "suppressed_outfit_tags": sorted(outfit_tags),
        }
    )
    return StepResult(
        status=result.status,
        input_hash=manifest["input_hash"],
        output_manifest=str(manifest_path),
        details=details,
    )


def _annotate_style_policy(result: StepResult, policy: str) -> StepResult:
    details = dict(result.details)
    details["caption_policy"] = policy
    return StepResult(
        status=result.status,
        input_hash=result.input_hash,
        output_manifest=result.output_manifest,
        details=details,
    )


def _invariant_tags(
    records: list[Mapping[str, Any]],
    *,
    minimum_coverage: float,
    predicate,
) -> set[str]:
    if len(records) < _MIN_INFERENCE_SAMPLES or minimum_coverage > 1.0:
        return set()
    counts: Counter[str] = Counter()
    for record in records:
        tags = {
            normalize_tag(tag)
            for tag in parse_caption(str(record.get("text") or ""))
            if normalize_tag(tag)
        }
        counts.update(tags)
    total = len(records)
    return {
        tag
        for tag, count in counts.items()
        if count / total >= minimum_coverage and predicate(tag)
    }
