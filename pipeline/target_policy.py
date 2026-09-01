from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .config import resolve_profiles, stable_hash, write_json_atomic
from .dataset.caption_cleaner import CATEGORY_PATTERNS, normalize_tag, parse_caption
from .dataset_semantics import attach_dataset_semantics_snapshot
from .models import PipelineError, StepResult, StepStatus
from .tokenizers import count_sdxl_tokens


_OUTFIT_INVARIANT_MIN_COVERAGE = 0.80
_RUNTIME_HOOK_INSTALLED = False


def target_caption_policy(target_type: str) -> dict[str, str]:
    """Return the semantic caption ownership rules for one training target.

    Character training keeps the existing dataset-owned character/outfit token model.
    Character-outfit training instead makes the TrainingConfig trigger the sole outfit
    concept token and suppresses stable garment descriptors so they are learned by it.
    """

    if target_type == "character_outfit":
        return {
            "training_trigger": "always",
            "character_anchor": "always",
            "character_token": "do_not_inject",
            "outfit_token": "do_not_inject",
            "character_features": "suppress",
            "outfit_features": "suppress",
            "invariant_outfit_tags": "suppress",
        }
    return {
        "character_token": "always",
        "outfit_token": "when_present",
        "character_features": "suppress",
        "outfit_features": "preserve",
    }


def attach_target_aware_dataset_semantics_snapshot(state, workspace):
    """Attach frozen dataset semantics without stealing an outfit LoRA trigger."""

    project = state.payload.get("project", {})
    configured_trigger = str(project.get("trigger") or "")
    state = attach_dataset_semantics_snapshot(state, workspace)
    project = state.payload.get("project", {})
    target_type = str(project.get("training_target_type", project.get("type", "")))
    project["semantic_caption_policy"] = target_caption_policy(target_type)

    if target_type == "character_outfit":
        restored = str(project.get("training_config_trigger") or configured_trigger).strip()
        if not restored:
            raise PipelineError("Character outfit training requires a TrainingConfig trigger")
        project["trigger"] = restored
        project["trigger_source"] = "training_config"
        project.pop("training_config_trigger", None)
        state.save()
    return state


def install_target_policy_runtime_hook() -> None:
    """Specialize semantic caption composition for training targets."""

    global _RUNTIME_HOOK_INSTALLED
    if _RUNTIME_HOOK_INSTALLED:
        return
    from . import semantic_runtime

    original = semantic_runtime._apply_semantic_captions

    def apply(state, result: StepResult) -> StepResult:
        project = state.payload.get("project", {})
        target_type = str(project.get("training_target_type", project.get("type", "")))
        if target_type != "character_outfit":
            return original(state, result)
        return _apply_character_outfit_semantic_captions(state, result)

    semantic_runtime._apply_semantic_captions = apply
    _RUNTIME_HOOK_INSTALLED = True


def _apply_character_outfit_semantic_captions(state, result: StepResult) -> StepResult:
    project = state.payload.get("project", {})
    snapshot = project.get("dataset_semantics_snapshot")
    if not snapshot or project.get("type") != "character":
        return result
    if result.status not in {StepStatus.DONE, StepStatus.SKIPPED} or not result.output_manifest:
        return result

    manifest_path = Path(result.output_manifest)
    if not manifest_path.is_file():
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") == "existing_passthrough":
        return result
    records = manifest.get("records", [])
    if not isinstance(records, list):
        return result

    profiles = resolve_profiles(
        str(project.get("hardware", "v100_16gb")),
        str(project["type"]),
        str(project.get("strategy", "quality")),
        project_overrides=project.get("overrides", {}),
    )
    max_tokens = int(
        profiles.merged.get("caption", {}).get(
            "max_token_length",
            profiles.hardware.get("caption", {}).get("default_max_token_length", 75),
        )
    )

    bindings = snapshot.get("images", {})
    outfits = snapshot.get("outfits", {})
    outfit_ids = {
        str(bindings.get(str(record.get("image") or ""), {}).get("outfit") or "default")
        for record in records
    }
    if len(outfit_ids) > 1:
        raise PipelineError(
            "character_outfit training contains multiple semantic outfits: "
            + ", ".join(sorted(outfit_ids))
            + ". Use a dataset/run containing one target outfit."
        )

    target_outfit_id = next(iter(outfit_ids), "default")
    target_outfit = outfits.get(target_outfit_id, outfits.get("default", {}))
    character = snapshot.get("character", {})
    character_token = str(character.get("token") or "").strip()
    outfit_token = str(target_outfit.get("token") or "").strip()
    character_features = {normalize_tag(tag) for tag in character.get("features", [])}
    manual_outfit_features = {normalize_tag(tag) for tag in target_outfit.get("features", [])}
    inferred_outfit_features = _infer_invariant_outfit_tags(records)
    suppressed_features = character_features | manual_outfit_features | inferred_outfit_features
    concept_tokens = {normalize_tag(value) for value in (character_token, outfit_token) if value}

    trigger = str(project.get("trigger") or "").strip()
    if not trigger:
        raise PipelineError("Character outfit training requires a non-empty trigger")
    anchors = [
        str(value).strip()
        for value in project.get("caption_anchor_tags", [])
        if str(value).strip()
    ]
    if not anchors:
        raise PipelineError("Character outfit training requires at least one character anchor")

    changed = 0
    total_suppressed = 0
    for record in records:
        required_prefix = [trigger, *anchors]
        retained: list[str] = []
        seen: set[str] = set()
        for tag in required_prefix:
            normalized = normalize_tag(tag)
            if normalized and normalized not in seen:
                seen.add(normalized)
                retained.append(tag)
        protected = len(retained)

        suppressed_here: list[str] = []
        for tag in parse_caption(str(record.get("text") or "")):
            normalized = normalize_tag(tag)
            if not normalized or normalized in seen:
                continue
            if normalized in concept_tokens or normalized in suppressed_features:
                suppressed_here.append(tag)
                continue
            seen.add(normalized)
            retained.append(tag)

        counts = count_sdxl_tokens(", ".join(retained))
        pruned = list(record.get("pruned", []))
        while len(retained) > protected and counts.maximum > max_tokens:
            pruned.append(retained.pop())
            counts = count_sdxl_tokens(", ".join(retained))
        if counts.maximum > max_tokens:
            raise PipelineError(
                "Required character-outfit trigger/anchor prefix exceeds the configured "
                f"{max_tokens}-token budget"
            )

        text = ", ".join(retained)
        if text != str(record.get("text") or ""):
            changed += 1
        total_suppressed += len(suppressed_here)
        destination = Path(str(record["caption"]))
        destination.write_text(text + "\n", encoding="utf-8")
        record["text"] = text
        record["pruned"] = pruned
        record["token_counts"] = {
            "clip_l": counts.clip_l,
            "clip_g": counts.clip_g,
            "exact": counts.exact,
            "backend": counts.backend,
            "error": counts.error,
        }
        record["semantic_concepts"] = {
            "training_target": "character_outfit",
            "training_trigger": trigger,
            "character_anchors": anchors,
            "dataset_character_token": character_token,
            "outfit": target_outfit_id,
            "dataset_outfit_token": outfit_token,
            "suppressed_target_features": sorted({normalize_tag(tag) for tag in suppressed_here}),
        }

    manifest.pop("input_hash", None)
    policy = target_caption_policy("character_outfit")
    manifest["dataset_semantics_snapshot_hash"] = snapshot.get("snapshot_hash")
    manifest["semantic_caption_policy"] = policy
    manifest["target_policy"] = {
        "target_type": "character_outfit",
        "outfit_id": target_outfit_id,
        "invariant_outfit_min_coverage": _OUTFIT_INVARIANT_MIN_COVERAGE,
        "manual_outfit_features": sorted(manual_outfit_features),
        "inferred_invariant_outfit_features": sorted(inferred_outfit_features),
    }
    manifest.setdefault("summary", {})["semantic_caption_updates"] = changed
    manifest["summary"]["target_feature_suppressions"] = total_suppressed
    manifest["input_hash"] = stable_hash(manifest)
    write_json_atomic(manifest_path, manifest)

    details = dict(result.details)
    details.update(
        {
            "dataset_semantics_snapshot_hash": snapshot.get("snapshot_hash"),
            "semantic_caption_updates": changed,
            "target_feature_suppressions": total_suppressed,
            "inferred_invariant_outfit_features": sorted(inferred_outfit_features),
        }
    )
    return StepResult(
        status=result.status,
        input_hash=manifest["input_hash"],
        output_manifest=str(manifest_path),
        details=details,
    )


def _infer_invariant_outfit_tags(records: list[Mapping[str, Any]]) -> set[str]:
    """Infer only high-coverage garment tags; avoid suppressing pose/scene/content tags."""

    total = len(records)
    if not total:
        return set()
    counts: Counter[str] = Counter()
    for record in records:
        normalized_tags = {
            normalize_tag(tag)
            for tag in parse_caption(str(record.get("text") or ""))
            if normalize_tag(tag)
        }
        counts.update(normalized_tags)

    outfit_patterns = CATEGORY_PATTERNS.get("outfit", ())
    return {
        tag
        for tag, count in counts.items()
        if count / total >= _OUTFIT_INVARIANT_MIN_COVERAGE
        and any(pattern.search(tag) for pattern in outfit_patterns)
    }
