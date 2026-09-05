from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import resolve_profiles, stable_hash, write_json_atomic
from .dataset.caption_cleaner import normalize_tag, parse_caption
from .models import PipelineError, StepResult, StepStatus
from .tokenizers import count_sdxl_tokens


_INSTALLED = False


def install_semantic_runtime_hooks() -> None:
    """Make frozen Dataset semantics part of caption generation and fingerprints."""
    global _INSTALLED
    if _INSTALLED:
        return
    from . import service
    from .steps import caption

    original_caption_run = caption.run
    original_signature = service.compute_step_signature

    def caption_run(state, *args, **kwargs):
        result = original_caption_run(state, *args, **kwargs)
        return _apply_semantic_captions(state, result)

    def signature(state, name: str, *, options: Mapping[str, Any] | None = None) -> str:
        value = original_signature(state, name, options=options)
        if name != "caption":
            return value
        snapshot = state.payload.get("project", {}).get("dataset_semantics_snapshot")
        if not snapshot:
            return value
        return stable_hash(
            {
                "base": value,
                "dataset_semantics_snapshot_hash": snapshot.get("snapshot_hash"),
                "semantic_caption_policy": state.payload["project"].get("semantic_caption_policy", {}),
            }
        )

    caption.run = caption_run
    service.compute_step_signature = signature
    _INSTALLED = True


def _apply_semantic_captions(state, result: StepResult) -> StepResult:
    """Apply semantic feature suppression without creating another trigger namespace.

    TrainingConfig/TriggerPolicy owns the runtime trigger. Dataset character/outfit
    tokens are retained only as metadata for grouping and compatibility; they are
    removed if they appear in generated captions instead of being injected.
    """

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
    character = snapshot.get("character", {})
    dataset_character_token = str(character.get("token") or "").strip()
    character_features = {normalize_tag(tag) for tag in character.get("features", [])}
    outfits = snapshot.get("outfits", {})
    bindings = snapshot.get("images", {})
    trigger = str(project.get("trigger") or "").strip()
    if not trigger:
        raise PipelineError("Character training requires a non-empty TrainingConfig trigger")
    anchors = [
        str(value).strip()
        for value in project.get("caption_anchor_tags", [])
        if str(value).strip()
    ]
    changed = 0

    for record in records:
        image_key = str(record.get("image") or "")
        outfit_id = str(bindings.get(image_key, {}).get("outfit") or "default")
        outfit = outfits.get(outfit_id, outfits.get("default", {}))
        dataset_outfit_token = str(outfit.get("token") or "").strip()
        concept_tokens = {
            normalize_tag(value)
            for value in (dataset_character_token, dataset_outfit_token)
            if value
        }
        tags = parse_caption(str(record.get("text") or ""))
        required_prefix = [trigger, *anchors]
        retained: list[str] = []
        seen: set[str] = set()
        for tag in required_prefix:
            normalized = normalize_tag(tag)
            if normalized and normalized not in seen:
                seen.add(normalized)
                retained.append(tag)
        protected = len(retained)
        for tag in tags:
            normalized = normalize_tag(tag)
            if (
                not normalized
                or normalized in seen
                or normalized in character_features
                or normalized in concept_tokens
            ):
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
                "Required TrainingConfig trigger/anchor prefix exceeds the configured "
                f"{max_tokens}-token budget"
            )
        text = ", ".join(retained)
        if text != str(record.get("text") or ""):
            changed += 1
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
            "training_trigger": trigger,
            "dataset_character_token": dataset_character_token,
            "outfit": outfit_id,
            "dataset_outfit_token": dataset_outfit_token,
        }

    manifest.pop("input_hash", None)
    manifest["dataset_semantics_snapshot_hash"] = snapshot.get("snapshot_hash")
    manifest["semantic_caption_policy"] = project.get("semantic_caption_policy", {})
    manifest.setdefault("summary", {})["semantic_caption_updates"] = changed
    manifest["input_hash"] = stable_hash(manifest)
    write_json_atomic(manifest_path, manifest)
    details = dict(result.details)
    details.update(
        {
            "dataset_semantics_snapshot_hash": snapshot.get("snapshot_hash"),
            "semantic_caption_updates": changed,
        }
    )
    return StepResult(
        status=result.status,
        input_hash=manifest["input_hash"],
        output_manifest=str(manifest_path),
        details=details,
    )
