from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import stable_hash, write_json_atomic
from .dataset.caption_cleaner import normalize_tag, parse_caption
from .models import PipelineError, StepResult, StepStatus
from .tokenizers import count_sdxl_tokens


def apply_activation_group_captions(state, result: StepResult) -> StepResult:
    """Insert the selected character group tag into each assigned image caption.

    Group tags are learned selectors, not display-only metadata. When a frozen
    ActivationRecipe contains groups, every active image has exactly one assignment
    and its caption prefix becomes ``trigger, group_tag, ...``. Existing protected
    anchors remain immediately after that pair.
    """

    project = state.payload.get("project", {})
    recipe = project.get("activation_recipe", {})
    groups = recipe.get("character_tags_groups", []) if isinstance(recipe, dict) else []
    if not groups:
        return result
    if result.status not in {StepStatus.DONE, StepStatus.SKIPPED} or not result.output_manifest:
        raise PipelineError(
            "Character tag groups are enabled but caption materialization produced no rewriteable manifest"
        )

    manifest_path = Path(result.output_manifest)
    if not manifest_path.is_file():
        raise PipelineError(f"Caption manifest is missing for character tag groups: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mode = str(manifest.get("mode", ""))
    if mode in {"existing_passthrough", "skip"}:
        raise PipelineError(
            "Character tag groups cannot be trained with existing_passthrough/skip because "
            "the group_tag must be inserted into every assigned caption"
        )

    assignments = recipe.get("assignments", {})
    if not isinstance(assignments, dict):
        raise PipelineError("ActivationRecipe assignments are invalid")
    by_name = {str(group["name"]): group for group in groups if isinstance(group, dict)}
    trigger = str(project.get("trigger") or "").strip()
    anchors = [
        str(value).strip()
        for value in project.get("caption_anchor_tags", [])
        if str(value).strip()
    ]

    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise PipelineError("Caption manifest records must be a list")
    changed = 0
    used_groups: dict[str, int] = {name: 0 for name in by_name}
    for record in records:
        if not isinstance(record, dict):
            continue
        image_key = str(record.get("image") or "")
        group_name = str(assignments.get(image_key) or "")
        group = by_name.get(group_name)
        if group is None:
            raise PipelineError(
                f"Prepared image {image_key!r} has no character tag group assignment"
            )
        group_tag = str(group.get("group_tag") or "").strip()
        if not group_tag:
            raise PipelineError(f"Character tag group {group_name!r} has an empty group_tag")

        original_tags = parse_caption(str(record.get("text") or ""))
        protected = [trigger, group_tag, *anchors]
        protected_norm = {normalize_tag(tag) for tag in protected if normalize_tag(tag)}
        rest: list[str] = []
        seen = set(protected_norm)
        for tag in original_tags:
            normalized = normalize_tag(tag)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            rest.append(tag)
        final_tags = [tag for tag in protected if tag] + rest
        text = ", ".join(final_tags)
        if text != str(record.get("text") or ""):
            changed += 1
        caption_path = Path(str(record.get("caption") or ""))
        if not caption_path:
            raise PipelineError(f"Prepared image {image_key!r} has no caption path")
        caption_path.write_text(text + "\n", encoding="utf-8")
        counts = count_sdxl_tokens(text)
        record["text"] = text
        record["token_counts"] = {
            "clip_l": counts.clip_l,
            "clip_g": counts.clip_g,
            "exact": counts.exact,
            "backend": counts.backend,
            "error": counts.error,
        }
        record["activation_group"] = {
            "name": group_name,
            "group_tag": group_tag,
        }
        used_groups[group_name] = used_groups.get(group_name, 0) + 1

    manifest.pop("input_hash", None)
    manifest["activation_recipe"] = {
        "schema_version": recipe.get("schema_version", 1),
        "snapshot_hash": recipe.get("snapshot_hash"),
        "groups": [
            {
                "name": str(group.get("name") or ""),
                "group_tag": str(group.get("group_tag") or ""),
                "images": used_groups.get(str(group.get("name") or ""), 0),
            }
            for group in groups
        ],
    }
    manifest.setdefault("summary", {})["activation_group_caption_updates"] = changed
    manifest["input_hash"] = stable_hash(manifest)
    write_json_atomic(manifest_path, manifest)

    details = dict(result.details)
    details.update(
        {
            "activation_group_caption_updates": changed,
            "activation_recipe_snapshot_hash": recipe.get("snapshot_hash"),
            "activation_groups": used_groups,
        }
    )
    return StepResult(
        status=result.status,
        input_hash=manifest["input_hash"],
        output_manifest=str(manifest_path),
        details=details,
    )
