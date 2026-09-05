from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import read_yaml, stable_hash, write_yaml_atomic
from .dataset.caption_cleaner import normalize_tag, parse_caption
from .dataset.tag_categories import is_identity_tag, is_outfit_tag
from .dataset_workspace import DatasetWorkspace
from .models import PipelineError, StateError
from .state import ProjectState, utc_now


ACTIVATION_RECIPE_SCHEMA_VERSION = 1


def activation_recipe_path(workspace: DatasetWorkspace) -> Path:
    return workspace.dataset_dir / "activation.yaml"


def new_activation_recipe(workspace: DatasetWorkspace) -> dict[str, Any]:
    if workspace.concept_type != "character":
        raise PipelineError("Character tag groups are only available for character datasets")
    return {
        "schema_version": ACTIVATION_RECIPE_SCHEMA_VERSION,
        "dataset": workspace.name,
        "character_tags_groups": [],
        "assignments": {},
        "updated_at": utc_now(),
    }


def load_activation_recipe(
    workspace: DatasetWorkspace,
    *,
    create: bool = False,
) -> dict[str, Any] | None:
    if workspace.concept_type != "character":
        return None
    path = activation_recipe_path(workspace)
    if not path.is_file():
        if not create:
            return None
        payload = new_activation_recipe(workspace)
        save_activation_recipe(workspace, payload)
        return payload
    payload = read_yaml(path)
    return _normalize_recipe(workspace, payload)


def save_activation_recipe(workspace: DatasetWorkspace, payload: Mapping[str, Any]) -> None:
    normalized = _normalize_recipe(workspace, copy.deepcopy(dict(payload)))
    normalized["updated_at"] = utc_now()
    write_yaml_atomic(activation_recipe_path(workspace), normalized)
    workspace.save()


def _normalize_recipe(workspace: DatasetWorkspace, payload: dict[str, Any]) -> dict[str, Any]:
    if workspace.concept_type != "character":
        raise PipelineError("Character tag groups cannot be attached to a style dataset")
    if int(payload.get("schema_version", -1)) != ACTIVATION_RECIPE_SCHEMA_VERSION:
        raise StateError(f"Unsupported activation recipe schema in {activation_recipe_path(workspace)}")
    payload["dataset"] = workspace.name
    groups = payload.setdefault("character_tags_groups", [])
    if not isinstance(groups, list):
        raise StateError("character_tags_groups must be a list")

    names: set[str] = set()
    group_tags: set[str] = set()
    normalized_groups: list[dict[str, Any]] = []
    for raw in groups:
        if not isinstance(raw, Mapping):
            raise StateError("Character tag group must be a mapping")
        name = _validate_group_name(raw.get("name"))
        name_key = name.casefold()
        if name_key in names:
            raise StateError(f"Character tag group names must be unique: {name}")
        names.add(name_key)
        group_tag = _validate_group_tag(raw.get("group_tag"))
        group_tag_key = normalize_tag(group_tag)
        if group_tag_key in group_tags:
            raise StateError(f"Character tag group tags must be unique: {group_tag}")
        group_tags.add(group_tag_key)
        identity_tags = _normalize_tags(raw.get("identity_tags", []))
        outfit_tags = _normalize_tags(raw.get("outfit_tags", []))
        if not identity_tags and not outfit_tags:
            raise StateError(f"Character tag group {name!r} must select at least one identity/outfit tag")
        normalized_groups.append(
            {
                "name": name,
                "group_tag": group_tag,
                "identity_tags": identity_tags,
                "outfit_tags": outfit_tags,
            }
        )
    payload["character_tags_groups"] = normalized_groups

    assignments = payload.setdefault("assignments", {})
    if not isinstance(assignments, dict):
        raise StateError("Activation recipe assignments must be a mapping")
    valid_names = {group["name"] for group in normalized_groups}
    valid_keys = {
        item.key for item in workspace.items(include_disabled=True, include_excluded=True)
    }
    normalized_assignments: dict[str, str] = {}
    for key, value in assignments.items():
        key = str(key)
        name = str(value)
        if key not in valid_keys:
            continue
        if name not in valid_names:
            raise StateError(f"Activation assignment references unknown group {name!r}: {key}")
        normalized_assignments[key] = name
    payload["assignments"] = normalized_assignments
    payload.setdefault("updated_at", utc_now())
    return payload


def _validate_group_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name or "\n" in name:
        raise PipelineError("Group name must be non-empty and single-line")
    if len(name) > 96:
        raise PipelineError("Group name must be at most 96 characters")
    return name


def _validate_group_tag(value: Any) -> str:
    tag = str(value or "").strip()
    if not tag or "," in tag or "\n" in tag:
        raise PipelineError("Group tag must be one non-empty token/phrase without commas/newlines")
    return tag


def _normalize_tags(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = parse_caption(values)
    if not isinstance(values, Sequence):
        raise StateError("Activation feature tags must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = normalize_tag(str(value))
        if not tag or tag in seen:
            continue
        seen.add(tag)
        result.append(tag)
    return result


def upsert_character_tags_group(
    workspace: DatasetWorkspace,
    *,
    name: str,
    group_tag: str,
    identity_tags: Sequence[str],
    outfit_tags: Sequence[str],
    previous_name: str | None = None,
) -> dict[str, Any]:
    payload = load_activation_recipe(workspace, create=True)
    assert payload is not None
    name = _validate_group_name(name)
    group_tag = _validate_group_tag(group_tag)
    record = {
        "name": name,
        "group_tag": group_tag,
        "identity_tags": _normalize_tags(identity_tags),
        "outfit_tags": _normalize_tags(outfit_tags),
    }
    if not record["identity_tags"] and not record["outfit_tags"]:
        raise PipelineError("Select at least one identity or outfit tag for the group")

    groups = list(payload["character_tags_groups"])
    previous_name = previous_name or name
    replaced = False
    for index, current in enumerate(groups):
        if str(current["name"]) == previous_name:
            groups[index] = record
            replaced = True
            break
    if not replaced:
        groups.append(record)
    payload["character_tags_groups"] = groups
    if previous_name != name:
        payload["assignments"] = {
            key: (name if value == previous_name else value)
            for key, value in payload["assignments"].items()
        }
    save_activation_recipe(workspace, payload)
    loaded = load_activation_recipe(workspace)
    assert loaded is not None
    return loaded


def delete_character_tags_group(workspace: DatasetWorkspace, name: str) -> dict[str, Any]:
    payload = load_activation_recipe(workspace, create=True)
    assert payload is not None
    payload["character_tags_groups"] = [
        group for group in payload["character_tags_groups"] if group["name"] != name
    ]
    payload["assignments"] = {
        key: value for key, value in payload["assignments"].items() if value != name
    }
    save_activation_recipe(workspace, payload)
    loaded = load_activation_recipe(workspace)
    assert loaded is not None
    return loaded


def set_group_images(
    workspace: DatasetWorkspace,
    name: str,
    image_keys: Sequence[str],
) -> dict[str, Any]:
    payload = load_activation_recipe(workspace, create=True)
    assert payload is not None
    if name not in {group["name"] for group in payload["character_tags_groups"]}:
        raise PipelineError(f"Unknown character tag group: {name}")
    valid = {
        item.key for item in workspace.items(include_disabled=True, include_excluded=True)
    }
    requested = {str(key) for key in image_keys}
    unknown = sorted(requested - valid)
    if unknown:
        raise PipelineError("Unknown dataset image(s): " + ", ".join(unknown[:5]))
    assignments = {
        key: value for key, value in payload["assignments"].items() if value != name
    }
    # One image belongs to at most one appearance group. Assigning it here moves it
    # from any previous group instead of creating overlapping learned selectors.
    for key in requested:
        assignments[key] = name
    payload["assignments"] = assignments
    save_activation_recipe(workspace, payload)
    loaded = load_activation_recipe(workspace)
    assert loaded is not None
    return loaded


def image_keys_for_group(
    workspace: DatasetWorkspace,
    payload: Mapping[str, Any],
    name: str,
    *,
    active_only: bool = True,
) -> list[str]:
    valid = {
        item.key
        for item in workspace.items(
            include_disabled=not active_only,
            include_excluded=not active_only,
        )
    }
    return sorted(
        key
        for key, value in payload.get("assignments", {}).items()
        if value == name and key in valid
    )


def tag_candidates(
    workspace: DatasetWorkspace,
    *,
    kind: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    if kind not in {"identity", "outfit"}:
        raise PipelineError("Tag candidate kind must be identity or outfit")
    predicate = is_identity_tag if kind == "identity" else is_outfit_tag
    items = workspace.items(include_disabled=False, include_excluded=False)
    total = len(items)
    if not total:
        return []
    counts: Counter[str] = Counter()
    for item in items:
        tags = {
            normalize_tag(tag)
            for tag in parse_caption(workspace.caption_text(item.key))
            if normalize_tag(tag)
        }
        counts.update(tag for tag in tags if predicate(tag))
    rows = [
        {
            "tag": tag,
            "count": count,
            "total": total,
            "coverage": count / total,
        }
        for tag, count in counts.items()
    ]
    rows.sort(key=lambda row: (-float(row["coverage"]), str(row["tag"])))
    return rows[:limit]


def suggest_group_images(
    workspace: DatasetWorkspace,
    *,
    identity_tags: Sequence[str],
    outfit_tags: Sequence[str],
) -> list[str]:
    identity = {normalize_tag(tag) for tag in identity_tags if normalize_tag(tag)}
    outfit = {normalize_tag(tag) for tag in outfit_tags if normalize_tag(tag)}
    selected: list[str] = []
    for item in workspace.items(include_disabled=False, include_excluded=False):
        tags = {
            normalize_tag(tag)
            for tag in parse_caption(workspace.caption_text(item.key))
            if normalize_tag(tag)
        }
        outfit_hits = len(tags & outfit)
        identity_hits = len(tags & identity)
        # Outfit features are the strongest discriminator between appearance
        # groups. Identity-only groups (for example hairstyle variants) remain
        # supported when no outfit features were selected.
        if outfit and outfit_hits == 0:
            continue
        if not outfit and identity and identity_hits == 0:
            continue
        if outfit_hits or identity_hits:
            selected.append(item.key)
    return selected


def validate_active_group_coverage(
    workspace: DatasetWorkspace,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload or load_activation_recipe(workspace, create=False)
    groups = list(payload.get("character_tags_groups", [])) if payload else []
    active = [
        item.key for item in workspace.items(include_disabled=False, include_excluded=False)
    ]
    assignments = dict(payload.get("assignments", {})) if payload else {}
    if not groups:
        return {
            "enabled": False,
            "active_images": len(active),
            "assigned_images": 0,
            "unassigned": active,
            "complete": True,
        }
    names = {group["name"] for group in groups}
    assigned = [key for key in active if assignments.get(key) in names]
    unassigned = [key for key in active if assignments.get(key) not in names]
    return {
        "enabled": True,
        "active_images": len(active),
        "assigned_images": len(assigned),
        "unassigned": unassigned,
        "complete": not unassigned,
    }


def activation_recipe_snapshot(
    workspace: DatasetWorkspace,
    *,
    trigger: str,
    character_anchors: Iterable[str] = (),
) -> dict[str, Any]:
    payload = load_activation_recipe(workspace, create=False)
    groups = list(payload.get("character_tags_groups", [])) if payload else []
    coverage = validate_active_group_coverage(workspace, payload)
    if groups and not coverage["complete"]:
        preview = ", ".join(coverage["unassigned"][:5])
        raise PipelineError(
            f"Character tag groups are enabled but {len(coverage['unassigned'])} active image(s) "
            f"are unassigned ({preview}). Assign every active image before training."
        )
    active_keys = {
        item.key for item in workspace.items(include_disabled=False, include_excluded=False)
    }
    assignments = {
        key: value
        for key, value in sorted((payload or {}).get("assignments", {}).items())
        if key in active_keys
    }
    group_records: list[dict[str, Any]] = []
    for group in groups:
        image_count = sum(value == group["name"] for value in assignments.values())
        combined = _normalize_tags([*group["identity_tags"], *group["outfit_tags"]])
        group_records.append(
            {
                "name": group["name"],
                "group_tag": group["group_tag"],
                "identity_tags": list(group["identity_tags"]),
                "outfit_tags": list(group["outfit_tags"]),
                "tags": combined,
                "image_count": image_count,
                "coverage": round(image_count / len(active_keys), 6) if active_keys else 0.0,
            }
        )
    basis = {
        "schema_version": ACTIVATION_RECIPE_SCHEMA_VERSION,
        "trigger": str(trigger).strip(),
        "character_anchors": _normalize_tags(character_anchors),
        "character_tags_groups": group_records,
        "assignments": assignments,
    }
    return {
        **basis,
        "snapshot_hash": stable_hash(basis),
        "created_at": utc_now(),
    }


def attach_activation_recipe_snapshot(
    state: ProjectState,
    workspace: DatasetWorkspace,
) -> ProjectState:
    project = state.payload["project"]
    snapshot = activation_recipe_snapshot(
        workspace,
        trigger=str(project.get("trigger") or ""),
        character_anchors=project.get("caption_anchor_tags", []),
    )
    project["activation_recipe"] = snapshot
    identity = dict(project.get("training_identity") or {})
    identity["activation_recipe_snapshot_hash"] = snapshot["snapshot_hash"]
    project["training_identity"] = identity
    state.save()
    return state


def public_activation_recipe(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(snapshot.get("schema_version", ACTIVATION_RECIPE_SCHEMA_VERSION)),
        "trigger": str(snapshot.get("trigger") or ""),
        "character_anchors": list(snapshot.get("character_anchors", [])),
        "character_tags_groups": [
            {
                "name": str(group.get("name") or ""),
                "group_tag": str(group.get("group_tag") or ""),
                "identity_tags": list(group.get("identity_tags", [])),
                "outfit_tags": list(group.get("outfit_tags", [])),
                "tags": list(group.get("tags", [])),
                "coverage": group.get("coverage"),
            }
            for group in snapshot.get("character_tags_groups", [])
        ],
    }


def activation_usage_hint(snapshot: Mapping[str, Any]) -> str:
    recipe = public_activation_recipe(snapshot)
    lines = ["Trigger word:", recipe["trigger"]]
    anchors = recipe.get("character_anchors", [])
    if anchors:
        lines.extend(["", "Character:", ", ".join(anchors)])
    for group in recipe["character_tags_groups"]:
        lines.extend(
            [
                "",
                f"Character Tags Group — {group['name']}",
                "Group tag:",
                str(group["group_tag"]),
                "Tags:",
                ", ".join(group["tags"]),
            ]
        )
    return "\n".join(lines).strip()


def activation_safetensors_metadata(snapshot: Mapping[str, Any]) -> dict[str, str]:
    recipe = public_activation_recipe(snapshot)
    return {
        "lora_pipeline.activation_schema": str(recipe["schema_version"]),
        "lora_pipeline.activation": json.dumps(
            recipe,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
