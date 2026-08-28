from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import stable_hash, write_json_atomic, write_yaml_atomic, read_yaml
from .dataset.caption_cleaner import normalize_tag, parse_caption
from .dataset_workspace import DatasetWorkspace
from .models import PipelineError, StateError
from .state import ProjectState, utc_now


SEMANTICS_SCHEMA_VERSION = 1
_OUTFIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def semantics_path(workspace: DatasetWorkspace) -> Path:
    return workspace.dataset_dir / "semantics.yaml"


def default_character_token(dataset_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", dataset_name).strip("_")
    return token or "character"


def default_outfit_token(character_token: str, outfit_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_]+", "_", outfit_id).strip("_") or "outfit"
    return f"{character_token}_{suffix}"


def new_semantics(workspace: DatasetWorkspace) -> dict[str, Any]:
    if workspace.concept_type != "character":
        raise PipelineError("Dataset semantics are currently defined for character datasets only")
    character_token = default_character_token(workspace.name)
    return {
        "schema_version": SEMANTICS_SCHEMA_VERSION,
        "dataset": workspace.name,
        "character": {
            "token": character_token,
            "features": [],
        },
        "outfits": {
            "default": {
                "label": "Default",
                "token": default_outfit_token(character_token, "default"),
                "features": [],
            }
        },
        "images": {},
        "updated_at": utc_now(),
    }


def load_semantics(
    workspace: DatasetWorkspace,
    *,
    create: bool = False,
) -> dict[str, Any] | None:
    if workspace.concept_type != "character":
        return None
    path = semantics_path(workspace)
    if not path.is_file():
        if not create:
            return None
        payload = new_semantics(workspace)
        save_semantics(workspace, payload)
        return payload
    payload = read_yaml(path)
    return _normalize_semantics(workspace, payload)


def save_semantics(workspace: DatasetWorkspace, payload: Mapping[str, Any]) -> None:
    normalized = _normalize_semantics(workspace, copy.deepcopy(dict(payload)))
    normalized["updated_at"] = utc_now()
    write_yaml_atomic(semantics_path(workspace), normalized)
    workspace.save()


def _normalize_semantics(
    workspace: DatasetWorkspace,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if workspace.concept_type != "character":
        raise PipelineError("Character/outfit semantics cannot be attached to a style dataset")
    if int(payload.get("schema_version", -1)) != SEMANTICS_SCHEMA_VERSION:
        raise StateError(f"Unsupported dataset semantics schema in {semantics_path(workspace)}")
    payload["dataset"] = workspace.name
    character = payload.setdefault("character", {})
    token = str(character.get("token") or default_character_token(workspace.name)).strip()
    _validate_token(token, "Character token")
    character["token"] = token
    character["features"] = _normalize_features(character.get("features", []))

    outfits = payload.setdefault("outfits", {})
    if not isinstance(outfits, dict):
        raise StateError("Dataset semantic outfits must be a mapping")
    default = outfits.setdefault("default", {})
    default.setdefault("label", "Default")
    default.setdefault("token", default_outfit_token(token, "default"))
    for outfit_id, outfit in list(outfits.items()):
        if not _OUTFIT_ID.fullmatch(str(outfit_id)) or outfit_id in {".", ".."}:
            raise StateError(f"Invalid outfit id: {outfit_id}")
        if not isinstance(outfit, dict):
            raise StateError(f"Invalid outfit record: {outfit_id}")
        outfit.setdefault("label", str(outfit_id))
        outfit_token = str(outfit.get("token") or default_outfit_token(token, str(outfit_id))).strip()
        _validate_token(outfit_token, f"Outfit token {outfit_id}")
        outfit["token"] = outfit_token
        outfit["features"] = _normalize_features(outfit.get("features", []))

    normalized_tokens: dict[str, str] = {normalize_tag(token): "character"}
    for outfit_id, outfit in outfits.items():
        normalized = normalize_tag(str(outfit["token"]))
        previous = normalized_tokens.get(normalized)
        if previous is not None:
            raise StateError(f"Concept tokens must be unique: {previous} and outfit {outfit_id}")
        normalized_tokens[normalized] = f"outfit {outfit_id}"

    images = payload.setdefault("images", {})
    if not isinstance(images, dict):
        raise StateError("Dataset semantic image bindings must be a mapping")
    for key, binding in list(images.items()):
        if not isinstance(binding, dict):
            raise StateError(f"Invalid semantic image binding: {key}")
        outfit_id = str(binding.get("outfit") or "default")
        if outfit_id not in outfits:
            raise StateError(f"Image {key} references unknown outfit: {outfit_id}")
        if outfit_id == "default":
            images.pop(key, None)
        else:
            binding["outfit"] = outfit_id
    payload.setdefault("updated_at", utc_now())
    return payload


def _validate_token(token: str, label: str) -> None:
    if not token or "," in token or "\n" in token:
        raise PipelineError(f"{label} must be non-empty and cannot contain commas/newlines")


def _normalize_features(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = parse_caption(values)
    if not isinstance(values, Sequence):
        raise StateError("Semantic feature tags must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = normalize_tag(str(value))
        if not tag or tag in seen:
            continue
        seen.add(tag)
        result.append(tag)
    return result


def set_character_token(workspace: DatasetWorkspace, token: str) -> dict[str, Any]:
    payload = load_semantics(workspace, create=True)
    assert payload is not None
    token = token.strip()
    _validate_token(token, "Character token")
    old = str(payload["character"]["token"])
    payload["character"]["token"] = token
    for outfit_id, outfit in payload["outfits"].items():
        if str(outfit.get("token")) == default_outfit_token(old, outfit_id):
            outfit["token"] = default_outfit_token(token, outfit_id)
    save_semantics(workspace, payload)
    return payload


def set_character_features(workspace: DatasetWorkspace, features: Sequence[str]) -> dict[str, Any]:
    payload = load_semantics(workspace, create=True)
    assert payload is not None
    payload["character"]["features"] = _normalize_features(features)
    save_semantics(workspace, payload)
    return payload


def add_outfit(
    workspace: DatasetWorkspace,
    outfit_id: str,
    *,
    label: str,
    token: str | None = None,
    image_keys: Sequence[str] = (),
) -> dict[str, Any]:
    payload = load_semantics(workspace, create=True)
    assert payload is not None
    outfit_id = outfit_id.strip()
    if outfit_id == "default" or not _OUTFIT_ID.fullmatch(outfit_id):
        raise PipelineError("Outfit id must be 1-64 letters, numbers, '.', '_' or '-' and cannot be 'default'")
    if outfit_id in payload["outfits"]:
        raise PipelineError(f"Outfit already exists: {outfit_id}")
    value = (token or default_outfit_token(str(payload["character"]["token"]), outfit_id)).strip()
    _validate_token(value, "Outfit token")
    payload["outfits"][outfit_id] = {
        "label": label.strip() or outfit_id,
        "token": value,
        "features": [],
    }
    _assign_outfit_images(workspace, payload, outfit_id, image_keys)
    save_semantics(workspace, payload)
    return payload


def set_outfit_token(workspace: DatasetWorkspace, outfit_id: str, token: str) -> dict[str, Any]:
    payload = load_semantics(workspace, create=True)
    assert payload is not None
    outfit = _require_outfit(payload, outfit_id)
    token = token.strip()
    _validate_token(token, "Outfit token")
    outfit["token"] = token
    save_semantics(workspace, payload)
    return payload


def set_outfit_features(
    workspace: DatasetWorkspace,
    outfit_id: str,
    features: Sequence[str],
) -> dict[str, Any]:
    payload = load_semantics(workspace, create=True)
    assert payload is not None
    outfit = _require_outfit(payload, outfit_id)
    outfit["features"] = _normalize_features(features)
    save_semantics(workspace, payload)
    return payload


def set_outfit_images(
    workspace: DatasetWorkspace,
    outfit_id: str,
    image_keys: Sequence[str],
) -> dict[str, Any]:
    payload = load_semantics(workspace, create=True)
    assert payload is not None
    _require_outfit(payload, outfit_id)
    _assign_outfit_images(workspace, payload, outfit_id, image_keys)
    save_semantics(workspace, payload)
    return payload


def _assign_outfit_images(
    workspace: DatasetWorkspace,
    payload: dict[str, Any],
    outfit_id: str,
    image_keys: Sequence[str],
) -> None:
    existing = {item.key for item in workspace.items(include_disabled=True, include_excluded=True)}
    requested = set(image_keys)
    unknown = sorted(requested - existing)
    if unknown:
        raise PipelineError("Unknown dataset image(s): " + ", ".join(unknown[:5]))
    bindings = payload["images"]
    for key, binding in list(bindings.items()):
        if str(binding.get("outfit")) == outfit_id and key not in requested:
            bindings.pop(key, None)
    if outfit_id == "default":
        for key in requested:
            bindings.pop(key, None)
        return
    for key in requested:
        bindings[key] = {"outfit": outfit_id}


def _require_outfit(payload: Mapping[str, Any], outfit_id: str) -> dict[str, Any]:
    try:
        return payload["outfits"][outfit_id]
    except KeyError as exc:
        raise PipelineError(f"Unknown outfit: {outfit_id}") from exc


def outfit_for_image(payload: Mapping[str, Any], key: str) -> str:
    binding = payload.get("images", {}).get(key, {})
    return str(binding.get("outfit") or "default")


def image_keys_for_outfit(
    workspace: DatasetWorkspace,
    payload: Mapping[str, Any],
    outfit_id: str,
    *,
    active_only: bool = True,
) -> list[str]:
    items = workspace.items(
        include_disabled=not active_only,
        include_excluded=not active_only,
    )
    return [item.key for item in items if outfit_for_image(payload, item.key) == outfit_id]


def character_feature_candidates(
    workspace: DatasetWorkspace,
    *,
    minimum_coverage: float = 0.5,
    limit: int = 160,
) -> list[dict[str, Any]]:
    items = workspace.items(include_disabled=False, include_excluded=False)
    return _tag_candidates(workspace, items, minimum_coverage=minimum_coverage, limit=limit)


def outfit_feature_candidates(
    workspace: DatasetWorkspace,
    payload: Mapping[str, Any],
    outfit_id: str,
    *,
    minimum_coverage: float = 0.35,
    limit: int = 160,
) -> list[dict[str, Any]]:
    active = workspace.items(include_disabled=False, include_excluded=False)
    selected = [item for item in active if outfit_for_image(payload, item.key) == outfit_id]
    others = [item for item in active if outfit_for_image(payload, item.key) != outfit_id]
    selected_counts = _tag_counts(workspace, selected)
    other_counts = _tag_counts(workspace, others)
    rows: list[dict[str, Any]] = []
    selected_total = len(selected)
    other_total = len(others)
    if selected_total == 0:
        return rows
    for tag, count in selected_counts.items():
        coverage = count / selected_total
        if coverage < minimum_coverage:
            continue
        other_coverage = other_counts.get(tag, 0) / other_total if other_total else 0.0
        rows.append(
            {
                "tag": tag,
                "coverage": coverage,
                "other_coverage": other_coverage,
                "specificity": coverage - other_coverage,
                "count": count,
                "total": selected_total,
            }
        )
    rows.sort(key=lambda row: (-float(row["specificity"]), -float(row["coverage"]), str(row["tag"])))
    return rows[:limit]


def _tag_candidates(
    workspace: DatasetWorkspace,
    items: Sequence[Any],
    *,
    minimum_coverage: float,
    limit: int,
) -> list[dict[str, Any]]:
    total = len(items)
    if not total:
        return []
    counts = _tag_counts(workspace, items)
    rows = [
        {"tag": tag, "coverage": count / total, "count": count, "total": total}
        for tag, count in counts.items()
        if count / total >= minimum_coverage
    ]
    rows.sort(key=lambda row: (-float(row["coverage"]), str(row["tag"])))
    return rows[:limit]


def _tag_counts(workspace: DatasetWorkspace, items: Sequence[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        tags = {normalize_tag(tag) for tag in parse_caption(workspace.caption_text(item.key))}
        counts.update(tag for tag in tags if tag)
    return counts


def semantics_snapshot(workspace: DatasetWorkspace) -> dict[str, Any] | None:
    payload = load_semantics(workspace, create=False)
    if payload is None:
        return None
    active_keys = {
        item.key for item in workspace.items(include_disabled=False, include_excluded=False)
    }
    basis = {
        "schema_version": SEMANTICS_SCHEMA_VERSION,
        "dataset": workspace.name,
        "character": copy.deepcopy(payload["character"]),
        "outfits": copy.deepcopy(payload["outfits"]),
        "images": {
            key: copy.deepcopy(binding)
            for key, binding in sorted(payload["images"].items())
            if key in active_keys
        },
    }
    return {
        **basis,
        "snapshot_hash": stable_hash(basis),
        "created_at": utc_now(),
    }


def attach_dataset_semantics_snapshot(
    state: ProjectState,
    workspace: DatasetWorkspace,
) -> ProjectState:
    if workspace.concept_type != "character":
        return state
    load_semantics(workspace, create=True)
    snapshot = semantics_snapshot(workspace)
    assert snapshot is not None
    project = state.payload["project"]
    old_trigger = str(project.get("trigger") or "")
    character_token = str(snapshot["character"]["token"])
    project["trigger"] = character_token
    if old_trigger and old_trigger != character_token:
        project["training_config_trigger"] = old_trigger
    project["trigger_source"] = "dataset_semantics"
    project["dataset_semantics_snapshot"] = snapshot
    project["semantic_caption_policy"] = {
        "character_token": "always",
        "outfit_token": "when_present",
        "character_features": "suppress",
        "outfit_features": "preserve",
    }
    identity = dict(project.get("training_identity") or {})
    identity["dataset_semantics_snapshot_hash"] = snapshot["snapshot_hash"]
    project["training_identity"] = identity
    write_json_atomic(state.project_dir / "dataset-semantics.json", snapshot)
    state.save()
    return state
