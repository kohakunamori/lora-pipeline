from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .models import BaseModel, ConfigurationError, ResolvedConfig


def repository_root() -> Path:
    configured = os.environ.get("LORA_PIPELINE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Expected a YAML mapping in {path}")
    return payload


def write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def deep_merge(*layers: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in layers:
        _merge_into(result, layer)
    return result


def _merge_into(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            nested = dict(target[key])
            _merge_into(nested, value)
            target[key] = nested
        else:
            target[key] = deepcopy(value)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def manifest_hash(paths: Iterable[Path], root: Path) -> str:
    records: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: item.as_posix().casefold()):
        stat = path.stat()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return stable_hash(records)


def load_base_registry(root: Path | None = None) -> dict[str, BaseModel]:
    root = root or repository_root()
    registry_path = root / "bases" / "registry.yaml"
    payload = read_yaml(registry_path) if registry_path.exists() else {"bases": {}}
    bases = payload.get("bases", {})
    if not isinstance(bases, dict):
        raise ConfigurationError("bases.registry.yaml: 'bases' must be a mapping")
    result: dict[str, BaseModel] = {}
    for base_id, item in bases.items():
        if not isinstance(item, dict):
            raise ConfigurationError(f"Base entry {base_id!r} must be a mapping")
        result[base_id] = BaseModel(
            id=base_id,
            name=str(item.get("name", base_id)),
            path=Path(str(item.get("path", ""))),
            family=str(item.get("family", "")),
            prediction_type=str(item.get("prediction_type", "epsilon")),
            sha256=item.get("sha256"),
            enabled=bool(item.get("enabled", True)),
            generation_defaults=item.get("generation_defaults", {}),
        )
    return result


def resolve_profiles(
    hardware_id: str,
    concept_id: str,
    training_id: str,
    *,
    project_overrides: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> ResolvedConfig:
    root = root or repository_root()
    hardware = read_yaml(root / "profiles" / "hardware" / f"{hardware_id}.yaml")
    concept = read_yaml(root / "profiles" / "concepts" / f"{concept_id}.yaml")
    training = read_yaml(root / "profiles" / "training" / f"{training_id}.yaml")
    merged = deep_merge(hardware, concept, training, project_overrides or {}, cli_overrides or {})
    validate_safety(hardware, training, merged)
    return ResolvedConfig(hardware=hardware, concept=concept, training=training, merged=merged)


def validate_safety(
    hardware: Mapping[str, Any],
    training: Mapping[str, Any],
    merged: Mapping[str, Any],
) -> None:
    resolution = int(merged.get("resolution", {}).get("default", 1024))
    max_area = int(merged.get("resolution", {}).get("max_bucket_area", 0))
    hardware_area = int(hardware.get("resolution", {}).get("max_bucket_area", 0))
    if resolution > int(hardware.get("resolution", {}).get("default", resolution)):
        raise ConfigurationError(f"Resolution {resolution} exceeds the validated V1 default envelope")
    if max_area > hardware_area:
        raise ConfigurationError(f"Bucket area {max_area} exceeds hardware limit {hardware_area}")

    training_values = merged.get("training", {})
    batch_size = int(training_values.get("batch_size", 1))
    accumulation = int(training_values.get("gradient_accumulation_steps", 1))
    if batch_size < 1:
        raise ConfigurationError("Physical batch must be at least 1")
    if accumulation < 1:
        raise ConfigurationError("Gradient accumulation must be at least 1")
    if int(training_values.get("network_dim", 16)) < 1:
        raise ConfigurationError("LoRA network_dim must be at least 1")
    if int(training_values.get("network_alpha", 8)) < 1:
        raise ConfigurationError("LoRA network_alpha must be at least 1")
    if float(training_values.get("unet_lr", 0.0001)) <= 0:
        raise ConfigurationError("UNet learning rate must be positive")

    cached = bool(training_values.get("cache_text_encoder_outputs", False))
    if cached:
        incompatible = {
            "shuffle_caption": merged.get("caption", {}).get("shuffle", False),
            "caption_dropout_rate": merged.get("caption", {}).get("dropout_rate", 0),
            "caption_tag_dropout_rate": merged.get("caption", {}).get("tag_dropout_rate", 0),
            "token_warmup_step": merged.get("caption", {}).get("token_warmup_step", 0),
        }
        enabled = [key for key, value in incompatible.items() if value]
        if enabled:
            raise ConfigurationError(
                "Text-encoder output caching is incompatible with: " + ", ".join(enabled)
            )
        if not training_values.get("network_train_unet_only", False):
            raise ConfigurationError("Text-encoder output caching requires U-Net-only network training")

    unet_only = bool(training_values.get("network_train_unet_only", True))
    text_encoder_lrs = (
        training_values.get("text_encoder_lr1"),
        training_values.get("text_encoder_lr2"),
    )
    if unet_only:
        if any(value is not None for value in text_encoder_lrs):
            raise ConfigurationError(
                "Text-encoder learning rates require network_train_unet_only=false"
            )
    else:
        if any(value is None for value in text_encoder_lrs):
            raise ConfigurationError(
                "SDXL text-encoder training requires text_encoder_lr1 and text_encoder_lr2"
            )
        if any(float(value) <= 0 for value in text_encoder_lrs):
            raise ConfigurationError("Text-encoder learning rates must be positive")
