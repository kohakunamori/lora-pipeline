from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import load_base_registry, read_yaml, sha256_file, write_yaml_atomic
from .models import BaseModel, ConfigurationError, PipelineError


def add_base(
    base_id: str,
    path: Path,
    *,
    name: str | None = None,
    family: str = "illustrious_sdxl",
    prediction_type: str = "epsilon",
    enabled: bool = True,
    root: Path,
) -> BaseModel:
    _validate_base_id(base_id)
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PipelineError(f"Checkpoint does not exist: {path}")
    registry_path = root / "bases" / "registry.yaml"
    payload = read_yaml(registry_path) if registry_path.exists() else {"bases": {}}
    bases = payload.setdefault("bases", {})
    if base_id in bases:
        raise ConfigurationError(f"Base id is already registered: {base_id}")
    bases[base_id] = {
        "name": name or path.stem,
        "path": str(path),
        "family": family,
        "prediction_type": prediction_type,
        "sha256": None,
        "enabled": enabled,
        "generation_defaults": {"sampler": "euler_a", "scheduler": "normal", "cfg": 4.5, "steps": 28},
    }
    write_yaml_atomic(registry_path, payload)
    return load_base_registry(root)[base_id]


def inspect_base(base_id: str, *, root: Path, persist_sha: bool = True) -> dict[str, Any]:
    registry = load_base_registry(root)
    if base_id not in registry:
        raise PipelineError(f"Unknown base id: {base_id}")
    base = registry[base_id]
    if not base.path.is_file():
        raise PipelineError(f"Checkpoint does not exist: {base.path}")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise PipelineError("safetensors is required to inspect checkpoint metadata") from exc
    with safe_open(base.path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        keys = list(checkpoint.keys())
    digest = sha256_file(base.path)
    result = {
        "id": base.id,
        "name": base.name,
        "path": str(base.path),
        "bytes": base.path.stat().st_size,
        "family": base.family,
        "prediction_type": base.prediction_type,
        "sha256": digest,
        "registered_sha256": base.sha256,
        "sha256_matches": base.sha256 in {None, digest},
        "tensor_count": len(keys),
        "tensor_key_sample": keys[:20],
        "metadata": metadata,
    }
    if persist_sha and base.sha256 != digest:
        registry_path = root / "bases" / "registry.yaml"
        payload = read_yaml(registry_path)
        payload["bases"][base_id]["sha256"] = digest
        write_yaml_atomic(registry_path, payload)
    return result


def scan_bases(directory: Path, *, root: Path) -> list[dict[str, Any]]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise PipelineError(f"Base scan directory does not exist: {directory}")
    registered = {str(base.path.resolve()): base.id for base in load_base_registry(root).values()}
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "registered_as": registered.get(str(path.resolve())),
            "suggested_id": _suggest_id(path.stem),
        }
        for path in sorted(directory.rglob("*.safetensors"), key=lambda item: item.as_posix().casefold())
    ]


def _validate_base_id(value: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", value):
        raise ConfigurationError("Base id must contain 2–64 lowercase letters, numbers, '-' or '_'")


def _suggest_id(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return (value or "checkpoint")[:64]
