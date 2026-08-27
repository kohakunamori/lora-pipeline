from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import load_base_registry, read_yaml, repository_root, sha256_file, write_yaml_atomic
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
        "sha256_stat": None,
        "enabled": enabled,
        "generation_defaults": {"sampler": "euler_a", "cfg": 4.5, "steps": 28},
    }
    write_yaml_atomic(registry_path, payload)
    return load_base_registry(root)[base_id]


def checkpoint_stat_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
    }


def resolve_base_sha256(
    base_id: str,
    *,
    root: Path | None = None,
    full: bool = False,
    persist: bool = True,
) -> tuple[str, bool, dict[str, int]]:
    """Return checkpoint SHA256 and whether a matching stat cache was reused."""

    root = root or repository_root()
    registry_path = root / "bases" / "registry.yaml"
    payload = read_yaml(registry_path) if registry_path.exists() else {"bases": {}}
    try:
        item = payload["bases"][base_id]
    except (KeyError, TypeError) as exc:
        raise PipelineError(f"Unknown base id: {base_id}") from exc
    path = Path(str(item.get("path", ""))).expanduser()
    if not path.is_file():
        raise PipelineError(f"Checkpoint does not exist: {path}")
    signature = checkpoint_stat_signature(path)
    cached_sha = item.get("sha256")
    cached_stat = item.get("sha256_stat")
    if not full and cached_sha and cached_stat == signature:
        return str(cached_sha), True, signature
    digest = sha256_file(path)
    if persist:
        item["sha256"] = digest
        item["sha256_stat"] = signature
        write_yaml_atomic(registry_path, payload)
    return digest, False, signature


def inspect_base(
    base_id: str,
    *,
    root: Path,
    persist_sha: bool = True,
    full_hash: bool = False,
) -> dict[str, Any]:
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
    digest, cache_reused, stat_signature = resolve_base_sha256(
        base_id, root=root, full=full_hash, persist=persist_sha
    )
    return {
        "id": base.id,
        "name": base.name,
        "path": str(base.path),
        "bytes": base.path.stat().st_size,
        "family": base.family,
        "prediction_type": base.prediction_type,
        "sha256": digest,
        "registered_sha256": base.sha256,
        "sha256_matches": base.sha256 in {None, digest},
        "sha256_cache_reused": cache_reused,
        "sha256_stat": stat_signature,
        "tensor_count": len(keys),
        "tensor_key_sample": keys[:20],
        "metadata": metadata,
    }


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
