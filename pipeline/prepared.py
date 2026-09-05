from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import write_json_atomic
from .models import PipelineError


@dataclass(frozen=True)
class PreparedGeneration:
    """Immutable prepared dataset generation selected for the project."""

    generation_id: str
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    legacy: bool = False


def generations_root(project_dir: Path) -> Path:
    return project_dir / "prepared" / "generations"


def current_pointer_path(project_dir: Path) -> Path:
    return project_dir / "prepared" / "current.json"


def generation_path(project_dir: Path, generation_id: str) -> Path:
    if not generation_id or any(part in {"", ".", ".."} for part in Path(generation_id).parts):
        raise PipelineError(f"Invalid prepared generation id: {generation_id!r}")
    root = generations_root(project_dir).resolve()
    candidate = (root / generation_id).resolve()
    if root not in candidate.parents:
        raise PipelineError(f"Prepared generation escapes its root: {generation_id!r}")
    return candidate


def set_current_generation(
    project_dir: Path,
    *,
    generation_id: str,
    manifest_hash: str,
    image_count: int,
) -> Path:
    path = current_pointer_path(project_dir)
    generation_root = generation_path(project_dir, generation_id)
    preview_rel: str | None = None
    manifest_path = generation_root / "manifest.json"
    if manifest_path.is_file():
        from .materialization.preview import write_generation_preview

        preview_path = write_generation_preview(project_dir, generation_root)
        preview_rel = preview_path.relative_to(project_dir).as_posix()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generation_id": generation_id,
        "manifest_hash": manifest_hash,
        "image_count": image_count,
    }
    if preview_rel is not None:
        payload["preview"] = preview_rel
    write_json_atomic(path, payload)
    return path


def load_current_generation(project_dir: Path) -> PreparedGeneration:
    pointer = current_pointer_path(project_dir)
    if pointer.is_file():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise PipelineError(f"Prepared current pointer is invalid: {pointer}: {exc}") from exc
        generation_id = str(payload.get("generation_id", ""))
        root = generation_path(project_dir, generation_id)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise PipelineError(
                f"Prepared generation {generation_id!r} is missing its manifest: {manifest_path}"
            )
        manifest = _read_manifest(manifest_path)
        expected = payload.get("manifest_hash")
        actual = manifest.get("manifest_hash") or manifest.get("input_hash")
        if expected and actual and expected != actual:
            raise PipelineError(
                f"Prepared generation pointer hash mismatch: expected {expected}, got {actual}"
            )
        return PreparedGeneration(
            generation_id=generation_id,
            root=root,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    # Compatibility path for projects created before immutable generations were introduced.
    legacy_manifest = project_dir / "prepared" / "manifest.json"
    if legacy_manifest.is_file():
        manifest = _read_manifest(legacy_manifest)
        generation_id = str(manifest.get("manifest_hash") or manifest.get("input_hash") or "legacy")
        return PreparedGeneration(
            generation_id=generation_id,
            root=legacy_manifest.parent,
            manifest_path=legacy_manifest,
            manifest=manifest,
            legacy=True,
        )
    raise PipelineError("Prepared dataset is missing; run prepare first")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise PipelineError(f"Prepared manifest is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"Prepared manifest must be a JSON object: {path}")
    return payload
