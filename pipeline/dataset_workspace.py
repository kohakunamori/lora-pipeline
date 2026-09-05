from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .config import (
    read_yaml,
    repository_root,
    sha256_file,
    stable_hash,
    write_json_atomic,
    write_yaml_atomic,
)
from .dataset.character import analyze_identity as analyze_character_identity
from .dataset.caption_cleaner import normalize_tag, parse_caption
from .dataset.duplicates import find_duplicates_from_paths
from .dataset.image_info import discover_images, inspect_image
from .dataset.tagger import CachedTagger, ImgutilsWdTagger, TaggerBackend
from .models import PipelineError, StateError
from .state import ProjectState, utc_now


DATASET_SCHEMA_VERSION = 1
DATASET_CURATION_SCHEMA_VERSION = 1
DATASET_DUPLICATE_ANALYZER_VERSION = 1
DATASET_IDENTITY_ANALYZER_VERSION = 1
_DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class DatasetItem:
    key: str
    source_id: str
    relative: Path
    image: Path
    caption: Path
    excluded: bool
    source_enabled: bool


class DatasetWorkspace:
    """Mutable, source-aware dataset workspace outside immutable training projects.

    A dataset can be curated repeatedly. Training projects consume an immutable
    snapshot of the currently enabled, non-excluded items, so later dataset edits
    never mutate an already-created project's raw/ directory.
    """

    def __init__(self, path: Path, payload: dict[str, Any]):
        self.path = path
        self.payload = payload
        self._normalize()

    @classmethod
    def create(
        cls,
        name: str,
        *,
        concept_type: str = "character",
        root: Path | None = None,
    ) -> "DatasetWorkspace":
        if not _DATASET_NAME.fullmatch(name) or name in {".", ".."}:
            raise StateError("Dataset name must be 1-64 letters, numbers, '.', '_' or '-'")
        if concept_type not in {"character", "style"}:
            raise PipelineError("Dataset concept type must be character or style")
        path = dataset_path(name, root=root)
        if path.exists():
            raise StateError(f"Dataset already exists: {name}")
        now = utc_now()
        payload: dict[str, Any] = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset": {
                "name": name,
                "type": concept_type,
                "created_at": now,
                "updated_at": now,
            },
            "sources": {},
        }
        for relative in ("sources", "review", "cache/tagger", "cache/work"):
            (path / relative).mkdir(parents=True, exist_ok=True)
        workspace = cls(path / "dataset.yaml", payload)
        workspace.save()
        return workspace

    @classmethod
    def load(cls, name: str, *, root: Path | None = None) -> "DatasetWorkspace":
        path = dataset_path(name, root=root) / "dataset.yaml"
        if not path.is_file():
            raise StateError(f"Dataset does not exist: {name}")
        return cls(path, read_yaml(path))

    @property
    def dataset_dir(self) -> Path:
        return self.path.parent

    @property
    def name(self) -> str:
        return str(self.payload["dataset"]["name"])

    @property
    def concept_type(self) -> str:
        return str(self.payload["dataset"]["type"])

    @property
    def sources(self) -> dict[str, dict[str, Any]]:
        return self.payload["sources"]

    def _normalize(self) -> None:
        self.payload.setdefault("schema_version", DATASET_SCHEMA_VERSION)
        dataset = self.payload.setdefault("dataset", {})
        if not dataset.get("name"):
            raise StateError(f"Dataset name is missing in {self.path}")
        dataset.setdefault("type", "character")
        dataset.setdefault("created_at", utc_now())
        dataset.setdefault("updated_at", dataset["created_at"])
        sources = self.payload.setdefault("sources", {})
        if not isinstance(sources, dict):
            raise StateError(f"Dataset sources must be a mapping in {self.path}")
        for source_id, source in sources.items():
            if not isinstance(source, dict):
                raise StateError(f"Invalid source record: {source_id}")
            source.setdefault("id", source_id)
            source.setdefault("label", source_id)
            source.setdefault("enabled", True)

    def save(self) -> None:
        self.payload["dataset"]["updated_at"] = utc_now()
        write_yaml_atomic(self.path, self.payload)

    def source_dir(self, source_id: str) -> Path:
        self._require_source(source_id)
        return self.dataset_dir / "sources" / source_id

    def source_images_dir(self, source_id: str) -> Path:
        return self.source_dir(source_id) / "images"

    def add_source_from_directory(
        self,
        directory: Path,
        *,
        kind: str,
        label: str | None = None,
        origin: str | None = None,
        parent_source: str | None = None,
        processing: Mapping[str, Any] | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            raise PipelineError(f"Source directory does not exist: {directory}")
        images = discover_images(directory)
        if not images:
            raise PipelineError(f"No supported images were found under {directory}")
        if parent_source is not None:
            self._require_source(parent_source)

        source_id = self._next_source_id(kind)
        target_root = self.dataset_dir / "sources" / source_id / "images"
        target_root.mkdir(parents=True, exist_ok=False)
        caption_count = 0
        try:
            for image in images:
                relative = image.relative_to(directory)
                target = target_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, target)
                sidecar = image.with_suffix(".txt")
                if sidecar.is_file():
                    shutil.copy2(sidecar, target.with_suffix(".txt"))
                    caption_count += 1
        except BaseException:
            shutil.rmtree(target_root.parent, ignore_errors=True)
            raise

        record: dict[str, Any] = {
            "id": source_id,
            "kind": kind,
            "label": (label or source_id).strip() or source_id,
            "enabled": bool(enabled),
            "origin": origin or str(directory),
            "parent_source": parent_source,
            "created_at": utc_now(),
            "imported_images": len(images),
            "imported_captions": caption_count,
            "processing": dict(processing or {}),
        }
        self.sources[source_id] = record
        self.save()
        return dict(record)

    def set_source_enabled(self, source_id: str, enabled: bool) -> None:
        source = self._require_source(source_id)
        source["enabled"] = bool(enabled)
        source["updated_at"] = utc_now()
        self.save()

    def items(
        self,
        *,
        source_id: str | None = None,
        include_disabled: bool = True,
        include_excluded: bool = True,
    ) -> list[DatasetItem]:
        source_ids = [source_id] if source_id else sorted(self.sources)
        exclusions = self._load_exclusions()
        result: list[DatasetItem] = []
        for current_id in source_ids:
            source = self._require_source(current_id)
            source_enabled = bool(source.get("enabled", True))
            if not include_disabled and not source_enabled:
                continue
            root = self.source_images_dir(current_id)
            for image in discover_images(root):
                relative = image.relative_to(root)
                key = f"{current_id}/{relative.as_posix()}"
                excluded = key in exclusions
                if excluded and not include_excluded:
                    continue
                result.append(
                    DatasetItem(
                        key=key,
                        source_id=current_id,
                        relative=relative,
                        image=image,
                        caption=image.with_suffix(".txt"),
                        excluded=excluded,
                        source_enabled=source_enabled,
                    )
                )
        return sorted(result, key=lambda item: item.key.casefold())

    def summary(self) -> dict[str, Any]:
        all_items = self.items(include_disabled=True, include_excluded=True)
        active = [item for item in all_items if item.source_enabled and not item.excluded]
        return {
            "name": self.name,
            "type": self.concept_type,
            "sources": len(self.sources),
            "enabled_sources": sum(bool(source.get("enabled", True)) for source in self.sources.values()),
            "images": len(all_items),
            "active_images": len(active),
            "excluded_images": sum(item.excluded for item in all_items),
            "captioned_active_images": sum(item.caption.is_file() for item in active),
            "updated_at": self.payload["dataset"].get("updated_at"),
        }

    def exclude(
        self,
        keys: Sequence[str],
        *,
        reason: str = "manual review",
        mode: str = "manual",
    ) -> int:
        existing = {item.key for item in self.items(include_disabled=True, include_excluded=True)}
        unknown = [key for key in keys if key not in existing]
        if unknown:
            raise PipelineError("Cannot exclude unknown dataset item(s): " + ", ".join(unknown[:5]))
        payload = self._load_exclusions()
        changed = 0
        for key in keys:
            if key in payload:
                continue
            payload[key] = {
                "reason": reason,
                "mode": mode,
                "excluded_at": utc_now(),
            }
            changed += 1
        self._save_exclusions(payload)
        if changed:
            self.save()
        return changed

    def restore(self, keys: Sequence[str]) -> int:
        payload = self._load_exclusions()
        changed = 0
        for key in keys:
            if key in payload:
                del payload[key]
                changed += 1
        self._save_exclusions(payload)
        if changed:
            self.save()
        return changed

    def audit(self, *, source_id: str | None = None) -> dict[str, Any]:
        items = self.items(
            source_id=source_id,
            include_disabled=False,
            include_excluded=True,
        )
        records: list[dict[str, Any]] = []
        by_sha: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            root = self.source_images_dir(item.source_id)
            inspected = inspect_image(item.image, root)
            inspected.update(
                {
                    "key": item.key,
                    "source_id": item.source_id,
                    "excluded": item.excluded,
                    "flags": [],
                    "safe_exclude": False,
                }
            )
            flags: list[dict[str, str]] = inspected["flags"]
            if inspected.get("corrupt"):
                flags.append({"code": "corrupt", "severity": "reject"})
                inspected["safe_exclude"] = True
            else:
                ratio = float(inspected.get("aspect_ratio") or 1.0)
                if inspected.get("very_small"):
                    flags.append({"code": "very_small", "severity": "review"})
                if ratio < 0.25 or ratio > 4.0:
                    flags.append({"code": "extreme_aspect_ratio", "severity": "review"})
                if inspected.get("animated"):
                    flags.append({"code": "animated_image", "severity": "review"})
            records.append(inspected)
            digest = str(inspected.get("sha256") or "")
            if digest:
                by_sha.setdefault(digest, []).append(inspected)

        duplicate_images = 0
        for group in by_sha.values():
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda record: str(record["key"]).casefold())
            canonical = str(ordered[0]["key"])
            for record in ordered[1:]:
                flags = record["flags"]
                flags.append(
                    {
                        "code": "exact_duplicate",
                        "severity": "reject",
                        "canonical": canonical,
                    }
                )
                record["safe_exclude"] = True
                duplicate_images += 1

        payload = {
            "schema_version": 1,
            "dataset": self.name,
            "source_id": source_id,
            "generated_at": utc_now(),
            "records": records,
            "summary": {
                "images": len(records),
                "already_excluded": sum(bool(record.get("excluded")) for record in records),
                "flagged": sum(bool(record.get("flags")) for record in records),
                "safe_exclude_suggestions": sum(bool(record.get("safe_exclude")) for record in records),
                "exact_duplicate_images": duplicate_images,
                "corrupt_images": sum(bool(record.get("corrupt")) for record in records),
                "review_only": sum(
                    bool(record.get("flags")) and not bool(record.get("safe_exclude"))
                    for record in records
                ),
            },
        }
        name = f"audit-{source_id}.json" if source_id else "audit.json"
        write_json_atomic(self.dataset_dir / "review" / name, payload)
        return payload

    def apply_safe_audit_exclusions(self, *, source_id: str | None = None) -> dict[str, Any]:
        audit = self.audit(source_id=source_id)
        records = [
            record
            for record in audit["records"]
            if record.get("safe_exclude") and not record.get("excluded")
        ]
        changed = 0
        for record in records:
            codes = [str(flag.get("code")) for flag in record.get("flags", [])]
            changed += self.exclude(
                [str(record["key"])],
                reason="automatic safe exclusion: " + ", ".join(codes),
                mode="automatic_safe",
            )
        return {"excluded": changed, "audit": audit}

    def analyze_duplicates(self, *, phash_distance: int = 6) -> dict[str, Any]:
        """Analyze exact/perceptual duplicates for the current active image set."""

        if phash_distance < 0:
            raise PipelineError("pHash distance must be non-negative")
        items = self.items(include_disabled=False, include_excluded=False)
        if not items:
            raise PipelineError("Dataset has no enabled, non-excluded images")

        manifest = find_duplicates_from_paths(
            ((item.key, item.image) for item in items),
            phash_distance=phash_distance,
        )
        image_set_hash = stable_hash(
            [
                {"key": str(record["path"]), "sha256": str(record["sha256"])}
                for record in manifest["images"]
            ]
        )
        manifest.update(
            {
                "dataset": self.name,
                "generated_at": utc_now(),
                "image_set_hash": image_set_hash,
            }
        )
        path = self.dataset_dir / "review" / "duplicates" / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, manifest)
        self._record_curation_analysis(
            "dedup",
            analyzer_version=DATASET_DUPLICATE_ANALYZER_VERSION,
            image_set_hash=image_set_hash,
            parameters={"phash_distance": int(phash_distance)},
            manifest=path,
            summary=dict(manifest["summary"]),
        )
        return manifest

    def analyze_identity(self, *, min_samples: int = 2) -> dict[str, Any]:
        """Run CCIP identity analysis for the current active character images."""

        if self.concept_type != "character":
            raise PipelineError("Identity analysis is only applicable to character datasets")
        if min_samples < 1:
            raise PipelineError("Identity min_samples must be at least 1")
        items = self.items(include_disabled=False, include_excluded=False)
        if not items:
            raise PipelineError("Dataset has no enabled, non-excluded images")

        image_set_hash = self._active_image_set_hash(items)
        result = analyze_character_identity(
            [item.image for item in items],
            min_samples=min_samples,
        )
        key_by_path = {str(item.image): item.key for item in items}
        for field in ("main_cluster", "possible_outliers", "possible_mixed_characters"):
            result[field] = [
                key_by_path.get(str(path), str(path))
                for path in result.get(field, [])
            ]
        summary = {
            "main_cluster": len(result.get("main_cluster", [])),
            "possible_outliers": len(result.get("possible_outliers", [])),
            "possible_mixed_characters": len(
                result.get("possible_mixed_characters", [])
            ),
        }
        result.update(
            {
                "dataset": self.name,
                "generated_at": utc_now(),
                "image_set_hash": image_set_hash,
                "summary": summary,
            }
        )
        path = self.dataset_dir / "review" / "outliers" / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, result)
        self._record_curation_analysis(
            "identity",
            analyzer_version=DATASET_IDENTITY_ANALYZER_VERSION,
            image_set_hash=image_set_hash,
            parameters={"min_samples": int(min_samples)},
            manifest=path,
            summary=summary,
        )
        return result

    def curation_status(
        self,
        *,
        image_set_hash: str | None = None,
        phash_distance: int = 6,
        identity_min_samples: int = 2,
    ) -> dict[str, Any]:
        """Report whether reusable curation analyses match the active image set."""

        current_hash = image_set_hash or self._active_image_set_hash()
        payload = self._load_curation_manifest()
        analyses = dict(payload.get("analyses", {}))
        expected = {
            "dedup": {
                "analyzer_version": DATASET_DUPLICATE_ANALYZER_VERSION,
                "parameters": {"phash_distance": int(phash_distance)},
            },
            "identity": {
                "analyzer_version": DATASET_IDENTITY_ANALYZER_VERSION,
                "parameters": {"min_samples": int(identity_min_samples)},
            },
        }
        resolved: dict[str, Any] = {}
        for name, expectation in expected.items():
            record = dict(analyses.get(name, {}))
            applicable = not (name == "identity" and self.concept_type != "character")
            fresh = bool(
                applicable
                and record
                and int(record.get("analyzer_version", -1))
                == int(expectation["analyzer_version"])
                and dict(record.get("parameters", {})) == expectation["parameters"]
                and str(record.get("image_set_hash") or "") == current_hash
                and self._curation_manifest_target_exists(record)
            )
            resolved[name] = {
                **record,
                "applicable": applicable,
                "fresh": fresh,
                "expected_analyzer_version": expectation["analyzer_version"],
                "expected_parameters": expectation["parameters"],
            }
        ready = resolved["dedup"]["fresh"] and (
            self.concept_type != "character" or resolved["identity"]["fresh"]
        )
        return {
            "schema_version": DATASET_CURATION_SCHEMA_VERSION,
            "dataset": self.name,
            "image_set_hash": current_hash,
            "ready": bool(ready),
            "analyses": resolved,
        }

    def auto_tag(
        self,
        *,
        source_id: str | None = None,
        threshold: float = 0.35,
        overwrite: bool = False,
        tagger: TaggerBackend | None = None,
    ) -> dict[str, Any]:
        if not 0.0 <= threshold <= 1.0:
            raise PipelineError("Tag threshold must be between 0 and 1")
        tagger = tagger or CachedTagger(
            ImgutilsWdTagger(model_name="EVA02_Large", threshold=threshold),
            self.dataset_dir / "cache" / "tagger",
        )
        records: list[dict[str, Any]] = []
        tagged = 0
        skipped_existing = 0
        for item in self.items(
            source_id=source_id,
            include_disabled=False,
            include_excluded=False,
        ):
            if item.caption.is_file() and not overwrite:
                skipped_existing += 1
                continue
            result = tagger.tag(item.image)
            tags = [
                str(tag)
                for tag, score in sorted(
                    result.tags.items(),
                    key=lambda pair: (-float(pair[1]), str(pair[0]).casefold()),
                )
                if float(score) >= threshold
            ]
            item.caption.write_text(", ".join(tags) + "\n", encoding="utf-8")
            tagged += 1
            records.append(
                {
                    "key": item.key,
                    "backend": result.backend,
                    "tags": len(tags),
                    "cache_hit": bool(result.metadata.get("cache_hit")),
                    "character_suggestions": sorted(
                        str(name)
                        for name, score in result.characters.items()
                        if float(score) >= 0.85
                    ),
                }
            )
        payload = {
            "schema_version": 1,
            "dataset": self.name,
            "source_id": source_id,
            "threshold": threshold,
            "overwrite": overwrite,
            "tagged": tagged,
            "skipped_existing": skipped_existing,
            "records": records,
            "finished_at": utc_now(),
        }
        write_json_atomic(self.dataset_dir / "review" / "tagging-last.json", payload)
        if tagged:
            self.save()
        return payload

    def caption_text(self, key: str) -> str:
        item = self._item_by_key(key)
        if not item.caption.is_file():
            return ""
        return item.caption.read_text(encoding="utf-8", errors="replace").strip()

    def replace_caption(self, key: str, text: str) -> str:
        item = self._item_by_key(key)
        tags = _unique_tags(parse_caption(text))
        if tags:
            item.caption.write_text(", ".join(tags) + "\n", encoding="utf-8")
        else:
            item.caption.unlink(missing_ok=True)
        self.save()
        return ", ".join(tags)

    def add_tags(self, key: str, tags: Iterable[str]) -> str:
        existing = parse_caption(self.caption_text(key))
        return self.replace_caption(key, ", ".join([*existing, *tags]))

    def remove_tags(self, key: str, tags: Iterable[str]) -> str:
        remove = {normalize_tag(tag) for tag in tags if normalize_tag(tag)}
        existing = [
            tag for tag in parse_caption(self.caption_text(key)) if normalize_tag(tag) not in remove
        ]
        return self.replace_caption(key, ", ".join(existing))

    def snapshot(self) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        inspection_records: list[dict[str, Any]] = []
        for item in self.items(include_disabled=False, include_excluded=False):
            caption_sha = None
            if item.caption.is_file():
                caption_sha = hashlib.sha256(item.caption.read_bytes()).hexdigest()
            inspected = inspect_image(item.image, self.source_images_dir(item.source_id))
            inspected["path"] = item.key
            inspected["caption"] = item.caption.is_file()
            if inspected.get("corrupt"):
                raise PipelineError(
                    f"Dataset contains a corrupt active image: {item.key}. "
                    "Run Dataset audit and exclude it before creating a Project."
                )
            inspection_records.append(inspected)
            records.append(
                {
                    "key": item.key,
                    "source_id": item.source_id,
                    "relative": item.relative.as_posix(),
                    "image_sha256": str(inspected["sha256"]),
                    "caption_sha256": caption_sha,
                }
            )
        if not records:
            raise PipelineError("Dataset has no enabled, non-excluded images")

        widths = [int(record["width"]) for record in inspection_records]
        heights = [int(record["height"]) for record in inspection_records]
        megapixels = [float(record["megapixels"]) for record in inspection_records]
        formats = Counter(
            str(record.get("format") or "unknown") for record in inspection_records
        )
        inspection_manifest = {
            "schema_version": 1,
            "root": str(self.dataset_dir),
            "input_hash": stable_hash(
                [
                    {
                        "path": record["path"],
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                    }
                    for record in inspection_records
                ]
            ),
            "summary": {
                "image_count": len(inspection_records),
                "valid_images": len(inspection_records),
                "corrupt_images": 0,
                "formats": dict(sorted(formats.items())),
                "caption_count": sum(
                    bool(record["caption"]) for record in inspection_records
                ),
                "very_small_images": sum(
                    bool(record.get("very_small")) for record in inspection_records
                ),
                "alpha_images": sum(
                    bool(record.get("alpha")) for record in inspection_records
                ),
                "animated_images": sum(
                    bool(record.get("animated")) for record in inspection_records
                ),
                "exif_oriented_images": sum(
                    record.get("exif_orientation") not in (None, 1)
                    for record in inspection_records
                ),
                "width": _range_summary(widths),
                "height": _range_summary(heights),
                "megapixels": _range_summary(megapixels),
            },
            "images": inspection_records,
        }
        image_set_hash = _image_set_hash_from_snapshot_records(records)
        source_ids = sorted({str(record["source_id"]) for record in records})
        basis = {
            "schema_version": 1,
            "dataset": self.name,
            "type": self.concept_type,
            "sources": [
                {
                    "id": source_id,
                    "kind": self.sources[source_id].get("kind"),
                    "label": self.sources[source_id].get("label"),
                    "parent_source": self.sources[source_id].get("parent_source"),
                }
                for source_id in source_ids
            ],
            "images": records,
            "image_set_hash": image_set_hash,
            "inspection": inspection_manifest,
        }
        return {
            **basis,
            "snapshot_hash": stable_hash(basis),
            "image_count": len(records),
            "caption_count": sum(record["caption_sha256"] is not None for record in records),
            "created_at": utc_now(),
        }

    def export_active(self, destination: Path) -> dict[str, Any]:
        snapshot = self.snapshot()
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise PipelineError(f"Dataset export destination must be empty: {destination}")
        for item in self.items(include_disabled=False, include_excluded=False):
            target = destination / item.source_id / item.relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.image, target)
            if item.caption.is_file():
                shutil.copy2(item.caption, target.with_suffix(".txt"))
        return snapshot

    def export_source_active(self, source_id: str, destination: Path) -> int:
        self._require_source(source_id)
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        count = 0
        for item in self.items(
            source_id=source_id,
            include_disabled=True,
            include_excluded=False,
        ):
            target = destination / item.relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.image, target)
            if item.caption.is_file():
                shutil.copy2(item.caption, target.with_suffix(".txt"))
            count += 1
        return count

    def _active_image_set_hash(
        self,
        items: Sequence[DatasetItem] | None = None,
    ) -> str:
        active = list(
            items
            if items is not None
            else self.items(include_disabled=False, include_excluded=False)
        )
        return stable_hash(
            [
                {"key": item.key, "sha256": sha256_file(item.image)}
                for item in active
            ]
        )

    def _curation_manifest_path(self) -> Path:
        return self.dataset_dir / "review" / "curation-manifest.json"

    def _load_curation_manifest(self) -> dict[str, Any]:
        path = self._curation_manifest_path()
        if not path.is_file():
            return {
                "schema_version": DATASET_CURATION_SCHEMA_VERSION,
                "dataset": self.name,
                "analyses": {},
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StateError(f"Invalid Dataset curation manifest: {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise StateError(f"Invalid Dataset curation manifest: {path}")
        if int(payload.get("schema_version", -1)) != DATASET_CURATION_SCHEMA_VERSION:
            return {
                "schema_version": DATASET_CURATION_SCHEMA_VERSION,
                "dataset": self.name,
                "analyses": {},
            }
        analyses = payload.setdefault("analyses", {})
        if not isinstance(analyses, dict):
            raise StateError(f"Invalid Dataset curation analyses: {path}")
        return payload

    def _record_curation_analysis(
        self,
        name: str,
        *,
        analyzer_version: int,
        image_set_hash: str,
        parameters: Mapping[str, Any],
        manifest: Path,
        summary: Mapping[str, Any],
    ) -> None:
        payload = self._load_curation_manifest()
        analyses = payload.setdefault("analyses", {})
        analyses[name] = {
            "analyzer_version": int(analyzer_version),
            "image_set_hash": image_set_hash,
            "parameters": dict(parameters),
            "manifest": manifest.relative_to(self.dataset_dir).as_posix(),
            "summary": dict(summary),
            "finished_at": utc_now(),
        }
        payload["updated_at"] = utc_now()
        write_json_atomic(self._curation_manifest_path(), payload)

    def _curation_manifest_target_exists(self, record: Mapping[str, Any]) -> bool:
        relative = str(record.get("manifest") or "").strip()
        if not relative:
            return False
        target = (self.dataset_dir / relative).resolve()
        try:
            target.relative_to(self.dataset_dir.resolve())
        except ValueError:
            return False
        return target.is_file()

    def _load_exclusions(self) -> dict[str, dict[str, Any]]:
        path = self.dataset_dir / "review" / "exclusions.yaml"
        if not path.is_file():
            return {}
        payload = read_yaml(path)
        raw = payload.get("excluded", {})
        if isinstance(raw, list):
            return {
                str(key): {"reason": "legacy exclusion", "mode": "manual"}
                for key in raw
            }
        if not isinstance(raw, dict):
            raise StateError(f"Invalid dataset exclusions file: {path}")
        return {str(key): dict(value or {}) for key, value in raw.items()}

    def _save_exclusions(self, exclusions: Mapping[str, Mapping[str, Any]]) -> None:
        write_yaml_atomic(
            self.dataset_dir / "review" / "exclusions.yaml",
            {"excluded": dict(sorted(exclusions.items()))},
        )

    def _next_source_id(self, kind: str) -> str:
        prefix = re.sub(r"[^a-z0-9]+", "-", kind.casefold()).strip("-") or "source"
        prefix = prefix[:20]
        index = 1
        while f"{prefix}-{index:03d}" in self.sources:
            index += 1
        return f"{prefix}-{index:03d}"

    def _require_source(self, source_id: str) -> dict[str, Any]:
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise StateError(f"Unknown dataset source: {source_id}") from exc

    def _item_by_key(self, key: str) -> DatasetItem:
        for item in self.items(include_disabled=True, include_excluded=True):
            if item.key == key:
                return item
        raise PipelineError(f"Unknown dataset item: {key}")


def datasets_root(root: Path | None = None) -> Path:
    return (root or repository_root()) / "datasets"


def dataset_path(name: str, *, root: Path | None = None) -> Path:
    if not _DATASET_NAME.fullmatch(name) or name in {".", ".."}:
        raise StateError("Dataset name must be 1-64 letters, numbers, '.', '_' or '-'")
    return datasets_root(root) / name


def list_datasets(*, root: Path | None = None) -> list[DatasetWorkspace]:
    base = datasets_root(root)
    if not base.is_dir():
        return []
    result: list[DatasetWorkspace] = []
    for path in sorted(base.iterdir(), key=lambda item: item.name.casefold()):
        if (path / "dataset.yaml").is_file():
            result.append(DatasetWorkspace(path / "dataset.yaml", read_yaml(path / "dataset.yaml")))
    return result


def create_project_from_dataset(
    workspace: DatasetWorkspace,
    *,
    name: str,
    base: str,
    trigger: str,
    strategy: str,
    images_seen: int,
    hardware: str = "v100_16gb",
    root: Path | None = None,
) -> ProjectState:
    """Create an immutable Project raw snapshot from the mutable Dataset workspace."""

    root = root or repository_root()
    cache = root / "cache" / "dataset-exports"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{workspace.name}-", dir=cache) as temporary:
        export_dir = Path(temporary) / "active"
        snapshot = workspace.export_active(export_dir)
        from .service import create_project

        state = create_project(
            name=name,
            concept_type=workspace.concept_type,
            base=base,
            trigger=trigger,
            strategy=strategy,
            dataset=export_dir,
            images_seen=images_seen,
            hardware=hardware,
            root=root,
        )
    project = state.payload["project"]
    project["raw_source"] = str(workspace.dataset_dir)
    project["dataset_snapshot"] = snapshot

    frozen_inspection = dict(snapshot["inspection"])
    frozen_inspection["root"] = str(state.project_dir / "raw")
    inspection_path = state.project_dir / "dataset-manifest.json"
    write_json_atomic(inspection_path, frozen_inspection)

    curation = workspace.curation_status(
        image_set_hash=str(snapshot["image_set_hash"]),
    )
    project["dataset_curation"] = curation
    frozen_targets = {
        "dedup": "review/duplicates/manifest.json",
        "identity": "review/outliers/manifest.json",
    }
    for analysis_name, target_relative in frozen_targets.items():
        record = curation["analyses"].get(analysis_name)
        if not isinstance(record, dict) or not record.get("fresh"):
            continue
        source_relative = str(record.get("manifest") or "")
        if not source_relative:
            continue
        source_manifest = workspace.dataset_dir / source_relative
        target_manifest = state.project_dir / target_relative
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_manifest, target_manifest)
        record["frozen"] = True
        record["frozen_manifest"] = target_relative
        record["source_manifest"] = source_relative

    preferences = dict(project.get("interactive_preferences", {}))
    if snapshot["caption_count"] == snapshot["image_count"]:
        preferences.setdefault("caption_mode", "existing_taglist_clean")
    project["interactive_preferences"] = preferences
    state.save()
    return state


def _range_summary(values: list[int] | list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": median(values), "max": max(values)}


def _image_set_hash_from_snapshot_records(records: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash(
        [
            {
                "key": str(record["key"]),
                "sha256": str(record["image_sha256"]),
            }
            for record in records
        ]
    )


def parse_number_selection(text: str, *, minimum: int = 1, maximum: int) -> list[int]:
    """Parse compact interactive selections such as ``1,3-5,9``."""

    selected: set[int] = set()
    for raw in text.replace("，", ",").split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            if not left.strip().isdigit() or not right.strip().isdigit():
                raise PipelineError(f"Invalid number range: {token}")
            start, end = int(left), int(right)
            if start > end:
                start, end = end, start
            values = range(start, end + 1)
        else:
            if not token.isdigit():
                raise PipelineError(f"Invalid item number: {token}")
            values = [int(token)]
        for value in values:
            if value < minimum or value > maximum:
                raise PipelineError(f"Item number out of range: {value} (expected {minimum}-{maximum})")
            selected.add(value)
    if not selected:
        raise PipelineError("No item numbers were selected")
    return sorted(selected)


def _unique_tags(tags: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = tag.strip()
        normalized = normalize_tag(cleaned)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(cleaned)
    return result
