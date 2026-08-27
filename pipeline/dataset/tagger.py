from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..config import sha256_file, stable_hash, write_json_atomic
from ..models import OptionalBackendUnavailable, PipelineError


@dataclass(frozen=True)
class TagResult:
    ratings: Mapping[str, float]
    tags: Mapping[str, float]
    characters: Mapping[str, float]
    backend: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TaggerBackend(ABC):
    @abstractmethod
    def tag(self, image: Path) -> TagResult:
        raise NotImplementedError

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "config": {
                key: value
                for key, value in vars(self).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        }


class CachedTagger(TaggerBackend):
    """Cache raw tagger output by image content and backend configuration."""

    def __init__(self, backend: TaggerBackend, cache_dir: Path):
        self.backend = backend
        self.cache_dir = cache_dir

    def cache_identity(self) -> Mapping[str, Any]:
        return {"cached": self.backend.cache_identity(), "schema_version": 1}

    def tag(self, image: Path) -> TagResult:
        image_sha = sha256_file(image)
        identity = self.backend.cache_identity()
        key = stable_hash(
            {
                "schema_version": 1,
                "image_sha256": image_sha,
                "backend": identity,
            }
        )
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return _tag_result_from_json(payload["result"], cache_hit=True, cache_key=key)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                path.unlink(missing_ok=True)
        result = self.backend.tag(image)
        write_json_atomic(
            path,
            {
                "schema_version": 1,
                "cache_key": key,
                "image": str(image),
                "image_sha256": image_sha,
                "backend_identity": identity,
                "result": _tag_result_to_json(result),
            },
        )
        return TagResult(
            ratings=result.ratings,
            tags=result.tags,
            characters=result.characters,
            backend=result.backend,
            metadata={**dict(result.metadata), "cache_hit": False, "cache_key": key},
        )


class ImgutilsWdTagger(TaggerBackend):
    def __init__(
        self,
        *,
        model_name: str = "EVA02_Large",
        threshold: float = 0.35,
        character_threshold: float = 0.85,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.character_threshold = character_threshold

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "backend": "imgutils-wd14",
            "model_name": self.model_name,
            "general_threshold": self.threshold,
            "character_threshold": self.character_threshold,
        }

    def tag(self, image: Path) -> TagResult:
        try:
            from imgutils.tagging import get_wd14_tags
        except ImportError as exc:
            raise OptionalBackendUnavailable("imgutils WD14 tagging backend is unavailable") from exc
        ratings, tags, characters = get_wd14_tags(
            str(image),
            model_name=self.model_name,
            general_threshold=self.threshold,
            character_threshold=self.character_threshold,
        )
        return TagResult(
            ratings={str(key): float(value) for key, value in ratings.items()},
            tags={str(key): float(value) for key, value in tags.items()},
            characters={str(key): float(value) for key, value in characters.items()},
            backend="imgutils-wd14",
            metadata={
                "model_name": self.model_name,
                "runtime": "imgutils-managed ONNX",
                "general_threshold": self.threshold,
                "character_threshold": self.character_threshold,
            },
        )


@dataclass(frozen=True)
class DualTagResult:
    stable: TagResult
    challenger: TagResult
    agreement: list[str]
    stable_only: list[str]
    challenger_only: list[str]
    conflicts: list[str]


class DualTagger(TaggerBackend):
    def __init__(self, stable: TaggerBackend, challenger: TaggerBackend, *, conflict_delta: float = 0.35):
        self.stable = stable
        self.challenger = challenger
        self.conflict_delta = conflict_delta

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "backend": "dual",
            "stable": self.stable.cache_identity(),
            "challenger": self.challenger.cache_identity(),
            "conflict_delta": self.conflict_delta,
        }

    def compare(self, image: Path) -> DualTagResult:
        stable = self.stable.tag(image)
        challenger = self.challenger.tag(image)
        stable_keys, challenger_keys = set(stable.tags), set(challenger.tags)
        shared = stable_keys & challenger_keys
        conflicts = sorted(
            key
            for key in shared
            if abs(float(stable.tags[key]) - float(challenger.tags[key])) >= self.conflict_delta
        )
        return DualTagResult(
            stable=stable,
            challenger=challenger,
            agreement=sorted(shared - set(conflicts)),
            stable_only=sorted(stable_keys - challenger_keys),
            challenger_only=sorted(challenger_keys - stable_keys),
            conflicts=conflicts,
        )

    def tag(self, image: Path) -> TagResult:
        return self.compare(image).stable


def _tag_result_to_json(result: TagResult) -> dict[str, Any]:
    return {
        "ratings": dict(result.ratings),
        "tags": dict(result.tags),
        "characters": dict(result.characters),
        "backend": result.backend,
        "metadata": dict(result.metadata),
    }


def _tag_result_from_json(
    payload: Mapping[str, Any], *, cache_hit: bool, cache_key: str
) -> TagResult:
    try:
        return TagResult(
            ratings={str(key): float(value) for key, value in payload.get("ratings", {}).items()},
            tags={str(key): float(value) for key, value in payload.get("tags", {}).items()},
            characters={
                str(key): float(value) for key, value in payload.get("characters", {}).items()
            },
            backend=str(payload["backend"]),
            metadata={
                **dict(payload.get("metadata", {})),
                "cache_hit": cache_hit,
                "cache_key": cache_key,
            },
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"Invalid cached tagger result: {exc}") from exc
