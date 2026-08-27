from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..models import OptionalBackendUnavailable


@dataclass(frozen=True)
class TagResult:
    ratings: Mapping[str, float]
    tags: Mapping[str, float]
    characters: Mapping[str, float]
    backend: str
    metadata: Mapping[str, str] = field(default_factory=dict)


class TaggerBackend(ABC):
    @abstractmethod
    def tag(self, image: Path) -> TagResult:
        raise NotImplementedError


class ImgutilsWdTagger(TaggerBackend):
    def __init__(self, *, model_name: str = "EVA02_Large", threshold: float = 0.35):
        self.model_name = model_name
        self.threshold = threshold

    def tag(self, image: Path) -> TagResult:
        try:
            from imgutils.tagging import get_wd14_tags
        except ImportError as exc:
            raise OptionalBackendUnavailable("imgutils WD14 tagging backend is unavailable") from exc
        ratings, tags, characters = get_wd14_tags(
            str(image),
            model_name=self.model_name,
            general_threshold=self.threshold,
            character_threshold=0.85,
        )
        return TagResult(
            ratings={str(key): float(value) for key, value in ratings.items()},
            tags={str(key): float(value) for key, value in tags.items()},
            characters={str(key): float(value) for key, value in characters.items()},
            backend="imgutils-wd14",
            metadata={"model_name": self.model_name, "runtime": "imgutils-managed ONNX"},
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

    def compare(self, image: Path) -> DualTagResult:
        stable = self.stable.tag(image)
        challenger = self.challenger.tag(image)
        stable_keys, challenger_keys = set(stable.tags), set(challenger.tags)
        shared = stable_keys & challenger_keys
        conflicts = sorted(
            key for key in shared if abs(float(stable.tags[key]) - float(challenger.tags[key])) >= self.conflict_delta
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
