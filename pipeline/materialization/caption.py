from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .caption_core import *  # noqa: F401,F403
from .caption_core import run as _run_core
from ..dataset.tagger import ImgutilsWdTagger, TagResult, TaggerBackend
from ..semantic_factorization import apply_character_semantic_factorization
from ..semantic_runtime import _apply_semantic_captions
from ..style_caption_policy import apply_style_caption_policy
from ..target_policy import _apply_character_outfit_semantic_captions


class _VisualPathTagger(TaggerBackend):
    """Route raw-image tag requests to the exact materialized visual pixels."""

    def __init__(
        self,
        backend: TaggerBackend,
        overrides: Mapping[Path, Path],
    ) -> None:
        self.backend = backend
        self.overrides = {
            source.resolve(): destination.resolve()
            for source, destination in overrides.items()
        }

    def cache_identity(self) -> dict[str, object]:
        return {
            "adapter": "materialized-visual-v1",
            "backend": self.backend.cache_identity(),
        }

    def tag(self, image: Path) -> TagResult:
        resolved = self.overrides.get(image.resolve(), image)
        return self.backend.tag(resolved)


def run(state, *args, **kwargs):
    """Transform captions, then apply target/semantic/style policies explicitly.

    Materialization may provide ``tag_image_overrides`` so generate/hybrid modes
    tag the crop+normalized visual that will actually be trained rather than the
    wider raw source image. Existing-caption modes remain byte/semantic compatible.
    """

    visual_overrides = kwargs.pop("tag_image_overrides", None)
    mode = str(kwargs.get("mode", "generate"))
    if visual_overrides and mode in {"generate", "hybrid"}:
        supplied = kwargs.get("tagger")
        if supplied is None:
            threshold = float(kwargs.get("threshold", 0.35))
            supplied = ImgutilsWdTagger(model_name="EVA02_Large", threshold=threshold)
        kwargs["tagger"] = _VisualPathTagger(supplied, visual_overrides)

    result = _run_core(state, *args, **kwargs)
    project = state.payload.get("project", {})
    target_type = str(project.get("training_target_type", project.get("type", "")))
    if target_type == "character_outfit":
        result = _apply_character_outfit_semantic_captions(state, result)
    else:
        result = _apply_semantic_captions(state, result)
        if target_type == "character":
            result = apply_character_semantic_factorization(state, result)
    return apply_style_caption_policy(state, result)
