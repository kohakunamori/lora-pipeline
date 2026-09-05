from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .caption_core import *  # noqa: F401,F403
from .caption_core import run as _run_core
from ..caption_policy import apply_target_caption_policy
from ..dataset.tagger import ImgutilsWdTagger, TagResult, TaggerBackend
from ..style_caption_policy import apply_style_caption_policy


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
    """Compile raw tags into target-aware training captions.

    Dataset identity/outfit semantic metadata no longer owns runtime caption
    composition. The protected TriggerPolicy prefix comes from TrainingConfig;
    CaptionPolicy only decides which stable visual attributes remain explicit.
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
    result = apply_target_caption_policy(state, result)
    return apply_style_caption_policy(state, result)
