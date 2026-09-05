from __future__ import annotations

from .caption_core import *  # noqa: F401,F403
from .caption_core import run as _run_core
from ..semantic_factorization import apply_character_semantic_factorization
from ..semantic_runtime import _apply_semantic_captions
from ..style_caption_policy import apply_style_caption_policy
from ..target_policy import _apply_character_outfit_semantic_captions


def run(state, *args, **kwargs):
    """Transform captions, then apply target/semantic/style policies explicitly."""

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
