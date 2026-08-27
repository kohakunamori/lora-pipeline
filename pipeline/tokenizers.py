from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .dataset.caption_cleaner import estimate_tokens, parse_caption


SDXL_TOKENIZER_IDS = (
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
)


@dataclass(frozen=True)
class TokenCounts:
    clip_l: int
    clip_g: int
    exact: bool
    backend: str
    error: str | None = None

    @property
    def maximum(self) -> int:
        return max(self.clip_l, self.clip_g)


def count_sdxl_tokens(text: str) -> TokenCounts:
    """Count content tokens with the two tokenizers used by SDXL.

    Tokenizer assets are loaded from the existing Hugging Face cache. A clearly
    marked estimate is returned only when the validated runtime has not cached
    those assets yet; training can then download them before preflight is rerun.
    """

    try:
        tokenizers = _load_tokenizers()
        counts = [
            len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
            for tokenizer in tokenizers
        ]
        return TokenCounts(
            clip_l=int(counts[0]),
            clip_g=int(counts[1]),
            exact=True,
            backend="transformers.CLIPTokenizer",
        )
    except Exception as exc:  # optional runtime/cache failure, reported to caller
        estimated = estimate_tokens(parse_caption(text))
        return TokenCounts(
            clip_l=estimated,
            clip_g=estimated,
            exact=False,
            backend="heuristic-fallback",
            error=f"{type(exc).__name__}: {exc}",
        )


@lru_cache(maxsize=1)
def _load_tokenizers() -> tuple[Any, Any]:
    from transformers import CLIPTokenizer

    loaded = []
    for model_id in SDXL_TOKENIZER_IDS:
        loaded.append(CLIPTokenizer.from_pretrained(model_id, local_files_only=True))
    return loaded[0], loaded[1]
