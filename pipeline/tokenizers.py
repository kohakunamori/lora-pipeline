from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .config import load_base_registry, repository_root
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
    qwen3: int | None = None

    @property
    def maximum(self) -> int:
        return self.qwen3 if self.qwen3 is not None else max(self.clip_l, self.clip_g)

    def as_dict(self) -> dict[str, Any]:
        return {
            "clip_l": self.clip_l,
            "clip_g": self.clip_g,
            "qwen3": self.qwen3,
            "exact": self.exact,
            "backend": self.backend,
            "error": self.error,
        }


def count_sdxl_tokens(text: str) -> TokenCounts:
    """Count content tokens with the two tokenizers used by SDXL."""
    try:
        tokenizers = _load_sdxl_tokenizers()
        counts = [
            len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
            for tokenizer in tokenizers
        ]
        return TokenCounts(
            clip_l=int(counts[0]), clip_g=int(counts[1]), exact=True,
            backend="transformers.CLIPTokenizer",
        )
    except Exception as exc:
        estimated = estimate_tokens(parse_caption(text))
        return TokenCounts(
            clip_l=estimated, clip_g=estimated, exact=False,
            backend="heuristic-fallback", error=f"{type(exc).__name__}: {exc}",
        )


def count_project_tokens(
    text: str, project: Mapping[str, Any], *, root: Path | None = None
) -> TokenCounts:
    """Count tokens with the tokenizer belonging to the registered base family."""
    root = root or repository_root()
    family, source = _base_tokenizer_source(str(root), str(project.get("base", "")))
    if family != "anima":
        return count_sdxl_tokens(text)
    if source and Path(source).is_dir():
        tokenizer_path = str(Path(source).expanduser())
    else:
        tokenizer_path = _qwen3_config_dir(root)
    try:
        tokenizer = _load_qwen3_tokenizer(tokenizer_path)
        count = len(tokenizer(text, truncation=False)["input_ids"])
        return TokenCounts(
            clip_l=0, clip_g=0, exact=True,
            backend="transformers.Qwen3Tokenizer", qwen3=int(count),
        )
    except Exception as exc:
        estimated = estimate_tokens(parse_caption(text))
        return TokenCounts(
            clip_l=0, clip_g=0, exact=False, backend="heuristic-fallback",
            error=f"{type(exc).__name__}: {exc}", qwen3=estimated,
        )


@lru_cache(maxsize=64)
def _base_tokenizer_source(root: str, base_id: str) -> tuple[str | None, str]:
    base = load_base_registry(Path(root)).get(base_id)
    if base is None:
        return None, ""
    return base.family, str(base.components.get("qwen3", ""))


def _qwen3_config_dir(root: Path) -> str:
    try:
        info = json.loads(
            (root / "environment" / "environment-info.json").read_text(encoding="utf-8")
        )
        candidate = Path(info["sd_scripts_path"]) / "configs" / "qwen3_06b"
        if candidate.is_dir():
            return str(candidate)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        pass
    return ""


@lru_cache(maxsize=1)
def _load_sdxl_tokenizers() -> tuple[Any, Any]:
    from transformers import CLIPTokenizer
    loaded = [
        CLIPTokenizer.from_pretrained(model_id, local_files_only=True)
        for model_id in SDXL_TOKENIZER_IDS
    ]
    return loaded[0], loaded[1]


@lru_cache(maxsize=4)
def _load_qwen3_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)
