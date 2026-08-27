from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping


STYLE_DESCRIPTOR_PATTERNS = (
    re.compile(r"\b(?:painterly|painting|lineart|line art|coloring style|shading style)\b", re.I),
    re.compile(r"\b(?:soft|hard) shading\b", re.I),
)


@dataclass(frozen=True)
class CleanCaption:
    text: str
    tags: tuple[str, ...]
    estimated_tokens: int
    pruned: tuple[str, ...]


def normalize_tag(tag: str) -> str:
    return re.sub(r"\s+", " ", tag.replace("_", " ").strip()).casefold()


def estimate_tokens(tags: Iterable[str]) -> int:
    # Conservative CLIP-oriented estimate: words plus separator overhead.
    return sum(max(1, len(re.findall(r"[\w'-]+|[^\w\s]", tag))) + 1 for tag in tags)


def clean_caption(
    tags: Mapping[str, float] | Iterable[str],
    *,
    trigger: str,
    threshold: float = 0.35,
    replacements: Mapping[str, str] | None = None,
    blacklist: Iterable[str] = (),
    max_token_length: int = 75,
    concept_type: str = "character",
    identity_features: Iterable[str] = (),
    identity_mode: str = "conservative",
    preserve_existing_style_descriptors: bool = False,
    ordering: Iterable[str] = (),
) -> CleanCaption:
    replacements = {normalize_tag(key): value for key, value in (replacements or {}).items()}
    blocked = {normalize_tag(tag) for tag in blacklist}
    identity = {normalize_tag(tag) for tag in identity_features}
    if isinstance(tags, Mapping):
        candidates = sorted(
            ((normalize_tag(tag), float(score)) for tag, score in tags.items() if float(score) >= threshold),
            key=lambda item: (-item[1], item[0]),
        )
    else:
        candidates = [(normalize_tag(tag), 1.0) for tag in tags]

    trigger_normalized = normalize_tag(trigger)
    ordered: list[tuple[str, float]] = []
    seen: set[str] = set()
    for tag, score in candidates:
        tag = normalize_tag(replacements.get(tag, tag))
        if not tag or tag in seen or tag in blocked or tag == trigger_normalized:
            continue
        if identity_mode == "exclude" and tag in identity:
            continue
        if concept_type == "style" and not preserve_existing_style_descriptors:
            if any(pattern.search(tag) for pattern in STYLE_DESCRIPTOR_PATTERNS):
                continue
        seen.add(tag)
        ordered.append((tag, score))

    priority = {normalize_tag(value): index for index, value in enumerate(ordering)}
    if priority:
        ordered.sort(key=lambda item: (priority.get(item[0], len(priority)), -item[1], item[0]))
    retained = [trigger.strip()] + [tag for tag, _ in ordered]
    pruned: list[str] = []
    while len(retained) > 1 and estimate_tokens(retained) > max_token_length:
        removed = retained.pop()
        pruned.append(removed)
    return CleanCaption(
        text=", ".join(retained),
        tags=tuple(retained),
        estimated_tokens=estimate_tokens(retained),
        pruned=tuple(pruned),
    )


def parse_caption(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
