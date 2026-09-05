from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .models import PipelineError


TRIGGER_STRATEGIES = (
    "explicit",
    "rare_token",
    "name",
    "multi_anchor",
)


@dataclass(frozen=True)
class TriggerPolicy:
    strategy: str
    requested: str
    trigger: str
    anchors: tuple[str, ...]

    @property
    def protected_prefix(self) -> tuple[str, ...]:
        values = [self.trigger, *self.anchors]
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = re.sub(r"\s+", " ", str(value).strip())
            normalized = text.replace("_", " ").casefold()
            if not text or normalized in seen:
                continue
            seen.add(normalized)
            result.append(text)
        return tuple(result)

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "requested": self.requested,
            "trigger": self.trigger,
            "anchors": list(self.anchors),
            "protected_prefix": list(self.protected_prefix),
        }


def resolve_trigger_policy(
    requested: str,
    *,
    strategy: str = "explicit",
    anchors: Iterable[str] = (),
) -> TriggerPolicy:
    """Resolve a small, predictable set of LoRA trigger strategies.

    ``explicit`` keeps the supplied trigger unchanged.
    ``rare_token`` converts a human label to a deterministic ``zz_<slug>`` token.
    ``name`` keeps a natural character/style name as the trigger.
    ``multi_anchor`` keeps the supplied trigger and protects additional anchor tags;
    this is primarily intended for character-outfit recipes.
    """

    strategy = str(strategy or "explicit").strip().casefold()
    if strategy not in TRIGGER_STRATEGIES:
        raise PipelineError(
            "Trigger strategy must be one of: " + ", ".join(TRIGGER_STRATEGIES)
        )
    requested = re.sub(r"\s+", " ", str(requested).strip())
    if not requested or "," in requested or "\n" in requested:
        raise PipelineError("Trigger must be non-empty and cannot contain commas/newlines")

    if strategy == "rare_token":
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", requested).strip("_").casefold()
        if not slug:
            raise PipelineError("Rare-token trigger must contain letters or numbers")
        trigger = requested if requested.casefold().startswith("zz_") else f"zz_{slug}"
    else:
        trigger = requested

    normalized_anchors: list[str] = []
    seen = {trigger.replace("_", " ").casefold()}
    for value in anchors:
        text = re.sub(r"\s+", " ", str(value).strip())
        normalized = text.replace("_", " ").casefold()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        normalized_anchors.append(text)

    if strategy == "multi_anchor" and not normalized_anchors:
        raise PipelineError("multi_anchor trigger strategy requires at least one anchor tag")

    return TriggerPolicy(
        strategy=strategy,
        requested=requested,
        trigger=trigger,
        anchors=tuple(normalized_anchors),
    )
