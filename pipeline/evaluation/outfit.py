from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

from ..models import GeneratedImage


def outfit_retention_proxy(generated: Sequence[GeneratedImage]) -> dict[str, Any]:
    """Describe coverage for reviewing whether the learned outfit is retained.

    There is deliberately no fake automatic clothing score here. The available
    model-independent evidence is a controlled generation matrix; final outfit
    fidelity remains a visual review item until a validated garment metric exists.
    """

    positive = [item for item in generated if item.case.contains_trigger]
    return {
        "status": "manual_review_required",
        "positive_samples": len(positive),
        "prompt_variants": len({item.case.prompt_id for item in positive}),
        "checkpoints": len({item.case.checkpoint_label for item in positive}),
        "strengths": len({item.case.strength for item in positive}),
        "review_goal": (
            "Confirm that trigger-on generations preserve the target outfit while pose, "
            "expression, framing, lighting, and background vary."
        ),
        "ground_truth": False,
    }


def outfit_trigger_leakage_proxy(
    generated: Sequence[GeneratedImage], *, anchor_tags: Iterable[str] = ()
) -> dict[str, Any]:
    """Summarize aligned trigger-on/off pairs for outfit leakage review."""

    groups: dict[tuple[str, str, float, int], set[bool]] = defaultdict(set)
    for item in generated:
        key = (
            item.case.checkpoint_label,
            item.case.prompt_id,
            float(item.case.strength),
            int(item.case.seed),
        )
        groups[key].add(bool(item.case.contains_trigger))
    paired = sum(values == {False, True} for values in groups.values())
    positive = sum(item.case.contains_trigger for item in generated)
    negative = len(generated) - positive
    return {
        "status": "manual_review_required",
        "anchor_tags": [str(tag) for tag in anchor_tags],
        "positive_samples": positive,
        "no_trigger_samples": negative,
        "aligned_on_off_pairs": paired,
        "pair_groups": len(groups),
        "pair_coverage_fraction": round(paired / len(groups), 6) if groups else 0.0,
        "review_goal": (
            "Compare aligned trigger-on/off cells. The target outfit should be strongly "
            "associated with the LoRA trigger and should not appear systematically in the "
            "anchor-only no-trigger baseline."
        ),
        "ground_truth": False,
    }
