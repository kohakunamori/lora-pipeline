from __future__ import annotations

from typing import Any, Sequence

from ..models import GeneratedImage


def cross_content_metrics(generated: Sequence[GeneratedImage], *, dataset_bias: dict[str, Any] | None = None) -> dict[str, Any]:
    positive = [item for item in generated if item.case.contains_trigger]
    prompts = {item.case.prompt_id for item in positive}
    return {
        "status": "manual_review_required",
        "content_prompt_coverage": len(prompts),
        "generated_positive_samples": len(positive),
        "dataset_bias": dataset_bias or {},
        "possible_subject_bias": bool((dataset_bias or {}).get("potential_subject_composition_bias", False)),
        "note": "Style consistency, subject diversity, and prompt compliance require contact-sheet review.",
        "ground_truth": False,
    }
