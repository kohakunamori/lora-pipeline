from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Sequence

from ..models import GeneratedImage, OptionalBackendUnavailable


def identity_metrics(reference_images: Sequence[Path], generated: Sequence[GeneratedImage]) -> dict[str, Any]:
    positive = [item for item in generated if item.case.contains_trigger]
    if not reference_images or not positive:
        return {"status": "not_available", "reason": "reference or positive images missing"}
    try:
        from imgutils.metrics import ccip_default_threshold, ccip_difference, ccip_merge
    except ImportError as exc:
        raise OptionalBackendUnavailable("imgutils CCIP evaluation backend is unavailable") from exc
    reference = ccip_merge([str(path) for path in reference_images])
    differences = [float(ccip_difference(reference, str(item.path))) for item in positive]
    threshold = float(ccip_default_threshold())
    return {
        "status": "auxiliary",
        "method": "CCIP difference to merged prepared reference",
        "lower_is_more_similar": True,
        "default_same_identity_threshold": threshold,
        "median_difference": round(statistics.median(differences), 6),
        "min_difference": round(min(differences), 6),
        "max_difference": round(max(differences), 6),
        "within_default_threshold_fraction": round(sum(value <= threshold for value in differences) / len(differences), 6),
        "sample_count": len(differences),
        "ground_truth": False,
    }


def controllability_proxy(generated: Sequence[GeneratedImage]) -> dict[str, Any]:
    prompt_ids = {item.case.prompt_id for item in generated if item.case.contains_trigger}
    checkpoints = {item.case.checkpoint_label for item in generated}
    strengths = {item.case.strength for item in generated}
    return {
        "status": "manual_review_required",
        "coverage": {
            "prompt_variants": len(prompt_ids),
            "checkpoints": len(checkpoints),
            "strengths": len(strengths),
        },
        "note": "Prompt coverage is recorded, but semantic controllability is not inferred from filenames or tags.",
    }
