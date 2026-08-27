from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Sequence

from ..models import GeneratedImage, OptionalBackendUnavailable


def character_trigger_leakage(reference_images: Sequence[Path], generated: Sequence[GeneratedImage]) -> dict[str, Any]:
    negative = [item for item in generated if not item.case.contains_trigger]
    if not reference_images or not negative:
        return {"status": "not_available", "reason": "reference or no-trigger images missing"}
    try:
        from imgutils.metrics import ccip_default_threshold, ccip_difference, ccip_merge
    except ImportError as exc:
        raise OptionalBackendUnavailable("imgutils CCIP leakage backend is unavailable") from exc
    reference = ccip_merge([str(path) for path in reference_images])
    differences = [float(ccip_difference(reference, str(item.path))) for item in negative]
    threshold = float(ccip_default_threshold())
    fraction = sum(value <= threshold for value in differences) / len(differences)
    return {
        "status": "review" if fraction else "low_signal",
        "method": "CCIP similarity of no-trigger generations to prepared identity reference",
        "median_difference": round(statistics.median(differences), 6),
        "default_same_identity_threshold": threshold,
        "within_default_threshold_fraction": round(fraction, 6),
        "sample_count": len(differences),
        "interpretation": "A high fraction is a possible identity-leakage signal, not proof of overtraining.",
        "ground_truth": False,
    }


def style_trigger_leakage(generated: Sequence[GeneratedImage]) -> dict[str, Any]:
    positive = sum(item.case.contains_trigger for item in generated)
    negative = len(generated) - positive
    return {
        "status": "manual_review_required",
        "positive_samples": positive,
        "no_trigger_samples": negative,
        "note": "Compare aligned positive/no-trigger images for unintended style transfer; no identity metric is applied.",
        "ground_truth": False,
    }
