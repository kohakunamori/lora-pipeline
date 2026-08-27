from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


def distribution_summary(
    captions: Iterable[Iterable[str]],
    distribution_config: Mapping[str, Any],
    *,
    aspect_ratios: Iterable[float] = (),
) -> dict[str, Any]:
    rows = [{tag.replace("_", " ").casefold() for tag in caption} for caption in captions]
    total = len(rows)
    buckets: dict[str, dict[str, float | int]] = {}
    for name, configured_tags in distribution_config.get("tags", {}).items():
        choices = {str(tag).replace("_", " ").casefold() for tag in configured_tags}
        count = sum(bool(row & choices) for row in rows)
        buckets[name] = {"count": count, "fraction": round(count / total, 4) if total else 0.0}

    subject_vocabulary = {
        "1girl",
        "1boy",
        "solo",
        "multiple girls",
        "multiple boys",
        "group",
        "landscape",
        "scenery",
        "no humans",
        "animal",
    }
    subject_counts = Counter(tag for row in rows for tag in row if tag in subject_vocabulary)
    dominant_subject, dominant_count = subject_counts.most_common(1)[0] if subject_counts else (None, 0)
    dominant_fraction = dominant_count / total if total else 0.0

    ratios = list(aspect_ratios)
    aspect_distribution = {
        "portrait": sum(value < 0.9 for value in ratios),
        "square": sum(0.9 <= value <= 1.1 for value in ratios),
        "landscape": sum(value > 1.1 for value in ratios),
    }
    aspect_total = len(ratios)
    aspect_fractions = {
        key: round(value / aspect_total, 4) if aspect_total else 0.0
        for key, value in aspect_distribution.items()
    }

    warning_config = distribution_config.get("bias_warning", {})
    dominant_limit = float(warning_config.get("dominant_subject_fraction", 0.8))
    portrait_limit = float(warning_config.get("portrait_fraction", 0.8))
    simple_background_limit = float(warning_config.get("simple_background_fraction", 0.8))
    min_multiple_people = float(warning_config.get("minimum_multiple_people_fraction", 0.05))
    warnings: list[dict[str, Any]] = []
    if dominant_fraction >= dominant_limit:
        warnings.append(
            {
                "code": "high_subject_concentration",
                "message": "Potential subject bias: one subject tag dominates the dataset",
                "value": round(dominant_fraction, 4),
                "threshold": dominant_limit,
                "dominant_subject": dominant_subject,
            }
        )
    portrait_fraction = float(buckets.get("portrait", {}).get("fraction", 0.0))
    if portrait_fraction >= portrait_limit:
        warnings.append(
            {
                "code": "high_portrait_concentration",
                "message": "Potential composition bias: portraits dominate the dataset",
                "value": portrait_fraction,
                "threshold": portrait_limit,
            }
        )
    simple_fraction = float(buckets.get("simple_background", {}).get("fraction", 0.0))
    if simple_fraction >= simple_background_limit:
        warnings.append(
            {
                "code": "high_simple_background_concentration",
                "message": "Potential background bias: simple backgrounds dominate the dataset",
                "value": simple_fraction,
                "threshold": simple_background_limit,
            }
        )
    multiple_fraction = float(buckets.get("multiple_people", {}).get("fraction", 0.0))
    if total >= 20 and multiple_fraction < min_multiple_people:
        warnings.append(
            {
                "code": "low_multi_subject_coverage",
                "message": "Low multi-subject coverage may bind the style to solo compositions",
                "value": multiple_fraction,
                "threshold": min_multiple_people,
            }
        )
    represented_aspects = sum(value > 0 for value in aspect_distribution.values())
    if aspect_total >= 10 and represented_aspects < 2:
        warnings.append(
            {
                "code": "low_aspect_ratio_diversity",
                "message": "Only one broad aspect-ratio class is represented",
                "distribution": aspect_distribution,
            }
        )

    return {
        "image_count": total,
        "distribution": buckets,
        "dominant_subject": {
            "tag": dominant_subject,
            "count": dominant_count,
            "fraction": round(dominant_fraction, 4),
        },
        "dominant_subject_tags": subject_counts.most_common(10),
        "aspect_ratio_distribution": aspect_distribution,
        "aspect_ratio_fractions": aspect_fractions,
        "warnings": warnings,
        "potential_subject_composition_bias": any(
            warning["code"] in {"high_subject_concentration", "high_portrait_concentration"}
            for warning in warnings
        ),
    }
