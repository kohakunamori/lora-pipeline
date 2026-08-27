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
    subject_counts = Counter(tag for row in rows for tag in row if tag in {"1girl", "1boy", "solo", "landscape"})
    warning_config = distribution_config.get("bias_warning", {})
    dominant_fraction = max((value / total for value in subject_counts.values()), default=0.0)
    portrait_fraction = float(buckets.get("portrait", {}).get("fraction", 0.0))
    biased = dominant_fraction >= float(warning_config.get("dominant_subject_fraction", 0.8)) and portrait_fraction >= float(
        warning_config.get("portrait_fraction", 0.8)
    )
    ratios = list(aspect_ratios)
    aspect_distribution = {
        "portrait": sum(value < 0.9 for value in ratios),
        "square": sum(0.9 <= value <= 1.1 for value in ratios),
        "landscape": sum(value > 1.1 for value in ratios),
    }
    return {
        "image_count": total,
        "distribution": buckets,
        "dominant_subject_tags": subject_counts.most_common(10),
        "potential_subject_composition_bias": biased,
        "warning": warning_config.get("message") if biased else None,
        "aspect_ratio_distribution": aspect_distribution,
    }
