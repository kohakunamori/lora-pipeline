from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .dataset.caption_cleaner import CATEGORY_PATTERNS, normalize_tag, parse_caption


_DEFAULT_LIMITS = {
    "minimum_images": 12,
    "concentration_warning_fraction": 0.80,
    "character_min_semantic_outfits": 2,
    "outfit_min_full_body_fraction": 0.15,
    "minimum_category_caption_coverage": 0.25,
}


def target_dataset_diagnostics(
    target_type: str,
    *,
    caption_records: Sequence[Mapping[str, Any]],
    dataset_semantics: Mapping[str, Any] | None = None,
    limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe target/data mismatches that can hurt LoRA disentanglement.

    Diagnostics are deliberately non-destructive. They report evidence from the
    frozen caption/semantic snapshot; they never auto-exclude samples or claim image
    quality ground truth.
    """

    configured = {**_DEFAULT_LIMITS, **dict(limits or {})}
    records = list(caption_records)
    total = len(records)
    warnings: list[str] = []
    observations: list[str] = []
    checks: dict[str, Any] = {}

    if target_type not in {"character", "character_outfit", "style"}:
        return {
            "status": "not_applicable",
            "target_type": target_type,
            "image_count": total,
            "warnings": [],
            "observations": [],
            "checks": {},
            "ground_truth": False,
        }

    if target_type == "style":
        return {
            "status": "delegated_to_style_guardrail",
            "target_type": target_type,
            "image_count": total,
            "warnings": [],
            "observations": [
                "Style subject/composition concentration is evaluated by the dedicated style-bias guardrail."
            ],
            "checks": {},
            "ground_truth": False,
        }

    minimum_images = int(configured["minimum_images"])
    concentration = float(configured["concentration_warning_fraction"])
    min_caption_coverage = float(configured["minimum_category_caption_coverage"])

    category_names = ("pose", "expression", "composition", "background", "lighting")
    for category in category_names:
        summary = _category_summary(records, category)
        checks[category] = summary
        if total < minimum_images:
            continue
        if summary["caption_coverage_fraction"] < min_caption_coverage:
            observations.append(
                f"{category} diversity is hard to assess because only "
                f"{summary['caption_coverage_fraction']:.1%} of captions contain a recognized {category} tag."
            )
            continue
        if summary["dominant_fraction"] >= concentration:
            warnings.append(
                f"{target_type} dataset has concentrated {category}: "
                f"{summary['dominant_tag']!r} appears in {summary['dominant_fraction']:.1%} of images. "
                "The learned concept may bind to this condition."
            )

    if target_type == "character":
        outfit_summary = _semantic_outfit_summary(records, dataset_semantics or {})
        checks["semantic_outfits"] = outfit_summary
        required_outfits = max(1, int(configured["character_min_semantic_outfits"]))
        if total >= minimum_images and outfit_summary["represented_outfits"] < required_outfits:
            warnings.append(
                "Character target has only one represented semantic outfit. Character identity can entangle "
                "with clothing; add another outfit when the intended LoRA should generalize across clothes, "
                "or verify Dataset semantics if multiple outfits are already present."
            )
        elif total >= minimum_images and outfit_summary["dominant_fraction"] >= concentration:
            warnings.append(
                f"Character target is dominated by outfit {outfit_summary['dominant_outfit']!r} "
                f"({outfit_summary['dominant_fraction']:.1%} of images); clothing leakage into identity is more likely."
            )

    if target_type == "character_outfit":
        full_body_fraction = _tag_fraction(records, {"full body"})
        portrait_fraction = _tag_fraction(records, {"portrait", "close up", "face"})
        checks["outfit_framing"] = {
            "full_body_fraction": full_body_fraction,
            "portrait_fraction": portrait_fraction,
            "minimum_full_body_fraction": float(configured["outfit_min_full_body_fraction"]),
        }
        if total >= minimum_images and full_body_fraction < float(
            configured["outfit_min_full_body_fraction"]
        ):
            warnings.append(
                f"Character-outfit target has only {full_body_fraction:.1%} full-body coverage. "
                "If the outfit includes lower-body/footwear details, those details may be underlearned."
            )
        if total >= minimum_images and portrait_fraction >= concentration:
            warnings.append(
                f"Character-outfit target is portrait-heavy ({portrait_fraction:.1%}); "
                "the outfit may bind to close framing and lose full-garment consistency."
            )

    if total < minimum_images:
        observations.append(
            f"Only {total} prepared image(s) are available; concentration warnings require at least "
            f"{minimum_images} samples to reduce false positives."
        )

    return {
        "status": "warning" if warnings else "ok",
        "target_type": target_type,
        "image_count": total,
        "limits": configured,
        "warnings": warnings,
        "observations": observations,
        "checks": checks,
        "ground_truth": False,
        "note": (
            "These diagnostics estimate concept/condition entanglement from tags and frozen Dataset semantics. "
            "They do not automatically remove data and are not image-quality ground truth."
        ),
    }


def _category_summary(
    records: Sequence[Mapping[str, Any]], category: str
) -> dict[str, Any]:
    patterns = CATEGORY_PATTERNS.get(category, ())
    total = len(records)
    counts: Counter[str] = Counter()
    captioned = 0
    for record in records:
        matched = {
            normalize_tag(tag)
            for tag in parse_caption(str(record.get("text") or ""))
            if any(pattern.search(normalize_tag(tag)) for pattern in patterns)
        }
        if matched:
            captioned += 1
            counts.update(matched)
    dominant_tag, dominant_count = counts.most_common(1)[0] if counts else (None, 0)
    return {
        "category": category,
        "captioned_images": captioned,
        "caption_coverage_fraction": round(captioned / total, 6) if total else 0.0,
        "distinct_tags": len(counts),
        "dominant_tag": dominant_tag,
        "dominant_count": dominant_count,
        "dominant_fraction": round(dominant_count / total, 6) if total else 0.0,
        "top_tags": counts.most_common(10),
    }


def _semantic_outfit_summary(
    records: Sequence[Mapping[str, Any]], semantics: Mapping[str, Any]
) -> dict[str, Any]:
    bindings = semantics.get("images", {}) if isinstance(semantics, Mapping) else {}
    counts: Counter[str] = Counter()
    for record in records:
        key = str(record.get("image") or "")
        binding = bindings.get(key, {}) if isinstance(bindings, Mapping) else {}
        outfit_id = str(binding.get("outfit") or "default") if isinstance(binding, Mapping) else "default"
        counts[outfit_id] += 1
    total = len(records)
    dominant_outfit, dominant_count = counts.most_common(1)[0] if counts else (None, 0)
    return {
        "represented_outfits": len(counts),
        "counts": dict(sorted(counts.items())),
        "dominant_outfit": dominant_outfit,
        "dominant_count": dominant_count,
        "dominant_fraction": round(dominant_count / total, 6) if total else 0.0,
        "defined_outfits": len(semantics.get("outfits", {})) if isinstance(semantics, Mapping) else 0,
    }


def _tag_fraction(
    records: Sequence[Mapping[str, Any]], choices: set[str]
) -> float:
    normalized_choices = {normalize_tag(value) for value in choices}
    total = len(records)
    if not total:
        return 0.0
    matched = 0
    for record in records:
        tags = {
            normalize_tag(tag)
            for tag in parse_caption(str(record.get("text") or ""))
        }
        matched += bool(tags & normalized_choices)
    return round(matched / total, 6)
