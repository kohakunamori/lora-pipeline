from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_TARGETS = {"character", "character_outfit", "style"}


@dataclass(frozen=True)
class AdvisoryRange:
    minimum: float | int
    preferred: float | int
    maximum: float | int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "minimum": self.minimum,
            "preferred": self.preferred,
            "maximum": self.maximum,
        }


def target_training_advice(
    target_type: str,
    *,
    image_count: int,
    current_training: Mapping[str, Any] | None = None,
    current_images_seen: int | None = None,
    style_distribution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a conservative target-aware training advisory.

    These values are intentionally heuristics rather than automatic overrides.  The
    pipeline records them so users can compare the current recipe against a target-
    and dataset-aware starting point without silently changing an explicit config.
    """

    if target_type not in _TARGETS:
        raise ValueError(f"Unsupported training target: {target_type}")
    count = max(1, int(image_count))
    current = dict(current_training or {})

    exposure = _exposure_range(target_type, count)
    rank = _rank_range(target_type, count)
    preferred_rank = int(rank.preferred)
    alpha = AdvisoryRange(
        minimum=max(1, int(rank.minimum) // 2),
        preferred=max(1, preferred_rank // 2),
        maximum=max(1, int(rank.maximum)),
    )
    lr = _learning_rate_range(target_type)

    observations: list[str] = []
    warnings: list[str] = []
    if count < 12:
        observations.append(
            "Very small datasets make any rank/LR recommendation uncertain; rely on checkpoint evaluation and stop early on overfit."
        )
    if target_type == "style":
        bias = _style_bias_summary(style_distribution or {})
        observations.extend(bias["observations"])
        warnings.extend(bias["warnings"])

    current_images = int(current_images_seen) if current_images_seen is not None else None
    if current_images is not None:
        if current_images < int(exposure.minimum):
            warnings.append(
                f"images_seen={current_images} is below the advisory floor {int(exposure.minimum)} for this target/data size; under-training is possible."
            )
        elif current_images > int(exposure.maximum):
            warnings.append(
                f"images_seen={current_images} exceeds the advisory ceiling {int(exposure.maximum)} for this target/data size; over-training risk is elevated."
            )

    _compare_current_numeric(
        warnings,
        current,
        key="network_dim",
        advisory=rank,
        label="LoRA rank",
    )
    _compare_current_numeric(
        warnings,
        current,
        key="unet_lr",
        advisory=lr,
        label="UNet learning rate",
    )

    preferred = {
        "images_seen": int(exposure.preferred),
        "network_dim": preferred_rank,
        "network_alpha": int(alpha.preferred),
        "unet_lr": float(lr.preferred),
    }
    return {
        "status": "heuristic_advisory",
        "ground_truth": False,
        "target_type": target_type,
        "image_count": count,
        "recommended": {
            "images_seen": exposure.as_dict(),
            "network_dim": rank.as_dict(),
            "network_alpha": alpha.as_dict(),
            "unet_lr": lr.as_dict(),
        },
        "preferred_start": preferred,
        "current": {
            "images_seen": current_images,
            "network_dim": current.get("network_dim"),
            "network_alpha": current.get("network_alpha"),
            "unet_lr": current.get("unet_lr"),
        },
        "equivalent_epochs": {
            "current": round(current_images / count, 3) if current_images is not None else None,
            "preferred": round(int(exposure.preferred) / count, 3),
        },
        "observations": observations,
        "warnings": warnings,
        "note": (
            "Recommendations are conservative starting points, not automatic quality truth. "
            "Use fixed evaluation prompts/checkpoints to decide whether to extend training."
        ),
    }


def _exposure_range(target_type: str, image_count: int) -> AdvisoryRange:
    if target_type == "character_outfit":
        preferred = _clamp(image_count * 50, 1500, 3500)
        minimum = _clamp(image_count * 30, 900, preferred)
        maximum = _clamp(image_count * 75, preferred, 5000)
    elif target_type == "style":
        preferred = _clamp(image_count * 30, 2000, 6000)
        minimum = _clamp(image_count * 18, 1200, preferred)
        maximum = _clamp(image_count * 50, preferred, 8000)
    else:
        preferred = _clamp(image_count * 40, 1200, 3000)
        minimum = _clamp(image_count * 24, 800, preferred)
        maximum = _clamp(image_count * 65, preferred, 4500)
    return AdvisoryRange(int(minimum), int(preferred), int(maximum))


def _rank_range(target_type: str, image_count: int) -> AdvisoryRange:
    if target_type == "character_outfit":
        preferred = 32 if image_count >= 24 else 16
        return AdvisoryRange(16, preferred, 64 if image_count >= 48 else 32)
    if target_type == "style":
        preferred = 32 if image_count >= 40 else 16
        return AdvisoryRange(16, preferred, 64 if image_count >= 100 else 32)
    return AdvisoryRange(8, 16, 32)


def _learning_rate_range(target_type: str) -> AdvisoryRange:
    if target_type == "character_outfit":
        return AdvisoryRange(5e-5, 8e-5, 1e-4)
    return AdvisoryRange(5e-5, 1e-4, 1e-4)


def _style_bias_summary(distribution: Mapping[str, Any]) -> dict[str, list[str]]:
    warnings: list[str] = []
    observations: list[str] = []
    dominant = float(distribution.get("dominant_subject", {}).get("fraction", 0.0) or 0.0)
    buckets = distribution.get("distribution", {})
    portrait = float(buckets.get("portrait", {}).get("fraction", 0.0) or 0.0)
    simple = float(buckets.get("simple_background", {}).get("fraction", 0.0) or 0.0)
    multi = float(buckets.get("multiple_people", {}).get("fraction", 0.0) or 0.0)
    if dominant >= 0.8:
        warnings.append(
            "Style dataset has high subject concentration; more training will reinforce subject/style entanglement rather than fix it."
        )
    if portrait >= 0.8:
        warnings.append(
            "Portrait-heavy style data may bind the learned style to portrait framing."
        )
    if simple >= 0.8:
        warnings.append(
            "Simple-background-heavy style data may bind the learned style to sparse backgrounds."
        )
    if multi == 0 and int(distribution.get("image_count", 0) or 0) >= 20:
        observations.append(
            "No multi-subject examples are represented; add some if cross-subject composition is an intended use case."
        )
    return {"warnings": warnings, "observations": observations}


def _compare_current_numeric(
    warnings: list[str],
    current: Mapping[str, Any],
    *,
    key: str,
    advisory: AdvisoryRange,
    label: str,
) -> None:
    if key not in current or current[key] is None:
        return
    try:
        value = float(current[key])
    except (TypeError, ValueError):
        return
    if value < float(advisory.minimum):
        warnings.append(
            f"{label}={current[key]} is below the advisory range minimum {advisory.minimum}."
        )
    elif value > float(advisory.maximum):
        warnings.append(
            f"{label}={current[key]} is above the advisory range maximum {advisory.maximum}."
        )


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
