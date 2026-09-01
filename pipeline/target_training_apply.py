from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .training_parameters import MANAGED_TRAINING_KEYS, update_key_training_overrides


def apply_target_preferred_start(
    *,
    strategy: str,
    overrides: Mapping[str, Any] | None,
    current_training: Mapping[str, Any],
    advice: Mapping[str, Any],
    root: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Apply only the advisor-owned knobs while preserving runtime/batch choices.

    Target advice owns exposure, rank, alpha and UNet LR. Physical batch, gradient
    accumulation, seed, and any unknown expert overrides keep their current effective
    behavior. The caller must obtain explicit user approval before invoking this helper.
    """

    preferred = advice.get("preferred_start", {})
    required = {"images_seen", "network_dim", "network_alpha", "unet_lr"}
    missing = required - set(preferred)
    if missing:
        raise ValueError("Target training advice is missing preferred values: " + ", ".join(sorted(missing)))

    values = {
        key: current_training[key]
        for key in MANAGED_TRAINING_KEYS
        if key in current_training
    }
    values.update(
        {
            "network_dim": int(preferred["network_dim"]),
            "network_alpha": int(preferred["network_alpha"]),
            "unet_lr": float(preferred["unet_lr"]),
        }
    )
    updated = update_key_training_overrides(
        overrides,
        strategy=strategy,
        values=values,
        root=root,
    )
    return int(preferred["images_seen"]), updated
