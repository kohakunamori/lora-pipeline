from __future__ import annotations

import pytest

from pipeline.target_training_advisor import target_training_advice
from pipeline.target_training_apply import apply_target_preferred_start
from pipeline.training_parameters import effective_training_settings


def test_apply_preferred_start_preserves_runtime_and_expert_overrides() -> None:
    overrides = {
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 3,
            "seed": 99,
            "network_dim": 16,
            "max_grad_norm": 0.7,
        },
        "metadata": {"title": "keep-me"},
    }
    current = effective_training_settings("quality", overrides)
    advice = target_training_advice(
        "character_outfit",
        image_count=48,
        current_training=current,
        current_images_seen=2000,
    )

    images_seen, updated = apply_target_preferred_start(
        strategy="quality",
        overrides=overrides,
        current_training=current,
        advice=advice,
    )
    effective = effective_training_settings("quality", updated)

    assert images_seen == advice["preferred_start"]["images_seen"]
    assert effective["network_dim"] == 32
    assert effective["network_alpha"] == 16
    assert effective["unet_lr"] == 0.00008
    assert effective["batch_size"] == 4
    assert effective["gradient_accumulation_steps"] == 3
    assert effective["seed"] == 99
    assert effective["max_grad_norm"] == 0.7
    assert updated["metadata"] == {"title": "keep-me"}


def test_apply_preferred_start_keeps_strategy_defaults_sparse() -> None:
    overrides = {"training": {"batch_size": 2}}
    current = effective_training_settings("quality", overrides)
    advice = target_training_advice(
        "character",
        image_count=30,
        current_training=current,
        current_images_seen=1200,
    )

    _, updated = apply_target_preferred_start(
        strategy="quality",
        overrides=overrides,
        current_training=current,
        advice=advice,
    )

    # Character preferred rank/LR match the quality preset, so they should not be
    # redundantly stored. The user's explicit Batch choice remains.
    assert updated["training"] == {"batch_size": 2}


def test_apply_preferred_start_rejects_incomplete_advice() -> None:
    with pytest.raises(ValueError, match="missing preferred values"):
        apply_target_preferred_start(
            strategy="quality",
            overrides={},
            current_training=effective_training_settings("quality", {}),
            advice={"preferred_start": {"images_seen": 1000}},
        )
