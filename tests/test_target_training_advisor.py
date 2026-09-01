from __future__ import annotations

from pipeline.target_training_advisor import target_training_advice


def test_character_outfit_advice_is_more_conservative_on_lr() -> None:
    advice = target_training_advice(
        "character_outfit",
        image_count=48,
        current_training={"network_dim": 16, "network_alpha": 8, "unet_lr": 0.0001},
        current_images_seen=2000,
    )
    assert advice["status"] == "heuristic_advisory"
    assert advice["ground_truth"] is False
    assert advice["preferred_start"]["network_dim"] == 32
    assert advice["preferred_start"]["network_alpha"] == 16
    assert advice["preferred_start"]["unet_lr"] == 0.00008
    assert advice["recommended"]["images_seen"]["minimum"] <= 2000 <= advice["recommended"]["images_seen"]["maximum"]


def test_style_advice_warns_that_more_training_does_not_fix_bias() -> None:
    advice = target_training_advice(
        "style",
        image_count=40,
        current_training={"network_dim": 16, "unet_lr": 0.0001},
        current_images_seen=2500,
        style_distribution={
            "image_count": 40,
            "dominant_subject": {"fraction": 0.9},
            "distribution": {
                "portrait": {"fraction": 0.85},
                "simple_background": {"fraction": 0.2},
                "multiple_people": {"fraction": 0.0},
            },
        },
    )
    joined = " ".join(advice["warnings"])
    assert "subject concentration" in joined
    assert "Portrait-heavy" in joined
    assert any("No multi-subject" in item for item in advice["observations"])
    assert advice["preferred_start"]["network_dim"] == 32


def test_small_character_dataset_keeps_rank_recommendation_conservative() -> None:
    advice = target_training_advice(
        "character",
        image_count=8,
        current_training={"network_dim": 64, "unet_lr": 0.0002},
        current_images_seen=5000,
    )
    assert advice["preferred_start"]["network_dim"] == 16
    assert advice["recommended"]["network_dim"]["maximum"] == 32
    joined = " ".join(advice["warnings"])
    assert "LoRA rank=64" in joined
    assert "UNet learning rate=0.0002" in joined
    assert "over-training risk" in joined
