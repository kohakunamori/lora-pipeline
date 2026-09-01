from __future__ import annotations

from pipeline.target_preflight import assess_style_distribution


def _distribution(*, count: int, dominant: float, portrait: float, simple: float) -> dict:
    return {
        "image_count": count,
        "dominant_subject": {"fraction": dominant},
        "distribution": {
            "portrait": {"fraction": portrait},
            "simple_background": {"fraction": simple},
        },
        "warnings": [],
    }


def test_style_guardrail_blocks_only_joint_extreme_entanglement() -> None:
    result = assess_style_distribution(
        _distribution(count=40, dominant=0.98, portrait=0.97, simple=0.2),
        limits={"minimum_images": 12, "extreme_fraction": 0.95},
    )
    assert result["status"] == "blocked"
    assert result["joint_extreme_subject_portrait"] is True
    assert result["blocking"]


def test_style_guardrail_does_not_block_portrait_style_without_subject_collapse() -> None:
    result = assess_style_distribution(
        _distribution(count=40, dominant=0.55, portrait=1.0, simple=0.2),
        limits={"minimum_images": 12, "extreme_fraction": 0.95},
    )
    assert result["status"] == "low"
    assert result["blocking"] == []


def test_style_guardrail_small_dataset_warns_instead_of_blocking() -> None:
    result = assess_style_distribution(
        _distribution(count=8, dominant=1.0, portrait=1.0, simple=1.0),
        limits={"minimum_images": 12, "extreme_fraction": 0.95},
    )
    assert result["blocking"] == []
    assert result["status"] == "warning"
    assert any("requires at least 12" in item for item in result["warnings"])


def test_existing_distribution_warnings_are_preserved() -> None:
    distribution = _distribution(count=30, dominant=0.85, portrait=0.7, simple=0.2)
    distribution["warnings"] = [
        {
            "code": "high_subject_concentration",
            "message": "Potential subject bias: one subject tag dominates the dataset",
            "value": 0.85,
        }
    ]
    result = assess_style_distribution(distribution)
    assert result["status"] == "warning"
    assert any("high_subject_concentration" in item for item in result["warnings"])
