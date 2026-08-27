from __future__ import annotations

from pipeline.dataset.style import distribution_summary
from pipeline.config import resolve_profiles


def _config() -> dict:
    return dict(resolve_profiles("v100_16gb", "style", "quality").concept["distribution"])


def test_style_bias_dimensions_warn_independently() -> None:
    captions = [["1girl", "outdoors", "full body"] for _ in range(20)]
    summary = distribution_summary(captions, _config(), aspect_ratios=[0.7, 1.4] * 10)
    codes = {warning["code"] for warning in summary["warnings"]}
    assert "high_subject_concentration" in codes
    assert "high_portrait_concentration" not in codes
    assert "low_multi_subject_coverage" in codes
    assert "low_aspect_ratio_diversity" not in codes


def test_style_portrait_and_background_bias_are_reported_separately() -> None:
    captions = [["portrait", "simple background", "1girl"] for _ in range(20)]
    summary = distribution_summary(captions, _config(), aspect_ratios=[1.0] * 20)
    codes = {warning["code"] for warning in summary["warnings"]}
    assert "high_subject_concentration" in codes
    assert "high_portrait_concentration" in codes
    assert "high_simple_background_concentration" in codes
    assert "low_aspect_ratio_diversity" in codes
