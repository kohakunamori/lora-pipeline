from __future__ import annotations

from pipeline.dataset.caption_cleaner import clean_caption, estimate_tokens


def test_trigger_is_inserted_once_and_first() -> None:
    result = clean_caption(
        {"red_hair": 0.9, "smile": 0.8, "zz_hutao": 0.99},
        trigger="zz_hutao",
    )
    assert result.tags[0] == "zz_hutao"
    assert result.tags.count("zz_hutao") == 1
    assert result.text.startswith("zz_hutao, ")


def test_caption_prunes_low_confidence_tail_to_token_budget() -> None:
    result = clean_caption(
        {"very_detailed_background": 0.4, "dynamic_pose": 0.8, "smile": 0.9, "blue_sky": 0.5},
        trigger="zz_test",
        max_token_length=9,
    )
    assert estimate_tokens(result.tags) <= 9
    assert result.pruned
    assert "smile" in result.tags


def test_style_generated_caption_suppresses_style_descriptors() -> None:
    result = clean_caption(
        ["1girl", "soft painterly shading", "outdoors"],
        trigger="zz_style",
        concept_type="style",
    )
    assert "soft painterly shading" not in result.tags
    assert "1girl" in result.tags


def test_existing_style_caption_can_preserve_user_descriptor() -> None:
    result = clean_caption(
        ["1girl", "soft painterly shading"],
        trigger="zz_style",
        concept_type="style",
        preserve_existing_style_descriptors=True,
    )
    assert "soft painterly shading" in result.tags
