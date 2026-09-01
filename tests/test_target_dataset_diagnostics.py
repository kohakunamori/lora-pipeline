from __future__ import annotations

from pipeline.target_dataset_diagnostics import target_dataset_diagnostics


def _records(texts: list[str]) -> list[dict[str, str]]:
    return [
        {"image": f"{index}.png", "text": text}
        for index, text in enumerate(texts, start=1)
    ]


def test_character_warns_when_only_one_semantic_outfit_is_represented() -> None:
    records = _records(
        [
            f"char_token, default_outfit, 1girl, standing, smile, {('indoors' if i % 2 else 'outdoors')}"
            for i in range(20)
        ]
    )
    result = target_dataset_diagnostics(
        "character",
        caption_records=records,
        dataset_semantics={
            "outfits": {"default": {"token": "default_outfit"}},
            "images": {},
        },
    )

    assert result["status"] == "warning"
    assert result["checks"]["semantic_outfits"]["represented_outfits"] == 1
    assert any("only one represented semantic outfit" in item for item in result["warnings"])


def test_character_balanced_outfits_avoid_clothing_entanglement_warning() -> None:
    texts = []
    bindings = {}
    for i in range(20):
        pose = "standing" if i % 2 else "sitting"
        background = "indoors" if i % 2 else "outdoors"
        texts.append(f"char_token, outfit_token, 1girl, {pose}, {background}")
        bindings[f"{i + 1}.png"] = {"outfit": "casual" if i < 10 else "stage"}
    result = target_dataset_diagnostics(
        "character",
        caption_records=_records(texts),
        dataset_semantics={
            "outfits": {"casual": {}, "stage": {}},
            "images": bindings,
        },
    )

    assert result["checks"]["semantic_outfits"]["represented_outfits"] == 2
    assert not any("clothing" in item for item in result["warnings"])


def test_character_outfit_warns_on_portrait_heavy_low_full_body_data() -> None:
    records = _records(
        [
            "outfit_trigger, character_anchor, 1girl, portrait, standing, simple background"
            for _ in range(20)
        ]
    )
    result = target_dataset_diagnostics(
        "character_outfit",
        caption_records=records,
    )

    assert result["status"] == "warning"
    framing = result["checks"]["outfit_framing"]
    assert framing["full_body_fraction"] == 0.0
    assert framing["portrait_fraction"] == 1.0
    joined = " ".join(result["warnings"])
    assert "full-body coverage" in joined
    assert "portrait-heavy" in joined


def test_character_outfit_with_varied_conditions_and_full_body_coverage_is_clean() -> None:
    texts = []
    poses = ["standing", "sitting", "running", "walking"]
    backgrounds = ["indoors", "outdoors", "city", "forest"]
    expressions = ["smile", "frown", "surprised", "expressionless"]
    for i in range(20):
        framing = "full body" if i < 8 else "upper body"
        texts.append(
            f"outfit_trigger, character_anchor, 1girl, {framing}, {poses[i % 4]}, "
            f"{expressions[i % 4]}, {backgrounds[i % 4]}"
        )
    result = target_dataset_diagnostics(
        "character_outfit",
        caption_records=_records(texts),
    )

    assert result["checks"]["outfit_framing"]["full_body_fraction"] == 0.4
    assert not any("full-body coverage" in item for item in result["warnings"])
    assert not any("portrait-heavy" in item for item in result["warnings"])


def test_low_caption_coverage_emits_observation_instead_of_concentration_warning() -> None:
    records = _records(
        ["char_token, 1girl, standing"] * 3
        + ["char_token, 1girl"] * 17
    )
    result = target_dataset_diagnostics(
        "character",
        caption_records=records,
        dataset_semantics={
            "outfits": {"casual": {}, "stage": {}},
            "images": {
                f"{i + 1}.png": {"outfit": "casual" if i < 10 else "stage"}
                for i in range(20)
            },
        },
    )

    assert any("pose diversity is hard to assess" in item for item in result["observations"])
    assert not any("concentrated pose" in item for item in result["warnings"])


def test_style_is_delegated_to_existing_style_guardrail() -> None:
    result = target_dataset_diagnostics(
        "style",
        caption_records=_records(["style_trigger, 1girl"] * 20),
    )
    assert result["status"] == "delegated_to_style_guardrail"
    assert result["warnings"] == []
