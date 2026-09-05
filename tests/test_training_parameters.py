from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from pipeline import i18n
from pipeline.interactive import InteractiveWizard
from pipeline.models import PipelineError
from pipeline.training_parameters import (
    TRAINING_PARAMETER_SPECS,
    effective_training_settings,
    reset_key_training_overrides,
    update_key_training_overrides,
    validate_training_override_values,
)


def test_parameter_guide_covers_every_user_facing_key() -> None:
    keys = {item.key for item in TRAINING_PARAMETER_SPECS}
    assert keys == {
        "images_seen",
        "network_dim",
        "network_alpha",
        "unet_lr",
        "batch_size",
        "gradient_accumulation_steps",
        "seed",
    }
    assert all(item.description_zh and item.recommendation_zh for item in TRAINING_PARAMETER_SPECS)


def test_custom_overrides_store_only_differences_from_strategy_defaults() -> None:
    overrides = update_key_training_overrides(
        {},
        strategy="quality",
        values={
            "network_dim": 16,
            "network_alpha": 8,
            "unet_lr": 0.0001,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "seed": 42,
        },
    )
    assert overrides == {"training": {"batch_size": 4}}
    effective = effective_training_settings("quality", overrides)
    assert effective["network_dim"] == 16
    assert effective["batch_size"] == 4


def test_changing_strategy_keeps_explicit_overrides_and_uses_new_defaults_for_rest() -> None:
    overrides = {"training": {"batch_size": 4}}
    effective = effective_training_settings("fast", overrides)
    assert effective["batch_size"] == 4
    assert effective["network_dim"] == 16
    assert effective["gradient_accumulation_steps"] == 1


def test_reset_removes_only_managed_key_overrides() -> None:
    overrides = {
        "training": {
            "batch_size": 4,
            "network_dim": 32,
            "max_grad_norm": 0.5,
        },
        "caption": {"shuffle": False},
    }
    reset = reset_key_training_overrides(overrides)
    assert reset == {
        "training": {"max_grad_norm": 0.5},
        "caption": {"shuffle": False},
    }


def test_custom_batch_has_no_artificial_upper_bound() -> None:
    validate_training_override_values({"training": {"batch_size": 16}})


def test_cli_parameter_guide_renders_descriptions() -> None:
    i18n.set_language("zh-CN")
    stream = StringIO()
    wizard = InteractiveWizard(
        console=Console(file=stream, force_terminal=False, color_system=None, width=180)
    )
    wizard._render_training_parameter_help("quality", images_seen=2000)
    output = stream.getvalue()
    assert "关键训练参数说明" in output
    assert "物理 Batch Size" in output
    assert "不设置人工上限" in output
    assert "有效 Batch" in output


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("batch_size", 0),
        ("gradient_accumulation_steps", 0),
        ("network_dim", 0),
        ("network_alpha", 0),
        ("unet_lr", 0),
        ("seed", -1),
    ],
)
def test_invalid_key_parameter_values_are_rejected(key: str, value: int | float) -> None:
    with pytest.raises(PipelineError):
        validate_training_override_values({"training": {key: value}})
