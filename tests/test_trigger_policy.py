import pytest

from pipeline.models import PipelineError
from pipeline.trigger_policy import resolve_trigger_policy


def test_explicit_and_name_keep_requested_trigger() -> None:
    assert resolve_trigger_policy("misuzu_demo").trigger == "misuzu_demo"
    assert resolve_trigger_policy("Hataya Misuzu", strategy="name").trigger == "Hataya Misuzu"


def test_rare_token_generates_deterministic_rare_trigger() -> None:
    policy = resolve_trigger_policy("Hataya Misuzu", strategy="rare_token")
    assert policy.trigger == "zz_hataya_misuzu"
    assert policy.protected_prefix == ("zz_hataya_misuzu",)


def test_multi_anchor_protects_trigger_and_anchor_order() -> None:
    policy = resolve_trigger_policy(
        "misuzu_swimsuit",
        strategy="multi_anchor",
        anchors=["hataya misuzu", "1girl"],
    )
    assert policy.protected_prefix == (
        "misuzu_swimsuit",
        "hataya misuzu",
        "1girl",
    )


def test_multi_anchor_requires_anchor() -> None:
    with pytest.raises(PipelineError, match="requires at least one anchor"):
        resolve_trigger_policy("demo", strategy="multi_anchor")
