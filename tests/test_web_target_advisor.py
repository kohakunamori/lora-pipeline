from __future__ import annotations

from types import SimpleNamespace

from pipeline.target_training_advisor import target_training_advice
from pipeline.training_parameters import effective_training_settings
from pipeline.web_target_advisor import TargetAdvisorHandler, _advice_table_html


class _Config:
    target_type = "style"
    images_seen = 2000


def test_advice_table_renders_current_range_and_preferred_values() -> None:
    current = effective_training_settings("quality", {})
    advice = target_training_advice(
        "style",
        image_count=40,
        current_training=current,
        current_images_seen=2000,
    )

    html = _advice_table_html(_Config(), current, advice)

    assert "style" in html
    assert "network_dim" in html
    assert "16 – 32" in html
    assert "<b>32</b>" in html
    assert "images_seen" in html


def test_final_web_handler_includes_target_advisor_layer() -> None:
    # Import after the target advisor class so web_full can install its final hook
    # chain and protected-deletion subclassing.
    from pipeline.web_full import FullHandler

    assert issubclass(FullHandler, TargetAdvisorHandler)
