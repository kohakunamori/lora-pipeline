from __future__ import annotations

from pipeline import i18n
from pipeline.interactive_menu_descriptions import with_menu_descriptions
from pipeline.wizard import MenuItem


def test_menu_descriptions_preserve_explicit_text_and_fill_common_actions() -> None:
    i18n.set_language("en")
    items = with_menu_descriptions(
        [
            MenuItem("custom", "Custom", "Existing explanation."),
            MenuItem("back", "Back"),
            MenuItem("tag", "Auto-tag images"),
        ]
    )

    assert items[0].description == "Existing explanation."
    assert "previous menu" in items[1].description
    assert "Auto-tag" in items[2].description
    assert all(item.description.strip() for item in items)


def test_unknown_menu_action_still_gets_a_localized_non_empty_description() -> None:
    i18n.set_language("zh-CN")
    item = with_menu_descriptions([MenuItem("future_action", "未来功能")])[0]

    assert item.description.strip()
    assert "未来功能" in item.description
