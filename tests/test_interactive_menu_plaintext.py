from __future__ import annotations

from pipeline import interactive_menu_navigation as navigation
from pipeline.interactive_menu_plaintext import install_plain_menu_labels, plain_rich_label
from pipeline.wizard import MenuItem


def test_plain_rich_label_removes_destructive_action_markup() -> None:
    assert plain_rich_label("[red]删除这个训练结果[/red]") == "删除这个训练结果"
    assert plain_rich_label("[bold yellow]Danger[/bold yellow]") == "Danger"


def test_plain_rich_label_keeps_invalid_user_brackets_literal() -> None:
    assert plain_rich_label("model[custom].safetensors") == "model[custom].safetensors"


def test_installed_menu_prompt_never_exposes_rich_tags() -> None:
    install_plain_menu_labels()
    items = [
        MenuItem("delete", "[red]删除这个训练结果[/red]", "delete"),
        MenuItem("back", "返回", "back"),
    ]
    state = navigation.NumberMenuState(size=2, cursor=0)

    prompt = navigation._menu_prompt(items, state, root_menu=False)

    assert "删除这个训练结果" in prompt
    assert "[red]" not in prompt
    assert "[/red]" not in prompt
