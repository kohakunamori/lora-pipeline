from __future__ import annotations

from rich.errors import MarkupError
from rich.text import Text

from . import interactive_menu_navigation as navigation
from .wizard import MenuItem


def plain_rich_label(value: str) -> str:
    """Return the visible text of a Rich-markup menu label.

    Numbered menu tables may intentionally use Rich markup (for example red
    destructive actions), but the lightweight prompt line and breadcrumb are
    written as literal text. Feeding markup strings into those literal surfaces
    exposes tags such as ``[red]`` to the user.
    """

    text = str(value)
    try:
        return Text.from_markup(text).plain
    except MarkupError:
        # A user-controlled filename may legitimately contain square brackets.
        # If it is not valid Rich markup, keep the original text unchanged.
        return text


def install_plain_menu_labels() -> None:
    """Keep table styling while using plain labels in prompts and breadcrumbs."""

    current_prompt = navigation._menu_prompt
    if getattr(current_prompt, "_lora_plain_labels", False):
        return

    original_prompt = current_prompt
    original_record = navigation._record_selected_label

    def menu_prompt(items, state, *, root_menu):
        plain_items = [
            MenuItem(item.value, plain_rich_label(item.label), item.description)
            for item in items
        ]
        return original_prompt(plain_items, state, root_menu=root_menu)

    def record_selected_label(wizard, caller, label):
        return original_record(
            wizard,
            caller,
            plain_rich_label(label) if label is not None else None,
        )

    setattr(menu_prompt, "_lora_plain_labels", True)
    navigation._menu_prompt = menu_prompt
    navigation._record_selected_label = record_selected_label
