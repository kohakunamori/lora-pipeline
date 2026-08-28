from __future__ import annotations

import re

from . import interactive_menu_navigation as navigation
from .wizard import MenuItem


_RICH_STYLE_WORDS = frozenset(
    {
        "black",
        "blue",
        "bold",
        "bright_black",
        "bright_blue",
        "bright_cyan",
        "bright_green",
        "bright_magenta",
        "bright_red",
        "bright_white",
        "bright_yellow",
        "cyan",
        "dim",
        "green",
        "italic",
        "magenta",
        "red",
        "reverse",
        "strike",
        "underline",
        "white",
        "yellow",
    }
)
_STYLE_TAG = re.compile(r"\[(/?)([^\[\]]+)\]")


def plain_rich_label(value: str) -> str:
    """Strip only known presentation tags from a menu label.

    Numbered menu tables may intentionally use Rich markup (for example red
    destructive actions), but the lightweight prompt line and breadcrumb are
    written as literal text. Unknown bracketed text is preserved because a
    user-controlled filename such as ``model[custom].safetensors`` is valid
    content, not presentation markup.
    """

    def replace(match: re.Match[str]) -> str:
        words = match.group(2).strip().casefold().split()
        if words and all(word in _RICH_STYLE_WORDS for word in words):
            return ""
        return match.group(0)

    return _STYLE_TAG.sub(replace, str(value))


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
