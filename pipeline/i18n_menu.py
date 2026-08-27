from __future__ import annotations

from typing import Any, Sequence

from .i18n import get_language, save_language, set_language
from .wizard import MenuItem, Wizard


_INSTALLED = False


def install_language_menu() -> None:
    """Add an in-app language switcher without changing workflow semantics."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    original_menu = Wizard._menu

    def menu(
        self: Wizard,
        title: str,
        items: Sequence[MenuItem],
        *,
        default: str | None = None,
    ) -> str:
        if title != "Home":
            return original_menu(self, title, items, default=default)

        augmented = list(items)
        language_item = MenuItem(
            "__language__",
            "语言 / Language",
            "切换简体中文或 English；选择会自动保存。",
        )
        # Keep Exit as the last home item when possible.
        exit_index = next((i for i, item in enumerate(augmented) if item.value == "quit"), len(augmented))
        augmented.insert(exit_index, language_item)

        while True:
            result = original_menu(self, title, augmented, default=default)
            if result != "__language__":
                return result
            selected = original_menu(
                self,
                "界面语言 / Interface language",
                [
                    MenuItem("zh-CN", "简体中文", "默认语言"),
                    MenuItem("en", "English", "English interface"),
                ],
                default=get_language(),
            )
            set_language(selected)
            save_language(selected)
            if selected == "zh-CN":
                self.console.print("[green]界面语言已切换为简体中文。[/green]")
            else:
                self.console.print("[green]Interface language changed to English.[/green]")

    Wizard._menu = menu  # type: ignore[method-assign]
