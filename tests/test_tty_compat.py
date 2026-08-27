from __future__ import annotations

from pipeline import tty_compat


class _FakeReadline:
    def __init__(self) -> None:
        self.bindings: list[str] = []

    def parse_and_bind(self, binding: str) -> None:
        self.bindings.append(binding)


class _NotATty:
    def isatty(self) -> bool:
        return False


def test_readline_binds_ctrl_h_and_del_to_backward_delete() -> None:
    readline = _FakeReadline()

    assert tty_compat._configure_readline(readline) is True
    assert readline.bindings == [
        r'"\C-H": backward-delete-char',
        r'"\C-?": backward-delete-char',
    ]


def test_non_tty_input_is_not_modified() -> None:
    assert tty_compat.configure_interactive_input(_NotATty()) is False


def test_readline_binding_failure_is_nonfatal() -> None:
    class BrokenReadline:
        def parse_and_bind(self, binding: str) -> None:
            del binding
            raise RuntimeError("unsupported")

    assert tty_compat._configure_readline(BrokenReadline()) is False


def test_rich_console_input_passes_complete_prompt_to_readline() -> None:
    seen: list[str] = []

    class FakeConsole:
        def input(self, prompt: object = "", *args: object, **kwargs: object) -> str:
            del prompt, args, kwargs
            return "original"

    assert tty_compat._configure_rich_console_input(
        FakeConsole,
        input_fn=lambda prompt: seen.append(prompt) or "2",
    ) is True

    console = FakeConsole()
    result = console.input("[bold]请输入编号[/bold] [1/2/3/4/5/6] (1): ")

    assert result == "2"
    assert seen == ["请输入编号 [1/2/3/4/5/6] (1): "]


def test_rich_console_input_keeps_special_stream_path_unchanged() -> None:
    class FakeConsole:
        def input(self, prompt: object = "", *args: object, **kwargs: object) -> str:
            del prompt, args, kwargs
            return "original"

    assert tty_compat._configure_rich_console_input(
        FakeConsole,
        input_fn=lambda prompt: "patched",
    ) is True
    assert FakeConsole().input("Prompt: ", stream=object()) == "original"
