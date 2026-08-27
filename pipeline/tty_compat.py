from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any, TextIO


_READLINE_BINDINGS = (
    r'"\C-H": backward-delete-char',
    r'"\C-?": backward-delete-char',
)


def configure_interactive_input(stream: TextIO | None = None) -> bool:
    """Enable reliable line editing for the interactive SSH/TUI workflow.

    Some terminals send Backspace as Ctrl-H (0x08), while others send DEL
    (0x7f). GNU readline can accept both without changing the parent shell's
    TTY settings.

    Rich normally renders its prompt first and then calls ``input()`` with an
    empty prompt string. Readline cannot account for those already-rendered
    columns when it redisplays the edit buffer, which may visually erase the
    entire Rich prompt on Backspace. We therefore also patch ``Console.input``
    so the complete visible Rich prompt is passed directly to
    ``input(prompt)``. Readline then owns the whole editable line and can
    redisplay Backspace correctly.

    This setup is deliberately a no-op for non-interactive stdin and on
    platforms where readline is unavailable.
    """

    input_stream = stream or sys.stdin
    try:
        if not input_stream.isatty():
            return False
    except (AttributeError, OSError):
        return False

    try:
        import readline
    except ImportError:
        return False

    if not _configure_readline(readline):
        return False
    return _configure_rich_console_input()


def prompt_input(
    prompt: str,
    *,
    default: str | None = None,
    choices: Sequence[str] | None = None,
    input_fn: Callable[[str], str] | None = None,
    on_invalid: Callable[[str, Sequence[str]], None] | None = None,
) -> str:
    """Read one simple prompt while giving readline the whole visible line."""

    read = input_fn or input
    allowed = tuple(str(value) for value in choices) if choices is not None else None
    suffix = ""
    if allowed:
        suffix += " [" + "/".join(allowed) + "]"
    if default is not None:
        suffix += f" ({default})"
    visible_prompt = f"{prompt}{suffix}: "

    while True:
        raw = read(visible_prompt)
        value = raw if raw != "" else default
        if value is None:
            value = ""
        value = str(value)
        if allowed is None or value in allowed:
            return value
        if on_invalid is not None:
            on_invalid(value, allowed)


def _configure_readline(readline_module: Any) -> bool:
    try:
        for binding in _READLINE_BINDINGS:
            readline_module.parse_and_bind(binding)
    except (AttributeError, RuntimeError, ValueError):
        return False
    return True


def _configure_rich_console_input(
    console_class: type[Any] | None = None,
    *,
    input_fn: Callable[[str], str] | None = None,
) -> bool:
    """Make Rich prompts readline-aware without changing parent TTY settings."""

    try:
        if console_class is None:
            from rich.console import Console

            console_class = Console
        from rich.text import Text
    except ImportError:
        return False

    current = getattr(console_class, "input", None)
    if current is None:
        return False
    if getattr(current, "_lora_readline_safe", False):
        return True

    original = current
    read = input_fn or input

    def console_input(self: Any, prompt: Any = "", *args: Any, **kwargs: Any) -> str:
        # Password input and explicit streams have special Rich semantics; keep
        # those paths untouched. Normal interactive prompts use the safe path.
        if kwargs.get("password") or kwargs.get("stream") is not None:
            return original(self, prompt, *args, **kwargs)

        try:
            if isinstance(prompt, Text):
                visible = prompt.plain
            elif isinstance(prompt, str):
                if kwargs.get("markup", True):
                    visible = Text.from_markup(prompt).plain
                else:
                    visible = prompt
            else:
                visible = str(prompt)
        except (TypeError, ValueError):
            return original(self, prompt, *args, **kwargs)
        return read(visible)

    setattr(console_input, "_lora_readline_safe", True)
    setattr(console_input, "_lora_original", original)
    console_class.input = console_input
    return True
