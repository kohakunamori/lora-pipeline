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
    entire Rich prompt on Backspace. Interactive code should therefore use
    :func:`prompt_input`, which passes the complete visible prompt directly to
    ``input(prompt)`` so readline owns the whole editable line.

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

    return _configure_readline(readline)


def prompt_input(
    prompt: str,
    *,
    default: str | None = None,
    choices: Sequence[str] | None = None,
    input_fn: Callable[[str], str] | None = None,
    on_invalid: Callable[[str, Sequence[str]], None] | None = None,
) -> str:
    """Read one line while giving readline the complete visible prompt.

    The function intentionally keeps formatting simple. Callers may translate
    and strip Rich markup before passing ``prompt``. When ``choices`` are
    supplied, invalid values are re-prompted instead of relying on Rich Prompt's
    separate render/input cycle.
    """

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
