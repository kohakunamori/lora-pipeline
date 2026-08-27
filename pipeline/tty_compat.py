from __future__ import annotations

import sys
from typing import Any, TextIO


_READLINE_BINDINGS = (
    r'"\C-H": backward-delete-char',
    r'"\C-?": backward-delete-char',
)


def configure_interactive_input(stream: TextIO | None = None) -> bool:
    """Enable reliable line editing for the interactive SSH/TUI workflow.

    Some terminals send Backspace as Ctrl-H (0x08), while others send DEL
    (0x7f).  GNU readline can accept both without changing the parent shell's
    TTY settings.  Importing readline also makes Python's ``input()`` -- and
    therefore Rich ``Prompt.ask()`` -- use readline editing.

    This is deliberately a no-op for non-interactive stdin and on platforms
    where readline is unavailable.
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


def _configure_readline(readline_module: Any) -> bool:
    try:
        for binding in _READLINE_BINDINGS:
            readline_module.parse_and_bind(binding)
    except (AttributeError, RuntimeError, ValueError):
        return False
    return True
