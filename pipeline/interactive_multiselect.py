from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .i18n import get_language
from .models import PipelineError


_ARROW_KEYS = {
    "A": "up",
    "B": "down",
    "C": "right",
    "D": "left",
}

_PLAIN_KEYS = {
    "a": "all",
    "n": "none",
    "q": "cancel",
    "h": "left",
    "j": "down",
    "k": "up",
    "l": "right",
}


@dataclass(frozen=True)
class MultiSelectOption:
    value: str
    label: str
    detail: str = ""


@dataclass
class MultiSelectState:
    size: int
    columns: int = 1
    cursor: int = 0
    selected: set[int] | None = None
    cancelled: bool = False

    def __post_init__(self) -> None:
        self.columns = max(1, self.columns)
        self.selected = set(self.selected or set())
        if self.size <= 0:
            self.cursor = 0
        else:
            self.cursor = min(max(0, self.cursor), self.size - 1)

    def apply(self, key: str) -> bool:
        """Apply one logical key. Return True when the selector should close."""
        if key == "cancel":
            self.cancelled = True
            return True
        if self.size <= 0:
            return key == "enter"
        if key == "left":
            self.cursor = max(0, self.cursor - 1)
        elif key == "right":
            self.cursor = min(self.size - 1, self.cursor + 1)
        elif key == "up":
            self.cursor = max(0, self.cursor - self.columns)
        elif key == "down":
            self.cursor = min(self.size - 1, self.cursor + self.columns)
        elif key == "space":
            assert self.selected is not None
            if self.cursor in self.selected:
                self.selected.remove(self.cursor)
            else:
                self.selected.add(self.cursor)
        elif key == "all":
            self.selected = set(range(self.size))
        elif key == "none":
            self.selected = set()
        elif key == "enter":
            return True
        return False


def select_many(
    console: Console,
    title: str,
    options: Sequence[MultiSelectOption],
    *,
    selected: Iterable[str] = (),
    columns: int = 1,
    page_size: int = 30,
    key_reader: Callable[[], str] | None = None,
) -> list[str] | None:
    """Keyboard-first selector. Enter commits; Q/Esc cancels without a result.

    The selector deliberately avoids Rich ``Live``. Some SSH/WebTTY frontends do
    not implement the cursor-up sequences Live uses to replace previous frames,
    causing every key press to append another copy of the selector diagonally
    across the terminal. Checkbox state must still update after Space/A/N, so each
    logical change uses the much simpler terminal contract: clear screen, home the
    cursor, and print one complete static frame.
    """
    if not options:
        return []
    selected_values = set(selected)
    initial = {index for index, option in enumerate(options) if option.value in selected_values}
    state = MultiSelectState(size=len(options), columns=columns, selected=initial)
    if key_reader is None:
        try:
            is_tty = sys.stdin.isatty()
        except (AttributeError, OSError):
            is_tty = False
        if not is_tty:
            raise PipelineError("Keyboard multi-select requires an interactive TTY")
        key_reader = read_key

    _draw_selector(console, title, options, state, page_size=page_size)
    while True:
        key = key_reader()
        if state.apply(key):
            break
        _draw_selector(console, title, options, state, page_size=page_size)

    # Match the old transient selector semantics: once the user confirms/cancels,
    # remove the selector page before the caller renders its next screen.
    _clear_selector_screen(console)
    if state.cancelled:
        return None
    assert state.selected is not None
    return [option.value for index, option in enumerate(options) if index in state.selected]


def _draw_selector(
    console: Console,
    title: str,
    options: Sequence[MultiSelectOption],
    state: MultiSelectState,
    *,
    page_size: int,
) -> None:
    _clear_selector_screen(console)
    console.print(_render(title, options, state, page_size=page_size))


def _clear_selector_screen(console: Console) -> None:
    """Use only the basic erase-display + cursor-home terminal operation."""
    if not console.is_terminal:
        return
    try:
        console.clear(home=True)
    except (AttributeError, OSError):
        # A non-standard output stream should not make selection unusable.
        pass


def _option_cell(option: MultiSelectOption, *, cursor: bool, selected: bool) -> Text:
    """Build a literal option row without letting checkbox text be parsed as Rich markup."""
    cell = Text()
    cell.append("> " if cursor else "  ", style="bold cyan" if cursor else None)
    cell.append("[x]" if selected else "[ ]", style="green" if selected else "dim")
    cell.append(" ")
    cell.append(option.label)
    return cell


def _render(
    title: str,
    options: Sequence[MultiSelectOption],
    state: MultiSelectState,
    *,
    page_size: int,
):
    page_size = max(state.columns, page_size)
    page_size -= page_size % state.columns
    page_size = max(state.columns, page_size)
    page = state.cursor // page_size
    start = page * page_size
    end = min(len(options), start + page_size)
    rows = (end - start + state.columns - 1) // state.columns

    # This is a selector, not a dashboard: size the list to its contents instead
    # of stretching both the table and surrounding panels across the terminal.
    table = Table.grid(padding=(0, 2))
    for _ in range(state.columns):
        table.add_column()
    assert state.selected is not None
    for row in range(rows):
        cells: list[Text] = []
        for column in range(state.columns):
            index = start + row * state.columns + column
            if index >= end:
                cells.append(Text())
                continue
            option = options[index]
            cells.append(
                _option_cell(
                    option,
                    cursor=index == state.cursor,
                    selected=index in state.selected,
                )
            )
        table.add_row(*cells)

    current = options[state.cursor]
    pages = max(1, (len(options) + page_size - 1) // page_size)
    detail = Text()
    if current.detail:
        detail.append(_tr("当前：", "Current: "), style="bold")
        detail.append(current.detail)
    status = Text(
        _tr(
            f"已选择 {len(state.selected)}/{len(options)} · 第 {page + 1}/{pages} 页",
            f"selected {len(state.selected)}/{len(options)} · page {page + 1}/{pages}",
        )
    )
    help_text = Text(
        _tr(
            "↑↓←→ / HJKL 移动 · Space 选择/取消 · Enter 保存 · Q/Esc 取消 · A 全选 · N 清空",
            "↑↓←→ / HJKL move · Space toggle · Enter save · Q/Esc cancel · A all · N clear",
        ),
        style="dim",
    )
    return Group(
        Panel.fit(table, title=title, border_style="cyan", padding=(0, 1)),
        detail,
        status,
        help_text,
    )


def _decode_plain_key(value: str) -> str:
    if value in {"\r", "\n"}:
        return "enter"
    if value == " ":
        return "space"
    if value == "\x1b":
        return "cancel"
    return _PLAIN_KEYS.get(value.casefold(), "unknown")


def _decode_escape_sequence(sequence: str) -> str:
    """Decode bytes following ESC; an ESC without a sequence means cancel."""
    if not sequence:
        return "cancel"
    if len(sequence) < 2 or sequence[0] not in {"[", "O"}:
        return "unknown"
    final = sequence[-1]
    action = _ARROW_KEYS.get(final)
    if action is None:
        return "unknown"
    body = sequence[1:-1]
    if sequence[0] == "O":
        return action if not body else "unknown"
    if body and not all(char.isdigit() or char == ";" for char in body):
        return "unknown"
    return action


def read_key() -> str:
    if os.name == "nt":
        return _read_key_windows()
    return _read_key_posix()


def _read_key_windows() -> str:
    import msvcrt

    value = msvcrt.getwch()
    if value in {"\x00", "\xe0"}:
        code = msvcrt.getwch()
        return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(code, "unknown")
    return _decode_plain_key(value)


def _read_key_posix() -> str:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # Read directly from the terminal fd. Mixing TextIOWrapper.read() with
        # select(fd) can strand the rest of an escape sequence in Python's
        # userspace buffer, making normal arrow keys look like a lone ESC.
        first = os.read(fd, 1)
        if not first:
            return "unknown"
        if first != b"\x1b":
            return _decode_plain_key(first.decode("ascii", errors="ignore"))

        sequence = bytearray()
        for _ in range(16):
            timeout = 0.20 if not sequence else 0.05
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                break
            value = os.read(fd, 1)
            if not value:
                break
            sequence.extend(value)
            if len(sequence) >= 2 and sequence[-1:] in b"ABCD~":
                break
        return _decode_escape_sequence(sequence.decode("ascii", errors="ignore"))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _tr(zh: str, en: str) -> str:
    return zh if get_language() == "zh-CN" else en
