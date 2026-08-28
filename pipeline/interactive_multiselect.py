from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .models import PipelineError


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

    def __post_init__(self) -> None:
        self.columns = max(1, self.columns)
        self.selected = set(self.selected or set())
        if self.size <= 0:
            self.cursor = 0
        else:
            self.cursor = min(max(0, self.cursor), self.size - 1)

    def apply(self, key: str) -> bool:
        """Apply one logical key. Return True when Enter confirms."""
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
) -> list[str]:
    """Keyboard-first bulk selector: arrows move, Space toggles, Enter confirms."""
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

    with Live(
        _render(title, options, state, page_size=page_size),
        console=console,
        refresh_per_second=12,
        transient=True,
    ) as live:
        while True:
            key = key_reader()
            if state.apply(key):
                break
            live.update(_render(title, options, state, page_size=page_size), refresh=True)
    assert state.selected is not None
    return [option.value for index, option in enumerate(options) if index in state.selected]


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
    table = Table.grid(expand=True)
    for _ in range(state.columns):
        table.add_column(ratio=1)
    assert state.selected is not None
    for row in range(rows):
        cells: list[str] = []
        for column in range(state.columns):
            index = start + row * state.columns + column
            if index >= end:
                cells.append("")
                continue
            option = options[index]
            cursor = "[bold cyan]>[/bold cyan]" if index == state.cursor else " "
            mark = "[green][x][/green]" if index in state.selected else "[dim][ ][/dim]"
            cells.append(f"{cursor} {mark} {escape(option.label)}")
        table.add_row(*cells)
    current = options[state.cursor]
    footer = (
        f"↑↓←→ move · Space select · Enter confirm · A all · N clear\n"
        f"selected {len(state.selected)}/{len(options)} · page {page + 1}/{max(1, (len(options) + page_size - 1) // page_size)}"
    )
    detail = escape(current.detail) if current.detail else ""
    return Group(
        Panel(table, title=title),
        Panel(detail or footer, subtitle=footer if detail else None, border_style="dim"),
    )


def read_key() -> str:
    if os.name == "nt":
        return _read_key_windows()
    return _read_key_posix()


def _read_key_windows() -> str:
    import msvcrt

    value = msvcrt.getwch()
    if value in {"\r", "\n"}:
        return "enter"
    if value == " ":
        return "space"
    if value.casefold() == "a":
        return "all"
    if value.casefold() == "n":
        return "none"
    if value in {"\x00", "\xe0"}:
        code = msvcrt.getwch()
        return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(code, "unknown")
    return "unknown"


def _read_key_posix() -> str:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        value = sys.stdin.read(1)
        if value in {"\r", "\n"}:
            return "enter"
        if value == " ":
            return "space"
        if value.casefold() == "a":
            return "all"
        if value.casefold() == "n":
            return "none"
        if value == "\x1b":
            sequence = ""
            for _ in range(2):
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    break
                sequence += sys.stdin.read(1)
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(sequence, "unknown")
        return "unknown"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
