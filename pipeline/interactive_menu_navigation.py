from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from types import FrameType
from typing import Sequence

from rich.cells import cell_len
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from .i18n import get_language
from .interactive_menu_descriptions import with_menu_descriptions
from .interactive_multiselect import _decode_escape_sequence
from .wizard import MenuItem, Wizard


@dataclass
class _MenuFrameState:
    frame: FrameType
    title: str
    selected_label: str | None = None


@dataclass
class NumberMenuState:
    size: int
    cursor: int = 0
    typed: str = ""
    exit_armed: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("NumberMenuState requires at least one item")
        self.cursor = min(max(0, self.cursor), self.size - 1)

    def apply(
        self,
        key: str,
        *,
        escape_target: str | None,
        quit_target: str | None,
        root_menu: bool,
    ) -> str | None:
        """Apply one key and return a special action when the menu should close."""
        self.message = ""
        if key == "double_escape":
            self.typed = ""
            self.exit_armed = False
            if root_menu and quit_target is not None:
                return quit_target
            if escape_target is not None:
                return escape_target
            self.message = _tr("当前菜单没有可返回的上一层。", "This menu has no previous-menu action.")
            return None
        if key == "escape":
            self.typed = ""
            if root_menu and quit_target is not None:
                if self.exit_armed:
                    return quit_target
                self.exit_armed = True
                return None
            self.exit_armed = False
            if escape_target is not None:
                return escape_target
            self.message = _tr("当前菜单没有可返回的上一层。", "This menu has no previous-menu action.")
            return None

        # Only two consecutive Esc presses may exit the root menu.
        self.exit_armed = False

        if key == "up":
            self.typed = ""
            self.cursor = max(0, self.cursor - 1)
        elif key == "down":
            self.typed = ""
            self.cursor = min(self.size - 1, self.cursor + 1)
        elif key == "backspace":
            self.typed = self.typed[:-1]
            self._sync_cursor_from_typed()
        elif key.startswith("digit:"):
            digit = key.split(":", 1)[1]
            candidate = self.typed + digit
            # Keep a plausible prefix so menus with >=10 rows remain usable.
            number = int(candidate) if candidate else 0
            if 1 <= number <= self.size:
                self.typed = candidate
                self.cursor = number - 1
            else:
                self.message = _tr("编号超出当前菜单范围。", "That number is outside this menu.")
        elif key == "enter":
            if self.typed:
                number = int(self.typed)
                if not 1 <= number <= self.size:
                    self.message = _tr("请输入有效编号。", "Enter a valid menu number.")
                    return None
                self.cursor = number - 1
            return "select"
        return None

    def _sync_cursor_from_typed(self) -> None:
        if not self.typed:
            return
        number = int(self.typed)
        if 1 <= number <= self.size:
            self.cursor = number - 1


def install_menu_navigation() -> None:
    """Install breadcrumb, stable numbered input, screen refresh, and Esc semantics."""
    current = Wizard._menu
    if getattr(current, "_lora_navigation_menu", False):
        return

    def menu(
        self: Wizard,
        title: str,
        items: Sequence[MenuItem],
        *,
        default: str | None = None,
    ) -> str:
        return _run_numbered_menu(self, title, items, default=default)

    setattr(menu, "_lora_navigation_menu", True)
    setattr(menu, "_lora_original", current)
    Wizard._menu = menu


def _run_numbered_menu(
    wizard: Wizard,
    title: str,
    items: Sequence[MenuItem],
    *,
    default: str | None,
    key_reader=None,
) -> str:
    """Render a numbered menu once and keep only the small prompt line dynamic.

    Rich ``Live`` previously rewrote the whole table after every key press. Some
    SSH/WebTTY terminals do not reliably implement the cursor-up sequences Live
    needs, so every refresh was appended to the screen instead of replacing the
    previous frame. A numbered CLI menu does not need full-screen animation:
    render the breadcrumb/table once, update only the final prompt with carriage
    return, then clear the completed page before the next screen is rendered.
    """

    items = with_menu_descriptions(items)
    if not items:
        raise ValueError("Menu requires at least one item")

    caller = inspect.currentframe()
    assert caller is not None
    caller = caller.f_back
    assert caller is not None
    # _run_numbered_menu is normally called by the installed Wizard._menu wrapper.
    if caller.f_code.co_name == "menu" and caller.f_back is not None:
        caller = caller.f_back

    path = _menu_path(wizard, caller, title)
    by_value = {item.value: index for index, item in enumerate(items)}
    default_index = by_value.get(default or "", 0)
    state = NumberMenuState(size=len(items), cursor=default_index)
    escape_target = _escape_target(items)
    quit_target = "quit" if any(item.value == "quit" for item in items) else None
    root_menu = len(path) == 1 and quit_target is not None
    read_key = key_reader or read_number_menu_key

    # Important: do not clear here. Many pages render a dataset/project summary
    # immediately before calling _menu(); the static numbered menu must remain on
    # the same page as that context. The previous menu cleared itself on exit.
    wizard.console.print(
        _render_numbered_menu(
            title,
            items,
            state,
            path,
            escape_target=escape_target,
            root_menu=root_menu,
        )
    )

    prompt_width = 0
    chosen_value: str | None = None
    chosen_label: str | None = None
    while chosen_value is None:
        prompt_width = _rewrite_menu_prompt(
            wizard.console,
            _menu_prompt(items, state, root_menu=root_menu),
            previous_width=prompt_width,
        )
        key = read_key()
        result = state.apply(
            key,
            escape_target=escape_target,
            quit_target=quit_target,
            root_menu=root_menu,
        )
        if result == "select":
            selected = items[state.cursor]
            chosen_value = selected.value
            chosen_label = selected.label
            break
        if result is not None:
            chosen_value = result
            chosen_label = next((item.label for item in items if item.value == result), None)
            break

    _finish_menu_prompt(wizard.console, prompt_width)
    assert chosen_value is not None
    _record_selected_label(wizard, caller, chosen_label)
    _clear_terminal(wizard.console)
    return chosen_value


def _menu_path(wizard: Wizard, caller: FrameType, title: str) -> list[str]:
    chain: list[FrameType] = []
    frame: FrameType | None = caller
    while frame is not None:
        chain.append(frame)
        frame = frame.f_back

    registry: dict[int, _MenuFrameState] = getattr(wizard, "_menu_navigation_frames", {})
    active = {id(frame): frame for frame in chain}
    registry = {
        frame_id: entry
        for frame_id, entry in registry.items()
        if frame_id in active and active[frame_id] is entry.frame
    }
    previous = registry.get(id(caller))
    registry[id(caller)] = _MenuFrameState(
        frame=caller,
        title=title,
        selected_label=previous.selected_label if previous is not None else None,
    )
    setattr(wizard, "_menu_navigation_frames", registry)

    parts: list[str] = []
    for active_frame in reversed(chain):
        entry = registry.get(id(active_frame))
        if entry is None or entry.frame is not active_frame:
            continue
        _append_path_part(parts, entry.title)
        if active_frame is not caller and entry.selected_label:
            _append_path_part(parts, entry.selected_label)

    root = _tr("主页", "Home")
    if not parts:
        parts = [root, title] if title != root else [root]
    elif parts[0].casefold() not in {"home", "主页"}:
        parts.insert(0, root)
    return parts


def _record_selected_label(wizard: Wizard, caller: FrameType, label: str | None) -> None:
    registry: dict[int, _MenuFrameState] = getattr(wizard, "_menu_navigation_frames", {})
    entry = registry.get(id(caller))
    if entry is not None and entry.frame is caller:
        entry.selected_label = label


def _append_path_part(parts: list[str], value: str) -> None:
    value = str(value).strip()
    if not value:
        return
    if parts and parts[-1].casefold() == value.casefold():
        return
    parts.append(value)


def _escape_target(items: Sequence[MenuItem]) -> str | None:
    values = {item.value for item in items}
    if "back" in values:
        return "back"
    if "cancel" in values:
        return "cancel"
    return None


def _render_numbered_menu(
    title: str,
    items: Sequence[MenuItem],
    state: NumberMenuState,
    path: Sequence[str],
    *,
    escape_target: str | None,
    root_menu: bool,
):
    del state  # Selection state is shown on the lightweight prompt line below.

    breadcrumb = Text()
    breadcrumb.append(_tr("路径：", "Path: "), style="dim")
    for index, part in enumerate(path):
        if index:
            breadcrumb.append(" › ", style="dim")
        breadcrumb.append(part, style="bold cyan" if index == len(path) - 1 else "dim")

    table = Table(title=title)
    table.add_column("#", justify="right", style="bold cyan")
    table.add_column(_tr("操作", "Action"), style="bold")
    table.add_column(_tr("说明", "Description"))
    for index, item in enumerate(items, start=1):
        table.add_row(str(index), item.label, item.description)

    if root_menu:
        help_line = _tr(
            "↑/↓ 改变当前编号 · 也可直接输入编号 · Enter 确认 · Esc×2 退出",
            "↑/↓ change number · or type a number · Enter confirm · Esc×2 exit",
        )
    elif escape_target is not None:
        help_line = _tr(
            "↑/↓ 改变当前编号 · 也可直接输入编号 · Enter 确认 · Esc 返回上一菜单",
            "↑/↓ change number · or type a number · Enter confirm · Esc back",
        )
    else:
        help_line = _tr(
            "↑/↓ 改变当前编号 · 也可直接输入编号 · Enter 确认",
            "↑/↓ change number · or type a number · Enter confirm",
        )
    return Group(breadcrumb, table, Text(help_line, style="dim"))


def _menu_prompt(items: Sequence[MenuItem], state: NumberMenuState, *, root_menu: bool) -> str:
    number = int(state.typed) if state.typed else state.cursor + 1
    label = items[state.cursor].label
    prompt = _tr(
        f"当前：{number} · {label}",
        f"Current: {number} · {label}",
    )
    if state.exit_armed and root_menu:
        prompt += _tr("  · 再按一次 Esc 退出", "  · press Esc again to exit")
    elif state.message:
        prompt += "  · " + state.message
    return prompt


def _rewrite_menu_prompt(console: Console, text: str, *, previous_width: int) -> int:
    """Rewrite one prompt line using only carriage return and spaces.

    Avoid ANSI cursor-up / erase-display sequences here: those are exactly what
    caused full menu frames to accumulate on some WebTTY/SSH frontends. Width is
    measured in terminal cells so CJK text is erased cleanly as labels change.
    """

    plain = str(text)
    current_width = cell_len(plain)
    width = max(previous_width, current_width)
    stream = console.file
    try:
        stream.write("\r" + plain + (" " * max(0, width - current_width)))
        stream.flush()
    except (AttributeError, OSError):
        console.print(plain, end="")
    return current_width


def _finish_menu_prompt(console: Console, previous_width: int) -> None:
    stream = console.file
    try:
        stream.write("\r" + (" " * previous_width) + "\r\n")
        stream.flush()
    except (AttributeError, OSError):
        console.print()


def read_number_menu_key() -> str:
    if os.name == "nt":
        return _read_number_menu_key_windows()
    return _read_number_menu_key_posix()


def _read_number_menu_key_windows() -> str:
    import msvcrt

    value = msvcrt.getwch()
    if value in {"\x00", "\xe0"}:
        code = msvcrt.getwch()
        return {"H": "up", "P": "down"}.get(code, "unknown")
    return _decode_number_plain_key(value)


def _read_number_menu_key_posix() -> str:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = os.read(fd, 1)
        if not first:
            return "unknown"
        if first != b"\x1b":
            return _decode_number_plain_key(first.decode("ascii", errors="ignore"))

        sequence = bytearray()
        for _ in range(16):
            timeout = 0.20 if not sequence else 0.05
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                break
            value = os.read(fd, 1)
            if not value:
                break
            if not sequence and value == b"\x1b":
                return "double_escape"
            sequence.extend(value)
            if len(sequence) >= 2 and sequence[-1:] in b"ABCD~":
                break
        if not sequence:
            return "escape"
        decoded = _decode_escape_sequence(sequence.decode("ascii", errors="ignore"))
        return decoded if decoded in {"up", "down"} else "unknown"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _decode_number_plain_key(value: str) -> str:
    if value in {"\r", "\n"}:
        return "enter"
    if value == "\x1b":
        return "escape"
    if value in {"\x08", "\x7f"}:
        return "backspace"
    if value.isdigit() and len(value) == 1:
        return f"digit:{value}"
    return "unknown"


def _clear_terminal(console: Console) -> None:
    """Clear the completed page before the next page starts rendering."""
    if not console.is_terminal:
        return
    try:
        console.clear(home=True)
    except (AttributeError, OSError):
        # A non-standard console should not make menu navigation unusable.
        pass


def _tr(zh: str, en: str) -> str:
    return zh if get_language() == "zh-CN" else en
