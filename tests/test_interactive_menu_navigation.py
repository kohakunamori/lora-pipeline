from __future__ import annotations

from io import StringIO

from rich.console import Console

from pipeline.interactive_menu_navigation import (
    NumberMenuState,
    _decode_number_plain_key,
    _render_numbered_menu,
    _run_numbered_menu,
)
from pipeline.wizard import MenuItem, Wizard


def test_submenu_escape_returns_back_immediately() -> None:
    state = NumberMenuState(size=3, cursor=1)
    assert state.apply("escape", escape_target="back", quit_target=None, root_menu=False) == "back"
    assert not state.exit_armed


def test_root_requires_two_consecutive_escape_presses_to_exit() -> None:
    state = NumberMenuState(size=4)
    assert state.apply("escape", escape_target=None, quit_target="quit", root_menu=True) is None
    assert state.exit_armed
    assert state.apply("escape", escape_target=None, quit_target="quit", root_menu=True) == "quit"


def test_fast_double_escape_event_exits_root_and_backs_out_of_submenu() -> None:
    root = NumberMenuState(size=4)
    assert root.apply("double_escape", escape_target=None, quit_target="quit", root_menu=True) == "quit"

    child = NumberMenuState(size=3)
    assert child.apply("double_escape", escape_target="back", quit_target=None, root_menu=False) == "back"


def test_non_escape_key_disarms_root_exit_confirmation() -> None:
    state = NumberMenuState(size=4)
    assert state.apply("escape", escape_target=None, quit_target="quit", root_menu=True) is None
    assert state.exit_armed
    assert state.apply("down", escape_target=None, quit_target="quit", root_menu=True) is None
    assert not state.exit_armed
    assert state.cursor == 1


def test_number_entry_and_arrow_navigation_share_one_state() -> None:
    state = NumberMenuState(size=12)
    state.apply("digit:1", escape_target="back", quit_target=None, root_menu=False)
    state.apply("digit:0", escape_target="back", quit_target=None, root_menu=False)
    assert state.cursor == 9
    assert state.typed == "10"
    assert state.apply("enter", escape_target="back", quit_target=None, root_menu=False) == "select"

    state = NumberMenuState(size=3)
    state.apply("down", escape_target="back", quit_target=None, root_menu=False)
    assert state.cursor == 1


def test_zero_is_not_a_valid_menu_number() -> None:
    state = NumberMenuState(size=3, cursor=1)
    assert state.apply("digit:0", escape_target="back", quit_target=None, root_menu=False) is None
    assert state.cursor == 1
    assert state.typed == ""
    assert state.message


def test_render_places_breadcrumb_above_numbered_table() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    items = [
        MenuItem("one", "One", "First action"),
        MenuItem("back", "Back", "Return to the parent"),
    ]
    state = NumberMenuState(size=2)
    console.print(
        _render_numbered_menu(
            "Child",
            items,
            state,
            ["Home", "Datasets", "Child"],
            escape_target="back",
            root_menu=False,
        )
    )
    rendered = output.getvalue()
    assert "Home › Datasets › Child" in rendered
    assert "Child" in rendered
    assert "Esc" in rendered


def test_numbered_menu_returns_selected_item_with_fake_key_reader() -> None:
    keys = iter(["digit:2", "enter"])
    wizard = Wizard(console=Console(file=StringIO(), force_terminal=False, width=100))
    selected = _run_numbered_menu(
        wizard,
        "Demo",
        [
            MenuItem("first", "First", "First option"),
            MenuItem("second", "Second", "Second option"),
            MenuItem("back", "Back", "Return"),
        ],
        default="first",
        key_reader=lambda: next(keys),
    )
    assert selected == "second"


def test_plain_menu_key_decoder_supports_numbers_escape_and_backspace() -> None:
    assert _decode_number_plain_key("7") == "digit:7"
    assert _decode_number_plain_key("\r") == "enter"
    assert _decode_number_plain_key("\x1b") == "escape"
    assert _decode_number_plain_key("\x7f") == "backspace"
