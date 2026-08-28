from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from pipeline.interactive_multiselect import (
    MultiSelectOption,
    MultiSelectState,
    _decode_escape_sequence,
    _decode_plain_key,
    _render,
    select_many,
)


def test_multiselect_state_uses_space_to_toggle_and_enter_to_confirm() -> None:
    state = MultiSelectState(size=6, columns=2)
    assert not state.apply("right")
    assert state.cursor == 1
    assert not state.apply("space")
    assert state.selected == {1}
    assert not state.apply("down")
    assert state.cursor == 3
    assert not state.apply("space")
    assert state.selected == {1, 3}
    assert state.apply("enter")
    assert not state.cancelled


def test_multiselect_state_cancel_closes_without_confirming() -> None:
    state = MultiSelectState(size=3, selected={0})
    assert not state.apply("down")
    assert not state.apply("space")
    assert state.selected == {0, 1}
    assert state.apply("cancel")
    assert state.cancelled


def test_select_many_keyboard_contract() -> None:
    keys = iter(["right", "space", "down", "space", "enter"])
    console = Console(file=StringIO(), force_terminal=False, width=100)
    options = [MultiSelectOption(str(index), f"image-{index}") for index in range(6)]
    selected = select_many(
        console,
        "images",
        options,
        columns=2,
        key_reader=lambda: next(keys),
    )
    assert selected == ["1", "3"]


def test_select_many_cancel_returns_none_instead_of_modified_selection() -> None:
    keys = iter(["down", "space", "cancel"])
    console = Console(file=StringIO(), force_terminal=False, width=100)
    options = [MultiSelectOption(str(index), f"tag-{index}") for index in range(3)]
    selected = select_many(
        console,
        "tags",
        options,
        selected=["0"],
        key_reader=lambda: next(keys),
    )
    assert selected is None


def test_multiselect_render_is_compact_and_shows_controls() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=100)
    options = [
        MultiSelectOption("selected", "selected tag", "Selected detail"),
        MultiSelectOption("plain", "plain tag", "Plain detail"),
    ]
    state = MultiSelectState(size=2, cursor=0, selected={0})

    console.print(_render("tags", options, state, page_size=30))
    rendered = output.getvalue()

    assert "> [x] selected tag" in rendered
    assert "[ ] plain tag" in rendered
    assert "selected 1/2" in rendered
    assert "Current: Selected detail" in rendered
    assert "Q/Esc cancel" in rendered
    # Panel.fit must not stretch this tiny selector across the whole 100-column console.
    assert max(len(line) for line in rendered.splitlines()) < 90


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("", "cancel"),
        ("[A", "up"),
        ("[B", "down"),
        ("[C", "right"),
        ("[D", "left"),
        ("OA", "up"),
        ("OB", "down"),
        ("OC", "right"),
        ("OD", "left"),
        ("[1;2A", "up"),
        ("[1;5D", "left"),
    ],
)
def test_decode_escape_sequence_accepts_common_terminal_forms(
    sequence: str,
    expected: str,
) -> None:
    assert _decode_escape_sequence(sequence) == expected


def test_decode_escape_sequence_rejects_unrelated_sequences() -> None:
    assert _decode_escape_sequence("[H") == "unknown"
    assert _decode_escape_sequence("[fooA") == "unknown"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("h", "left"),
        ("j", "down"),
        ("k", "up"),
        ("l", "right"),
        ("H", "left"),
        ("J", "down"),
        ("K", "up"),
        ("L", "right"),
        ("a", "all"),
        ("n", "none"),
        ("q", "cancel"),
        ("Q", "cancel"),
        ("\x1b", "cancel"),
        (" ", "space"),
        ("\r", "enter"),
    ],
)
def test_decode_plain_key_supports_navigation_and_cancel(value: str, expected: str) -> None:
    assert _decode_plain_key(value) == expected
