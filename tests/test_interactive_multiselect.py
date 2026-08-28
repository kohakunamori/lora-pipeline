from __future__ import annotations

from io import StringIO

from rich.console import Console

from pipeline.interactive_multiselect import MultiSelectOption, MultiSelectState, select_many


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
