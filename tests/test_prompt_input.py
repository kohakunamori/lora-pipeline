from __future__ import annotations

from pipeline.tty_compat import prompt_input


def test_prompt_input_passes_the_full_visible_prompt_to_builtin_input() -> None:
    seen: list[str] = []

    def fake_input(prompt: str) -> str:
        seen.append(prompt)
        return ""

    value = prompt_input(
        "请输入编号",
        default="1",
        choices=["1", "2", "3"],
        input_fn=fake_input,
    )

    assert value == "1"
    assert seen == ["请输入编号 [1/2/3] (1): "]


def test_prompt_input_reprompts_for_invalid_choice() -> None:
    answers = iter(["9", "2"])
    seen: list[str] = []

    def fake_input(prompt: str) -> str:
        seen.append(prompt)
        return next(answers)

    value = prompt_input(
        "Choose a number",
        default="1",
        choices=["1", "2", "3"],
        input_fn=fake_input,
        on_invalid=lambda value, choices: None,
    )

    assert value == "2"
    assert len(seen) == 2


def test_prompt_input_without_choices_supports_empty_default() -> None:
    value = prompt_input("Name", default="demo", input_fn=lambda prompt: "")
    assert value == "demo"
