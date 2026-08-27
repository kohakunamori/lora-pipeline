from __future__ import annotations

import sys

from pipeline import cli, interactive


def test_installed_entry_opens_dashboard_without_arguments(monkeypatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(sys, "argv", ["lora-pipeline"])
    monkeypatch.setattr(interactive.Wizard, "home", lambda self: called.append("home"))

    interactive.main()

    assert called == ["home"]


def test_installed_entry_preserves_advanced_subcommands(monkeypatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(sys, "argv", ["lora-pipeline", "doctor"])
    monkeypatch.setattr(cli, "main", lambda: called.append("cli"))

    interactive.main()

    assert called == ["cli"]
