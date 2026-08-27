from __future__ import annotations

from io import StringIO

from rich.console import Console

from pipeline import i18n
from pipeline.interactive_lifecycle import InteractiveWizard


def test_home_exposes_four_primary_work_areas(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(tmp_path))
    i18n.set_language("zh-CN")
    stream = StringIO()
    wizard = InteractiveWizard(console=Console(file=stream, force_terminal=False, width=140))
    seen: list[str] = []

    def menu(title, items, default=None):
        del title, default
        seen.extend(item.value for item in items)
        return "quit"

    monkeypatch.setattr(wizard, "_menu", menu)
    wizard.home()

    assert seen[:4] == ["datasets", "configs", "status", "results"]
    output = stream.getvalue()
    assert "LoRA 工作台" in output
    assert "数据集" in output
    assert "训练配置" in output
    assert "训练状态" in output
    assert "训练结果" in output
