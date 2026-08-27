from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.table import Table

from pipeline import i18n


def test_chinese_translation_covers_core_interactive_terms() -> None:
    i18n.set_language("zh-CN")
    try:
        assert i18n.translate("Home") == "主页"
        assert i18n.translate("Create a project") == "创建项目"
        assert i18n.translate("Training strategy") == "训练策略"
        assert i18n.translate("Project: demo") == "项目：demo"
        assert "视频下载/导入失败" in i18n.translate("[red]Video download/import failed[/red]")
    finally:
        i18n.set_language("en")


def test_english_mode_is_passthrough() -> None:
    i18n.set_language("en")
    assert i18n.translate("Create a project") == "Create a project"


def test_language_preference_round_trip(tmp_path, monkeypatch) -> None:
    config = tmp_path / "ui.json"
    monkeypatch.setenv("LORA_PIPELINE_UI_CONFIG", str(config))
    monkeypatch.delenv("LORA_PIPELINE_LANG", raising=False)

    i18n.save_language("zh-CN")

    assert i18n.load_saved_language() == "zh-CN"
    assert '"language": "zh-CN"' in config.read_text(encoding="utf-8")


def test_environment_language_overrides_saved_setting(tmp_path, monkeypatch) -> None:
    config = tmp_path / "ui.json"
    monkeypatch.setenv("LORA_PIPELINE_UI_CONFIG", str(config))
    i18n.save_language("zh-CN")
    monkeypatch.setenv("LORA_PIPELINE_LANG", "en")

    assert i18n.load_saved_language() == "en"


def test_rich_hooks_translate_tables_at_render_time() -> None:
    i18n.install_rich_hooks()
    i18n.set_language("zh-CN")
    try:
        output = StringIO()
        console = Console(file=output, force_terminal=False, width=100)
        table = Table(title="Project summary")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Strategy", "Quality")
        console.print(table)
        rendered = output.getvalue()
        assert "项目摘要" in rendered
        assert "字段" in rendered
        assert "策略" in rendered
        assert "质量优先" in rendered
    finally:
        i18n.set_language("en")
