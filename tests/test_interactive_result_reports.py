from __future__ import annotations

from pipeline.interactive_result_reports import (
    install_result_report_menu,
    report_type_items,
    result_action_items,
)


class ChineseWizard:
    @staticmethod
    def _b(zh: str, en: str) -> str:
        del en
        return zh


def test_result_actions_merge_evaluation_into_one_report_entry() -> None:
    wizard = ChineseWizard()
    items = result_action_items(wizard, evidence={}, can_delete=True)
    values = [item.value for item in items]

    assert values == ["report", "promote", "paths", "technical", "delete", "back"]
    assert sum(item.value == "report" for item in items) == 1
    assert all(item.description.strip() for item in items)
    assert not any("尚未提供专用说明" in item.description for item in items)


def test_report_types_explain_workflow_without_internal_stage_terms() -> None:
    wizard = ChineseWizard()
    items = report_type_items(wizard, evidence={})
    text = "\n".join(f"{item.label}\n{item.description}" for item in items)

    assert [item.value for item in items] == ["quick", "deep", "back"]
    assert "快速比较" in text
    assert "1–2" in text
    assert "耗时较低" in text
    assert "耗时更高" in text
    assert "Screening" not in text
    assert "Full" not in text
    assert "finalist" not in text.casefold()


def test_report_type_labels_show_existing_evidence_in_plain_language() -> None:
    wizard = ChineseWizard()
    items = report_type_items(wizard, evidence={"screening": {}, "full": {}})

    assert "已生成" in items[0].label
    assert "已生成" in items[1].label


def test_installer_replaces_final_wizard_result_methods() -> None:
    class Wizard:
        def _training_result_detail(self):
            return "old"

    previous = Wizard._training_result_detail
    install_result_report_menu(Wizard)

    assert Wizard._training_result_detail is not previous
    assert hasattr(Wizard, "_evaluation_report_menu")
    assert hasattr(Wizard, "_evaluate_selected_run")
    assert hasattr(Wizard, "_promote_selected_run")
