from __future__ import annotations

from typing import Sequence

from .i18n import get_language
from .wizard import MenuItem, Wizard


_INSTALLED = False

_COMMON_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "back": (
        "返回上一层；不会在当前菜单执行其他操作。",
        "Return to the previous menu without running another action here.",
    ),
    "quit": (
        "退出交互模式；已经保存的项目和数据不会被修改。",
        "Exit interactive mode without changing already saved project data.",
    ),
    "cancel": (
        "取消当前流程并返回；尚未确认的更改不会提交。",
        "Cancel the current flow and return without committing unconfirmed changes.",
    ),
    "import": (
        "向当前数据集导入新的图片、视频或其他受支持的数据来源。",
        "Import a new supported image, video, or other data source into the current dataset.",
    ),
    "sources": (
        "查看已导入的数据来源，并按来源检查或管理对应图片。",
        "Inspect imported sources and manage the images associated with each source.",
    ),
    "audit": (
        "对整个数据集运行自动检查，汇总质量、重复项和潜在问题。",
        "Run automatic whole-dataset checks for quality, duplicates, and potential issues.",
    ),
    "curate": (
        "进入高级清洗与分析工具，对数据集进行更细致的筛选和整理。",
        "Open advanced curation and analysis tools for finer dataset filtering and cleanup.",
    ),
    "tag": (
        "对数据集运行自动 Tag，并保存可供后续语义和训练流程使用的标签。",
        "Auto-tag dataset images and save tags for later semantic and training workflows.",
    ),
    "concepts": (
        "管理角色特征、服饰及其触发词，并把语义关系绑定到数据集图片。",
        "Manage character features, outfits, trigger tokens, and their image assignments.",
    ),
    "review": (
        "人工浏览图片并确认保留、排除或需要进一步处理的项目。",
        "Review images manually and decide what to keep, exclude, or process further.",
    ),
    "edit_tags": (
        "人工检查并修改图片 Tag，修正自动标签中的遗漏或错误。",
        "Review and edit image tags manually to correct missing or inaccurate auto-tags.",
    ),
    "training": (
        "使用当前数据集及已保存语义配置进入训练配置流程。",
        "Start the training configuration flow using the current dataset and saved semantics.",
    ),
    "artifacts": (
        "查看当前项目的路径、训练记录、评测报告和生成产物。",
        "Inspect project paths, training runs, evaluation reports, and generated artifacts.",
    ),
    "continue": (
        "按照当前项目状态执行下一项推荐工作。",
        "Continue with the next action recommended by the current project state.",
    ),
    "save": (
        "保存当前设置或编辑结果，然后继续。",
        "Save the current settings or edits and continue.",
    ),
    "apply": (
        "应用当前选择或配置，并进入后续步骤。",
        "Apply the current selection or configuration and continue.",
    ),
    "skip": (
        "跳过当前可选步骤，并按现有状态继续后续流程。",
        "Skip this optional step and continue using the existing state.",
    ),
    "preview": (
        "先查看本操作将产生的结果或影响，不立即提交更改。",
        "Preview the result or impact of this action without committing changes immediately.",
    ),
    "delete": (
        "进入删除流程；真正删除前仍会显示目标和安全确认。",
        "Open the deletion flow; targets and safety confirmation are shown before deletion.",
    ),
    "remove": (
        "从当前配置或集合中移除此项；后续步骤会显示具体影响。",
        "Remove this item from the current configuration or collection; later steps show the impact.",
    ),
    "edit": (
        "打开该项的编辑流程，修改现有配置或元数据。",
        "Open the edit flow for this item to change its existing configuration or metadata.",
    ),
}


def _fallback_description(item: MenuItem) -> str:
    """Make missing documentation visible instead of inventing semantic meaning."""

    zh = f"⚠ “{item.label}”尚未提供专用说明；当前界面不能可靠解释这个选项的具体含义和影响。"
    en = f"⚠ “{item.label}” does not yet have dedicated help text; this UI cannot reliably explain its exact meaning or effects."
    return zh if get_language() == "zh-CN" else en


def describe_menu_item(item: MenuItem) -> MenuItem:
    """Return a menu item whose description is guaranteed to be non-empty.

    Explicit descriptions are authoritative. Generic descriptions are only used for
    well-known navigation/action values. Unknown domain-specific items receive a
    visible missing-help warning rather than fabricated semantic guidance.
    """

    if item.description.strip():
        return item
    pair = _COMMON_DESCRIPTIONS.get(item.value)
    if pair is None:
        description = _fallback_description(item)
    else:
        description = pair[0] if get_language() == "zh-CN" else pair[1]
    return MenuItem(item.value, item.label, description)


def with_menu_descriptions(items: Sequence[MenuItem]) -> list[MenuItem]:
    """Fill missing menu descriptions while preserving explicit descriptions verbatim."""

    return [describe_menu_item(item) for item in items]


def install_menu_descriptions() -> None:
    """Guarantee that every numbered CLI menu option renders explanatory text."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    original_menu = Wizard._menu

    def menu(
        self: Wizard,
        title: str,
        items: Sequence[MenuItem],
        *,
        default: str | None = None,
    ) -> str:
        return original_menu(
            self,
            title,
            with_menu_descriptions(items),
            default=default,
        )

    Wizard._menu = menu  # type: ignore[method-assign]
