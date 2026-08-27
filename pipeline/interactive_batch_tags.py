from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from .dataset_tag_editor import BatchTagAction, batch_edit_tags, parse_tag_input
from .dataset_workspace import DatasetWorkspace, parse_number_selection
from .interactive_deletion import InteractiveWizard as BaseInteractiveWizard
from .wizard import MenuItem


class InteractiveWizard(BaseInteractiveWizard):
    """Add batch tag operations on top of the existing Dataset editor."""

    def _choose_and_edit_tag(
        self,
        workspace: DatasetWorkspace,
        *,
        source_id: str | None = None,
    ) -> None:
        action = self._menu(
            self._b("Tag 编辑", "Tag editor"),
            [
                MenuItem("single", self._b("修改单张图片", "Edit one image")),
                MenuItem("prepend", self._b("批量添加到 Tag 首部", "Batch prepend tags")),
                MenuItem("append", self._b("批量添加到 Tag 尾部", "Batch append tags")),
                MenuItem("remove", self._b("批量删除指定 Tag", "Batch remove tags")),
                MenuItem("back", self._b("返回", "Back")),
            ],
            default="single",
        )
        if action == "back":
            return
        if action == "single":
            super()._choose_and_edit_tag(workspace, source_id=source_id)
            return
        self._batch_edit_tags(workspace, action=action, source_id=source_id)

    def _batch_edit_tags(
        self,
        workspace: DatasetWorkspace,
        *,
        action: BatchTagAction,
        source_id: str | None,
    ) -> None:
        active_items = workspace.items(
            source_id=source_id,
            include_disabled=False,
            include_excluded=False,
        )
        all_items = workspace.items(
            source_id=source_id,
            include_disabled=True,
            include_excluded=True,
        )
        if not all_items:
            self.console.print(self._b("[yellow]没有可编辑图片。[/yellow]", "[yellow]No editable images.[/yellow]"))
            return

        scope_items = [
            MenuItem(
                "active",
                self._b("全部当前可训练图片", "All currently trainable images"),
                self._b(
                    f"仅启用来源且未排除，共 {len(active_items)} 张。",
                    f"Enabled sources and non-excluded items only: {len(active_items)} image(s).",
                ),
            ),
            MenuItem(
                "select",
                self._b("按编号 / 范围选择", "Select by number / range"),
                self._b("可以包含已排除图片或停用来源。", "May include excluded images or disabled sources."),
            ),
            MenuItem("back", self._b("返回", "Back")),
        ]
        scope = self._menu(self._b("批量修改范围", "Batch edit scope"), scope_items, default="active")
        if scope == "back":
            return
        if scope == "active":
            if not active_items:
                self.console.print(
                    self._b(
                        "[yellow]当前范围没有启用且未排除的图片。[/yellow]",
                        "[yellow]There are no enabled, non-excluded images in this scope.[/yellow]",
                    )
                )
                return
            target_items = active_items
        else:
            limit = min(len(all_items), 100)
            table = Table(title=self._b("选择要批量修改的图片", "Select images for batch tag editing"))
            table.add_column("#", justify="right")
            table.add_column(self._b("状态", "State"))
            table.add_column(self._b("来源", "Source"))
            table.add_column(self._b("文件", "File"))
            table.add_column("Tags")
            for index, item in enumerate(all_items[:limit], start=1):
                state = self._b("排除", "excluded") if item.excluded else self._b("保留", "keep")
                table.add_row(
                    str(index),
                    state,
                    item.source_id,
                    item.relative.as_posix(),
                    self._truncate(workspace.caption_text(item.key), 55),
                )
            self.console.print(table)
            if len(all_items) > limit:
                self.console.print(
                    self._b(
                        f"[dim]当前选择器显示前 {limit} 张；大量图片建议使用“全部当前可训练图片”。[/dim]",
                        f"[dim]The selector shows the first {limit}; use all trainable images for larger batches.[/dim]",
                    )
                )
            raw = self._ask_text(
                self._b(
                    "输入编号或范围（例如 1,3-5）",
                    "Enter numbers/ranges (for example 1,3-5)",
                )
            )
            numbers = parse_number_selection(raw, maximum=limit)
            target_items = [all_items[number - 1] for number in numbers]

        text = self._ask_text(
            self._b(
                "Tag（逗号或换行分隔）",
                "Tags (comma- or newline-separated)",
            )
        )
        tags = parse_tag_input(text)
        operation = {
            "prepend": self._b("添加到首部", "prepend"),
            "append": self._b("添加到尾部", "append"),
            "remove": self._b("删除", "remove"),
        }[action]
        self.console.print(
            Panel.fit(
                self._b(
                    f"操作：{operation}\n图片：{len(target_items)}\nTag：{', '.join(tags)}",
                    f"Operation: {operation}\nImages: {len(target_items)}\nTags: {', '.join(tags)}",
                ),
                title=self._b("批量 Tag 预览", "Batch tag preview"),
            )
        )
        if not self._confirm(self._b("应用这次批量修改吗？", "Apply this batch edit?"), default=False):
            return

        result = batch_edit_tags(
            workspace,
            [item.key for item in target_items],
            tags,
            action=action,
        )
        self.console.print(
            self._b(
                f"[green]批量 Tag 修改完成：{result['changed']} 张已修改，{result['unchanged']} 张无需修改。[/green]",
                f"[green]Batch tag edit complete: {result['changed']} changed, {result['unchanged']} unchanged.[/green]",
            )
        )
