from __future__ import annotations

from rich.panel import Panel

from .dataset_tag_editor import BatchTagAction, batch_edit_tags, parse_tag_input
from .dataset_workspace import DatasetWorkspace
from .interactive_multiselect import MultiSelectOption, select_many
from .interactive_protected_deletion import InteractiveWizard as BaseInteractiveWizard
from .wizard import MenuItem


class InteractiveWizard(BaseInteractiveWizard):
    """Replace number/range batch flows with arrows + Space + Enter."""

    def _review_dataset_items(self, workspace: DatasetWorkspace, *, source_id: str | None = None) -> None:
        while True:
            workspace = DatasetWorkspace.load(workspace.name)
            items = workspace.items(source_id=source_id, include_disabled=True, include_excluded=True)
            if not items:
                self.console.print(self._b("[yellow]没有可审核图片。[/yellow]", "[yellow]No images to review.[/yellow]"))
                return
            action = self._menu(
                self._b("图片审核", "Image review"),
                [
                    MenuItem("exclude", self._b("批量选择要排除的图片", "Select images to exclude")),
                    MenuItem("restore", self._b("批量选择要恢复的图片", "Select images to restore")),
                    MenuItem("edit", self._b("修改图片 Tag", "Edit image tags")),
                    MenuItem("back", self._b("返回", "Back")),
                ],
                default="exclude",
            )
            if action == "back":
                return
            if action == "edit":
                super()._choose_and_edit_tag(workspace, source_id=source_id)
                continue
            candidates = [item for item in items if item.excluded == (action == "restore")]
            if not candidates:
                self.console.print(self._b("[dim]当前没有符合条件的图片。[/dim]", "[dim]No matching images.[/dim]"))
                continue
            audit = workspace.audit(source_id=source_id)
            audit_map = {str(record["key"]): record for record in audit["records"]}
            selected = select_many(
                self.console,
                self._b("选择图片", "Select images"),
                [self._image_option(workspace, item, audit_map.get(item.key, {})) for item in candidates],
                columns=3,
                page_size=30,
            )
            if not selected:
                continue
            if action == "exclude":
                reason = self._ask_text(
                    self._b("排除原因", "Exclusion reason"),
                    default=self._b("人工审核", "manual review"),
                ).strip()
                changed = workspace.exclude(selected, reason=reason or "manual review")
                self.console.print(self._b(f"[green]已排除 {changed} 张。[/green]", f"[green]Excluded {changed} image(s).[/green]"))
            else:
                changed = workspace.restore(selected)
                self.console.print(self._b(f"[green]已恢复 {changed} 张。[/green]", f"[green]Restored {changed} image(s).[/green]"))

    def _batch_edit_tags(
        self,
        workspace: DatasetWorkspace,
        *,
        action: BatchTagAction,
        source_id: str | None,
    ) -> None:
        active = workspace.items(source_id=source_id, include_disabled=False, include_excluded=False)
        all_items = workspace.items(source_id=source_id, include_disabled=True, include_excluded=True)
        if not all_items:
            self.console.print(self._b("[yellow]没有可编辑图片。[/yellow]", "[yellow]No editable images.[/yellow]"))
            return
        scope = self._menu(
            self._b("批量修改范围", "Batch edit scope"),
            [
                MenuItem("active", self._b("全部当前可训练图片", "All currently trainable images"), f"{len(active)} images"),
                MenuItem("select", self._b("交互式选择图片", "Interactively select images"), self._b("方向键移动，Space 选择，Enter 确认。", "Arrows move, Space selects, Enter confirms.")),
                MenuItem("back", self._b("返回", "Back")),
            ],
            default="active",
        )
        if scope == "back":
            return
        target = active
        if scope == "select":
            selected = set(
                select_many(
                    self.console,
                    self._b("选择要批量修改的图片", "Select images for batch tag editing"),
                    [self._image_option(workspace, item) for item in all_items],
                    columns=3,
                    page_size=30,
                )
            )
            target = [item for item in all_items if item.key in selected]
        if not target:
            return
        tags = parse_tag_input(
            self._ask_text(self._b("Tag（逗号或换行分隔）", "Tags (comma- or newline-separated)"))
        )
        operation = {"prepend": "prepend", "append": "append", "remove": "remove"}[action]
        self.console.print(
            Panel.fit(
                f"Operation: {operation}\nImages: {len(target)}\nTags: {', '.join(tags)}",
                title=self._b("批量 Tag 预览", "Batch tag preview"),
            )
        )
        if not self._confirm(self._b("应用这次批量修改吗？", "Apply this batch edit?"), default=False):
            return
        result = batch_edit_tags(workspace, [item.key for item in target], tags, action=action)
        self.console.print(
            self._b(
                f"[green]批量 Tag 修改完成：{result['changed']} 张已修改，{result['unchanged']} 张无需修改。[/green]",
                f"[green]Batch tag edit complete: {result['changed']} changed, {result['unchanged']} unchanged.[/green]",
            )
        )

    def _image_option(self, workspace: DatasetWorkspace, item, audit: dict | None = None) -> MultiSelectOption:
        flags = ""
        if audit:
            values = [str(flag.get("code")) for flag in audit.get("flags", [])]
            flags = (" · flags: " + ", ".join(values)) if values else ""
        state = self._b("排除", "excluded") if item.excluded else self._b("保留", "active")
        return MultiSelectOption(
            item.key,
            item.relative.as_posix(),
            self._b(
                f"来源 {item.source_id} · {state}{flags} · Tags: {self._truncate(workspace.caption_text(item.key), 100)}",
                f"Source {item.source_id} · {state}{flags} · Tags: {self._truncate(workspace.caption_text(item.key), 100)}",
            ),
        )
