from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from .dataset_deletion import (
    delete_dataset_items,
    delete_dataset_source,
    delete_dataset_workspace,
)
from .dataset_workspace import DatasetWorkspace, parse_number_selection
from .interactive_lifecycle import InteractiveWizard as BaseInteractiveWizard
from .wizard import MenuItem


class InteractiveWizard(BaseInteractiveWizard):
    """Add explicit destructive Dataset controls on top of the four-part UI."""

    def dataset_dashboard(self, name: str) -> None:
        while True:
            workspace = DatasetWorkspace.load(name)
            self._render_dataset_dashboard(workspace)
            action = self._menu(
                self._b("数据集操作", "Dataset actions"),
                [
                    MenuItem("import", self._b("导入新的数据来源", "Import a new data source")),
                    MenuItem("sources", self._b("按来源管理", "Manage by source")),
                    MenuItem("audit", self._b("全数据集自动检查", "Audit the whole dataset")),
                    MenuItem("tag", self._b("自动打 Tag", "Auto-tag images")),
                    MenuItem("review", self._b("人工图片审核 / 排除", "Manual image review / exclusions")),
                    MenuItem("edit_tags", self._b("人工修改 Tag", "Edit tags manually")),
                    MenuItem(
                        "training",
                        self._b("用此数据集开始训练", "Start training with this dataset"),
                        self._b(
                            "转到训练状态：再选择一份训练配置，并同时冻结两个快照。",
                            "Go through Training Status, select a config, and freeze both snapshots together.",
                        ),
                    ),
                    MenuItem(
                        "delete",
                        self._b("[red]删除数据 / 来源 / 图片[/red]", "[red]Delete dataset / sources / images[/red]"),
                        self._b(
                            "永久删除 Dataset 工作区中的副本；不会删除原始导入文件或已有训练快照。",
                            "Permanently delete Dataset-owned copies; originals and existing training snapshots are untouched.",
                        ),
                    ),
                    MenuItem("back", self._b("返回数据集列表", "Back to dataset list")),
                ],
                default="import" if not workspace.sources else "sources",
            )
            if action == "back":
                return
            if action == "import":
                self._import_dataset_source(workspace)
            elif action == "sources":
                self._manage_dataset_sources(workspace.name)
            elif action == "audit":
                self._audit_dataset(workspace)
            elif action == "tag":
                self._auto_tag_dataset(workspace)
            elif action == "review":
                self._review_dataset_items(workspace)
            elif action == "edit_tags":
                self._choose_and_edit_tag(workspace)
            elif action == "training":
                self._start_training_from_dataset_config(prefilled_workspace=workspace)
            elif action == "delete":
                if self._dataset_delete_menu(workspace):
                    return

    def _dataset_delete_menu(self, workspace: DatasetWorkspace) -> bool:
        """Return True when the whole Dataset was deleted."""

        while True:
            workspace = DatasetWorkspace.load(workspace.name)
            summary = workspace.summary()
            self.console.print(
                Panel.fit(
                    self._b(
                        "[bold red]永久删除[/bold red]\n"
                        "这里删除的是 Dataset 工作区中的副本。\n"
                        "不会删除最初导入的图片目录/原视频，也不会修改已经创建的训练 Project、Run、权重或结果。\n"
                        "如果只是暂时不想让某张图训练，请优先使用“排除”。",
                        "[bold red]Permanent deletion[/bold red]\n"
                        "These actions delete copies owned by the Dataset workspace.\n"
                        "Original import folders/videos and existing Projects, Runs, weights, and results are not modified.\n"
                        "If you only want an image omitted from training, prefer Exclude.",
                    ),
                    border_style="red",
                )
            )
            actions: list[MenuItem] = []
            if summary["images"]:
                actions.append(
                    MenuItem(
                        "images",
                        self._b("删除某个来源中的部分图片", "Delete selected images from a source"),
                    )
                )
            if workspace.sources:
                actions.append(MenuItem("source", self._b("删除整个来源", "Delete an entire source")))
            actions.extend(
                [
                    MenuItem(
                        "dataset",
                        self._b("[red]删除整个数据集[/red]", "[red]Delete the entire dataset[/red]"),
                    ),
                    MenuItem("back", self._b("返回", "Back")),
                ]
            )
            action = self._menu(
                self._b("删除操作", "Deletion actions"),
                actions,
                default="images" if summary["images"] else "back",
            )
            if action == "back":
                return False
            if action == "images":
                self._delete_images_interactive(workspace)
            elif action == "source":
                self._delete_source_interactive(workspace)
            elif action == "dataset":
                return self._delete_dataset_interactive(workspace)

    def _select_source_for_deletion(self, workspace: DatasetWorkspace, title: str) -> str | None:
        choices = []
        for source_id, source in sorted(workspace.sources.items()):
            image_count = len(
                workspace.items(
                    source_id=source_id,
                    include_disabled=True,
                    include_excluded=True,
                )
            )
            choices.append(
                MenuItem(
                    source_id,
                    str(source.get("label") or source_id),
                    self._b(
                        f"{source.get('kind')} · {image_count} 张图片 · ID {source_id}",
                        f"{source.get('kind')} · {image_count} images · ID {source_id}",
                    ),
                )
            )
        choices.append(MenuItem("back", self._b("返回", "Back")))
        selected = self._menu(title, choices, default=choices[0].value)
        return None if selected == "back" else selected

    def _delete_source_interactive(self, workspace: DatasetWorkspace) -> None:
        source_id = self._select_source_for_deletion(
            workspace,
            self._b("选择要删除的来源", "Select source to delete"),
        )
        if source_id is None:
            return
        source = workspace.sources[source_id]
        items = workspace.items(
            source_id=source_id,
            include_disabled=True,
            include_excluded=True,
        )
        children = [
            child_id
            for child_id, child in workspace.sources.items()
            if child.get("parent_source") == source_id
        ]
        child_note = ""
        if children:
            child_note = self._b(
                f"\n派生来源不会级联删除：{', '.join(children)}",
                f"\nDerived sources are retained: {', '.join(children)}",
            )
        self.console.print(
            Panel.fit(
                self._b(
                    f"[bold red]删除来源[/bold red]\n"
                    f"名称：{source.get('label', source_id)}\nID：{source_id}\n图片：{len(items)}"
                    f"{child_note}\n\n原始外部素材不会被删除。",
                    f"[bold red]Delete source[/bold red]\n"
                    f"Label: {source.get('label', source_id)}\nID: {source_id}\nImages: {len(items)}"
                    f"{child_note}\n\nOriginal external media will not be deleted.",
                ),
                border_style="red",
            )
        )
        if not self._confirm(
            self._b(
                "永久删除这个来源及其 Dataset 副本？",
                "Permanently delete this source and its Dataset copies?",
            ),
            default=False,
        ):
            return
        typed = self._ask_text(
            self._b(
                f"输入来源 ID {source_id} 以确认",
                f"Type source ID {source_id} to confirm",
            )
        ).strip()
        if typed != source_id:
            self.console.print(
                self._b(
                    "[yellow]确认不匹配，已取消。[/yellow]",
                    "[yellow]Confirmation did not match; cancelled.[/yellow]",
                )
            )
            return
        result = delete_dataset_source(workspace, source_id)
        self.console.print(
            self._b(
                f"[green]已删除来源 {result['label']}，共 {result['deleted_images']} 张 Dataset 图片副本。[/green]",
                f"[green]Deleted source {result['label']} with {result['deleted_images']} Dataset image copies.[/green]",
            )
        )

    def _delete_images_interactive(self, workspace: DatasetWorkspace) -> None:
        source_id = self._select_source_for_deletion(
            workspace,
            self._b("选择图片所属来源", "Select the source containing the images"),
        )
        if source_id is None:
            return

        page = 0
        page_size = 30
        while True:
            workspace = DatasetWorkspace.load(workspace.name)
            if source_id not in workspace.sources:
                return
            items = workspace.items(
                source_id=source_id,
                include_disabled=True,
                include_excluded=True,
            )
            if not items:
                self.console.print(
                    self._b(
                        "[yellow]这个来源已经没有图片。[/yellow]",
                        "[yellow]This source has no images left.[/yellow]",
                    )
                )
                return
            page_count = max(1, (len(items) + page_size - 1) // page_size)
            page = min(page, page_count - 1)
            start = page * page_size
            end = min(len(items), start + page_size)

            table = Table(
                title=self._b(
                    f"永久删除图片 · 第 {page + 1}/{page_count} 页",
                    f"Permanent image deletion · page {page + 1}/{page_count}",
                )
            )
            table.add_column("#", justify="right")
            table.add_column(self._b("状态", "State"))
            table.add_column(self._b("文件", "File"))
            table.add_column("Tags")
            for index in range(start, end):
                item = items[index]
                table.add_row(
                    str(index + 1),
                    self._b("已排除", "excluded") if item.excluded else self._b("保留", "active"),
                    item.relative.as_posix(),
                    self._truncate(workspace.caption_text(item.key), 60),
                )
            self.console.print(table)

            actions = [
                MenuItem(
                    "delete",
                    self._b("按编号/范围永久删除", "Permanently delete by number/range"),
                )
            ]
            if page > 0:
                actions.append(MenuItem("prev", self._b("上一页", "Previous page")))
            if page + 1 < page_count:
                actions.append(MenuItem("next", self._b("下一页", "Next page")))
            actions.append(MenuItem("back", self._b("返回", "Back")))
            action = self._menu(self._b("图片删除", "Image deletion"), actions, default="delete")
            if action == "back":
                return
            if action == "prev":
                page -= 1
                continue
            if action == "next":
                page += 1
                continue

            raw = self._ask_text(
                self._b(
                    "输入编号或范围（例如 1,3-5）",
                    "Enter numbers/ranges (for example 1,3-5)",
                )
            )
            numbers = parse_number_selection(raw, maximum=len(items))
            selected = [items[number - 1] for number in numbers]
            preview = "\n".join(
                f"  {number}. {items[number - 1].relative.as_posix()}"
                for number in numbers[:12]
            )
            if len(numbers) > 12:
                preview += self._b(
                    f"\n  ... 另有 {len(numbers) - 12} 张",
                    f"\n  ... plus {len(numbers) - 12} more",
                )
            self.console.print(
                Panel.fit(
                    self._b(
                        f"[bold red]将永久删除 {len(selected)} 张图片[/bold red]\n{preview}\n\n同名 .txt Tag 也会删除。",
                        f"[bold red]Permanently delete {len(selected)} image(s)[/bold red]\n{preview}\n\nMatching .txt tag files will also be deleted.",
                    ),
                    border_style="red",
                )
            )
            if not self._confirm(
                self._b("确认永久删除？", "Confirm permanent deletion?"),
                default=False,
            ):
                continue
            result = delete_dataset_items(workspace, [item.key for item in selected])
            self.console.print(
                self._b(
                    f"[green]已永久删除 {result['deleted_images']} 张图片和 {result['deleted_captions']} 个 Tag 文件。[/green]",
                    f"[green]Permanently deleted {result['deleted_images']} image(s) and {result['deleted_captions']} tag file(s).[/green]",
                )
            )

    def _delete_dataset_interactive(self, workspace: DatasetWorkspace) -> bool:
        summary = workspace.summary()
        self.console.print(
            Panel.fit(
                self._b(
                    f"[bold red]删除整个数据集：{workspace.name}[/bold red]\n"
                    f"来源：{summary['sources']}\n图片：{summary['images']}\n路径：{workspace.dataset_dir}\n\n"
                    "这不会删除原始导入文件，也不会删除已经冻结的训练 Project/Run/结果。",
                    f"[bold red]Delete entire dataset: {workspace.name}[/bold red]\n"
                    f"Sources: {summary['sources']}\nImages: {summary['images']}\nPath: {workspace.dataset_dir}\n\n"
                    "Original imports and already-frozen training Projects/Runs/results are not deleted.",
                ),
                border_style="red",
            )
        )
        if not self._confirm(
            self._b(
                "永久删除整个 Dataset 工作区？",
                "Permanently delete the entire Dataset workspace?",
            ),
            default=False,
        ):
            return False
        typed = self._ask_text(
            self._b(
                f"输入数据集名称 {workspace.name} 以确认",
                f"Type dataset name {workspace.name} to confirm",
            )
        ).strip()
        if typed != workspace.name:
            self.console.print(
                self._b(
                    "[yellow]确认不匹配，已取消。[/yellow]",
                    "[yellow]Confirmation did not match; cancelled.[/yellow]",
                )
            )
            return False
        result = delete_dataset_workspace(workspace)
        self.console.print(
            self._b(
                f"[green]数据集 {result['dataset']} 已删除：{result['sources']} 个来源，{result['images']} 张 Dataset 图片副本。[/green]",
                f"[green]Deleted dataset {result['dataset']}: {result['sources']} sources and {result['images']} Dataset image copies.[/green]",
            )
        )
        return True
