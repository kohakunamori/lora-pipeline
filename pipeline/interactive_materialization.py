from __future__ import annotations

from typing import Any, Mapping, Sequence

from rich.panel import Panel
from rich.table import Table

from .activation_recipe import (
    delete_character_tags_group,
    image_keys_for_group,
    load_activation_recipe,
    set_group_images,
    suggest_group_images,
    tag_candidates,
    upsert_character_tags_group,
    validate_active_group_coverage,
)
from .dataset_workspace import DatasetWorkspace
from .interactive_multiselect import MultiSelectOption, select_many
from .interactive_semantic_concepts import InteractiveWizard as BaseInteractiveWizard
from .models import PipelineError
from .wizard import MenuItem


class InteractiveWizard(BaseInteractiveWizard):
    """Final Dataset UX aligned with the materialization compiler contract.

    Imported Dataset sources stay unchanged. Target-aware crop, downscale, caption
    policy and TriggerPolicy are applied exactly once when a TrainingConfig is
    materialized into an immutable prepared generation.
    """

    def dataset_dashboard(self, name: str) -> None:
        while True:
            workspace = DatasetWorkspace.load(name)
            self._render_dataset_dashboard(workspace)
            items = [
                MenuItem(
                    "import",
                    self._b("导入素材", "Import material"),
                    self._b(
                        "保留原始图片/视频抽帧；训练目标相关裁剪只在 materialize 阶段执行一次。",
                        "Keep imported images/video frames unchanged; target-aware crop runs once during materialization.",
                    ),
                ),
                MenuItem(
                    "sources",
                    self._b("来源管理", "Manage sources"),
                    self._b(
                        "查看来源并启用/停用；RAW 来源不会被训练流程改写。",
                        "Inspect and enable/disable sources; training never rewrites RAW sources.",
                    ),
                ),
                MenuItem(
                    "audit",
                    self._b("输入安全检查", "Input sanity check"),
                    self._b(
                        "检查损坏图片和完全重复项；只有确定安全的项目才建议自动排除。",
                        "Check corrupt images and exact duplicates; only deterministic-safe items are suggested for exclusion.",
                    ),
                ),
                MenuItem(
                    "tag",
                    self._b("自动打 Tag", "Auto-tag"),
                    self._b(
                        "可在 Dataset 阶段准备/修正 caption；角色 Tag 分组会从这些 Tags 提取候选。",
                        "Prepare/correct captions in Dataset; character tag groups extract their candidates from these tags.",
                    ),
                ),
            ]
            if workspace.concept_type == "character":
                items.append(
                    MenuItem(
                        "groups",
                        self._b("角色 Tag 分组", "Character tag groups"),
                        self._b(
                            "自动提取 Identity / Outfit Tag 候选，人工勾选每组特征并填写 Group Name / Group Tag。Group Tag 会进入对应图片的训练 Caption，并写入最终 LoRA metadata。",
                            "Extract identity/outfit tag candidates, select each group's features, and enter its Group Name / Group Tag. The Group Tag is trained in assigned captions and exported in final LoRA metadata.",
                        ),
                    )
                )
            items.extend(
                [
                    MenuItem(
                        "edit_tags",
                        self._b("修正 Tag", "Edit tags"),
                        self._b(
                            "只在需要时人工修正 Tag。",
                            "Manually correct tags only when necessary.",
                        ),
                    ),
                    MenuItem(
                        "review",
                        self._b("人工排除 / 恢复", "Manual exclude / restore"),
                        self._b(
                            "可选人工兜底；不会物理删除来源文件。",
                            "Optional manual fallback; source files are not physically deleted.",
                        ),
                    ),
                    MenuItem(
                        "training",
                        self._b("开始训练", "Start training"),
                        self._b(
                            "选择 TrainingConfig；materialize 会冻结 crop、约 1MP 训练图、caption/trigger/Group Tag，并生成 preview.html 供检查。",
                            "Choose a TrainingConfig; materialize freezes crop, ~1MP training pixels and caption/trigger/Group Tags, then writes preview.html for review.",
                        ),
                    ),
                    MenuItem("back", self._b("返回", "Back")),
                ]
            )
            action = self._menu(
                self._b("数据集操作", "Dataset actions"),
                items,
                default="import" if not workspace.sources else "tag",
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
            elif action == "groups":
                self._activation_group_manager(workspace.name)
            elif action == "edit_tags":
                self._choose_and_edit_tag(workspace)
            elif action == "review":
                self._review_dataset_items(workspace)
            elif action == "training":
                self._start_training_from_dataset_config(prefilled_workspace=workspace)

    def _activation_group_manager(self, dataset_name: str) -> None:
        while True:
            workspace = DatasetWorkspace.load(dataset_name)
            recipe = load_activation_recipe(workspace, create=True)
            assert recipe is not None
            self._render_activation_groups(workspace, recipe)
            groups = list(recipe["character_tags_groups"])
            actions = [
                MenuItem(
                    "add",
                    self._b("添加 Character Tags Group", "Add Character Tags Group"),
                    self._b(
                        "填写 Group Name / Group Tag，再从自动候选中分别选择 Identity 和 Outfit Tags。",
                        "Enter Group Name / Group Tag, then select identity and outfit tags from automatic candidates.",
                    ),
                )
            ]
            if groups:
                actions.append(MenuItem("manage", self._b("管理已有 Group", "Manage existing group")))
            actions.append(MenuItem("back", self._b("返回", "Back")))
            action = self._menu(
                self._b("角色 Tag 分组", "Character tag groups"),
                actions,
                default="manage" if groups else "add",
            )
            if action == "back":
                return
            if action == "add":
                self._create_activation_group(workspace)
            elif action == "manage":
                selected = self._choose_activation_group(recipe)
                if selected is not None:
                    self._manage_activation_group(workspace.name, selected)

    def _render_activation_groups(self, workspace: DatasetWorkspace, recipe: Mapping[str, Any]) -> None:
        coverage = validate_active_group_coverage(workspace, recipe)
        table = Table(title=self._b("Character Tags Groups", "Character Tags Groups"))
        table.add_column(self._b("Group Name", "Group Name"), style="bold")
        table.add_column(self._b("Group Tag", "Group Tag"))
        table.add_column(self._b("Identity Tags", "Identity Tags"))
        table.add_column(self._b("Outfit Tags", "Outfit Tags"))
        table.add_column(self._b("图片", "Images"), justify="right")
        for group in recipe.get("character_tags_groups", []):
            name = str(group["name"])
            table.add_row(
                name,
                str(group["group_tag"]),
                self._truncate(", ".join(group.get("identity_tags", [])), 56) or "—",
                self._truncate(", ".join(group.get("outfit_tags", [])), 56) or "—",
                str(len(image_keys_for_group(workspace, recipe, name))),
            )
        if not recipe.get("character_tags_groups"):
            table.add_row("—", "—", "—", "—", "0")
        self.console.print(table)
        if coverage["enabled"]:
            style = "green" if coverage["complete"] else "yellow"
            self.console.print(
                f"[{style}]Assigned {coverage['assigned_images']}/{coverage['active_images']} active images"
                + ("[/" + style + "]" if coverage["complete"] else f" · unassigned {len(coverage['unassigned'])}[/{style}]")
            )

    def _activation_group_help(self) -> None:
        self.console.print(
            Panel.fit(
                self._b(
                    "[bold]Group Name[/bold]\n"
                    "给人看的分组名称，同时作为该分组在配置中的唯一名称；不再另外填写 ID。\n"
                    "例：NIC26 Swimsuit\n\n"
                    "[bold]Group Tag[/bold]\n"
                    "模型需要学习的分组选择 Tag。只写一个 token/短语，不写逗号列表。\n"
                    "例：misuzu_nic26\n\n"
                    "[bold]Identity Tags[/bold]\n"
                    "从自动 Tags 中选择这组形态的角色外观特征，例如发色、眼色、发型。\n\n"
                    "[bold]Outfit Tags[/bold]\n"
                    "选择这组形态的服饰/配件特征。Identity + Outfit 合起来就是完整 Group 使用配方。",
                    "[bold]Group Name[/bold]\n"
                    "Human-readable unique group name. There is no separate public ID.\n"
                    "Example: NIC26 Swimsuit\n\n"
                    "[bold]Group Tag[/bold]\n"
                    "A learned selector inserted into captions for this group. Enter one token/phrase, not a comma list.\n"
                    "Example: misuzu_nic26\n\n"
                    "[bold]Identity Tags[/bold]\n"
                    "Select this appearance group's character traits such as hair color, eye color, and hairstyle.\n\n"
                    "[bold]Outfit Tags[/bold]\n"
                    "Select clothing/accessory traits. Identity + Outfit together form the complete group recipe.",
                ),
                title=self._b("字段含义", "Field meanings"),
                border_style="cyan",
            )
        )

    def _create_activation_group(self, workspace: DatasetWorkspace) -> None:
        self._activation_group_help()
        name = self._ask_text(self._b("Group Name", "Group Name")).strip()
        group_tag = self._ask_group_tag()
        identity_tags = self._select_activation_tags(workspace, kind="identity", selected=())
        if identity_tags is None:
            return
        outfit_tags = self._select_activation_tags(workspace, kind="outfit", selected=())
        if outfit_tags is None:
            return
        if not identity_tags and not outfit_tags:
            self.console.print(
                self._b(
                    "[red]至少选择一个 Identity 或 Outfit Tag。[/red]",
                    "[red]Select at least one identity or outfit tag.[/red]",
                )
            )
            return
        recipe = upsert_character_tags_group(
            workspace,
            name=name,
            group_tag=group_tag,
            identity_tags=identity_tags,
            outfit_tags=outfit_tags,
        )
        suggested = suggest_group_images(
            workspace,
            identity_tags=identity_tags,
            outfit_tags=outfit_tags,
        )
        selected_images = self._select_group_images(
            workspace,
            recipe,
            name,
            selected=suggested,
            suggested_count=len(suggested),
        )
        if selected_images is not None:
            set_group_images(workspace, name, selected_images)

    def _manage_activation_group(self, dataset_name: str, group_name: str) -> None:
        while True:
            workspace = DatasetWorkspace.load(dataset_name)
            recipe = load_activation_recipe(workspace, create=True)
            assert recipe is not None
            group = next(
                (row for row in recipe["character_tags_groups"] if row["name"] == group_name),
                None,
            )
            if group is None:
                return
            action = self._menu(
                f"{group['name']} · {group['group_tag']}",
                [
                    MenuItem("fields", self._b("修改 Group Name / Group Tag", "Edit Group Name / Group Tag")),
                    MenuItem("identity", self._b("选择 Identity Tags", "Select Identity Tags")),
                    MenuItem("outfit", self._b("选择 Outfit Tags", "Select Outfit Tags")),
                    MenuItem("images", self._b("选择所属图片", "Select assigned images")),
                    MenuItem("delete", self._b("删除 Group", "Delete Group")),
                    MenuItem("back", self._b("返回", "Back")),
                ],
                default="identity",
            )
            if action == "back":
                return
            if action == "fields":
                self._activation_group_help()
                new_name = self._ask_text("Group Name", default=str(group["name"])).strip()
                group_tag = self._ask_group_tag(default=str(group["group_tag"]))
                upsert_character_tags_group(
                    workspace,
                    previous_name=group_name,
                    name=new_name,
                    group_tag=group_tag,
                    identity_tags=group.get("identity_tags", []),
                    outfit_tags=group.get("outfit_tags", []),
                )
                group_name = new_name
            elif action == "identity":
                selected = self._select_activation_tags(
                    workspace,
                    kind="identity",
                    selected=group.get("identity_tags", []),
                )
                if selected is not None and (selected or group.get("outfit_tags")):
                    upsert_character_tags_group(
                        workspace,
                        previous_name=group_name,
                        name=group_name,
                        group_tag=str(group["group_tag"]),
                        identity_tags=selected,
                        outfit_tags=group.get("outfit_tags", []),
                    )
            elif action == "outfit":
                selected = self._select_activation_tags(
                    workspace,
                    kind="outfit",
                    selected=group.get("outfit_tags", []),
                )
                if selected is not None and (selected or group.get("identity_tags")):
                    upsert_character_tags_group(
                        workspace,
                        previous_name=group_name,
                        name=group_name,
                        group_tag=str(group["group_tag"]),
                        identity_tags=group.get("identity_tags", []),
                        outfit_tags=selected,
                    )
            elif action == "images":
                current = image_keys_for_group(workspace, recipe, group_name)
                selected = self._select_group_images(
                    workspace,
                    recipe,
                    group_name,
                    selected=current,
                )
                if selected is not None:
                    set_group_images(workspace, group_name, selected)
            elif action == "delete":
                if self._confirm(
                    self._b(
                        f"删除 Group {group_name}？其图片会变为未分组。",
                        f"Delete group {group_name}? Its images will become unassigned.",
                    ),
                    default=False,
                ):
                    delete_character_tags_group(workspace, group_name)
                    return

    def _ask_group_tag(self, *, default: str | None = None) -> str:
        while True:
            value = self._ask_text(
                self._b(
                    "Group Tag（推理时选择该组；例如 misuzu_nic26）",
                    "Group Tag (selects this group at inference; e.g. misuzu_nic26)",
                ),
                default=default,
            ).strip()
            if not value or "," in value or "\n" in value:
                self.console.print(
                    self._b(
                        "[red]Group Tag 必须是一个非空 token/短语，不能包含逗号或换行。[/red]",
                        "[red]Group Tag must be one non-empty token/phrase without commas/newlines.[/red]",
                    )
                )
                continue
            return value

    def _select_activation_tags(
        self,
        workspace: DatasetWorkspace,
        *,
        kind: str,
        selected: Sequence[str],
    ) -> list[str] | None:
        rows = tag_candidates(workspace, kind=kind)
        known = {str(row["tag"]) for row in rows}
        for tag in selected:
            if tag not in known:
                rows.append({"tag": tag, "count": 0, "total": 0, "coverage": 0.0})
        if not rows:
            self.console.print(
                self._b(
                    "[yellow]没有可选 Tag。请先运行 Auto-tag，或人工补充图片 .txt Tags。[/yellow]",
                    "[yellow]No candidate tags. Run Auto-tag first or add image .txt tags manually.[/yellow]",
                )
            )
            return []
        options = [
            MultiSelectOption(
                str(row["tag"]),
                f"{row['tag']} · {float(row.get('coverage', 0.0)):.0%}",
                self._b(
                    f"训练集覆盖率 {float(row.get('coverage', 0.0)):.0%} · {row.get('count', 0)}/{row.get('total', 0)} images",
                    f"Dataset coverage {float(row.get('coverage', 0.0)):.0%} · {row.get('count', 0)}/{row.get('total', 0)} images",
                ),
            )
            for row in rows
        ]
        title = self._b("选择 Identity Tags", "Select Identity Tags") if kind == "identity" else self._b("选择 Outfit Tags", "Select Outfit Tags")
        return select_many(
            self.console,
            title,
            options,
            selected=selected,
            page_size=28,
        )

    def _select_group_images(
        self,
        workspace: DatasetWorkspace,
        recipe: Mapping[str, Any],
        group_name: str,
        *,
        selected: Sequence[str],
        suggested_count: int | None = None,
    ) -> list[str] | None:
        if suggested_count is not None:
            self.console.print(
                self._b(
                    f"[dim]根据刚选的 Tags 自动预选了 {suggested_count} 张图片；请在下面复核。[/dim]",
                    f"[dim]Auto-selected {suggested_count} image(s) from the chosen tags; review below.[/dim]",
                )
            )
        assignments = recipe.get("assignments", {})
        options = [
            MultiSelectOption(
                item.key,
                item.relative.as_posix(),
                self._b(
                    f"来源 {item.source_id} · 当前 Group {assignments.get(item.key, '未分组')} · Tags: {self._truncate(workspace.caption_text(item.key), 110)}",
                    f"Source {item.source_id} · current Group {assignments.get(item.key, 'unassigned')} · Tags: {self._truncate(workspace.caption_text(item.key), 110)}",
                ),
            )
            for item in workspace.items(include_disabled=False, include_excluded=False)
        ]
        return select_many(
            self.console,
            self._b(
                f"选择属于 {group_name} 的图片",
                f"Select images for {group_name}",
            ),
            options,
            selected=selected,
            columns=3,
            page_size=30,
        )

    def _choose_activation_group(self, recipe: Mapping[str, Any]) -> str | None:
        groups = list(recipe.get("character_tags_groups", []))
        if not groups:
            return None
        items = [
            MenuItem(
                str(group["name"]),
                str(group["name"]),
                f"Group Tag: {group['group_tag']} · Identity {len(group.get('identity_tags', []))} · Outfit {len(group.get('outfit_tags', []))}",
            )
            for group in groups
        ]
        items.append(MenuItem("back", self._b("返回", "Back")))
        selected = self._menu(
            self._b("选择 Group", "Select group"),
            items,
            default=str(groups[0]["name"]),
        )
        return None if selected == "back" else selected
