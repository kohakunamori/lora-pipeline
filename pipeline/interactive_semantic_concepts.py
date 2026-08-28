from __future__ import annotations

import re
from typing import Sequence

from rich.panel import Panel

from . import resource_deletion as _resource_deletion
from .dataset_metadata_snapshot import attach_dataset_metadata_snapshot
from .dataset_semantics import (
    add_outfit,
    attach_dataset_semantics_snapshot,
    character_feature_candidates,
    image_keys_for_outfit,
    load_semantics,
    outfit_feature_candidates,
    outfit_for_image,
    set_character_features,
    set_character_token,
    set_outfit_features,
    set_outfit_images,
    set_outfit_token,
)
from .dataset_workspace import DatasetWorkspace
from .interactive_bulk_selection import InteractiveWizard as BaseInteractiveWizard
from .interactive_multiselect import MultiSelectOption, select_many
from .semantic_runtime import install_semantic_runtime_hooks
from .wizard import MenuItem


_ORIGINAL_CREATE = _resource_deletion._create_project_from_training_config


def _create_with_dataset_snapshots(workspace, config, *, project_name, root=None):
    state = _ORIGINAL_CREATE(workspace, config, project_name=project_name, root=root)
    state = attach_dataset_metadata_snapshot(state, workspace)
    return attach_dataset_semantics_snapshot(state, workspace)


if not getattr(_resource_deletion._create_project_from_training_config, "_dataset_semantics_wrapped", False):
    _create_with_dataset_snapshots._dataset_semantics_wrapped = True
    _resource_deletion._create_project_from_training_config = _create_with_dataset_snapshots
install_semantic_runtime_hooks()


class InteractiveWizard(BaseInteractiveWizard):
    """Dataset-owned Character + Outfit concepts and interactive feature selection."""

    def _create_dataset(self) -> DatasetWorkspace | None:
        workspace = super()._create_dataset()
        if workspace is not None and workspace.concept_type == "character":
            load_semantics(workspace, create=True)
        return workspace

    def dataset_dashboard(self, name: str) -> None:
        while True:
            workspace = DatasetWorkspace.load(name)
            if workspace.concept_type == "character":
                load_semantics(workspace, create=True)
            self._render_dataset_dashboard(workspace)
            items = [
                MenuItem("import", self._b("导入新的数据来源", "Import a new data source")),
                MenuItem("sources", self._b("按来源管理", "Manage by source")),
                MenuItem("audit", self._b("全数据集自动检查", "Audit the whole dataset")),
                MenuItem("curate", self._b("高级清洗分析", "Advanced curation analysis")),
                MenuItem("tag", self._b("自动打 Tag", "Auto-tag images")),
            ]
            if workspace.concept_type == "character":
                items.append(MenuItem("concepts", self._b("角色 / 服饰语义", "Character / outfit concepts")))
            items += [
                MenuItem("review", self._b("人工图片审核 / 排除", "Manual image review / exclusions")),
                MenuItem("edit_tags", self._b("人工修改 Tag", "Edit tags manually")),
                MenuItem("training", self._b("用此数据集开始训练", "Start training with this dataset")),
                MenuItem("back", self._b("返回数据集列表", "Back to dataset list")),
            ]
            action = self._menu(
                self._b("数据集操作", "Dataset actions"),
                items,
                default="import" if not workspace.sources else ("concepts" if workspace.concept_type == "character" else "sources"),
            )
            if action == "back":
                return
            if action == "import": self._import_dataset_source(workspace)
            elif action == "sources": self._manage_dataset_sources(workspace.name)
            elif action == "audit": self._audit_dataset(workspace)
            elif action == "curate": self._curate_dataset(workspace)
            elif action == "tag": self._auto_tag_dataset(workspace)
            elif action == "concepts": self._semantic_manager(workspace.name)
            elif action == "review": self._review_dataset_items(workspace)
            elif action == "edit_tags": self._choose_and_edit_tag(workspace)
            elif action == "training": self._start_training_from_dataset_config(prefilled_workspace=workspace)

    def _render_dataset_dashboard(self, workspace: DatasetWorkspace) -> None:
        super()._render_dataset_dashboard(workspace)
        if workspace.concept_type != "character":
            return
        semantic = load_semantics(workspace, create=True)
        assert semantic is not None
        self.console.print(
            Panel.fit(
                self._b(
                    f"角色 Token：{semantic['character']['token']} · 角色特征：{len(semantic['character']['features'])} · 服饰：{len(semantic['outfits'])}",
                    f"Character token: {semantic['character']['token']} · character features: {len(semantic['character']['features'])} · outfits: {len(semantic['outfits'])}",
                ),
                title=self._b("角色语义", "Character semantics"),
                border_style="cyan",
            )
        )

    def _semantic_manager(self, name: str) -> None:
        while True:
            workspace = DatasetWorkspace.load(name)
            semantic = load_semantics(workspace, create=True)
            assert semantic is not None
            self._show_semantics(workspace, semantic)
            action = self._menu(
                self._b("角色语义", "Character semantics"),
                [
                    MenuItem("token", self._b("修改角色 Token", "Edit character token")),
                    MenuItem("features", self._b("选择角色特征 Tags", "Select character feature tags")),
                    MenuItem("default", self._b("管理 Default 服饰", "Manage Default outfit")),
                    MenuItem("add", self._b("添加服饰", "Add outfit")),
                    MenuItem("manage", self._b("管理已有服饰", "Manage existing outfit")),
                    MenuItem("back", self._b("返回", "Back")),
                ],
                default="features",
            )
            if action == "back": return
            if action == "token":
                set_character_token(workspace, self._ask_text("Character token", default=semantic["character"]["token"]).strip())
            elif action == "features": self._choose_character_features(workspace)
            elif action == "default": self._manage_outfit(workspace, "default")
            elif action == "add": self._add_outfit(workspace)
            elif action == "manage":
                outfit_id = self._choose_outfit(semantic)
                if outfit_id: self._manage_outfit(workspace, outfit_id)

    def _show_semantics(self, workspace: DatasetWorkspace, semantic: dict) -> None:
        lines = [
            f"[bold]{semantic['character']['token']}[/bold]",
            self._b("角色特征：", "Character features: ") + (", ".join(semantic["character"]["features"]) or self._b("未选择", "none")),
            "",
        ]
        for outfit_id, outfit in semantic["outfits"].items():
            count = len(image_keys_for_outfit(workspace, semantic, outfit_id))
            features = ", ".join(outfit.get("features", [])) or self._b("未选择", "none")
            lines += [f"[bold]{outfit.get('label', outfit_id)}[/bold] · {outfit['token']} · {count} images", f"  {features}"]
        self.console.print(Panel.fit("\n".join(lines), title="Character / Outfit Concepts"))

    def _choose_character_features(self, workspace: DatasetWorkspace) -> None:
        semantic = load_semantics(workspace, create=True); assert semantic is not None
        current = list(semantic["character"]["features"])
        rows = character_feature_candidates(workspace, minimum_coverage=0.5)
        options = self._feature_options(rows, current, outfit=False)
        if not options:
            self.console.print(self._b("[yellow]没有共有 Tag 候选；请先 Auto-tag。[/yellow]", "[yellow]No common tag candidates; run Auto-tag first.[/yellow]")); return
        selected = select_many(self.console, self._b("选择角色特征 Tags", "Select character feature tags"), options, selected=current, page_size=24)
        set_character_features(workspace, selected)

    def _add_outfit(self, workspace: DatasetWorkspace) -> None:
        semantic = load_semantics(workspace, create=True); assert semantic is not None
        label = self._ask_text(self._b("服饰名称", "Outfit name")).strip()
        outfit_id = self._ask_text(self._b("服饰 ID", "Outfit id"), default=_slug(label)).strip()
        token = self._ask_text(self._b("服饰 Token", "Outfit token"), default=f"{semantic['character']['token']}_{_slug(outfit_id)}").strip()
        images = self._choose_outfit_images(workspace, semantic, None)
        if not images:
            self.console.print(self._b("[yellow]未选择图片，已取消。[/yellow]", "[yellow]No images selected; cancelled.[/yellow]")); return
        add_outfit(workspace, outfit_id, label=label, token=token, image_keys=images)
        self._choose_outfit_features(workspace, outfit_id)

    def _manage_outfit(self, workspace: DatasetWorkspace, outfit_id: str) -> None:
        while True:
            semantic = load_semantics(workspace, create=True); assert semantic is not None
            outfit = semantic["outfits"][outfit_id]
            items = [MenuItem("token", self._b("修改服饰 Token", "Edit outfit token")), MenuItem("features", self._b("选择服饰 Tags", "Select outfit feature tags"))]
            if outfit_id != "default": items.append(MenuItem("images", self._b("重新选择所属图片", "Reselect outfit images")))
            items.append(MenuItem("back", self._b("返回", "Back")))
            action = self._menu(f"{outfit.get('label', outfit_id)} · {outfit['token']}", items, default="features")
            if action == "back": return
            if action == "token": set_outfit_token(workspace, outfit_id, self._ask_text("Outfit token", default=outfit["token"]).strip())
            elif action == "features": self._choose_outfit_features(workspace, outfit_id)
            elif action == "images":
                images = self._choose_outfit_images(workspace, semantic, outfit_id)
                if images: set_outfit_images(workspace, outfit_id, images)

    def _choose_outfit_features(self, workspace: DatasetWorkspace, outfit_id: str) -> None:
        semantic = load_semantics(workspace, create=True); assert semantic is not None
        current = list(semantic["outfits"][outfit_id].get("features", []))
        rows = outfit_feature_candidates(workspace, semantic, outfit_id, minimum_coverage=0.35)
        options = self._feature_options(rows, current, outfit=True)
        if not options:
            self.console.print(self._b("[yellow]没有服饰 Tag 候选。[/yellow]", "[yellow]No outfit tag candidates.[/yellow]")); return
        selected = select_many(self.console, self._b("选择服饰 Tags", "Select outfit feature tags"), options, selected=current, page_size=24)
        set_outfit_features(workspace, outfit_id, selected)

    def _feature_options(self, rows: Sequence[dict], current: Sequence[str], *, outfit: bool) -> list[MultiSelectOption]:
        known = {str(row["tag"]) for row in rows}
        rows = list(rows) + [{"tag": tag, "coverage": 0.0, "specificity": 0.0} for tag in current if tag not in known]
        result = []
        for row in rows:
            tag, coverage = str(row["tag"]), float(row.get("coverage", 0.0))
            specificity = float(row.get("specificity", 0.0))
            label = f"{tag} · {coverage:.0%}" + (f" · Δ{specificity:+.2f}" if outfit else "")
            result.append(MultiSelectOption(tag, label, self._b(f"覆盖率 {coverage:.0%}" + (f" · 服饰区分度 Δ={specificity:+.2f}" if outfit else ""), f"Coverage {coverage:.0%}" + (f" · outfit specificity Δ={specificity:+.2f}" if outfit else ""))))
        return result

    def _choose_outfit_images(self, workspace: DatasetWorkspace, semantic: dict, outfit_id: str | None) -> list[str]:
        current = image_keys_for_outfit(workspace, semantic, outfit_id) if outfit_id else []
        options = [
            MultiSelectOption(
                item.key,
                item.relative.as_posix(),
                self._b(
                    f"来源 {item.source_id} · 当前服饰 {outfit_for_image(semantic, item.key)} · Tags: {self._truncate(workspace.caption_text(item.key), 100)}",
                    f"Source {item.source_id} · current outfit {outfit_for_image(semantic, item.key)} · Tags: {self._truncate(workspace.caption_text(item.key), 100)}",
                ),
            )
            for item in workspace.items(include_disabled=False, include_excluded=False)
        ]
        return select_many(self.console, self._b("选择属于该服饰的图片", "Select images belonging to this outfit"), options, selected=current, columns=3, page_size=30)

    def _choose_outfit(self, semantic: dict) -> str | None:
        ids = [outfit_id for outfit_id in semantic["outfits"] if outfit_id != "default"]
        if not ids:
            self.console.print(self._b("[dim]还没有额外服饰。[/dim]", "[dim]No additional outfits yet.[/dim]")); return None
        selected = self._menu(self._b("选择服饰", "Select outfit"), [MenuItem(i, semantic["outfits"][i].get("label", i), semantic["outfits"][i]["token"]) for i in ids] + [MenuItem("back", self._b("返回", "Back"))], default=ids[0])
        return None if selected == "back" else selected


def _slug(value: str) -> str:
    return (re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "outfit")[:64]
