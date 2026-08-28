from __future__ import annotations

import re
from typing import Sequence

from rich.panel import Panel

from .dataset_semantics import (
    add_outfit,
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
from . import semantic_project_hooks as _semantic_project_hooks  # noqa: F401
from .wizard import MenuItem


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
                MenuItem(
                    "import",
                    self._b("导入新的数据来源", "Import a new data source"),
                    self._b(
                        "向当前数据集追加图片目录、视频等受支持来源；来源与图片归属会记录在 Dataset Workspace 中，不会直接开始训练。",
                        "Add a supported image directory, video, or other source to this dataset. Source and image provenance are recorded in the Dataset Workspace; this does not start training.",
                    ),
                ),
                MenuItem(
                    "sources",
                    self._b("按来源管理", "Manage by source"),
                    self._b(
                        "按导入来源查看和管理图片，适合检查某一批素材、来源状态及其对应的数据项。",
                        "Inspect and manage images grouped by import source, including a source's status and associated dataset items.",
                    ),
                ),
                MenuItem(
                    "audit",
                    self._b("全数据集自动检查", "Audit the whole dataset"),
                    self._b(
                        "对整个数据集运行自动质量检查，汇总重复项、图像质量和其他需要人工关注的问题。",
                        "Run automatic whole-dataset checks for duplicates, image quality, and other issues that may need human attention.",
                    ),
                ),
                MenuItem(
                    "curate",
                    self._b("高级清洗分析", "Advanced curation analysis"),
                    self._b(
                        "进入更细粒度的数据清洗与分析工具，用于筛选异常样本、偏置和其他会影响 LoRA 学习质量的数据问题。",
                        "Open finer-grained curation and analysis tools for abnormal samples, bias, and other dataset issues that can affect LoRA quality.",
                    ),
                ),
                MenuItem(
                    "tag",
                    self._b("自动打 Tag", "Auto-tag images"),
                    self._b(
                        "为图片生成可编辑的自动 Tags；角色特征、服饰特征候选以及后续训练 caption 都会使用这些标签作为数据语义来源。",
                        "Generate editable auto-tags for images. Character/outfit feature candidates and later training captions use these tags as semantic input.",
                    ),
                ),
            ]
            if workspace.concept_type == "character":
                items.append(
                    MenuItem(
                        "concepts",
                        self._b("角色 / 服饰语义", "Character / outfit concepts"),
                        self._b(
                            "定义角色 Token、稳定角色特征、服饰 Token / 特征及图片归属；创建训练项目时这些定义会冻结为语义快照，并参与训练 caption 组合。",
                            "Define the character token, stable character features, outfit tokens/features, and image assignments. These definitions are frozen into a semantic snapshot when a training project is created and participate in caption composition.",
                        ),
                    )
                )
            items += [
                MenuItem(
                    "review",
                    self._b("人工图片审核 / 排除", "Manual image review / exclusions"),
                    self._b(
                        "人工浏览数据集图片并标记需要排除的样本；用于处理自动检查无法可靠判断的构图、质量或内容问题。",
                        "Review dataset images manually and mark samples for exclusion when composition, quality, or content cannot be judged reliably by automatic checks.",
                    ),
                ),
                MenuItem(
                    "edit_tags",
                    self._b("人工修改 Tag", "Edit tags manually"),
                    self._b(
                        "人工修正单张图片的 Tags，处理自动 Tagger 的误标、漏标或不希望进入训练描述的标签。",
                        "Correct tags on individual images to fix auto-tagger mistakes, omissions, or labels you do not want in training captions.",
                    ),
                ),
                MenuItem(
                    "training",
                    self._b("用此数据集开始训练", "Start training with this dataset"),
                    self._b(
                        "基于当前 Dataset Workspace 创建训练项目，并冻结当前数据、元数据和角色/服饰语义快照，之后再进入训练参数配置。",
                        "Create a training project from this Dataset Workspace, freezing the current dataset, metadata, and character/outfit semantic snapshots before configuring training parameters.",
                    ),
                ),
                MenuItem(
                    "back",
                    self._b("返回数据集列表", "Back to dataset list"),
                    self._b("返回数据集列表，不修改当前数据集。", "Return to the dataset list without changing this dataset."),
                ),
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
                    MenuItem(
                        "token",
                        self._b("修改角色 Token", "Edit character token"),
                        self._b(
                            "角色 Token 是代表“这个角色身份”的专用触发词。创建训练项目时它会冻结进语义快照；除 existing_passthrough 外，caption 阶段会把它作为必需前缀写入每张训练图的描述，推理时通常用它召回角色。若某服饰仍使用自动生成的默认 Token，修改角色 Token 时该服饰 Token 的角色前缀也会同步更新。",
                            "The character token is the dedicated trigger representing this character's identity. It is frozen into the semantic snapshot when a training project is created and, except in existing_passthrough mode, is inserted as a required prefix in every training caption. At inference it is normally used to recall the character. Outfit tokens still using their automatically generated defaults are renamed with the new character-token prefix as well.",
                        ),
                    ),
                    MenuItem(
                        "features",
                        self._b("选择角色特征 Tags", "Select character feature tags"),
                        self._b(
                            "角色特征 Tags 表示跨服饰仍属于角色本人的稳定身份/外观特征，例如发色、眼色、固定发型或其他固有特征，而不是衣服、姿势或背景。语义 caption 组合会从普通 Tags 中过滤这些已选特征，让角色 Token 承担它们，减少重复描述和角色/服饰概念纠缠。",
                            "Character feature tags describe stable identity/appearance traits that belong to the character across outfits, such as hair color, eye color, a fixed hairstyle, or other intrinsic traits—not clothing, pose, or background. Semantic caption composition removes selected character features from ordinary tags so the character token carries them, reducing duplicate description and character/outfit concept entanglement.",
                        ),
                    ),
                    MenuItem(
                        "default",
                        self._b("管理 Default 服饰", "Manage Default outfit"),
                        self._b(
                            "Default 服饰是所有未显式绑定到其他服饰的图片自动所属的兜底服饰。它有自己的服饰 Token 和服饰特征；这些图片的训练 caption 会同时包含角色 Token 与 Default 服饰 Token。",
                            "The Default outfit is the fallback outfit for every image not explicitly assigned to another outfit. It has its own outfit token and outfit features; captions for those images contain both the character token and the Default outfit token.",
                        ),
                    ),
                    MenuItem(
                        "add",
                        self._b("添加服饰", "Add outfit"),
                        self._b(
                            "为一套额外服饰创建独立概念：设置名称、ID 和服饰 Token，并选择属于它的图片与稳定服饰特征。绑定到该服饰的图片会在训练 caption 中使用它的服饰 Token，而不是 Default 服饰 Token。",
                            "Create a separate concept for an additional outfit: set its name, ID, and outfit token, then assign its images and stable outfit features. Images bound to it use this outfit token in training captions instead of the Default outfit token.",
                        ),
                    ),
                    MenuItem(
                        "manage",
                        self._b("管理已有服饰", "Manage existing outfit"),
                        self._b(
                            "选择一个已创建的非 Default 服饰，修改其服饰 Token、稳定服饰 Tags 或图片归属；用于把同一角色的不同服装拆成可单独触发和控制的概念。",
                            "Select an existing non-Default outfit and edit its outfit token, stable outfit tags, or image assignments. This separates different clothes for the same character into independently triggerable concepts.",
                        ),
                    ),
                    MenuItem(
                        "back",
                        self._b("返回", "Back"),
                        self._b("返回数据集操作页；已保存的角色/服饰语义保持不变。", "Return to dataset actions; already saved character/outfit semantics remain unchanged."),
                    ),
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
            items = [
                MenuItem(
                    "token",
                    self._b("修改服饰 Token", "Edit outfit token"),
                    self._b(
                        "服饰 Token 是代表当前这套衣服的专用触发词。属于该服饰的训练图片在语义 caption 组合时会同时写入角色 Token 和此服饰 Token；推理时可与角色 Token 一起使用，以召回和控制这套服装。",
                        "The outfit token is the dedicated trigger for this specific outfit. Semantic caption composition writes both the character token and this outfit token for images assigned to the outfit; at inference it can be used together with the character token to recall and control this clothing concept.",
                    ),
                ),
                MenuItem(
                    "features",
                    self._b("选择服饰 Tags", "Select outfit feature tags"),
                    self._b(
                        "服饰 Tags 用于记录这套服饰稳定出现的视觉元素，例如 jacket、skirt、ribbon 等，帮助定义和区分不同服饰概念。它们不决定图片属于哪套服饰；图片归属由“重新选择所属图片”单独管理。",
                        "Outfit tags record stable visual elements of this clothing concept, such as a jacket, skirt, or ribbon, helping define and distinguish outfits. They do not decide which images belong to the outfit; image assignment is managed separately by 'Reselect outfit images'.",
                    ),
                ),
            ]
            if outfit_id != "default":
                items.append(
                    MenuItem(
                        "images",
                        self._b("重新选择所属图片", "Reselect outfit images"),
                        self._b(
                            "决定哪些数据集图片绑定到当前服饰。这个绑定关系直接决定每张图片在训练 caption 中注入哪个服饰 Token；未绑定到额外服饰的图片自动归入 Default。",
                            "Choose which dataset images are bound to this outfit. This binding directly determines which outfit token is injected into each training caption; images not assigned to an additional outfit fall back to Default.",
                        ),
                    )
                )
            items.append(
                MenuItem(
                    "back",
                    self._b("返回", "Back"),
                    self._b("返回角色语义页；已保存的服饰设置保持不变。", "Return to Character semantics; already saved outfit settings remain unchanged."),
                )
            )
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
        items = []
        for outfit_id in ids:
            outfit = semantic["outfits"][outfit_id]
            features = ", ".join(outfit.get("features", [])) or self._b("未选择服饰特征", "no outfit features selected")
            items.append(
                MenuItem(
                    outfit_id,
                    outfit.get("label", outfit_id),
                    self._b(
                        f"服饰 Token：{outfit['token']} · 特征：{features}",
                        f"Outfit token: {outfit['token']} · features: {features}",
                    ),
                )
            )
        items.append(MenuItem("back", self._b("返回", "Back"), self._b("返回角色语义页，不选择服饰。", "Return to Character semantics without selecting an outfit.")))
        selected = self._menu(self._b("选择服饰", "Select outfit"), items, default=ids[0])
        return None if selected == "back" else selected


def _slug(value: str) -> str:
    return (re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "outfit")[:64]
