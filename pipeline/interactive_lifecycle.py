from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from rich.panel import Panel
from rich.table import Table

from .config import repository_root
from .dataset.image_info import discover_images
from .dataset_workspace import DatasetWorkspace, list_datasets
from .interactive_datasets import InteractiveWizard as BaseInteractiveWizard
from .models import PipelineError, StepStatus
from .service import load_project, project_path, run_single_step
from .state import ProjectState
from .steps import promote
from .training_config import (
    TrainingConfig,
    create_project_from_training_config,
    list_training_configs,
    make_training_workspace_name,
)
from .wizard import MenuItem, STRATEGIES


class InteractiveWizard(BaseInteractiveWizard):
    """Four-part UX: Dataset -> Training Config -> Training Status -> Results.

    Existing ProjectState directories remain the compatibility/runtime layer, but
    new interactive users no longer need to treat "Project" as the primary object.
    """

    # ------------------------------------------------------------------
    # Four-part home
    # ------------------------------------------------------------------
    def home(self) -> None:
        while True:
            datasets = list_datasets()
            configs = list_training_configs()
            states = self._all_project_states()
            active = self._active_training_count(states)
            results = len(self._result_entries(states))
            self.console.print(
                Panel.fit(
                    self._b(
                        "[bold blue]LoRA 工作台[/bold blue]\n"
                        "数据集负责整理数据；训练配置负责训练方法；训练状态负责运行/恢复；训练结果负责权重与示例图。\n\n"
                        f"数据集：{len(datasets)} · 训练配置：{len(configs)} · 活动/待处理训练：{active} · 已有结果：{results}",
                        "[bold blue]LoRA workspace[/bold blue]\n"
                        "Datasets curate data; Training Configs define recipes; Training Status runs/resumes work; Results owns weights and samples.\n\n"
                        f"Datasets: {len(datasets)} · configs: {len(configs)} · active/pending runs: {active} · results: {results}",
                    ),
                    border_style="blue",
                )
            )
            action = self._menu(
                self._b("主页", "Home"),
                [
                    MenuItem(
                        "datasets",
                        self._b("数据集", "Datasets"),
                        self._b("多来源导入、裁切、Tag、审核、排除/恢复。", "Import sources, crop, tag, review, exclude/restore."),
                    ),
                    MenuItem(
                        "configs",
                        self._b("训练配置", "Training configs"),
                        self._b("底模、Trigger、LoRA 参数、训练预算和工作流偏好。", "Base, trigger, LoRA parameters, budget, and workflow preferences."),
                    ),
                    MenuItem(
                        "status",
                        self._b("训练状态", "Training status"),
                        self._b("开始一次训练、查看进度、恢复中断或失败的训练。", "Start training, inspect progress, resume interrupted or failed work."),
                    ),
                    MenuItem(
                        "results",
                        self._b("训练结果", "Training results"),
                        self._b("权重、示例图片、评测、对比图和最佳模型。", "Weights, samples, evaluation sheets, and promoted best models."),
                    ),
                    MenuItem(
                        "system",
                        self._b("系统与高级", "System & advanced"),
                        self._b("底模管理、机器检查和旧 Project 技术视图。", "Base models, machine checks, and legacy Project technical view."),
                    ),
                    MenuItem("quit", self._b("退出", "Exit")),
                ],
                default="datasets" if not datasets else ("configs" if not configs else "status"),
            )
            if action == "quit":
                self.console.print(self._b("[dim]已退出。[/dim]", "[dim]Goodbye.[/dim]"))
                return
            self._guarded(
                {
                    "datasets": self.dataset_manager,
                    "configs": self.training_config_manager,
                    "status": self.training_status_manager,
                    "results": self.training_results_manager,
                    "system": self._system_menu,
                }[action]
            )

    def new_project(self):
        """Compatibility entry point: new work now starts from Training Status."""
        return self._start_training_from_dataset_config()

    def _system_menu(self) -> None:
        while True:
            states = self._all_project_states()
            items = [
                MenuItem("bases", self._b("管理底模", "Manage base models")),
                MenuItem("doctor", self._b("检查当前机器", "Check this machine")),
            ]
            if states:
                items.append(
                    MenuItem(
                        "projects",
                        self._b("技术 Project 视图", "Technical Project view"),
                        self._b("兼容旧项目和底层状态机；日常不需要使用。", "Compatibility view for legacy projects and the internal state machine."),
                    )
                )
            items.append(MenuItem("back", self._b("返回", "Back")))
            action = self._menu(self._b("系统与高级", "System & advanced"), items, default="bases")
            if action == "back":
                return
            if action == "bases":
                self.base_manager()
            elif action == "doctor":
                self.doctor()
            elif action == "projects":
                self._choose_and_open_project(states)

    # ------------------------------------------------------------------
    # Dataset dashboard override: keep curation separate from configuration.
    # ------------------------------------------------------------------
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
                        self._b("转到训练状态：再选择一份训练配置，并同时冻结两个快照。", "Go through Training Status, select a config, and freeze both snapshots together."),
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

    def _create_project_from_dataset_interactive(self, *, workspace: DatasetWorkspace | None = None):
        # Compatibility for old menu paths that still call this method.
        return self._start_training_from_dataset_config(prefilled_workspace=workspace)

    # ------------------------------------------------------------------
    # Training Configs
    # ------------------------------------------------------------------
    def training_config_manager(self) -> None:
        while True:
            configs = list_training_configs()
            self._render_training_configs(configs)
            items: list[MenuItem] = []
            if configs:
                items.append(MenuItem("open", self._b("打开训练配置", "Open training config")))
            items.extend(
                [
                    MenuItem("create", self._b("创建训练配置", "Create training config")),
                    MenuItem("back", self._b("返回", "Back")),
                ]
            )
            action = self._menu(
                self._b("训练配置", "Training configs"),
                items,
                default="open" if configs else "create",
            )
            if action == "back":
                return
            if action == "create":
                config = self._create_training_config()
                if config is not None:
                    self._training_config_dashboard(config.name)
            else:
                selected = self._select_training_config(configs)
                if selected is not None:
                    self._training_config_dashboard(selected.name)

    def _create_training_config(self) -> TrainingConfig | None:
        registry = self._enabled_bases()
        if not registry:
            self.console.print(self._b("[yellow]还没有已启用底模。[/yellow]", "[yellow]No enabled base model is registered.[/yellow]"))
            if self._confirm(self._b("现在管理底模吗？", "Manage base models now?"), default=True):
                self.base_manager()
                registry = self._enabled_bases()
            if not registry:
                return None

        while True:
            name = self._ask_text(self._b("训练配置名称", "Training config name")).strip()
            concept = self._menu(
                self._b("配置类型", "Config type"),
                [MenuItem("character", self._b("人物", "Character")), MenuItem("style", self._b("风格", "Style"))],
                default="character",
            )
            base = self._select_base(registry, title=self._b("底模", "Base checkpoint"))
            trigger = self._ask_text(self._b("Trigger", "Trigger"), default=f"zz_{name}").strip()
            strategy = self._menu(self._b("训练策略", "Training strategy"), list(STRATEGIES), default="quality")
            images_seen = self._ask_positive_int(self._b("图片曝光预算 images_seen", "Image exposure budget (images_seen)"), default=1000)
            overrides: dict[str, Any] = {}
            if not self._confirm(
                self._b("LoRA rank / alpha / 学习率使用策略默认值吗？", "Use strategy defaults for LoRA rank / alpha / learning rate?"),
                default=True,
            ):
                overrides["training"] = {
                    "network_dim": self._ask_positive_int("LoRA rank / network_dim", default=16),
                    "network_alpha": self._ask_positive_int("LoRA alpha / network_alpha", default=8),
                    "unet_lr": self._ask_positive_float("UNet learning rate", default=0.0001),
                }
            evaluation: dict[str, Any] = {}
            if concept == "character":
                evaluation["subject_prompt"] = self._ask_text(
                    self._b("评测主体 Prompt", "Evaluation subject prompt"), default="1girl"
                ).strip()
            try:
                config = TrainingConfig.create(
                    name,
                    concept_type=concept,
                    base=base,
                    trigger=trigger,
                    strategy=strategy,
                    images_seen=images_seen,
                    overrides=overrides,
                    evaluation=evaluation,
                )
            except (PipelineError, ValueError) as exc:
                self.console.print(f"[red]{exc}[/red]")
                if self._confirm(self._b("重新创建吗？", "Try again?"), default=True):
                    continue
                return None
            self.console.print(
                Panel.fit(
                    self._b(
                        f"[green bold]训练配置已创建[/green bold]\n{name}\n快照：{config.snapshot()['snapshot_hash'][:16]}",
                        f"[green bold]Training config created[/green bold]\n{name}\nSnapshot: {config.snapshot()['snapshot_hash'][:16]}",
                    )
                )
            )
            return config

    def _training_config_dashboard(self, name: str) -> None:
        while True:
            config = TrainingConfig.load(name)
            self._render_training_config_detail(config)
            action = self._menu(
                self._b("训练配置操作", "Training config actions"),
                [
                    MenuItem("core", self._b("修改基础训练设置", "Edit core training settings")),
                    MenuItem("lora", self._b("修改 LoRA 参数", "Edit LoRA parameters")),
                    MenuItem("workflow", self._b("修改数据处理工作流偏好", "Edit data-processing workflow preferences")),
                    MenuItem("evaluation", self._b("修改评测设置", "Edit evaluation settings")),
                    MenuItem("start", self._b("用这份配置开始训练", "Start training with this config")),
                    MenuItem("back", self._b("返回", "Back")),
                ],
                default="core",
            )
            if action == "back":
                return
            if action == "core":
                self._edit_training_config_core(config)
            elif action == "lora":
                self._edit_training_config_lora(config)
            elif action == "workflow":
                self._edit_training_config_workflow(config)
            elif action == "evaluation":
                self._edit_training_config_evaluation(config)
            elif action == "start":
                self._start_training_from_dataset_config(prefilled_config=config)

    def _edit_training_config_core(self, config: TrainingConfig) -> None:
        registry = self._enabled_bases()
        if registry:
            config.data["base"] = self._select_base(registry, title=self._b("底模", "Base checkpoint"))
        config.data["trigger"] = self._ask_text("Trigger", default=config.trigger).strip()
        config.data["strategy"] = self._menu(
            self._b("训练策略", "Training strategy"), list(STRATEGIES), default=config.strategy
        )
        config.data["images_seen"] = self._ask_positive_int(
            "images_seen", default=config.images_seen
        )
        config.save()
        self.console.print(self._b("[green]基础设置已保存。[/green]", "[green]Core settings saved.[/green]"))

    def _edit_training_config_lora(self, config: TrainingConfig) -> None:
        training = config.overrides.setdefault("training", {})
        action = self._menu(
            self._b("LoRA 参数", "LoRA parameters"),
            [
                MenuItem("custom", self._b("设置自定义 rank / alpha / LR", "Set custom rank / alpha / LR")),
                MenuItem("reset", self._b("恢复训练策略默认值", "Reset to strategy defaults")),
                MenuItem("back", self._b("返回", "Back")),
            ],
            default="custom",
        )
        if action == "back":
            return
        if action == "reset":
            for key in ("network_dim", "network_alpha", "unet_lr"):
                training.pop(key, None)
            if not training:
                config.overrides.pop("training", None)
        else:
            training["network_dim"] = self._ask_positive_int(
                "LoRA rank / network_dim", default=int(training.get("network_dim", 16))
            )
            training["network_alpha"] = self._ask_positive_int(
                "LoRA alpha / network_alpha", default=int(training.get("network_alpha", 8))
            )
            training["unet_lr"] = self._ask_positive_float(
                "UNet learning rate", default=float(training.get("unet_lr", 0.0001))
            )
        config.save()
        self.console.print(self._b("[green]LoRA 参数已保存。[/green]", "[green]LoRA parameters saved.[/green]"))

    def _edit_training_config_workflow(self, config: TrainingConfig) -> None:
        workflow = config.workflow
        workflow["run_dedup"] = self._confirm(
            self._b("训练快照仍运行感知/重复检查吗？", "Run duplicate checks on the training snapshot?"),
            default=bool(workflow.get("run_dedup", True)),
        )
        workflow["exclude_exact_duplicates"] = bool(workflow["run_dedup"]) and self._confirm(
            self._b("自动排除完全重复项吗？", "Automatically exclude exact duplicates?"),
            default=bool(workflow.get("exclude_exact_duplicates", False)),
        )
        if config.concept_type == "character":
            workflow["run_identity"] = self._confirm(
                self._b("运行人物身份一致性检查吗？", "Run character identity consistency checks?"),
                default=bool(workflow.get("run_identity", True)),
            )
        else:
            workflow["run_identity"] = False
        caption_items = [
            MenuItem("auto", self._b("自动（推荐）", "Auto (recommended)"), self._b("全有 Tag 则清洗已有 Tag，否则自动生成。", "Clean existing tags when complete, otherwise generate.")),
            MenuItem("generate", self._b("重新自动打标", "Generate captions")),
            MenuItem("existing_taglist_clean", self._b("使用并清洗已有 Tag", "Clean existing tag lists")),
            MenuItem("existing_passthrough", self._b("原样使用已有文本", "Pass existing captions through")),
            MenuItem("hybrid", self._b("已有 Tag + 自动建议", "Existing tags + generated suggestions")),
            MenuItem("skip", self._b("跳过 Caption 步骤", "Skip caption step")),
        ]
        current = str(workflow.get("caption_mode", "auto"))
        workflow["caption_mode"] = self._menu(
            self._b("Caption / Tag 模式", "Caption / tag mode"), caption_items, default=current
        )
        workflow["allow_trigger_only"] = self._confirm(
            self._b("无 Caption 时允许仅 Trigger 回退吗？", "Allow trigger-only fallback for missing captions?"),
            default=bool(workflow.get("allow_trigger_only", False)),
        )
        workflow["run_review"] = self._confirm(
            self._b("训练前生成最终审核摘要吗？", "Generate the final review summary before training?"),
            default=bool(workflow.get("run_review", True)),
        )
        workflow["run_screening_evaluation"] = False
        config.save()
        self.console.print(
            self._b(
                "[green]工作流偏好已保存。评测固定由“训练结果”区域触发。[/green]",
                "[green]Workflow saved. Evaluation is launched from Training Results.[/green]",
            )
        )

    def _edit_training_config_evaluation(self, config: TrainingConfig) -> None:
        if config.concept_type == "character":
            config.evaluation["subject_prompt"] = self._ask_text(
                self._b("评测主体 Prompt", "Evaluation subject prompt"),
                default=str(config.evaluation.get("subject_prompt", "1girl")),
            ).strip()
            config.save()
            self.console.print(self._b("[green]评测设置已保存。[/green]", "[green]Evaluation settings saved.[/green]"))
        else:
            self.console.print(
                self._b(
                    "[dim]Style 配置当前使用 profile 中的统一评测矩阵。[/dim]",
                    "[dim]Style configs currently use the profile evaluation matrix.[/dim]",
                )
            )

    def _render_training_configs(self, configs: Sequence[TrainingConfig]) -> None:
        if not configs:
            self.console.print(self._b("[dim]还没有训练配置。[/dim]", "[dim]No training configs yet.[/dim]"))
            return
        table = Table(title=self._b("训练配置", "Training configs"))
        table.add_column(self._b("名称", "Name"), style="bold")
        table.add_column(self._b("类型", "Type"))
        table.add_column(self._b("底模", "Base"))
        table.add_column(self._b("策略", "Strategy"))
        table.add_column("Rank", justify="right")
        table.add_column("images_seen", justify="right")
        table.add_column(self._b("快照", "Snapshot"))
        for config in configs:
            training = config.overrides.get("training", {})
            table.add_row(
                config.name,
                config.concept_type,
                config.base,
                config.strategy,
                str(training.get("network_dim", self._b("默认", "default"))),
                str(config.images_seen),
                config.snapshot()["snapshot_hash"][:10],
            )
        self.console.print(table)

    def _render_training_config_detail(self, config: TrainingConfig) -> None:
        training = config.overrides.get("training", {})
        workflow = config.workflow
        table = Table(title=self._b(f"训练配置 · {config.name}", f"Training config · {config.name}"), show_header=False)
        table.add_column(self._b("项目", "Field"), style="bold")
        table.add_column(self._b("值", "Value"))
        table.add_row(self._b("类型", "Type"), config.concept_type)
        table.add_row(self._b("底模", "Base"), config.base)
        table.add_row("Trigger", config.trigger)
        table.add_row(self._b("策略", "Strategy"), config.strategy)
        table.add_row("images_seen", str(config.images_seen))
        table.add_row("Rank", str(training.get("network_dim", self._b("策略默认", "strategy default"))))
        table.add_row("Alpha", str(training.get("network_alpha", self._b("策略默认", "strategy default"))))
        table.add_row("UNet LR", str(training.get("unet_lr", self._b("策略默认", "strategy default"))))
        table.add_row("Caption", str(workflow.get("caption_mode", "auto")))
        table.add_row(self._b("配置快照", "Config snapshot"), config.snapshot()["snapshot_hash"][:16])
        self.console.print(table)

    def _select_training_config(
        self,
        configs: Sequence[TrainingConfig],
        *,
        concept_type: str | None = None,
    ) -> TrainingConfig | None:
        filtered = [config for config in configs if concept_type is None or config.concept_type == concept_type]
        if not filtered:
            return None
        items = [
            MenuItem(
                config.name,
                config.name,
                f"{config.concept_type} · {config.base} · {config.strategy} · images_seen={config.images_seen}",
            )
            for config in filtered
        ] + [MenuItem("back", self._b("返回", "Back"))]
        selected = self._menu(self._b("选择训练配置", "Select training config"), items, default=filtered[0].name)
        if selected == "back":
            return None
        return next(config for config in filtered if config.name == selected)

    # ------------------------------------------------------------------
    # Training Status
    # ------------------------------------------------------------------
    def training_status_manager(self) -> None:
        while True:
            states = self._all_project_states()
            entries = self._status_entries(states)
            self._render_training_status(entries)
            items = [
                MenuItem("start", self._b("开始一次新训练", "Start a new training"), self._b("选择 Dataset + Training Config，并同时冻结快照。", "Choose Dataset + Training Config and freeze both snapshots.")),
            ]
            if entries:
                items.append(MenuItem("open", self._b("查看 / 恢复训练", "Inspect / resume training")))
            items.append(MenuItem("back", self._b("返回", "Back")))
            action = self._menu(self._b("训练状态", "Training status"), items, default="open" if entries else "start")
            if action == "back":
                return
            if action == "start":
                self._start_training_from_dataset_config()
            else:
                entry = self._select_status_entry(entries)
                if entry is not None:
                    self._training_status_detail(entry)

    def _start_training_from_dataset_config(
        self,
        *,
        prefilled_workspace: DatasetWorkspace | None = None,
        prefilled_config: TrainingConfig | None = None,
    ) -> ProjectState | None:
        datasets = list_datasets()
        if prefilled_workspace is None:
            if not datasets:
                self.console.print(self._b("[yellow]还没有数据集。[/yellow]", "[yellow]No datasets exist yet.[/yellow]"))
                if self._confirm(self._b("现在打开数据集管理吗？", "Open dataset manager now?"), default=True):
                    self.dataset_manager()
                return None
            prefilled_workspace = self._select_dataset(datasets)
            if prefilled_workspace is None:
                return None
        workspace = DatasetWorkspace.load(prefilled_workspace.name)
        summary = workspace.summary()
        if int(summary["active_images"]) < 1:
            raise PipelineError(self._b("数据集没有可训练图片。", "Dataset has no active training images."))

        configs = list_training_configs()
        if prefilled_config is None:
            compatible = [config for config in configs if config.concept_type == workspace.concept_type]
            if not compatible:
                self.console.print(
                    self._b(
                        f"[yellow]没有与 {workspace.concept_type} 数据集兼容的训练配置。[/yellow]",
                        f"[yellow]No training config is compatible with this {workspace.concept_type} dataset.[/yellow]",
                    )
                )
                if self._confirm(self._b("现在创建训练配置吗？", "Create a training config now?"), default=True):
                    created = self._create_training_config()
                    if created is not None and created.concept_type == workspace.concept_type:
                        prefilled_config = created
                if prefilled_config is None:
                    return None
            else:
                prefilled_config = self._select_training_config(compatible, concept_type=workspace.concept_type)
                if prefilled_config is None:
                    return None
        config = TrainingConfig.load(prefilled_config.name)
        if config.concept_type != workspace.concept_type:
            raise PipelineError(
                self._b(
                    f"数据集类型 {workspace.concept_type} 与配置类型 {config.concept_type} 不兼容。",
                    f"Dataset type {workspace.concept_type} is incompatible with config type {config.concept_type}.",
                )
            )
        config.validate(require_enabled_base=True)

        audit = workspace.audit()
        safe = int(audit["summary"]["safe_exclude_suggestions"])
        if safe and self._confirm(
            self._b(
                f"数据集还有 {safe} 个损坏/完全重复项可安全排除；训练前自动处理吗？",
                f"The dataset has {safe} safe corrupt/exact-duplicate exclusions; apply them before training?",
            ),
            default=True,
        ):
            workspace.apply_safe_audit_exclusions()
            workspace = DatasetWorkspace.load(workspace.name)
            summary = workspace.summary()

        dataset_snapshot = workspace.snapshot()
        config_snapshot = config.snapshot()
        table = Table(title=self._b("冻结训练输入", "Freeze training inputs"), show_header=False)
        table.add_column(self._b("项目", "Field"), style="bold")
        table.add_column(self._b("值", "Value"))
        table.add_row(self._b("数据集", "Dataset"), workspace.name)
        table.add_row(self._b("数据集快照", "Dataset snapshot"), dataset_snapshot["snapshot_hash"][:16])
        table.add_row(self._b("图片", "Images"), str(dataset_snapshot["image_count"]))
        table.add_row(self._b("训练配置", "Training config"), config.name)
        table.add_row(self._b("配置快照", "Config snapshot"), config_snapshot["snapshot_hash"][:16])
        table.add_row(self._b("底模", "Base"), config.base)
        table.add_row("Trigger", config.trigger)
        table.add_row(self._b("策略", "Strategy"), config.strategy)
        table.add_row("images_seen", str(config.images_seen))
        self.console.print(table)
        if not self._confirm(
            self._b(
                "创建这次训练的不可变 Dataset + Config 快照吗？",
                "Create immutable Dataset + Config snapshots for this training?",
            ),
            default=True,
        ):
            return None

        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        project_name = make_training_workspace_name(workspace.name, config.name, timestamp=timestamp)
        suffix = 1
        original = project_name
        while project_path(project_name).exists():
            tail = f"-{suffix}"
            project_name = (original[: 64 - len(tail)] + tail).rstrip("-._")
            suffix += 1
        state = create_project_from_training_config(
            workspace,
            config,
            project_name=project_name,
        )
        self.console.print(
            Panel.fit(
                self._b(
                    f"[green bold]训练工作区已冻结[/green bold]\n"
                    f"Dataset：{workspace.name} @ {state.payload['project']['dataset_snapshot']['snapshot_hash'][:16]}\n"
                    f"Config：{config.name} @ {state.payload['project']['training_config_snapshot']['snapshot_hash'][:16]}\n"
                    "后续修改 Dataset 或 Training Config 都不会改变这次训练。",
                    f"[green bold]Training workspace frozen[/green bold]\n"
                    f"Dataset: {workspace.name} @ {state.payload['project']['dataset_snapshot']['snapshot_hash'][:16]}\n"
                    f"Config: {config.name} @ {state.payload['project']['training_config_snapshot']['snapshot_hash'][:16]}\n"
                    "Later Dataset or Training Config edits cannot change this training.",
                )
            )
        )
        if self._confirm(self._b("现在准备并开始 GPU 训练吗？", "Prepare and start GPU training now?"), default=True):
            self.continue_project(state.name)
        return load_project(state.name)

    def _training_status_detail(self, entry: dict[str, Any]) -> None:
        while True:
            state = load_project(str(entry["project"]))
            run = self._find_run_record(state, entry.get("run_id"))
            project = state.payload["project"]
            identity = project.get("training_identity", {})
            status = str(run.get("status")) if run else self._workspace_status(state)
            table = Table(title=self._b("训练状态详情", "Training status detail"), show_header=False)
            table.add_column(self._b("项目", "Field"), style="bold")
            table.add_column(self._b("值", "Value"))
            table.add_row(self._b("数据集", "Dataset"), str(identity.get("dataset") or project.get("dataset_snapshot", {}).get("dataset") or "legacy"))
            table.add_row(self._b("训练配置", "Training config"), str(identity.get("config") or "legacy"))
            table.add_row(self._b("状态", "Status"), status)
            table.add_row(self._b("当前内部步骤", "Current internal step"), str(state.next_actionable_step() or "complete"))
            if run:
                table.add_row("Run ID", str(run.get("id")))
                budget = run.get("resolved_budget", {})
                if isinstance(budget, dict):
                    table.add_row("images_seen", str(budget.get("target_images_seen") or budget.get("images_seen") or ""))
                metrics = run.get("metrics", {})
                if isinstance(metrics, dict):
                    if metrics.get("seconds_per_step") is not None:
                        table.add_row("sec/step", str(metrics.get("seconds_per_step")))
                    if metrics.get("peak_vram_gb") is not None:
                        table.add_row("peak VRAM", str(metrics.get("peak_vram_gb")))
                if run.get("last_error"):
                    table.add_row(self._b("最后错误", "Last error"), self._truncate(str(run["last_error"]), 180))
            table.add_row(self._b("技术工作区", "Technical workspace"), state.name)
            self.console.print(table)

            actions = []
            if status not in {"trained", "evaluated", "promoted"} or state.next_actionable_step() is not None:
                actions.append(MenuItem("continue", self._b("继续 / 恢复训练", "Continue / resume training")))
            if run and run.get("status") in {"trained", "evaluated", "promoted"}:
                actions.append(MenuItem("results", self._b("查看这次训练结果", "Open this training result")))
            actions.extend(
                [
                    MenuItem("technical", self._b("打开技术 Project 仪表盘", "Open technical Project dashboard")),
                    MenuItem("back", self._b("返回", "Back")),
                ]
            )
            action = self._menu(self._b("训练状态操作", "Training status actions"), actions, default=actions[0].value)
            if action == "back":
                return
            if action == "continue":
                self.continue_project(state.name)
            elif action == "technical":
                self.project_dashboard(state.name)
            elif action == "results" and run:
                self._training_result_detail({"project": state.name, "run_id": str(run["id"])})

    def _status_entries(self, states: Sequence[ProjectState]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for state in states:
            project = state.payload["project"]
            identity = project.get("training_identity", {})
            runs = list(state.payload.get("runs", []))
            if runs:
                for run in runs:
                    entries.append(
                        {
                            "project": state.name,
                            "run_id": str(run.get("id")),
                            "status": str(run.get("status", "unknown")),
                            "dataset": identity.get("dataset") or project.get("dataset_snapshot", {}).get("dataset") or "legacy",
                            "config": identity.get("config") or "legacy",
                            "updated": run.get("finished_at") or run.get("interrupted_at") or run.get("started_at"),
                            "legacy": project.get("workspace_role") != "training_run",
                        }
                    )
            else:
                entries.append(
                    {
                        "project": state.name,
                        "run_id": None,
                        "status": self._workspace_status(state),
                        "dataset": identity.get("dataset") or project.get("dataset_snapshot", {}).get("dataset") or "legacy",
                        "config": identity.get("config") or "legacy",
                        "updated": project.get("updated_at"),
                        "legacy": project.get("workspace_role") != "training_run",
                    }
                )
        return sorted(entries, key=lambda item: str(item.get("updated") or ""), reverse=True)

    def _render_training_status(self, entries: Sequence[dict[str, Any]]) -> None:
        if not entries:
            self.console.print(self._b("[dim]还没有训练记录。[/dim]", "[dim]No training records yet.[/dim]"))
            return
        table = Table(title=self._b("训练状态", "Training status"))
        table.add_column(self._b("数据集", "Dataset"), style="bold")
        table.add_column(self._b("配置", "Config"))
        table.add_column("Run")
        table.add_column(self._b("状态", "Status"))
        table.add_column(self._b("兼容模式", "Mode"))
        for entry in entries[:40]:
            table.add_row(
                str(entry["dataset"]),
                str(entry["config"]),
                str(entry.get("run_id") or "—"),
                str(entry["status"]),
                self._b("旧 Project", "legacy Project") if entry.get("legacy") else self._b("四区运行", "four-part run"),
            )
        self.console.print(table)

    def _select_status_entry(self, entries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        if not entries:
            return None
        items: list[MenuItem] = []
        mapping: dict[str, dict[str, Any]] = {}
        for index, entry in enumerate(entries, start=1):
            key = f"entry-{index}"
            mapping[key] = entry
            items.append(
                MenuItem(
                    key,
                    f"{entry['dataset']} / {entry['config']}",
                    f"{entry.get('run_id') or 'pending'} · {entry['status']} · workspace={entry['project']}",
                )
            )
        items.append(MenuItem("back", self._b("返回", "Back")))
        selected = self._menu(self._b("选择训练", "Select training"), items, default=items[0].value)
        return None if selected == "back" else mapping[selected]

    # ------------------------------------------------------------------
    # Training Results
    # ------------------------------------------------------------------
    def training_results_manager(self) -> None:
        while True:
            entries = self._result_entries(self._all_project_states())
            self._render_training_results(entries)
            if not entries:
                self._ask_text(self._b("按 Enter 返回", "Press Enter to return"), default="")
                return
            action = self._menu(
                self._b("训练结果", "Training results"),
                [
                    MenuItem("open", self._b("打开一个训练结果", "Open a training result")),
                    MenuItem("back", self._b("返回", "Back")),
                ],
                default="open",
            )
            if action == "back":
                return
            selected = self._select_result_entry(entries)
            if selected is not None:
                self._training_result_detail(selected)

    def _training_result_detail(self, entry: dict[str, Any]) -> None:
        while True:
            state = load_project(str(entry["project"]))
            run = self._find_run_record(state, str(entry["run_id"]))
            if run is None:
                return
            self._render_result_detail(state, run)
            evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
            actions = [
                MenuItem(
                    "screening",
                    self._b("运行 / 重跑 Screening 评测", "Run / rerun screening evaluation"),
                    self._b("生成示例图和 checkpoint × strength 对比图。", "Generate samples and checkpoint × strength sheets."),
                ),
                MenuItem(
                    "full",
                    self._b("运行 Full 评测", "Run full evaluation"),
                    self._b("明确选择 1–2 个 finalist 后生成完整评测。", "Explicitly select 1–2 finalists for full evaluation."),
                ),
                MenuItem(
                    "promote",
                    self._b("选择最佳权重", "Promote best weight"),
                    self._b("基于人工查看过的评测证据创建 best.safetensors。", "Create best.safetensors after human review of evaluation evidence."),
                ),
                MenuItem("paths", self._b("查看权重 / 示例图片路径", "Show weight / sample paths")),
                MenuItem("technical", self._b("打开技术 Project 仪表盘", "Open technical Project dashboard")),
                MenuItem("back", self._b("返回", "Back")),
            ]
            default = "screening" if "screening" not in evidence else ("full" if "full" not in evidence else "promote")
            action = self._menu(self._b("结果操作", "Result actions"), actions, default=default)
            if action == "back":
                return
            if action == "screening":
                self._evaluate_selected_run(state, run, stage="screening")
            elif action == "full":
                self._evaluate_selected_run(state, run, stage="full")
            elif action == "promote":
                self._promote_selected_run(state, run)
            elif action == "paths":
                self._render_result_paths(run)
            elif action == "technical":
                self.project_dashboard(state.name)

    def _evaluate_selected_run(self, state: ProjectState, run: dict[str, Any], *, stage: str) -> None:
        checkpoints = [Path(value) for value in run.get("checkpoints", []) if Path(value).is_file()]
        if not checkpoints:
            raise PipelineError(self._b("没有可评测权重。", "No checkpoints are available for evaluation."))
        checkpoint_names: list[str] | None = None
        if stage == "full":
            finalists = self._select_checkpoints(
                checkpoints,
                title=self._b("选择 1–2 个 Full 评测 finalist", "Choose 1–2 finalists for full evaluation"),
                minimum=1,
                maximum=2,
            )
            checkpoint_names = [path.name for path in finalists]
        evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
        force = stage in evidence
        if force and not self._confirm(
            self._b(f"{stage} 已存在，确定重跑吗？", f"{stage} already exists. Rerun it?"),
            default=False,
        ):
            return
        if not self._confirm(self._b("现在使用 GPU 生成评测示例吗？", "Generate evaluation samples on the GPU now?"), default=True):
            return
        result = self._run_with_lock_retry(
            lambda break_lock: run_single_step(
                load_project(state.name),
                "evaluate",
                force=force,
                break_lock=break_lock,
                verbose=self.verbose,
                evaluation_stage=stage,
                evaluation_run=str(run["id"]),
                evaluation_checkpoints=checkpoint_names,
            )
        )
        self._print_step_result("evaluate", result)

    def _promote_selected_run(self, state: ProjectState, run: dict[str, Any]) -> None:
        evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
        evaluated = {
            value
            for record in evidence.values()
            if isinstance(record, dict)
            for value in record.get("checkpoints", [])
        }
        checkpoints = [
            Path(value)
            for value in run.get("checkpoints", [])
            if Path(value).is_file() and (Path(value).name in evaluated or Path(value).stem in evaluated)
        ]
        if not checkpoints:
            self.console.print(
                self._b(
                    "[yellow]还没有经过评测的 checkpoint。先运行 Screening 或 Full。[/yellow]",
                    "[yellow]No evaluated checkpoint is available. Run screening or full evaluation first.[/yellow]",
                )
            )
            return
        choice = self._menu(
            self._b("选择最佳 checkpoint", "Choose best checkpoint"),
            [MenuItem(path.name, path.name, str(path)) for path in checkpoints],
            default=checkpoints[-1].name,
        )
        strength = self._ask_positive_float(
            self._b("推荐 LoRA 强度", "Recommended LoRA strength"), default=0.8
        )
        if not self._confirm(
            self._b(f"确认把 {choice} 设为最佳权重吗？", f"Promote {choice} as the best weight?"),
            default=False,
        ):
            return
        payload = promote.run(
            load_project(state.name),
            run_id=str(run["id"]),
            checkpoint_name=choice,
            strength=strength,
        )
        self.console.print(
            Panel.fit(
                self._b(
                    f"[green bold]最佳权重已生成[/green bold]\n{payload['artifacts']['promoted_lora']}",
                    f"[green bold]Best weight promoted[/green bold]\n{payload['artifacts']['promoted_lora']}",
                )
            )
        )

    def _result_entries(self, states: Sequence[ProjectState]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for state in states:
            project = state.payload["project"]
            identity = project.get("training_identity", {})
            for run in state.payload.get("runs", []):
                if run.get("status") not in {"trained", "evaluated", "promoted"}:
                    continue
                checkpoints = [Path(value) for value in run.get("checkpoints", []) if Path(value).is_file()]
                if not checkpoints:
                    continue
                run_dir = Path(str(run.get("path")))
                samples = len(discover_images(run_dir / "samples")) if (run_dir / "samples").is_dir() else 0
                entries.append(
                    {
                        "project": state.name,
                        "run_id": str(run["id"]),
                        "status": str(run.get("status")),
                        "dataset": identity.get("dataset") or project.get("dataset_snapshot", {}).get("dataset") or "legacy",
                        "config": identity.get("config") or "legacy",
                        "checkpoints": len(checkpoints),
                        "samples": samples,
                        "promoted": bool(run.get("promotion")),
                        "updated": run.get("finished_at") or run.get("started_at"),
                    }
                )
        return sorted(entries, key=lambda item: str(item.get("updated") or ""), reverse=True)

    def _render_training_results(self, entries: Sequence[dict[str, Any]]) -> None:
        if not entries:
            self.console.print(self._b("[dim]还没有完成的训练结果。[/dim]", "[dim]No completed training results yet.[/dim]"))
            return
        table = Table(title=self._b("训练结果", "Training results"))
        table.add_column(self._b("数据集", "Dataset"), style="bold")
        table.add_column(self._b("配置", "Config"))
        table.add_column("Run")
        table.add_column(self._b("状态", "Status"))
        table.add_column(self._b("权重", "Weights"), justify="right")
        table.add_column(self._b("示例图", "Samples"), justify="right")
        table.add_column("Best")
        for entry in entries:
            table.add_row(
                str(entry["dataset"]),
                str(entry["config"]),
                str(entry["run_id"]),
                str(entry["status"]),
                str(entry["checkpoints"]),
                str(entry["samples"]),
                "★" if entry["promoted"] else "",
            )
        self.console.print(table)

    def _select_result_entry(self, entries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        items: list[MenuItem] = []
        mapping: dict[str, dict[str, Any]] = {}
        for index, entry in enumerate(entries, start=1):
            key = f"result-{index}"
            mapping[key] = entry
            items.append(
                MenuItem(
                    key,
                    f"{entry['dataset']} / {entry['config']} / {entry['run_id']}",
                    f"{entry['status']} · weights={entry['checkpoints']} · samples={entry['samples']}",
                )
            )
        items.append(MenuItem("back", self._b("返回", "Back")))
        selected = self._menu(self._b("选择训练结果", "Select training result"), items, default=items[0].value)
        return None if selected == "back" else mapping[selected]

    def _render_result_detail(self, state: ProjectState, run: dict[str, Any]) -> None:
        project = state.payload["project"]
        identity = project.get("training_identity", {})
        run_dir = Path(str(run["path"]))
        samples = len(discover_images(run_dir / "samples")) if (run_dir / "samples").is_dir() else 0
        evidence = run.get("evaluation", {}) if isinstance(run.get("evaluation"), dict) else {}
        table = Table(title=self._b("训练结果详情", "Training result detail"), show_header=False)
        table.add_column(self._b("项目", "Field"), style="bold")
        table.add_column(self._b("值", "Value"))
        table.add_row(self._b("数据集", "Dataset"), str(identity.get("dataset") or "legacy"))
        table.add_row(self._b("训练配置", "Training config"), str(identity.get("config") or "legacy"))
        table.add_row("Run ID", str(run["id"]))
        table.add_row(self._b("状态", "Status"), str(run.get("status")))
        table.add_row(self._b("权重数量", "Checkpoint count"), str(len(run.get("checkpoints", []))))
        table.add_row(self._b("示例图数量", "Sample images"), str(samples))
        table.add_row(self._b("评测阶段", "Evaluation stages"), ", ".join(sorted(evidence)) or self._b("未评测", "not evaluated"))
        promotion = run.get("promotion", {}) if isinstance(run.get("promotion"), dict) else {}
        table.add_row(self._b("最佳权重", "Best weight"), str(promotion.get("checkpoint") or "—"))
        table.add_row(self._b("结果目录", "Result directory"), str(run_dir))
        self.console.print(table)

    def _render_result_paths(self, run: dict[str, Any]) -> None:
        run_dir = Path(str(run["path"]))
        table = Table(title=self._b("结果文件", "Result files"))
        table.add_column(self._b("类型", "Type"))
        table.add_column(self._b("路径", "Path"))
        for value in run.get("checkpoints", []):
            path = Path(value)
            if path.is_file():
                table.add_row(self._b("权重", "weight"), str(path))
        best = run_dir / "best.safetensors"
        if best.is_file():
            table.add_row("BEST", str(best))
        if (run_dir / "samples").is_dir():
            for path in discover_images(run_dir / "samples")[:30]:
                table.add_row(self._b("示例图", "sample"), str(path))
        sheets = run_dir / "contact-sheets"
        if sheets.is_dir():
            for path in sorted(sheets.rglob("*.jpg")):
                table.add_row(self._b("对比图", "contact sheet"), str(path))
        report = run_dir / "report.html"
        if report.is_file():
            table.add_row(self._b("评测报告", "evaluation report"), str(report))
        self.console.print(table)

    # ------------------------------------------------------------------
    # Shared state helpers
    # ------------------------------------------------------------------
    def _all_project_states(self) -> list[ProjectState]:
        root = repository_root() / "projects"
        if not root.is_dir():
            return []
        states: list[ProjectState] = []
        for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
            if (path / "project.yaml").is_file():
                states.append(ProjectState.load(path))
        return states

    def _workspace_status(self, state: ProjectState) -> str:
        next_step = state.next_actionable_step()
        if next_step is None:
            return "complete"
        status = state.status(next_step)
        return f"{status.value}:{next_step}"

    def _active_training_count(self, states: Sequence[ProjectState]) -> int:
        active_status = {"running", "interrupted", "failed", "configuring", "dry-run"}
        count = 0
        for state in states:
            runs = state.payload.get("runs", [])
            if not runs and state.next_actionable_step() is not None:
                count += 1
                continue
            count += sum(str(run.get("status")) in active_status for run in runs)
        return count

    @staticmethod
    def _find_run_record(state: ProjectState, run_id: str | None) -> dict[str, Any] | None:
        if run_id is None:
            return None
        for run in state.payload.get("runs", []):
            if str(run.get("id")) == str(run_id):
                return run
        return None
