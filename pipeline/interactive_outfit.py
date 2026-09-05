from __future__ import annotations

from typing import Any, Sequence

from rich.panel import Panel
from rich.table import Table

from .interactive_final import InteractiveWizard as BaseInteractiveWizard
from .models import PipelineError
from .training_config import (
    TrainingConfig,
    parse_anchor_tags,
    prompt_contains_trigger,
)
from .training_parameters import (
    TRAINING_PARAMETER_SPECS,
    effective_training_settings,
    reset_key_training_overrides,
    strategy_training_defaults,
    update_key_training_overrides,
)
from .trigger_policy import TRIGGER_STRATEGIES, resolve_trigger_policy
from .wizard import MenuItem, STRATEGIES


class InteractiveWizard(BaseInteractiveWizard):
    """Final UX layer for explicit targets, trigger policy, and key parameter tuning."""

    def _create_training_config(self) -> TrainingConfig | None:
        registry = self._enabled_bases()
        if not registry:
            self.console.print(
                self._b(
                    "[yellow]还没有已启用底模。[/yellow]",
                    "[yellow]No enabled base model is registered.[/yellow]",
                )
            )
            if self._confirm(self._b("现在管理底模吗？", "Manage base models now?"), default=True):
                self.base_manager()
                registry = self._enabled_bases()
            if not registry:
                return None

        while True:
            name = self._ask_text(self._b("训练配置名称", "Training config name")).strip()
            target_type = self._menu(
                self._b("训练目标", "Training target"),
                [
                    MenuItem(
                        "character",
                        self._b("人物", "Character"),
                        self._b("学习人物身份。", "Learn character identity."),
                    ),
                    MenuItem(
                        "character_outfit",
                        self._b("人物衣装", "Character outfit"),
                        self._b(
                            "学习某个人物的特定衣装；Trigger 与人物锚点分离。",
                            "Learn a specific outfit with a separate trigger and character anchors.",
                        ),
                    ),
                    MenuItem("style", self._b("风格", "Style")),
                ],
                default="character",
            )
            concept = "style" if target_type == "style" else "character"
            base = self._select_base(registry, title=self._b("底模", "Base checkpoint"))
            trigger_strategy = self._ask_trigger_strategy(target_type)
            trigger_default = name if trigger_strategy == "rare_token" else f"zz_{name}"
            trigger = self._ask_training_trigger(name, default=trigger_default)
            anchor_tags: list[str] = []
            if target_type == "character_outfit":
                anchor_tags = self._ask_character_anchors()
            strategy = self._menu(
                self._b("训练策略", "Training strategy"),
                list(STRATEGIES),
                default="quality",
            )
            images_seen = self._ask_positive_int(
                self._b("图片曝光预算 images_seen", "Image exposure budget (images_seen)"),
                default=1000,
            )
            overrides: dict[str, Any] = {}
            if not self._confirm(
                self._b("关键训练参数使用策略预设吗？", "Use strategy defaults for key training parameters?"),
                default=True,
            ):
                self._render_training_parameter_help(strategy, images_seen=images_seen)
                overrides = self._ask_key_training_parameters(strategy, overrides)

            evaluation: dict[str, Any] = {}
            if concept == "character":
                label = (
                    self._b(
                        "评测主体基础 Prompt（自动附加人物锚点）",
                        "Evaluation base subject prompt (character anchors are appended automatically)",
                    )
                    if target_type == "character_outfit"
                    else self._b("评测主体 Prompt", "Evaluation subject prompt")
                )
                effective_trigger = resolve_trigger_policy(
                    trigger,
                    strategy=trigger_strategy,
                    anchors=anchor_tags,
                ).trigger
                evaluation["subject_prompt"] = self._ask_evaluation_subject(
                    effective_trigger, label=label, default="1girl"
                )
            try:
                config = TrainingConfig.create(
                    name,
                    concept_type=concept,
                    target_type=target_type,
                    base=base,
                    trigger=trigger,
                    trigger_strategy=trigger_strategy,
                    anchor_tags=anchor_tags,
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
                        f"[green bold]训练配置已创建[/green bold]\n{name}\n"
                        f"目标：{config.target_type}\nTrigger：{config.trigger}\n"
                        f"Trigger 策略：{config.trigger_strategy}\n快照：{config.snapshot()['snapshot_hash'][:16]}",
                        f"[green bold]Training config created[/green bold]\n{name}\n"
                        f"Target: {config.target_type}\nTrigger: {config.trigger}\n"
                        f"Trigger strategy: {config.trigger_strategy}\nSnapshot: {config.snapshot()['snapshot_hash'][:16]}",
                    )
                )
            )
            return config

    def _edit_training_config_core(self, config: TrainingConfig) -> None:
        registry = self._enabled_bases()
        if registry:
            config.data["base"] = self._select_base(
                registry, title=self._b("底模", "Base checkpoint")
            )
        if config.concept_type == "character":
            target = self._menu(
                self._b("人物训练目标", "Character training target"),
                [
                    MenuItem("character", self._b("人物", "Character")),
                    MenuItem("character_outfit", self._b("人物衣装", "Character outfit")),
                ],
                default=config.target_type,
            )
            config.data["target_type"] = target
            if target == "character_outfit":
                config.data["anchor_tags"] = self._ask_character_anchors(default=config.anchor_tags)
            else:
                config.data["anchor_tags"] = []
        else:
            target = config.target_type

        trigger_strategy = self._ask_trigger_strategy(
            target,
            default=config.trigger_strategy,
        )
        requested = self._ask_training_trigger(
            config.name,
            default=str(config.data.get("trigger_requested") or config.trigger),
        )
        policy = resolve_trigger_policy(
            requested,
            strategy=trigger_strategy,
            anchors=config.anchor_tags,
        )
        config.data["trigger"] = policy.trigger
        config.data["trigger_requested"] = policy.requested
        config.data["trigger_strategy"] = policy.strategy
        config.data["strategy"] = self._menu(
            self._b("训练策略", "Training strategy"),
            list(STRATEGIES),
            default=config.strategy,
        )
        config.data["images_seen"] = self._ask_positive_int(
            "images_seen", default=config.images_seen
        )
        config.save()
        self.console.print(
            self._b("[green]基础设置已保存。[/green]", "[green]Core settings saved.[/green]")
        )

    def _edit_training_config_lora(self, config: TrainingConfig) -> None:
        action = self._menu(
            self._b("关键训练参数", "Key training parameters"),
            [
                MenuItem(
                    "custom",
                    self._b("自定义关键参数", "Customize key parameters"),
                    self._b("只保存与当前策略预设不同的值。", "Only values that differ from the current strategy preset are stored."),
                ),
                MenuItem("reset", self._b("恢复关键参数预设", "Reset key parameters to preset")),
                MenuItem("back", self._b("返回", "Back")),
            ],
            default="custom",
        )
        if action == "back":
            return
        if action == "reset":
            config.data["overrides"] = reset_key_training_overrides(config.overrides)
        else:
            self._render_training_parameter_help(config.strategy, images_seen=config.images_seen)
            config.data["overrides"] = self._ask_key_training_parameters(
                config.strategy, config.overrides
            )
        config.save()
        self.console.print(
            self._b("[green]关键训练参数已保存。[/green]", "[green]Key training parameters saved.[/green]")
        )

    def _edit_training_config_evaluation(self, config: TrainingConfig) -> None:
        if config.concept_type == "character":
            label = (
                self._b(
                    "评测主体基础 Prompt（自动附加人物锚点）",
                    "Evaluation base subject prompt (character anchors are appended automatically)",
                )
                if config.target_type == "character_outfit"
                else self._b("评测主体 Prompt", "Evaluation subject prompt")
            )
            config.evaluation["subject_prompt"] = self._ask_evaluation_subject(
                config.trigger,
                label=label,
                default=str(config.evaluation.get("subject_prompt", "1girl")),
            )
            config.save()
            self.console.print(
                self._b("[green]评测设置已保存。[/green]", "[green]Evaluation settings saved.[/green]")
            )
        else:
            super()._edit_training_config_evaluation(config)

    def _render_training_configs(self, configs: Sequence[TrainingConfig]) -> None:
        if not configs:
            self.console.print(self._b("[dim]还没有训练配置。[/dim]", "[dim]No training configs yet.[/dim]"))
            return
        table = Table(title=self._b("训练配置", "Training configs"))
        table.add_column(self._b("名称", "Name"), style="bold")
        table.add_column(self._b("训练目标", "Target"))
        table.add_column(self._b("底模", "Base"))
        table.add_column(self._b("Trigger", "Trigger"))
        table.add_column(self._b("训练策略", "Strategy"))
        table.add_column("Rank", justify="right")
        table.add_column("Batch", justify="right")
        table.add_column("images_seen", justify="right")
        table.add_column(self._b("快照", "Snapshot"))
        for config in configs:
            training = effective_training_settings(config.strategy, config.overrides)
            table.add_row(
                config.name,
                config.target_type,
                config.base,
                f"{config.trigger} ({config.trigger_strategy})",
                config.strategy,
                str(training.get("network_dim", 16)),
                str(training.get("batch_size", 1)),
                str(config.images_seen),
                config.snapshot()["snapshot_hash"][:10],
            )
        self.console.print(table)

    def _render_training_config_detail(self, config: TrainingConfig) -> None:
        training = effective_training_settings(config.strategy, config.overrides)
        custom = config.overrides.get("training", {})
        workflow = config.workflow
        physical_batch = int(training.get("batch_size", 1))
        accumulation = int(training.get("gradient_accumulation_steps", 1))
        table = Table(
            title=self._b(f"训练配置 · {config.name}", f"Training config · {config.name}"),
            show_header=False,
        )
        table.add_column(self._b("项目", "Field"), style="bold")
        table.add_column(self._b("值", "Value"))
        table.add_row(self._b("训练目标", "Target"), config.target_type)
        table.add_row(self._b("运行时类型", "Runtime concept"), config.concept_type)
        table.add_row(self._b("底模", "Base"), config.base)
        table.add_row(self._b("Trigger", "Trigger"), config.trigger)
        table.add_row(self._b("Trigger 策略", "Trigger strategy"), config.trigger_strategy)
        if config.anchor_tags:
            table.add_row(self._b("人物锚点", "Character anchors"), ", ".join(config.anchor_tags))
        if config.concept_type == "character":
            table.add_row(
                self._b("实际评测主体", "Effective evaluation subject"),
                str(config.effective_evaluation().get("subject_prompt", "1girl")),
            )
        table.add_row(self._b("训练策略", "Training strategy"), config.strategy)
        table.add_row("images_seen", str(config.images_seen))
        table.add_row("Rank", self._value_source("network_dim", training, custom))
        table.add_row("Alpha", self._value_source("network_alpha", training, custom))
        table.add_row("UNet LR", self._value_source("unet_lr", training, custom))
        table.add_row(self._b("物理 Batch", "Physical batch"), self._value_source("batch_size", training, custom))
        table.add_row(self._b("梯度累积", "Gradient accumulation"), self._value_source("gradient_accumulation_steps", training, custom))
        table.add_row(self._b("有效 Batch", "Effective batch"), str(physical_batch * accumulation))
        table.add_row("Seed", self._value_source("seed", training, custom))
        table.add_row("Caption", str(workflow.get("caption_mode", "auto")))
        table.add_row(self._b("配置快照", "Config snapshot"), config.snapshot()["snapshot_hash"][:16])
        self.console.print(table)

    def _render_training_parameter_help(self, strategy: str, *, images_seen: int) -> None:
        defaults = strategy_training_defaults(strategy)
        table = Table(title=self._b(f"关键训练参数说明 · {strategy}", f"Key training parameter guide · {strategy}"))
        table.add_column(self._b("参数", "Parameter"), style="bold", no_wrap=True)
        table.add_column(self._b("当前预设", "Preset"), no_wrap=True)
        table.add_column(self._b("说明 / 建议", "Description / recommendation"))
        for spec in TRAINING_PARAMETER_SPECS:
            preset = str(images_seen) if spec.key == "images_seen" else str(defaults.get(spec.key, "—"))
            description = (
                f"{spec.description_zh}\n[dim]{spec.recommendation_zh}[/dim]"
                if self.zh
                else f"{spec.description_en}\n[dim]{spec.recommendation_en}[/dim]"
            )
            table.add_row(spec.label_zh if self.zh else spec.label_en, preset, description)
        self.console.print(table)
        self.console.print(
            self._b(
                "[dim]固定 images_seen 时，有效 Batch = 物理 Batch × 梯度累积；有效 Batch 越大，optimizer step 数越少。[/dim]",
                "[dim]With fixed images_seen, effective batch = physical batch × gradient accumulation; larger effective batches mean fewer optimizer steps.[/dim]",
            )
        )

    def _ask_key_training_parameters(self, strategy: str, overrides: dict[str, Any]) -> dict[str, Any]:
        effective = effective_training_settings(strategy, overrides)
        values = {
            "network_dim": self._ask_positive_int("LoRA Rank / network_dim", default=int(effective.get("network_dim", 16))),
            "network_alpha": self._ask_positive_int("LoRA Alpha / network_alpha", default=int(effective.get("network_alpha", 8))),
            "unet_lr": self._ask_positive_float("UNet learning rate", default=float(effective.get("unet_lr", 0.0001))),
            "batch_size": self._ask_positive_int(
                self._b("物理 Batch Size（无人工上限，以实际显存为准）", "Physical batch size (no artificial cap; actual VRAM decides)"),
                default=int(effective.get("batch_size", 1)),
            ),
            "gradient_accumulation_steps": self._ask_positive_int(
                self._b("梯度累积步数", "Gradient accumulation steps"),
                default=int(effective.get("gradient_accumulation_steps", 1)),
            ),
            "seed": self._ask_nonnegative_int(self._b("随机种子 Seed", "Random seed"), default=int(effective.get("seed", 42))),
        }
        return update_key_training_overrides(overrides, strategy=strategy, values=values)

    def _ask_nonnegative_int(self, label: str, *, default: int) -> int:
        while True:
            raw = self._ask_text(label, default=str(default)).strip()
            try:
                value = int(raw)
            except ValueError:
                self.console.print(self._b("[red]请输入整数。[/red]", "[red]Enter an integer.[/red]"))
                continue
            if value < 0:
                self.console.print(self._b("[red]数值不能小于 0。[/red]", "[red]Value must be at least 0.[/red]"))
                continue
            return value

    def _value_source(self, key: str, effective: dict[str, Any], custom: dict[str, Any]) -> str:
        source = self._b("自定义", "custom") if key in custom else self._b("预设", "preset")
        return f"{effective.get(key, '—')} ({source})"

    def _ask_trigger_strategy(self, target_type: str, *, default: str | None = None) -> str:
        choices = [
            MenuItem(
                "rare_token",
                self._b("稀有 Token（推荐）", "Rare token (recommended)"),
                self._b("把输入名称自动转成 zz_xxx，减少与底模已有概念冲突。", "Convert the supplied name to zz_xxx to reduce collisions with base-model concepts."),
            ),
            MenuItem(
                "name",
                self._b("自然名称", "Natural name"),
                self._b("直接使用人物名/风格名作为 Trigger。", "Use the natural character/style name as the trigger."),
            ),
            MenuItem(
                "explicit",
                self._b("自定义 Trigger", "Explicit trigger"),
                self._b("完全按输入值使用。", "Use the entered trigger exactly."),
            ),
        ]
        if target_type == "character_outfit":
            choices.insert(
                0,
                MenuItem(
                    "multi_anchor",
                    self._b("Trigger + 人物锚点（推荐衣装）", "Trigger + character anchors (recommended for outfit)"),
                    self._b("Trigger 学衣装，固定人物锚点保护角色身份。", "Use the trigger for the outfit while fixed character anchors protect identity."),
                ),
            )
        resolved_default = default or ("multi_anchor" if target_type == "character_outfit" else "rare_token")
        if resolved_default not in TRIGGER_STRATEGIES or not any(item.value == resolved_default for item in choices):
            resolved_default = choices[0].value
        return self._menu(self._b("Trigger 策略", "Trigger strategy"), choices, default=resolved_default)

    def _ask_training_trigger(self, name: str, *, default: str | None = None) -> str:
        default_value = default or f"zz_{name}"
        while True:
            value = self._ask_text(
                self._b("Trigger 名称 / token（不要填写逗号列表）", "Trigger name/token (not a comma-separated list)"),
                default=default_value,
            ).strip()
            if not value:
                self.console.print(self._b("[red]Trigger 不能为空。[/red]", "[red]Trigger cannot be empty.[/red]"))
                continue
            if "," in value:
                self.console.print(self._b("[red]Trigger 只能是一个 token/短语。[/red]", "[red]Trigger must be one token/phrase.[/red]"))
                continue
            return value

    def _ask_character_anchors(self, *, default: Sequence[str] = ()) -> list[str]:
        default_text = ", ".join(default)
        while True:
            raw = self._ask_text(
                self._b("人物锚点 Tag（逗号分隔，例如 hataya misuzu）", "Character anchor tags (comma-separated, e.g. hataya misuzu)"),
                default=default_text,
            ).strip()
            anchors = parse_anchor_tags(raw)
            if anchors:
                return anchors
            self.console.print(self._b("[red]人物衣装训练至少需要一个人物锚点。[/red]", "[red]Character outfit training requires at least one character anchor.[/red]"))

    def _ask_evaluation_subject(self, trigger: str, *, label: str, default: str) -> str:
        while True:
            value = self._ask_text(label, default=default).strip()
            if prompt_contains_trigger(value, trigger):
                self.console.print(
                    self._b(
                        "[red]评测主体不能包含 LoRA Trigger；ON/OFF 对照会自动添加 Trigger。[/red]",
                        "[red]Evaluation subject must not contain the LoRA trigger; ON/OFF cases add it automatically.[/red]",
                    )
                )
                continue
            return value
