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
from .wizard import MenuItem, STRATEGIES


class InteractiveWizard(BaseInteractiveWizard):
    """Final UX layer for explicit character-outfit training targets."""

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
                        self._b(
                            "学习人物身份，并验证跨服装泛化。",
                            "Learn character identity and test cross-outfit generalization.",
                        ),
                    ),
                    MenuItem(
                        "character_outfit",
                        self._b("人物衣装", "Character outfit"),
                        self._b(
                            "学习某个人物的特定衣装；Trigger 与人物锚点分离。",
                            "Learn a specific outfit for a character with a separate trigger and identity anchor.",
                        ),
                    ),
                    MenuItem("style", self._b("风格", "Style")),
                ],
                default="character",
            )
            concept = "style" if target_type == "style" else "character"
            base = self._select_base(registry, title=self._b("底模", "Base checkpoint"))
            trigger = self._ask_training_trigger(name)
            anchor_tags: list[str] = []
            if target_type == "character_outfit":
                anchor_tags = self._ask_character_anchors()
            strategy = self._menu(
                self._b("训练策略", "Training strategy"),
                list(STRATEGIES),
                default="quality",
            )
            images_seen = self._ask_positive_int(
                self._b(
                    "图片曝光预算 images_seen",
                    "Image exposure budget (images_seen)",
                ),
                default=1000,
            )
            overrides: dict[str, Any] = {}
            if not self._confirm(
                self._b(
                    "LoRA rank / alpha / 学习率使用策略默认值吗？",
                    "Use strategy defaults for LoRA rank / alpha / learning rate?",
                ),
                default=True,
            ):
                overrides["training"] = {
                    "network_dim": self._ask_positive_int("LoRA rank / network_dim", default=16),
                    "network_alpha": self._ask_positive_int("LoRA alpha / network_alpha", default=8),
                    "unet_lr": self._ask_positive_float("UNet learning rate", default=0.0001),
                }
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
                evaluation["subject_prompt"] = self._ask_evaluation_subject(
                    trigger, label=label, default="1girl"
                )
            try:
                config = TrainingConfig.create(
                    name,
                    concept_type=concept,
                    target_type=target_type,
                    base=base,
                    trigger=trigger,
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
                        f"目标：{config.target_type}\n快照：{config.snapshot()['snapshot_hash'][:16]}",
                        f"[green bold]Training config created[/green bold]\n{name}\n"
                        f"Target: {config.target_type}\nSnapshot: {config.snapshot()['snapshot_hash'][:16]}",
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
                config.data["anchor_tags"] = self._ask_character_anchors(
                    default=config.anchor_tags
                )
            else:
                config.data["anchor_tags"] = []
        config.data["trigger"] = self._ask_training_trigger(
            config.name, default=config.trigger
        )
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
                self._b(
                    "[green]评测设置已保存。[/green]",
                    "[green]Evaluation settings saved.[/green]",
                )
            )
        else:
            super()._edit_training_config_evaluation(config)

    def _render_training_configs(self, configs: Sequence[TrainingConfig]) -> None:
        if not configs:
            self.console.print(
                self._b("[dim]还没有训练配置。[/dim]", "[dim]No training configs yet.[/dim]")
            )
            return
        table = Table(title=self._b("训练配置", "Training configs"))
        table.add_column(self._b("名称", "Name"), style="bold")
        table.add_column(self._b("训练目标", "Target"))
        table.add_column(self._b("底模", "Base"))
        table.add_column(self._b("策略", "Strategy"))
        table.add_column("Rank", justify="right")
        table.add_column("images_seen", justify="right")
        table.add_column(self._b("快照", "Snapshot"))
        for config in configs:
            training = config.overrides.get("training", {})
            table.add_row(
                config.name,
                config.target_type,
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
        table = Table(
            title=self._b(
                f"训练配置 · {config.name}", f"Training config · {config.name}"
            ),
            show_header=False,
        )
        table.add_column(self._b("项目", "Field"), style="bold")
        table.add_column(self._b("值", "Value"))
        table.add_row(self._b("训练目标", "Target"), config.target_type)
        table.add_row(self._b("运行时类型", "Runtime concept"), config.concept_type)
        table.add_row(self._b("底模", "Base"), config.base)
        table.add_row(self._b("唯一 Trigger", "Trigger token"), config.trigger)
        if config.anchor_tags:
            table.add_row(
                self._b("人物锚点", "Character anchors"), ", ".join(config.anchor_tags)
            )
        if config.concept_type == "character":
            table.add_row(
                self._b("实际评测主体", "Effective evaluation subject"),
                str(config.effective_evaluation().get("subject_prompt", "1girl")),
            )
        table.add_row(self._b("策略", "Strategy"), config.strategy)
        table.add_row("images_seen", str(config.images_seen))
        table.add_row(
            "Rank", str(training.get("network_dim", self._b("策略默认", "strategy default")))
        )
        table.add_row(
            "Alpha", str(training.get("network_alpha", self._b("策略默认", "strategy default")))
        )
        table.add_row(
            "UNet LR", str(training.get("unet_lr", self._b("策略默认", "strategy default")))
        )
        table.add_row("Caption", str(workflow.get("caption_mode", "auto")))
        table.add_row(
            self._b("配置快照", "Config snapshot"), config.snapshot()["snapshot_hash"][:16]
        )
        self.console.print(table)

    def _ask_training_trigger(self, name: str, *, default: str | None = None) -> str:
        default_value = default or f"zz_{name}"
        while True:
            value = self._ask_text(
                self._b(
                    "唯一 Trigger token（不要填写逗号分隔的 Tag 列表）",
                    "Single trigger token (not a comma-separated tag list)",
                ),
                default=default_value,
            ).strip()
            if not value:
                self.console.print(self._b("[red]Trigger 不能为空。[/red]", "[red]Trigger cannot be empty.[/red]"))
                continue
            if "," in value:
                self.console.print(
                    self._b(
                        "[red]Trigger 只能是一个 token/短语；人物名请放到衣装模式的人物锚点中。[/red]",
                        "[red]Trigger must be one token/phrase; put character identity in outfit anchor tags.[/red]",
                    )
                )
                continue
            return value

    def _ask_character_anchors(self, *, default: Sequence[str] = ()) -> list[str]:
        default_text = ", ".join(default)
        while True:
            raw = self._ask_text(
                self._b(
                    "人物锚点 Tag（逗号分隔，例如 hataya misuzu）",
                    "Character anchor tags (comma-separated, e.g. hataya misuzu)",
                ),
                default=default_text,
            ).strip()
            anchors = parse_anchor_tags(raw)
            if anchors:
                return anchors
            self.console.print(
                self._b(
                    "[red]人物衣装训练至少需要一个人物锚点。[/red]",
                    "[red]Character outfit training requires at least one character anchor.[/red]",
                )
            )

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
