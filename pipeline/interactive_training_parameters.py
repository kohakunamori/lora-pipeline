from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from .interactive_outfit import InteractiveWizard as BaseInteractiveWizard
from .target_training_advisor import target_training_advice
from .target_training_apply import apply_target_preferred_start
from .training_parameters import (
    TRAINING_PARAMETER_SPECS,
    effective_training_settings,
    reset_key_training_overrides,
    strategy_training_defaults,
)
from .wizard import MenuItem


class InteractiveWizard(BaseInteractiveWizard):
    """Final CLI layer for documented and target-aware training-parameter guidance."""

    def _render_training_parameter_help(self, strategy: str, *, images_seen: int) -> None:
        defaults = strategy_training_defaults(strategy)
        table = Table(
            title=self._b(
                f"关键训练参数说明 · {strategy}",
                f"Key training parameter guide · {strategy}",
            )
        )
        table.add_column(self._b("参数", "Parameter"), style="bold", no_wrap=True)
        table.add_column(self._b("当前预设", "Preset"), no_wrap=True)
        table.add_column(self._b("说明 / 建议", "Description / recommendation"))
        for spec in TRAINING_PARAMETER_SPECS:
            preset = str(images_seen) if spec.key == "images_seen" else str(defaults.get(spec.key, "—"))
            description = self._b(
                f"{spec.description_zh}\n[dim]{spec.recommendation_zh}[/dim]",
                f"{spec.description_en}\n[dim]{spec.recommendation_en}[/dim]",
            )
            table.add_row(
                self._b(spec.label_zh, spec.label_en),
                preset,
                description,
            )
        self.console.print(table)
        self.console.print(
            self._b(
                "[dim]注意：固定 images_seen 时，有效 Batch = 物理 Batch × 梯度累积；有效 Batch 越大，optimizer step 数越少，因此并非完全等价的纯加速开关。[/dim]",
                "[dim]With fixed images_seen, effective batch = physical batch × gradient accumulation. A larger effective batch means fewer optimizer steps, so this is not a purely equivalent speed switch.[/dim]",
            )
        )

    def _edit_training_config_lora(self, config) -> None:
        action = self._menu(
            self._b("关键训练参数", "Key training parameters"),
            [
                MenuItem(
                    "advisor",
                    self._b("按训练目标生成推荐", "Generate target-aware recommendation"),
                    self._b(
                        "输入预计训练图片数，比较当前配置与人物 / 衣装 / 风格专用建议；确认后才应用。",
                        "Enter the expected training-image count, compare the current recipe with target-specific guidance, and apply only after confirmation.",
                    ),
                ),
                MenuItem("custom", self._b("自定义关键参数", "Customize key parameters")),
                MenuItem("reset", self._b("恢复关键参数预设", "Reset key parameters to preset")),
                MenuItem("back", self._b("返回", "Back")),
            ],
            default="advisor",
        )
        if action == "back":
            return
        if action == "reset":
            config.data["overrides"] = reset_key_training_overrides(config.overrides)
            config.save()
            self.console.print(
                self._b(
                    "[green]关键训练参数已恢复策略预设。[/green]",
                    "[green]Key training parameters reset to strategy defaults.[/green]",
                )
            )
            return
        if action == "custom":
            self._render_training_parameter_help(config.strategy, images_seen=config.images_seen)
            config.data["overrides"] = self._ask_key_training_parameters(
                config.strategy, config.overrides
            )
            config.save()
            self.console.print(
                self._b(
                    "[green]关键训练参数已保存。[/green]",
                    "[green]Key training parameters saved.[/green]",
                )
            )
            return

        image_count = self._ask_positive_int(
            self._b(
                "预计实际参与训练的图片数",
                "Expected number of images that will actually enter training",
            ),
            default=40,
        )
        current = effective_training_settings(config.strategy, config.overrides)
        advice = target_training_advice(
            config.target_type,
            image_count=image_count,
            current_training=current,
            current_images_seen=config.images_seen,
        )
        self._render_target_training_advice(config, advice, current)
        if not self._confirm(
            self._b(
                "采用首选起点吗？只修改 images_seen / Rank / Alpha / LR；Batch、梯度累积和 Seed 保持不变。",
                "Apply the preferred starting point? Only images_seen / Rank / Alpha / LR change; batch, accumulation, and seed stay unchanged.",
            ),
            default=False,
        ):
            self.console.print(
                self._b(
                    "[dim]仅查看建议，没有修改配置。[/dim]",
                    "[dim]Recommendation viewed; config was not changed.[/dim]",
                )
            )
            return

        images_seen, overrides = apply_target_preferred_start(
            strategy=config.strategy,
            overrides=config.overrides,
            current_training=current,
            advice=advice,
        )
        config.data["images_seen"] = images_seen
        config.data["overrides"] = overrides
        config.save()
        self.console.print(
            self._b(
                "[green]已应用训练目标首选起点；Batch / 梯度累积 / Seed 未改变。[/green]",
                "[green]Applied the target-aware preferred start; batch / accumulation / seed were preserved.[/green]",
            )
        )

    def _render_target_training_advice(self, config, advice: dict, current: dict) -> None:
        recommended = advice["recommended"]
        preferred = advice["preferred_start"]
        table = Table(
            title=self._b(
                f"训练目标建议 · {config.target_type} · {advice['image_count']} 张图",
                f"Target-aware advice · {config.target_type} · {advice['image_count']} images",
            )
        )
        table.add_column(self._b("参数", "Parameter"), style="bold")
        table.add_column(self._b("当前", "Current"))
        table.add_column(self._b("建议区间", "Advisory range"))
        table.add_column(self._b("首选起点", "Preferred start"))
        rows = (
            ("images_seen", config.images_seen),
            ("network_dim", current.get("network_dim")),
            ("network_alpha", current.get("network_alpha")),
            ("unet_lr", current.get("unet_lr")),
        )
        for key, current_value in rows:
            bounds = recommended[key]
            table.add_row(
                key,
                str(current_value),
                f"{bounds['minimum']} – {bounds['maximum']}",
                str(preferred[key]),
            )
        self.console.print(table)
        self.console.print(
            Panel.fit(
                self._b(
                    "这是保守起点建议，不是质量真值；真正的数据偏置、过拟合与是否继续训练仍以 Preflight + 固定评测矩阵为准。",
                    "These are conservative starting points, not quality ground truth. Actual data bias, overfitting, and whether to extend training are still decided by Preflight plus fixed evaluation matrices.",
                ),
                title=self._b("说明", "Note"),
            )
        )
        for warning in advice.get("warnings", []):
            self.console.print(f"[yellow]• {warning}[/yellow]")
