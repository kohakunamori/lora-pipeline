from __future__ import annotations

from rich.table import Table

from .interactive_outfit import InteractiveWizard as BaseInteractiveWizard
from .training_parameters import TRAINING_PARAMETER_SPECS, strategy_training_defaults


class InteractiveWizard(BaseInteractiveWizard):
    """Final CLI layer for the documented key training-parameter guide."""

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
