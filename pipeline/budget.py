from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .models import PipelineError, ResolvedConfig


@dataclass(frozen=True)
class ResolvedBudget:
    unit: str
    requested_value: int
    target_images_seen: int
    optimizer_steps: int
    physical_batch: int
    gradient_accumulation: int
    effective_batch: int
    actual_images_seen: int
    equivalent_epochs: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "requested_value": self.requested_value,
            "target_images_seen": self.target_images_seen,
            "optimizer_steps": self.optimizer_steps,
            "physical_batch": self.physical_batch,
            "gradient_accumulation": self.gradient_accumulation,
            "effective_batch": self.effective_batch,
            "actual_images_seen": self.actual_images_seen,
            "equivalent_epochs": self.equivalent_epochs,
        }


def resolve_budget(
    project: Mapping[str, Any],
    config: ResolvedConfig,
    *,
    image_count: int,
    images_seen_override: int | None = None,
    optimizer_steps_override: int | None = None,
) -> ResolvedBudget:
    if image_count < 1:
        raise PipelineError("Training budget cannot be resolved for an empty prepared dataset")
    if images_seen_override is not None and optimizer_steps_override is not None:
        raise PipelineError("Use either images_seen or optimizer_steps override, not both")

    training = config.merged.get("training", {})
    physical_batch = int(training.get("batch_size", 1))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if physical_batch < 1 or accumulation < 1:
        raise PipelineError("Physical batch and gradient accumulation must both be at least 1")
    effective_batch = physical_batch * accumulation

    budget = project.get("budget", {})
    unit = str(budget.get("unit", "images_seen"))
    requested = int(budget.get("value", budget.get("images_seen", 1000)))

    if images_seen_override is not None:
        unit = "images_seen"
        requested = int(images_seen_override)
    elif optimizer_steps_override is not None:
        unit = "optimizer_steps_override"
        requested = int(optimizer_steps_override)

    if requested < 1:
        raise PipelineError("Training budget must be at least 1")

    if unit in {"images_seen", "image_exposures"}:
        target_images_seen = requested
        optimizer_steps = max(1, math.ceil(target_images_seen / effective_batch))
    elif unit == "epochs":
        target_images_seen = requested * image_count
        optimizer_steps = max(1, math.ceil(target_images_seen / effective_batch))
    elif unit in {"optimizer_steps", "legacy_optimizer_steps", "optimizer_steps_override"}:
        optimizer_steps = requested
        target_images_seen = requested * effective_batch
    else:
        raise PipelineError(f"Unsupported training budget unit: {unit}")

    actual_images_seen = optimizer_steps * effective_batch
    return ResolvedBudget(
        unit=unit,
        requested_value=requested,
        target_images_seen=target_images_seen,
        optimizer_steps=optimizer_steps,
        physical_batch=physical_batch,
        gradient_accumulation=accumulation,
        effective_batch=effective_batch,
        actual_images_seen=actual_images_seen,
        equivalent_epochs=round(actual_images_seen / image_count, 6),
    )
