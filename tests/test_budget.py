from __future__ import annotations

from pipeline.budget import resolve_budget
from pipeline.config import resolve_profiles


def test_quality_and_fast_are_comparable_at_equal_images_seen() -> None:
    project = {"budget": {"unit": "images_seen", "value": 1000}}
    quality = resolve_budget(
        project,
        resolve_profiles("v100_16gb", "character", "quality"),
        image_count=50,
    )
    fast = resolve_budget(
        project,
        resolve_profiles("v100_16gb", "character", "fast"),
        image_count=50,
    )
    assert quality.target_images_seen == fast.target_images_seen == 1000
    assert quality.physical_batch == 1
    assert fast.physical_batch == 2
    assert quality.optimizer_steps == 1000
    assert fast.optimizer_steps == 500
    assert quality.actual_images_seen == fast.actual_images_seen == 1000
    assert quality.equivalent_epochs == fast.equivalent_epochs == 20


def test_image_budget_rounds_up_by_at_most_one_effective_batch() -> None:
    budget = resolve_budget(
        {"budget": {"unit": "images_seen", "value": 1001}},
        resolve_profiles("v100_16gb", "style", "fast"),
        image_count=37,
    )
    assert budget.actual_images_seen >= budget.target_images_seen
    assert budget.actual_images_seen - budget.target_images_seen < budget.effective_batch


def test_epoch_budget_resolves_after_prepared_image_count_is_known() -> None:
    budget = resolve_budget(
        {"budget": {"unit": "epochs", "value": 7}},
        resolve_profiles("v100_16gb", "character", "quality"),
        image_count=31,
    )
    assert budget.target_images_seen == 217
    assert budget.optimizer_steps == 217
    assert budget.equivalent_epochs == 7
