from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import deep_merge, load_base_registry, resolve_profiles
from pipeline.models import ConfigurationError


def test_deep_merge_preserves_nested_keys_and_last_layer_wins() -> None:
    result = deep_merge(
        {"training": {"batch_size": 1, "seed": 42}, "caption": {"shuffle": True}},
        {"training": {"batch_size": 2}},
    )
    assert result == {
        "training": {"batch_size": 2, "seed": 42},
        "caption": {"shuffle": True},
    }


def test_validated_profiles_resolve_independent_dimensions() -> None:
    config = resolve_profiles("v100_16gb", "style", "cached")
    assert config.hardware["id"] == "v100_16gb"
    assert config.concept["id"] == "style"
    assert config.training["id"] == "cached"
    assert config.merged["training"]["cache_text_encoder_outputs"] is True
    assert config.merged["caption"]["shuffle"] is False


def test_profile_validation_rejects_unvalidated_physical_batch() -> None:
    with pytest.raises(ConfigurationError, match="Physical batch"):
        resolve_profiles(
            "v100_16gb",
            "character",
            "quality",
            project_overrides={"training": {"batch_size": 3}},
        )


def test_cached_profile_rejects_caption_randomization() -> None:
    with pytest.raises(ConfigurationError, match="incompatible"):
        resolve_profiles(
            "v100_16gb",
            "style",
            "cached",
            project_overrides={"caption": {"shuffle": True}},
        )


def test_base_registry_loads_generic_entry(tmp_path: Path) -> None:
    registry = tmp_path / "bases" / "registry.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        """bases:
  example_base:
    name: Example Base
    path: /path/to/example.safetensors
    family: illustrious_sdxl
    prediction_type: epsilon
    sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
""",
        encoding="utf-8",
    )
    base = load_base_registry(tmp_path)["example_base"]
    assert base.family == "illustrious_sdxl"
    assert base.prediction_type == "epsilon"
    assert len(base.sha256 or "") == 64


def test_missing_base_registry_is_empty(tmp_path: Path) -> None:
    assert load_base_registry(tmp_path) == {}
