from pathlib import Path

import yaml

from pipeline.training_config import TrainingConfig


def _base_registry(root: Path) -> None:
    (root / "bases").mkdir(parents=True, exist_ok=True)
    checkpoint = root / "base.safetensors"
    checkpoint.write_bytes(b"base")
    (root / "bases" / "registry.yaml").write_text(
        yaml.safe_dump(
            {
                "bases": {
                    "base": {
                        "name": "Base",
                        "path": str(checkpoint),
                        "family": "illustrious_sdxl",
                        "prediction_type": "epsilon",
                        "sha256": None,
                        "enabled": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_rare_token_keeps_one_protected_caption_token(tmp_path: Path) -> None:
    _base_registry(tmp_path)
    config = TrainingConfig.create(
        "character",
        concept_type="character",
        base="base",
        trigger="Hataya Misuzu",
        trigger_strategy="rare_token",
        root=tmp_path,
    )

    assert config.trigger == "zz_hataya_misuzu"
    assert config.runtime_overrides()["caption"]["keep_tokens"] == 1


def test_multi_anchor_protects_entire_prefix(tmp_path: Path) -> None:
    _base_registry(tmp_path)
    config = TrainingConfig.create(
        "outfit",
        concept_type="character",
        target_type="character_outfit",
        base="base",
        trigger="misuzu_swimsuit",
        trigger_strategy="multi_anchor",
        anchor_tags=["hataya misuzu", "1girl"],
        root=tmp_path,
    )

    assert config.trigger_policy["protected_prefix"] == [
        "misuzu_swimsuit",
        "hataya misuzu",
        "1girl",
    ]
    runtime = config.runtime_overrides()
    assert runtime["caption"]["keep_tokens"] == 3
    assert runtime["caption"]["anchor_tags"] == ["hataya misuzu", "1girl"]
