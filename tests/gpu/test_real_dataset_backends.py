from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.gpu
def test_real_wd_eva02_tagger_backend() -> None:
    if os.environ.get("LORA_RUN_DATASET_BACKEND_SMOKE") != "1":
        pytest.skip("Set LORA_RUN_DATASET_BACKEND_SMOKE=1 in a configured GPU environment")
    from imgutils.tagging import get_wd14_tags

    root = Path(os.environ["LORA_PIPELINE_ROOT"])
    image = root / "smoke" / "dataset" / "smoke_01.png"
    ratings, tags, characters = get_wd14_tags(
        str(image), model_name="EVA02_Large", general_threshold=0.35, character_threshold=0.85
    )
    assert ratings
    assert isinstance(tags, dict)
    assert isinstance(characters, dict)


@pytest.mark.gpu
def test_real_ccip_backend() -> None:
    if os.environ.get("LORA_RUN_DATASET_BACKEND_SMOKE") != "1":
        pytest.skip("Set LORA_RUN_DATASET_BACKEND_SMOKE=1 in a configured GPU environment")
    from imgutils.metrics import ccip_batch_differences

    root = Path(os.environ["LORA_PIPELINE_ROOT"])
    paths = [root / "smoke" / "dataset" / f"smoke_{index:02d}.png" for index in (1, 2)]
    differences = ccip_batch_differences([str(path) for path in paths])
    assert differences.shape == (2, 2)
