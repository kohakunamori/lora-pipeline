from __future__ import annotations

import os
from pathlib import Path

import pytest

from pipeline.config import load_base_registry
from pipeline.evaluation.contact_sheet import create_contact_sheet
from pipeline.evaluation.generation import SdScriptsGenerator
from pipeline.models import GenerationCase


@pytest.mark.gpu
def test_real_sd_scripts_generation_with_lora(tmp_path) -> None:
    if os.environ.get("LORA_RUN_GPU_GENERATION_SMOKE") != "1":
        pytest.skip(
            "Set LORA_RUN_GPU_GENERATION_SMOKE=1 with LORA_GENERATION_CHECKPOINT and LORA_BASE_ID"
        )
    checkpoint = Path(os.environ["LORA_GENERATION_CHECKPOINT"])
    assert checkpoint.is_file()
    base = load_base_registry()[os.environ["LORA_BASE_ID"]]
    common = {
        "checkpoint": checkpoint,
        "checkpoint_label": checkpoint.stem,
        "strength": 0.8,
        "prompt_id": "portrait",
        "negative_prompt": "low quality, worst quality, text, watermark",
        "seed": 42,
    }
    cases = [
        GenerationCase(prompt="example_trigger, 1girl, portrait", contains_trigger=True, **common),
        GenerationCase(prompt="1girl, portrait", contains_trigger=False, **common),
    ]
    generated = SdScriptsGenerator().generate(
        cases,
        base_path=base.path,
        output_dir=tmp_path / "samples",
        settings={"width": 512, "height": 512, "steps": 4, "sampler": "euler_a", "cfg": 4.5},
        verbose=1,
    )
    assert len(generated) == 2
    assert all(item.path.is_file() for item in generated)
    assert create_contact_sheet(generated, tmp_path / "contact-sheet.jpg").is_file()
