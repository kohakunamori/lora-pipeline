from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from pipeline.service import load_project, run_single_step


@pytest.mark.gpu
def test_real_evaluation_produces_final_artifacts() -> None:
    if os.environ.get("LORA_RUN_GPU_EVALUATION_ACCEPTANCE") != "1":
        pytest.skip("Set LORA_RUN_GPU_EVALUATION_ACCEPTANCE=1 and LORA_ACCEPTANCE_PROJECT")
    project_name = os.environ["LORA_ACCEPTANCE_PROJECT"]
    state = load_project(project_name)
    # Keep the production profile matrix intact. This project-local override makes
    # deployment acceptance short while exercising the real backend and artifact path.
    state.payload["project"].setdefault("overrides", {}).update(
        {
            "checkpoints": {"target_candidates": 1},
            "evaluation": {"prompts": ["portrait"], "strengths": [0.8]},
        }
    )
    state.payload["project"]["acceptance_override"] = "one prompt × one strength × latest checkpoint"
    state.save()
    result = run_single_step(state, "evaluate", force=True, verbose=1)
    details = result.details
    for key in ("contact_sheet", "report", "best", "best_yaml"):
        assert Path(details[key]).is_file()
    best = yaml.safe_load(Path(details["best_yaml"]).read_text(encoding="utf-8"))
    assert best["base"]["sha256"]
    assert best["recommended"]["status"] == "provisional_pending_manual_contact_sheet_review"
