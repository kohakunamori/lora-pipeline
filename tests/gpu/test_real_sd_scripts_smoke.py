from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.gpu
def test_real_v100_sd_scripts_smoke() -> None:
    if os.environ.get("LORA_RUN_GPU_SMOKE") != "1":
        pytest.skip("Set LORA_RUN_GPU_SMOKE=1 for the explicit V100 smoke test")
    root = Path(os.environ.get("LORA_PIPELINE_ROOT", Path(__file__).resolve().parents[2]))
    script = root / "environment" / "smoke" / "run_sd_scripts_smoke.sh"
    result = subprocess.run([str(script), "1"], cwd=root, check=False)
    assert result.returncode == 0
