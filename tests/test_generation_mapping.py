from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pipeline.evaluation import generation as generation_module
from pipeline.evaluation.generation import SdScriptsGenerator
from pipeline.models import GenerationCase


def _case(tmp_path: Path, index: int) -> GenerationCase:
    checkpoint = tmp_path / "candidate.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    return GenerationCase(
        case_id=f"case-{index}",
        checkpoint=checkpoint,
        checkpoint_label="candidate",
        strength=0.6 + index * 0.1,
        prompt_id=f"prompt-{index}",
        prompt=f"zz_test, prompt {index}",
        negative_prompt="bad",
        seed=42 + index,
        contains_trigger=True,
    )


def test_sd_scripts_generator_maps_sequences_to_explicit_case_ids(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    environment = root / "environment"
    sd_scripts = root / "sd-scripts"
    environment.mkdir(parents=True)
    sd_scripts.mkdir()
    python = root / "python"
    python.write_text("", encoding="utf-8")
    (sd_scripts / "sdxl_gen_img.py").write_text("", encoding="utf-8")
    (environment / "environment-info.json").write_text(
        json.dumps(
            {
                "python_path": str(python),
                "sd_scripts_path": str(sd_scripts),
                "sd_scripts_commit": "fake",
            }
        ),
        encoding="utf-8",
    )

    def fake_run(command, *, cwd, env, log_path, verbose=0):
        del cwd, env, verbose
        output_dir = Path(command[command.index("--outdir") + 1])
        prompt_file = Path(command[command.index("--from_file") + 1])
        count = len(prompt_file.read_text(encoding="utf-8").splitlines())
        # Create in reverse order to prove directory ordering is irrelevant.
        for index in reversed(range(1, count + 1)):
            Image.new("RGB", (32, 32), (index * 20, 40, 80)).save(
                output_dir / f"im_{index:06d}.png"
            )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake generation\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(generation_module, "run_command_tee", fake_run)
    cases = [_case(tmp_path, index) for index in range(3)]
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base")
    output = root / "run" / "samples" / "screening"
    generated = SdScriptsGenerator(root=root, use_gpu_lease=False).generate(
        cases,
        base_path=base,
        output_dir=output,
        settings={"width": 32, "height": 32, "steps": 1, "sampler": "euler_a", "cfg": 4.5},
    )

    assert [item.case.case_id for item in generated] == ["case-0", "case-1", "case-2"]
    assert [item.path.name for item in generated] == ["case-0.png", "case-1.png", "case-2.png"]
    manifest = json.loads((output / "generation-manifest.json").read_text(encoding="utf-8"))
    assert [item["case_id"] for item in manifest["images"]] == ["case-0", "case-1", "case-2"]
    assert not list((output / "work").rglob("im_*.png"))
