from __future__ import annotations

import json
import tomllib
from pathlib import Path

from pipeline.config import resolve_profiles, write_json_atomic
from pipeline.models import BaseModel, TrainingRequest
from pipeline.trainer.sd_scripts import SdScriptsTrainer


def test_sd_scripts_config_defaults_do_not_enable_vpred_or_clip_skip(tmp_path) -> None:
    root = tmp_path / "repo"
    environment = root / "environment"
    environment.mkdir(parents=True)
    python = tmp_path / "conda" / "bin" / "python"
    accelerate = python.parent / "accelerate"
    sd_scripts = tmp_path / "sd-scripts"
    entrypoint = sd_scripts / "sdxl_train_network.py"
    python.parent.mkdir(parents=True)
    sd_scripts.mkdir()
    python.write_text("", encoding="utf-8")
    accelerate.write_text("", encoding="utf-8")
    entrypoint.write_text("", encoding="utf-8")
    write_json_atomic(
        environment / "environment-info.json",
        {
            "python_path": str(python),
            "sd_scripts_path": str(sd_scripts),
            "sd_scripts_commit": "abc",
        },
    )
    project = tmp_path / "project"
    prepared = project / "prepared"
    (prepared / "images").mkdir(parents=True)
    (prepared / "captions").mkdir()
    (prepared / "images" / "one.png").write_bytes(b"image")
    (prepared / "captions" / "one.txt").write_text(
        "zz_test, 1girl\n", encoding="utf-8"
    )
    (prepared / "images" / "two.png").write_bytes(b"image-two")
    (prepared / "captions" / "two.txt").write_text(
        "zz_test, portrait\n", encoding="utf-8"
    )
    write_json_atomic(
        prepared / "manifest.json",
        {
            "images": [
                {
                    "source": "source-a/one.png",
                    "image": "images/one.png",
                    "caption": "captions/one.txt",
                },
                {
                    "source": "source-b/two.png",
                    "image": "images/two.png",
                    "caption": "captions/two.txt",
                },
            ]
        },
    )
    run_dir = project / "runs" / "run"
    base_path = tmp_path / "base.safetensors"
    base_path.write_bytes(b"base")
    request = TrainingRequest(
        project_dir=project,
        run_dir=run_dir,
        base=BaseModel(
            id="base",
            name="Base",
            path=base_path,
            family="illustrious_sdxl",
            prediction_type="epsilon",
            sha256="deadbeef",
            enabled=True,
        ),
        config=resolve_profiles("v100_16gb", "character", "quality"),
        optimizer_steps=10,
        target_images_seen=10,
    )
    result = SdScriptsTrainer(root=root, use_gpu_lease=False).train(request, dry_run=True)
    text = (run_dir / "config" / "train.toml").read_text(encoding="utf-8")
    assert "v_parameterization" not in text
    assert "v_pred_like_loss" not in text
    assert "clip_skip" not in text
    assert "mixed_precision = \"fp16\"" in text
    assert "sdpa = true" in text
    assert "no_half_vae = true" in text
    assert "save_state = true" in text
    assert "max_data_loader_n_workers = 0" in text
    assert result.accounting["target_images_seen"] == 10
    assert result.accounting["images_seen"] == 10

    metadata = json.loads(
        (run_dir / "config" / "run-metadata.json").read_text(encoding="utf-8")
    )
    dataset_dir = Path(
        metadata["storage"]["dataset_dir"]
    )
    assert dataset_dir.is_dir()
    assert not any(path.is_symlink() for path in dataset_dir.rglob("*"))
    dataset_config = tomllib.loads(
        (run_dir / "config" / "dataset.toml").read_text(encoding="utf-8")
    )
    subset_dirs = [
        Path(subset["image_dir"])
        for subset in dataset_config["datasets"][0]["subsets"]
    ]
    assert subset_dirs == [dataset_dir / "source-a", dataset_dir / "source-b"]
    snapshot = json.loads(
        (run_dir / "config" / "dataset-snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot["schema_version"] == 3
    assert snapshot["subset_directories"] == ["source-a", "source-b"]
    assert snapshot["dataset_snapshot_hash"]
    assert snapshot["captions_hash"]
