from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest
import yaml

from pipeline.config import repository_root, resolve_profiles, write_json_atomic
from pipeline.models import BaseModel, ConfigurationError, TrainingRequest
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


def test_anima_sd_scripts_dry_run_uses_qwen3_backend(tmp_path) -> None:
    root = tmp_path / "repo"
    shutil.copytree(repository_root() / "profiles", root / "profiles")
    environment = root / "environment"
    environment.mkdir(parents=True)
    python = tmp_path / "conda" / "bin" / "python"
    accelerate = python.parent / "accelerate"
    sd_scripts = tmp_path / "sd-scripts"
    python.parent.mkdir(parents=True)
    sd_scripts.mkdir()
    python.write_text("", encoding="utf-8")
    accelerate.write_text("", encoding="utf-8")
    (sd_scripts / "anima_train_network.py").write_text("", encoding="utf-8")
    write_json_atomic(
        environment / "environment-info.json",
        {"python_path": str(python), "sd_scripts_path": str(sd_scripts), "sd_scripts_commit": "abc"},
    )

    project = tmp_path / "project"
    prepared = project / "prepared"
    (prepared / "images").mkdir(parents=True)
    (prepared / "captions").mkdir()
    for name, caption in [("one", "zz_anima, portrait"), ("two", "zz_anima, full body")]:
        (prepared / "images" / f"{name}.png").write_bytes(name.encode())
        (prepared / "captions" / f"{name}.txt").write_text(caption + "\n", encoding="utf-8")
    write_json_atomic(
        prepared / "manifest.json",
        {"images": [
            {"source": "source-a/one.png", "image": "images/one.png", "caption": "captions/one.txt"},
            {"source": "source-b/two.png", "image": "images/two.png", "caption": "captions/two.txt"},
        ]},
    )
    base_path = tmp_path / "anima.safetensors"
    qwen3_path = tmp_path / "qwen3.safetensors"
    vae_path = tmp_path / "vae.safetensors"
    for path in (base_path, qwen3_path, vae_path):
        path.write_bytes(path.name.encode())
    registry_dir = root / "bases"
    registry_dir.mkdir(parents=True)
    (registry_dir / "registry.yaml").write_text(
        yaml.safe_dump({"bases": {"anima": {
            "name": "Anima", "path": str(base_path), "family": "anima",
            "prediction_type": "rectified_flow", "enabled": True,
            "components": {
                "qwen3": {"path": str(qwen3_path)},
                "vae": {"path": str(vae_path)},
            },
        }}}),
        encoding="utf-8",
    )
    run_dir = project / "runs" / "anima-run"
    config = resolve_profiles(
        "v100_16gb", "character", "quality", root=root, base_family="anima"
    )
    cached_config = resolve_profiles(
        "v100_16gb", "character", "cached", root=root, base_family="anima"
    )
    assert config.merged["training"]["cache_text_encoder_outputs"] is False
    assert cached_config.merged["training"]["cache_text_encoder_outputs"] is True
    with pytest.raises(ConfigurationError, match="caption.shuffle"):
        resolve_profiles(
            "v100_16gb", "character", "quality", root=root, base_family="anima",
            cli_overrides={"training": {"cache_text_encoder_outputs_to_disk": True}},
        )
    request = TrainingRequest(
        project_dir=project,
        run_dir=run_dir,
        base=BaseModel(
            id="anima", name="Anima", path=base_path, family="anima",
            prediction_type="rectified_flow", sha256="deadbeef", enabled=True,
            components={"qwen3": qwen3_path, "vae": vae_path},
        ),
        config=config, optimizer_steps=10, target_images_seen=10,
    )

    result = SdScriptsTrainer(root=root, use_gpu_lease=False).train(request, dry_run=True)
    config_text = (run_dir / "config" / "train.toml").read_text(encoding="utf-8")
    assert result.dry_run is True
    assert "qwen3 = " + json.dumps(str(qwen3_path)) in config_text
    assert "vae = " + json.dumps(str(vae_path)) in config_text
    assert "network_module = \"networks.lora_anima\"" in config_text
    assert "qwen3_max_token_length = 512" in config_text
    assert "timestep_sampling = \"sigmoid\"" in config_text
    assert "discrete_flow_shift = 1.0" in config_text
    assert "attn_mode = \"sdpa\"" in config_text
    assert "mixed_precision = \"fp16\"" in config_text
    assert "\nmax_token_length =" not in config_text

    metadata = json.loads((run_dir / "config" / "run-metadata.json").read_text(encoding="utf-8"))
    assert any(part.endswith("anima_train_network.py") for part in metadata["command"])
    assert metadata["base"]["family"] == "anima"
    assert metadata["base"]["components"]["qwen3"]["path"] == str(qwen3_path)
    assert metadata["base"]["components"]["qwen3"]["sha256"]
