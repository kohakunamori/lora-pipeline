from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest
from PIL import Image

from pipeline.config import resolve_profiles, write_json_atomic
from pipeline.model_artifact import (
    build_modelspec_metadata,
    infer_trigger_phrase,
    normalize_metadata_config,
    prepare_thumbnail,
    resolve_model_metadata,
    rewrite_safetensors_metadata,
)
from pipeline.models import BaseModel, PipelineError, TrainingRequest
from pipeline.trainer.sd_scripts import SdScriptsTrainer


def _trainer_request(
    tmp_path: Path, *, overrides: dict | None = None
) -> tuple[Path, TrainingRequest]:
    root = tmp_path / "repo"
    shutil.copytree(Path(__file__).parents[1] / "profiles", root / "profiles")
    (root / "environment").mkdir(parents=True)
    python = tmp_path / "conda" / "bin" / "python"
    accelerate = python.parent / "accelerate"
    sd_scripts = tmp_path / "sd-scripts"
    (python.parent).mkdir(parents=True)
    sd_scripts.mkdir()
    python.write_text("", encoding="utf-8")
    accelerate.write_text("", encoding="utf-8")
    (sd_scripts / "sdxl_train_network.py").write_text("", encoding="utf-8")
    write_json_atomic(
        root / "environment" / "environment-info.json",
        {
            "python_path": str(python),
            "sd_scripts_path": str(sd_scripts),
            "sd_scripts_commit": "test",
        },
    )
    project = tmp_path / "project"
    prepared = project / "prepared"
    (prepared / "images").mkdir(parents=True)
    (prepared / "captions").mkdir()
    for index, caption in enumerate(
        ("kohaku_test, 1girl, solo", "kohaku_test, 1girl, outdoors")
    ):
        (prepared / "images" / f"image-{index}.png").write_bytes(b"image" + bytes([index]))
        (prepared / "captions" / f"image-{index}.txt").write_text(caption, encoding="utf-8")
    write_json_atomic(
        prepared / "manifest.json",
        {
            "images": [
                {
                    "source": f"image-{index}.png",
                    "image": f"images/image-{index}.png",
                    "caption": f"captions/image-{index}.txt",
                }
                for index in range(2)
            ]
        },
    )
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base")
    run_dir = project / "runs" / "run"
    return (
        root,
        TrainingRequest(
            project_dir=project,
            run_dir=run_dir,
            base=BaseModel(
                id="base",
                name="Base",
                path=base,
                family="illustrious_sdxl",
                prediction_type="epsilon",
                sha256="base-sha",
                enabled=True,
            ),
            config=resolve_profiles(
                "v100_16gb",
                "character",
                "quality",
                project_overrides=overrides or {},
                root=root,
            ),
            optimizer_steps=4,
            target_images_seen=4,
        ),
    )


def test_metadata_is_mapped_to_train_toml_and_run_metadata(tmp_path: Path) -> None:
    thumbnail = tmp_path / "thumbnail.png"
    Image.new("RGB", (640, 480), "green").save(thumbnail)
    root, request = _trainer_request(
        tmp_path,
        overrides={
            "metadata": {
                "title": "Test LoRA",
                "author": "tester",
                "tags": ["anime", "style"],
                "trigger_phrase": "test_trigger",
                "thumbnail": {"enabled": True, "path": str(thumbnail)},
            }
        },
    )
    SdScriptsTrainer(root=root, use_gpu_lease=False).train(request, dry_run=True)
    train_text = (request.run_dir / "config" / "train.toml").read_text(encoding="utf-8")
    assert 'metadata_title = "Test LoRA"' in train_text
    assert 'metadata_author = "tester"' in train_text
    assert 'metadata_tags = "anime, style"' in train_text
    assert 'metadata_trigger_phrase = "test_trigger"' in train_text
    assert 'metadata_thumbnail = ' in train_text
    metadata = json.loads(
        (request.run_dir / "config" / "run-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["model_metadata"]["trigger_source"] == "explicit"
    assert metadata["model_metadata"]["usage_hint"] == "Use trigger phrase: test_trigger"
    assert metadata["model_metadata"]["thumbnail_source"] == "explicit"


def test_unconfigured_metadata_adds_no_train_toml_fields(tmp_path: Path) -> None:
    root, request = _trainer_request(tmp_path)
    SdScriptsTrainer(root=root, use_gpu_lease=False).train(request, dry_run=True)
    values = tomllib.loads(
        (request.run_dir / "config" / "train.toml").read_text(encoding="utf-8")
    )
    assert not any(key.startswith("metadata_") for key in values)
    metadata = json.loads(
        (request.run_dir / "config" / "run-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["model_metadata"]["trigger_source"] == "none"
    assert metadata["model_metadata"]["tags"] == []


def test_trigger_inference_is_conservative_and_tracks_coverage() -> None:
    captions = [
        "kohaku_test, 1girl, solo, long hair",
        "kohaku_test, 1girl, outdoors",
        "kohaku_test, 1girl, smile",
    ]
    assert infer_trigger_phrase(captions) == "kohaku_test"
    assert infer_trigger_phrase(["1girl, solo", "1girl, smile", "1girl, outdoors"]) is None


def test_auto_trigger_records_candidates_and_user_usage_hint_wins(tmp_path: Path) -> None:
    metadata = resolve_model_metadata(
        {
            "metadata": {
                "trigger_phrase": "auto",
                "usage_hint": "Custom usage",
            }
        },
        run_dir=tmp_path / "run",
        captions=["kohaku_test, 1girl", "kohaku_test, outdoors"],
        allow_sample=False,
    )
    assert metadata.trigger_phrase == "kohaku_test"
    assert metadata.trigger_source == "auto"
    assert metadata.usage_hint == "Custom usage"
    assert metadata.trigger_confidence is not None
    assert metadata.trigger_candidates[0]["dataset_coverage"] == 1.0


def test_auto_trigger_without_reliable_candidate_records_none(tmp_path: Path) -> None:
    metadata = resolve_model_metadata(
        {"metadata": {"trigger_phrase": "auto"}},
        run_dir=tmp_path / "run",
        captions=["1girl, solo", "1girl, outdoors"],
        allow_sample=False,
    )
    assert metadata.trigger_phrase is None
    assert metadata.trigger_source == "none"
    assert metadata.usage_hint is None


def test_thumbnail_normalization_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (1024, 768), (255, 0, 0, 128)).save(source)
    original = source.read_bytes()
    destination = prepare_thumbnail(source, tmp_path / "derived", max_size=256)
    assert destination.is_file()
    assert max(Image.open(destination).size) <= 256
    assert source.read_bytes() == original


def test_explicit_thumbnail_must_exist_and_use_supported_format(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="does not exist"):
        resolve_model_metadata(
            {
                "metadata": {
                    "thumbnail": {"enabled": True, "path": str(tmp_path / "missing.png")}
                }
            },
            run_dir=tmp_path / "run",
            allow_sample=False,
        )
    unsupported = tmp_path / "preview.bmp"
    Image.new("RGB", (32, 32), "blue").save(unsupported)
    with pytest.raises(PipelineError, match="common image format"):
        prepare_thumbnail(unsupported, tmp_path / "derived")


def test_metadata_normalization_and_modelspec_mapping(tmp_path: Path) -> None:
    normalized = normalize_metadata_config({"tags": ["anime", "style"]})
    assert normalized["tags"] == ["anime", "style"]
    source = tmp_path / "preview.png"
    Image.new("RGB", (32, 32), "blue").save(source)
    metadata = resolve_model_metadata(
        {
            "metadata": {"trigger_phrase": "x", "thumbnail": {"enabled": True, "path": str(source)}}
        },
        run_dir=tmp_path / "run",
        captions=[],
        allow_sample=False,
    )
    spec = build_modelspec_metadata(metadata)
    assert spec["modelspec.trigger_phrase"] == "x"
    assert spec["modelspec.usage_hint"] == "Use trigger phrase: x"
    assert spec["modelspec.thumbnail"].startswith("data:image/jpeg;base64,")


def test_safetensors_metadata_rewrite_preserves_tensors(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open
    from safetensors.torch import save_file

    path = tmp_path / "model.safetensors"
    original = {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    save_file(original, str(path), metadata={"ss_network_dim": "16", "existing": "keep"})
    rewrite_safetensors_metadata(path, {"modelspec.trigger_phrase": "test"})
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor("weight"), original["weight"])
        assert handle.metadata() == {
            "ss_network_dim": "16",
            "existing": "keep",
            "modelspec.trigger_phrase": "test",
        }


def test_safetensors_rewrite_failure_keeps_original_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open
    from safetensors.torch import save_file
    import safetensors.torch

    path = tmp_path / "model.safetensors"
    original_tensor = torch.tensor([1, 2, 3])
    save_file(
        {"weight": original_tensor},
        str(path),
        metadata={"ss_network_dim": "16"},
    )
    original_bytes = path.read_bytes()

    def fail_save(*args, **kwargs) -> None:
        del args, kwargs
        raise OSError("injected write failure")

    monkeypatch.setattr(safetensors.torch, "save_file", fail_save)
    with pytest.raises(PipelineError, match="atomically rewrite"):
        rewrite_safetensors_metadata(path, {"modelspec.trigger_phrase": "test"})

    assert path.read_bytes() == original_bytes
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor("weight"), original_tensor)
        assert handle.metadata() == {"ss_network_dim": "16"}
