from __future__ import annotations

import pytest
import yaml

from pipeline.bases import add_base, resolve_base_component_fingerprints, resolve_base_sha256
from pipeline.config import read_yaml, sha256_file
from pipeline.models import ConfigurationError


def test_base_hash_cache_reuses_stat_and_never_accepts_changed_content(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "bases").mkdir(parents=True)
    checkpoint = root / "base.safetensors"
    checkpoint.write_bytes(b"version-one")
    registry = {
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
    registry_path = root / "bases" / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

    first, reused, _ = resolve_base_sha256("base", root=root)
    assert reused is False
    assert first == sha256_file(checkpoint)
    second, reused, _ = resolve_base_sha256("base", root=root)
    assert reused is True
    assert second == first

    checkpoint.write_bytes(b"version-two-is-different")
    changed, reused, _ = resolve_base_sha256("base", root=root)
    assert reused is False
    assert changed != first
    persisted = read_yaml(registry_path)["bases"]["base"]
    assert persisted["sha256"] == first
    assert persisted["sha256_stat"] != {
        "bytes": checkpoint.stat().st_size,
        "mtime_ns": checkpoint.stat().st_mtime_ns,
        "inode": checkpoint.stat().st_ino,
        "device": checkpoint.stat().st_dev,
    }


def test_anima_base_requires_and_fingerprints_components(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "bases").mkdir(parents=True)
    checkpoint = root / "anima.safetensors"
    qwen3 = root / "qwen3.safetensors"
    vae = root / "vae.safetensors"
    checkpoint.write_bytes(b"anima")
    qwen3.write_bytes(b"qwen3")
    vae.write_bytes(b"vae")

    with pytest.raises(ConfigurationError, match="--qwen3 and --vae"):
        add_base("anima_base", checkpoint, family="anima", root=root)

    base = add_base(
        "anima_base", checkpoint, family="anima",
        components={"qwen3": qwen3, "vae": vae}, root=root,
    )
    assert base.prediction_type == "rectified_flow"
    fingerprints = resolve_base_component_fingerprints("anima_base", root=root)
    assert fingerprints["qwen3"]["sha256"] == sha256_file(qwen3)
    assert fingerprints["vae"]["sha256"] == sha256_file(vae)
