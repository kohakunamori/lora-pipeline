from __future__ import annotations

import yaml

from pipeline.bases import resolve_base_sha256
from pipeline.config import read_yaml, sha256_file


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
