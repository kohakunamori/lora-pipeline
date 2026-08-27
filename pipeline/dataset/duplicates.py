from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .image_info import discover_images
from ..config import sha256_file, stable_hash
from ..models import OptionalBackendUnavailable


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def find_duplicates(raw_dir: Path, *, phash_distance: int = 6) -> dict[str, Any]:
    try:
        import imagehash
    except ImportError as exc:
        raise OptionalBackendUnavailable("ImageHash is required for perceptual duplicate detection") from exc

    paths = discover_images(raw_dir)
    exact: dict[str, list[str]] = defaultdict(list)
    hashes: list[Any] = []
    records: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(raw_dir).as_posix()
        sha = sha256_file(path)
        exact[sha].append(relative)
        try:
            with Image.open(path) as image:
                perceptual = imagehash.phash(image.convert("RGB"))
        except OSError:
            perceptual = None
        hashes.append(perceptual)
        records.append({"path": relative, "sha256": sha, "phash": str(perceptual) if perceptual else None})

    union = _UnionFind(len(paths))
    distances: list[dict[str, Any]] = []
    for left in range(len(paths)):
        if hashes[left] is None:
            continue
        for right in range(left + 1, len(paths)):
            if hashes[right] is None:
                continue
            distance = int(hashes[left] - hashes[right])
            if distance <= phash_distance:
                union.union(left, right)
                distances.append(
                    {"left": records[left]["path"], "right": records[right]["path"], "distance": distance}
                )

    near: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        near[union.find(index)].append(record["path"])
    exact_groups = [files for files in exact.values() if len(files) > 1]
    near_groups = [files for files in near.values() if len(files) > 1]
    return {
        "schema_version": 1,
        "input_hash": stable_hash(records),
        "phash_distance": phash_distance,
        "images": records,
        "exact_groups": sorted(exact_groups, key=lambda group: group[0]),
        "near_groups": sorted(near_groups, key=lambda group: group[0]),
        "near_pairs": distances,
        "summary": {
            "exact_groups": len(exact_groups),
            "exact_images": sum(len(group) for group in exact_groups),
            "near_groups": len(near_groups),
            "near_images": sum(len(group) for group in near_groups),
        },
    }


def suggested_exact_exclusions(manifest: dict[str, Any]) -> list[str]:
    return sorted(path for group in manifest.get("exact_groups", []) for path in group[1:])
