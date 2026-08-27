from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset.character import analyze_identity
from .dataset.image_info import discover_images
from .models import PipelineError


@dataclass(frozen=True)
class VideoIdentityCluster:
    cluster_id: int
    frames: tuple[Path, ...]
    representatives: tuple[Path, ...]

    @property
    def size(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class VideoIdentityReport:
    method: str
    clusters: tuple[VideoIdentityCluster, ...]
    outliers: tuple[Path, ...]
    total_frames: int

    def as_dict(self, *, root: Path | None = None) -> dict[str, Any]:
        def display(path: Path) -> str:
            if root is not None:
                try:
                    return path.relative_to(root).as_posix()
                except ValueError:
                    pass
            return str(path)

        return {
            "method": self.method,
            "total_frames": self.total_frames,
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "frames": cluster.size,
                    "representatives": [display(path) for path in cluster.representatives],
                }
                for cluster in self.clusters
            ],
            "outliers": len(self.outliers),
        }


def cluster_video_identity(frame_dir: Path, *, min_samples: int = 2) -> VideoIdentityReport:
    """Cluster filtered video frames by anime-character identity before project creation.

    This intentionally reuses the same CCIP backend as the normal Character identity step,
    but exposes every non-noise cluster so the user can choose the target character instead
    of assuming the largest cluster is correct.
    """

    images = discover_images(frame_dir)
    if not images:
        raise PipelineError(f"No supported video frames found under {frame_dir}")
    if len(images) == 1:
        only = images[0]
        return VideoIdentityReport(
            method="ccip",
            clusters=(VideoIdentityCluster(0, (only,), (only,)),),
            outliers=(),
            total_frames=1,
        )

    result = analyze_identity(images, min_samples=min_samples)
    labels = [int(label) for label in result.get("labels", [])]
    if len(labels) != len(images):
        raise PipelineError("CCIP identity clustering returned an invalid label count")

    grouped: dict[int, list[Path]] = {}
    outliers: list[Path] = []
    for image, label in zip(images, labels, strict=True):
        if label < 0:
            outliers.append(image)
        else:
            grouped.setdefault(label, []).append(image)

    clusters = tuple(
        VideoIdentityCluster(
            cluster_id=cluster_id,
            frames=tuple(frames),
            representatives=_representatives(frames),
        )
        for cluster_id, frames in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    )
    if not clusters:
        raise PipelineError(
            "CCIP could not form a stable character cluster from the filtered video frames"
        )
    return VideoIdentityReport(
        method=str(result.get("method", "ccip")),
        clusters=clusters,
        outliers=tuple(outliers),
        total_frames=len(images),
    )


def _representatives(frames: list[Path], *, limit: int = 3) -> tuple[Path, ...]:
    if len(frames) <= limit:
        return tuple(frames)
    if limit <= 1:
        return (frames[len(frames) // 2],)
    positions = [round(index * (len(frames) - 1) / (limit - 1)) for index in range(limit)]
    return tuple(frames[position] for position in positions)
