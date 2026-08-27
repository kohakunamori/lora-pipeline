from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..models import OptionalBackendUnavailable


def analyze_identity(images: list[Path], *, min_samples: int = 2) -> dict[str, Any]:
    if len(images) < 2:
        return {
            "method": "ccip",
            "labels": [0] * len(images),
            "main_cluster": [str(path) for path in images],
            "possible_outliers": [],
            "possible_mixed_characters": [],
        }
    try:
        from imgutils.metrics import ccip_clustering
    except ImportError as exc:
        raise OptionalBackendUnavailable("imgutils CCIP backend is unavailable") from exc
    labels = [int(label) for label in ccip_clustering([str(path) for path in images], min_samples=min_samples)]
    counts = Counter(label for label in labels if label >= 0)
    main_label = counts.most_common(1)[0][0] if counts else -1
    main = [str(path) for path, label in zip(images, labels, strict=True) if label == main_label]
    outliers = [str(path) for path, label in zip(images, labels, strict=True) if label < 0]
    mixed = [str(path) for path, label in zip(images, labels, strict=True) if label >= 0 and label != main_label]
    return {
        "method": "ccip",
        "labels": labels,
        "clusters": dict(sorted(Counter(labels).items())),
        "main_cluster": main,
        "possible_outliers": outliers,
        "possible_mixed_characters": mixed,
    }
