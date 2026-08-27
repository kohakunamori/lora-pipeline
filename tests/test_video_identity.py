from __future__ import annotations

from pathlib import Path

from PIL import Image

from pipeline import video_identity


def _image(path: Path, value: int) -> None:
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def test_cluster_video_identity_exposes_all_non_noise_clusters(tmp_path, monkeypatch) -> None:
    for index in range(6):
        _image(tmp_path / f"frame-{index}.png", 20 + index)

    monkeypatch.setattr(
        video_identity,
        "analyze_identity",
        lambda images, min_samples: {
            "method": "ccip",
            "labels": [2, 2, 2, 7, 7, -1],
        },
    )

    report = video_identity.cluster_video_identity(tmp_path)

    assert report.total_frames == 6
    assert [cluster.cluster_id for cluster in report.clusters] == [2, 7]
    assert [cluster.size for cluster in report.clusters] == [3, 2]
    assert len(report.outliers) == 1
    assert len(report.clusters[0].representatives) == 3


def test_identity_report_provenance_uses_relative_representative_names(tmp_path, monkeypatch) -> None:
    for index in range(4):
        _image(tmp_path / f"frame-{index}.png", 40 + index)

    monkeypatch.setattr(
        video_identity,
        "analyze_identity",
        lambda images, min_samples: {
            "method": "ccip",
            "labels": [0, 0, 0, 0],
        },
    )

    report = video_identity.cluster_video_identity(tmp_path)
    payload = report.as_dict(root=tmp_path)

    assert payload["clusters"][0]["frames"] == 4
    assert all("/" not in name for name in payload["clusters"][0]["representatives"])
