from __future__ import annotations

from PIL import Image

from pipeline.video_identity import cluster_video_identity


def test_single_video_frame_is_exposed_as_one_cluster(tmp_path) -> None:
    Image.new("RGB", (32, 32), "white").save(tmp_path / "only.png")
    report = cluster_video_identity(tmp_path)
    assert report.total_frames == 1
    assert len(report.clusters) == 1
    assert report.clusters[0].size == 1
    assert not report.outliers
