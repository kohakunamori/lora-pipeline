from __future__ import annotations

from pipeline import video_frame_selection as selector
from pipeline.video_color import VideoColorInfo


def test_candidate_grid_checks_five_nearby_frames_without_large_time_drift() -> None:
    # 2 second interval at 8 candidates/sec => nominal centers 0, 16, 32.
    assert selector._candidate_indices(period=16, target_count=3) == [
        0,
        1,
        2,
        14,
        15,
        16,
        17,
        18,
        30,
        31,
        32,
        33,
        34,
    ]


def test_blur_ranking_chooses_least_blurry_and_prefers_center_on_ties() -> None:
    candidates = [0, 1, 2, 14, 15, 16, 17, 18]
    scores = [0.40, 0.20, 0.30, 0.40, 0.10, 0.10, 0.10, 0.20]

    selected = selector._select_best_indices(
        candidates,
        scores,
        period=16,
        target_count=2,
    )

    assert selected == [1, 16]


def test_probe_filter_scores_a_small_proxy_before_blurdetect() -> None:
    info = VideoColorInfo(
        width=3840,
        height=2160,
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
    )

    chain = selector._probe_filter(16, info)

    assert chain.startswith("fps=8:start_time=0:round=near,select='")
    assert "scale=960:960:force_original_aspect_ratio=decrease:flags=area" in chain
    assert "blurdetect=block_width=32:block_height=32:block_pct=80" in chain
    assert "zscale=" not in chain


def test_hdr_proxy_is_downscaled_before_tonemap_but_final_extract_is_full_resolution() -> None:
    info = VideoColorInfo(
        width=3840,
        height=2160,
        pixel_format="yuv420p10le",
        color_space="bt2020nc",
        color_transfer="smpte2084",
        color_primaries="bt2020",
    )

    probe = selector._probe_filter(16, info)
    extract = selector._extract_filter([2, 16, 34], info)

    assert probe.index("scale=960:960") < probe.index("zscale=t=linear")
    assert "tonemap=mobius:param=0.3:desat=0" in probe
    assert "scale=960:960:force_original_aspect_ratio=decrease" not in extract
    assert "eq(n\\,2)" in extract
    assert "eq(n\\,16)" in extract
    assert "tonemap=mobius:param=0.3:desat=0" in extract


def test_blur_log_parser_ignores_summary_line() -> None:
    stderr = """
[Parsed_blurdetect_3] blur: 4.5000000
[Parsed_blurdetect_3] blur: 2.2500000
[Parsed_blurdetect_3] blur mean: 3.3750000
"""
    assert selector._parse_blur_scores(stderr) == [4.5, 2.25]


def test_window_expression_covers_plus_minus_250ms_for_two_second_interval() -> None:
    expression = selector._window_select_expression(16)
    for residue in (0, 1, 2, 14, 15):
        assert f"\\,{residue})" in expression
