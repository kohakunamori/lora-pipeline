from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from pipeline.dataset_optimizer import (
    FastQualityPolicy,
    OptimizeOptions,
    _append_relative_blur_flags,
    _caption_risk_flags,
    analyze_deep_quality,
    analyze_fast_quality,
    optimize_dataset,
)
from pipeline.dataset_workspace import DatasetWorkspace


def _image(path: Path, color: str = "red", size: tuple[int, int] = (768, 1024)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class FakeDeepBackend:
    def __init__(self, truncated_name: str | None = None):
        self.truncated_name = truncated_name

    def inspect(self, image: Path):
        return {
            "truncated": image.name == self.truncated_name,
            "monochrome": image.name.startswith("mono"),
            "image_type": "comic" if image.name.startswith("comic") else "illustration",
            "image_type_score": 0.95,
            "aesthetic_label": "low" if image.name.startswith("low") else "good",
            "aesthetic_percentile": 0.05 if image.name.startswith("low") else 0.7,
            "portrait_type": "person",
            "portrait_score": 0.9,
            "head_count": 2 if image.name.startswith("multi") else 1,
            "heads": [],
        }

    def lpips_clusters(self, images, *, threshold):
        del threshold
        return [0 if index < 2 else -1 for index, _path in enumerate(images)]


def test_fast_audit_keeps_captioned_copy_as_exact_duplicate_canonical(tmp_path) -> None:
    source = tmp_path / "source"
    _image(source / "a.png", "red")
    shutil.copy2(source / "a.png", source / "b.png")
    (source / "b.txt").write_text("1girl, portrait\n", encoding="utf-8")

    workspace = DatasetWorkspace.create("demo", concept_type="style", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")

    audit = analyze_fast_quality(workspace)
    assert len(audit["exact_groups"]) == 1
    group = audit["exact_groups"][0]
    assert group["canonical"].endswith("/b.png")
    assert group["safe_exclude_candidates"] == ["image-directory-001/a.png"]

    report = optimize_dataset(
        workspace,
        options=OptimizeOptions(apply_safe=True),
    )
    assert workspace.summary()["active_images"] == 1
    assert report["safe_exclusions_applied"][0]["key"].endswith("/a.png")


def test_caption_risk_flags_are_review_only_and_grouped() -> None:
    flags = _caption_risk_flags(
        "1girl, watermark, motion_blur, comic, monochrome, portrait"
    )
    groups = {flag["group"] for flag in flags}
    assert groups == {
        "text_overlay",
        "technical_quality",
        "layout_contamination",
        "monochrome",
    }
    assert {flag["severity"] for flag in flags} == {"review"}


def test_relative_blur_is_dataset_relative_not_absolute() -> None:
    records = []
    for index, sharpness in enumerate([100.0] * 8 + [0.1]):
        records.append(
            {
                "key": f"{index}.png",
                "corrupt": False,
                "technical": {"edge_variance": sharpness},
                "flags": [],
                "safe_exclude": False,
            }
        )
    _append_relative_blur_flags(
        records,
        FastQualityPolicy(relative_min_samples=8, blur_mad_z=3.5),
    )
    assert not any(
        flag["code"] == "relative_blur_outlier"
        for record in records
        for flag in record["flags"]
    )

    varied = []
    for index, sharpness in enumerate([70, 80, 90, 100, 110, 120, 130, 0.1]):
        varied.append(
            {
                "key": f"{index}.png",
                "corrupt": False,
                "technical": {"edge_variance": float(sharpness)},
                "flags": [],
                "safe_exclude": False,
            }
        )
    _append_relative_blur_flags(
        varied,
        FastQualityPolicy(relative_min_samples=8, blur_mad_z=3.5),
    )
    assert any(
        flag["code"] == "relative_blur_outlier"
        for flag in varied[-1]["flags"]
    )


def test_deep_model_signals_do_not_auto_exclude_advisory_quality(tmp_path) -> None:
    source = tmp_path / "source"
    for name in ("comic.png", "low.png", "multi.png", "mono.png"):
        _image(source / name, "red")
    workspace = DatasetWorkspace.create("demo", concept_type="character", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")

    result = analyze_deep_quality(workspace, backend=FakeDeepBackend())
    records = {record["key"].split("/")[-1]: record for record in result["records"]}
    assert records["comic.png"]["safe_exclude"] is False
    assert records["low.png"]["safe_exclude"] is False
    assert records["multi.png"]["safe_exclude"] is False
    assert records["mono.png"]["safe_exclude"] is False
    assert result["summary"]["non_target_image_type"] == 1
    assert result["summary"]["low_aesthetic"] == 1
    assert result["summary"]["head_count_issues"] == 1


def test_truncated_deep_signal_is_safe_and_lpips_is_review_only(tmp_path) -> None:
    source = tmp_path / "source"
    for name, color in (("a.png", "red"), ("b.png", "blue"), ("bad.png", "green")):
        _image(source / name, color)
    workspace = DatasetWorkspace.create("demo", concept_type="style", root=tmp_path)
    workspace.add_source_from_directory(source, kind="image_directory")
    backend = FakeDeepBackend(truncated_name="bad.png")

    deep = analyze_deep_quality(workspace, backend=backend)
    bad = next(record for record in deep["records"] if record["key"].endswith("/bad.png"))
    assert bad["safe_exclude"] is True

    report = optimize_dataset(
        workspace,
        options=OptimizeOptions(apply_safe=True, deep=True),
        deep_backend=backend,
    )
    assert workspace.summary()["active_images"] == 2
    assert any(item["key"].endswith("/bad.png") for item in report["safe_exclusions_applied"])
    assert report["lpips"]["summary"]["groups"] == 1
    assert workspace.summary()["active_images"] == 2
