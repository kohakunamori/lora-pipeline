from pathlib import Path

from PIL import Image

from pipeline.dataset.image_normalizer import (
    DEFAULT_MAX_PIXELS,
    normalize_training_image,
    plan_training_size,
)


def test_small_image_is_not_upscaled_or_reencoded(tmp_path: Path) -> None:
    source = tmp_path / "small.png"
    destination = tmp_path / "prepared" / "small.png"
    Image.new("RGB", (768, 1024), "red").save(source)
    original = source.read_bytes()

    plan = normalize_training_image(source, destination)

    assert plan.downscaled is False
    assert plan.target_size == (768, 1024)
    assert destination.read_bytes() == original


def test_large_image_is_downscaled_to_about_one_megapixel(tmp_path: Path) -> None:
    source = tmp_path / "large.jpg"
    destination = tmp_path / "prepared" / "large.jpg"
    Image.new("RGB", (3840, 2160), "blue").save(source)

    plan = normalize_training_image(source, destination)

    assert plan.downscaled is True
    assert plan.target_size[0] % 32 == 0
    assert plan.target_size[1] % 32 == 0
    assert plan.target_size[0] * plan.target_size[1] <= DEFAULT_MAX_PIXELS
    with Image.open(destination) as prepared:
        assert prepared.size == plan.target_size


def test_plan_preserves_orientation_without_upscale() -> None:
    landscape = plan_training_size(3000, 2000)
    portrait = plan_training_size(2000, 3000)

    assert landscape.downscaled and portrait.downscaled
    assert landscape.target_size == tuple(reversed(portrait.target_size))
    assert landscape.target_size[0] > landscape.target_size[1]
    assert portrait.target_size[1] > portrait.target_size[0]


def test_crop_happens_before_downscale(tmp_path: Path) -> None:
    source = tmp_path / "large.png"
    destination = tmp_path / "prepared" / "large.png"
    Image.new("RGB", (3000, 2000), "green").save(source)

    # The selected crop is already below the one-megapixel budget, so cropping
    # should remove the need for a second resize even though the raw source is 6 MP.
    plan = normalize_training_image(
        source,
        destination,
        crop_box=(1050, 500, 1950, 1500),
    )

    assert plan.cropped is True
    assert plan.input_size == (900, 1000)
    assert plan.target_size == (900, 1000)
    assert plan.downscaled is False
    with Image.open(destination) as prepared:
        assert prepared.size == (900, 1000)
