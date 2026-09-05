from pathlib import Path

from PIL import Image

from pipeline.dataset.crop import plan_target_crop
from pipeline.dataset.subject import SubjectObservation
from pipeline.models import OptionalBackendUnavailable


class StubDetector:
    def __init__(self, observation):
        self.observation = observation
        self.calls = 0

    def detect_path(self, path: Path):
        self.calls += 1
        return self.observation


class UnavailableDetector:
    def detect_path(self, path: Path):
        raise OptionalBackendUnavailable("detector unavailable")


def _image(path: Path, size=(1600, 1200)) -> None:
    Image.new("RGB", size, "white").save(path)


def test_style_preserves_composition_without_loading_detector(tmp_path: Path) -> None:
    source = tmp_path / "style.png"
    _image(source)
    detector = StubDetector(None)

    plan = plan_target_crop(source, target_type="style", detector=detector)

    assert plan.crop_box is None
    assert plan.reason == "style_preserves_composition"
    assert detector.calls == 0


def test_character_crops_small_confident_subject(tmp_path: Path) -> None:
    source = tmp_path / "character.png"
    _image(source)
    observation = SubjectObservation(
        source_size=(1600, 1200),
        person_bbox=(600, 300, 1000, 900),
        head_bbox=(700, 320, 900, 500),
        halfbody_bbox=None,
        person_score=0.95,
        head_score=0.9,
        halfbody_score=None,
        detection_kind="person",
        person_count=1,
        ambiguous=False,
    )
    plan = plan_target_crop(
        source,
        target_type="character",
        detector=StubDetector(observation),
    )

    assert plan.cropped is True
    assert plan.mode == "subject_crop"
    assert min(plan.crop_size) >= 512
    assert plan.crop_box is not None
    assert plan.crop_box[0] < observation.person_bbox[0]
    assert plan.crop_box[2] > observation.person_bbox[2]


def test_character_keeps_already_prominent_subject(tmp_path: Path) -> None:
    source = tmp_path / "character.png"
    _image(source)
    observation = SubjectObservation(
        source_size=(1600, 1200),
        person_bbox=(300, 100, 1300, 1100),
        head_bbox=(650, 120, 950, 380),
        halfbody_bbox=None,
        person_score=0.95,
        head_score=0.9,
        halfbody_score=None,
        detection_kind="person",
        person_count=1,
        ambiguous=False,
    )
    plan = plan_target_crop(
        source,
        target_type="character",
        detector=StubDetector(observation),
    )

    assert plan.crop_box is None
    assert plan.reason == "subject_already_prominent"


def test_outfit_never_crops_head_only_fallback(tmp_path: Path) -> None:
    source = tmp_path / "outfit.png"
    _image(source)
    observation = SubjectObservation(
        source_size=(1600, 1200),
        person_bbox=(650, 250, 950, 950),
        head_bbox=(700, 250, 900, 450),
        halfbody_bbox=None,
        person_score=None,
        head_score=0.95,
        halfbody_score=None,
        detection_kind="head_fallback",
        person_count=0,
        ambiguous=False,
    )
    plan = plan_target_crop(
        source,
        target_type="character_outfit",
        detector=StubDetector(observation),
    )

    assert plan.crop_box is None
    assert plan.reason == "head_fallback_cannot_preserve_outfit"


def test_ambiguous_multiple_people_preserve_original(tmp_path: Path) -> None:
    source = tmp_path / "character.png"
    _image(source)
    observation = SubjectObservation(
        source_size=(1600, 1200),
        person_bbox=(200, 200, 700, 1000),
        head_bbox=None,
        halfbody_bbox=None,
        person_score=0.9,
        head_score=None,
        halfbody_score=None,
        detection_kind="person",
        person_count=2,
        ambiguous=True,
    )
    plan = plan_target_crop(
        source,
        target_type="character",
        detector=StubDetector(observation),
    )

    assert plan.crop_box is None
    assert plan.reason == "multiple_subjects_ambiguous"


def test_tiny_source_skips_detector_because_crop_cannot_add_detail(tmp_path: Path) -> None:
    source = tmp_path / "tiny.png"
    _image(source, (400, 500))
    detector = StubDetector(None)

    plan = plan_target_crop(source, target_type="character", detector=detector)

    assert plan.crop_box is None
    assert plan.reason == "source_too_small_for_safe_crop"
    assert detector.calls == 0


def test_detector_unavailable_keeps_original_as_conservative_fallback(tmp_path: Path) -> None:
    source = tmp_path / "character.png"
    _image(source)

    plan = plan_target_crop(
        source,
        target_type="character",
        detector=UnavailableDetector(),
    )

    assert plan.crop_box is None
    assert plan.mode == "keep"
    assert plan.reason == "subject_detector_unavailable"
