from PIL import Image

from pipeline.dataset.subject import SubjectDetector


def test_subject_detector_maps_proxy_boxes_back_to_source() -> None:
    calls = {"half": 0}

    def person(image, **kwargs):
        assert image.size == (1600, 1200)
        return [((500, 150, 1100, 1100), "person", 0.9)]

    def heads(image, **kwargs):
        return [((700, 180, 900, 380), "head", 0.95)]

    def half(image, **kwargs):
        calls["half"] += 1
        return [((50, 20, image.width - 50, image.height // 2), "halfbody", 0.8)]

    detector = SubjectDetector(
        person_detector=person,
        head_detector=heads,
        halfbody_detector=half,
        proxy_long_edge=1600,
    )
    observation = detector.detect(Image.new("RGB", (3200, 2400), "white"))

    assert observation is not None
    assert observation.person_bbox == (1000, 300, 2200, 2200)
    assert observation.head_bbox == (1400, 360, 1800, 760)
    assert observation.halfbody_bbox is not None
    assert observation.person_count == 1
    assert observation.ambiguous is False
    assert calls["half"] == 1


def test_multiple_similar_people_are_marked_ambiguous() -> None:
    def person(image, **kwargs):
        return [
            ((100, 100, 500, 1000), "person", 0.9),
            ((600, 100, 1000, 1000), "person", 0.9),
        ]

    def heads(image, **kwargs):
        return []

    def half(image, **kwargs):
        return []

    detector = SubjectDetector(
        person_detector=person,
        head_detector=heads,
        halfbody_detector=half,
    )
    observation = detector.detect(Image.new("RGB", (1200, 1200), "white"))

    assert observation is not None
    assert observation.person_count == 2
    assert observation.ambiguous is True
