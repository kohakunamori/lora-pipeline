from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from ..models import OptionalBackendUnavailable

Box = tuple[int, int, int, int]
Detection = tuple[Box, str, float]
Detector = Callable[..., list[Detection]]
DEFAULT_DETECTION_PROXY_LONG_EDGE = 1600


@dataclass(frozen=True)
class SubjectObservation:
    source_size: tuple[int, int]
    person_bbox: Box
    head_bbox: Box | None
    halfbody_bbox: Box | None
    person_score: float | None
    head_score: float | None
    halfbody_score: float | None
    detection_kind: str
    person_count: int
    ambiguous: bool

    @property
    def person_area_fraction(self) -> float:
        return _box_area(self.person_bbox) / max(1, self.source_size[0] * self.source_size[1])

    @property
    def person_height_fraction(self) -> float:
        return _box_size(self.person_bbox)[1] / max(1, self.source_size[1])

    def as_dict(self) -> dict[str, object]:
        return {
            "source_size": list(self.source_size),
            "person_bbox": list(self.person_bbox),
            "head_bbox": list(self.head_bbox) if self.head_bbox else None,
            "halfbody_bbox": list(self.halfbody_bbox) if self.halfbody_bbox else None,
            "person_score": self.person_score,
            "head_score": self.head_score,
            "halfbody_score": self.halfbody_score,
            "detection_kind": self.detection_kind,
            "person_count": self.person_count,
            "ambiguous": self.ambiguous,
            "person_area_fraction": round(self.person_area_fraction, 6),
            "person_height_fraction": round(self.person_height_fraction, 6),
        }


class SubjectDetector:
    """DeepGHS detector shared by still-image and video materialization.

    Inference runs on a bounded proxy. Returned boxes always use EXIF-oriented
    source-image coordinates so the actual crop can be taken from original pixels.
    """

    def __init__(
        self,
        *,
        person_detector: Detector | None = None,
        head_detector: Detector | None = None,
        halfbody_detector: Detector | None = None,
        proxy_long_edge: int = DEFAULT_DETECTION_PROXY_LONG_EDGE,
    ) -> None:
        if proxy_long_edge < 512:
            raise ValueError("proxy_long_edge must be at least 512")
        self.proxy_long_edge = int(proxy_long_edge)
        self._person_detector = person_detector
        self._head_detector = head_detector
        self._halfbody_detector = halfbody_detector

    def detect_path(self, path: Path) -> SubjectObservation | None:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        return self.detect(image)

    def detect(self, image: Image.Image) -> SubjectObservation | None:
        detect_person, detect_heads, detect_halfbody = self._detectors()
        source = image.convert("RGB")
        proxy, scale_x, scale_y = _make_detection_proxy(source, self.proxy_long_edge)
        try:
            people = detect_person(
                proxy,
                level="m",
                version="v1.1",
                conf_threshold=0.3,
                iou_threshold=0.5,
            )
            heads = detect_heads(proxy, conf_threshold=0.4, iou_threshold=0.7)
        except Exception as exc:
            raise OptionalBackendUnavailable(
                f"DeepGHS anime subject detection failed: {exc}"
            ) from exc

        people = [
            (_clip_box(box, proxy.width, proxy.height), label, float(score))
            for box, label, score in people
        ]
        heads = [
            (_clip_box(box, proxy.width, proxy.height), label, float(score))
            for box, label, score in heads
        ]

        if people:
            primary_index, ambiguous = _primary_person(people, proxy.size)
            person_proxy, _label, person_score = people[primary_index]
            person_box = _map_box(
                person_proxy, scale_x, scale_y, source.width, source.height
            )
            head_index = _best_head_for_person(heads, person_proxy)
            head_box: Box | None = None
            head_score: float | None = None
            if head_index is not None:
                head_proxy, _head_label, head_score = heads[head_index]
                head_box = _map_box(
                    head_proxy, scale_x, scale_y, source.width, source.height
                )

            halfbody_box: Box | None = None
            halfbody_score: float | None = None
            x0, y0, x1, y1 = person_proxy
            person_image = proxy.crop((x0, y0, x1, y1))
            if person_image.width >= 96 and person_image.height >= 96:
                try:
                    halfbody = detect_halfbody(
                        person_image,
                        level="s",
                        version="v1.0",
                        conf_threshold=0.5,
                        iou_threshold=0.7,
                    )
                except Exception as exc:
                    raise OptionalBackendUnavailable(
                        f"DeepGHS anime half-body detection failed: {exc}"
                    ) from exc
                if halfbody:
                    local_box, _half_label, halfbody_score = max(
                        halfbody, key=lambda item: float(item[2])
                    )
                    hx0, hy0, hx1, hy1 = _clip_box(
                        local_box, person_image.width, person_image.height
                    )
                    halfbody_box = _map_box(
                        (x0 + hx0, y0 + hy0, x0 + hx1, y0 + hy1),
                        scale_x,
                        scale_y,
                        source.width,
                        source.height,
                    )

            return SubjectObservation(
                source_size=source.size,
                person_bbox=person_box,
                head_bbox=head_box,
                halfbody_bbox=halfbody_box,
                person_score=float(person_score),
                head_score=float(head_score) if head_score is not None else None,
                halfbody_score=(
                    float(halfbody_score) if halfbody_score is not None else None
                ),
                detection_kind="person",
                person_count=len(people),
                ambiguous=ambiguous,
            )

        # Head fallback is intentionally conservative: only one head may infer a
        # primary person. Outfit crop policy never crops from this weaker evidence.
        if len(heads) == 1:
            head_proxy, _label, head_score = heads[0]
            head_box = _map_box(
                head_proxy, scale_x, scale_y, source.width, source.height
            )
            return SubjectObservation(
                source_size=source.size,
                person_bbox=_infer_person_from_head(
                    head_box, source.width, source.height
                ),
                head_bbox=head_box,
                halfbody_bbox=None,
                person_score=None,
                head_score=float(head_score),
                halfbody_score=None,
                detection_kind="head_fallback",
                person_count=0,
                ambiguous=False,
            )
        return None

    def _detectors(self) -> tuple[Detector, Detector, Detector]:
        if all(
            detector is not None
            for detector in (
                self._person_detector,
                self._head_detector,
                self._halfbody_detector,
            )
        ):
            return (
                self._person_detector,  # type: ignore[return-value]
                self._head_detector,  # type: ignore[return-value]
                self._halfbody_detector,  # type: ignore[return-value]
            )
        try:
            from imgutils.detect import detect_halfbody, detect_heads, detect_person
        except ImportError as exc:
            raise OptionalBackendUnavailable(
                "DeepGHS imgutils anime subject detectors are unavailable"
            ) from exc
        self._person_detector = detect_person
        self._head_detector = detect_heads
        self._halfbody_detector = detect_halfbody
        return detect_person, detect_heads, detect_halfbody


def _make_detection_proxy(
    image: Image.Image, long_edge: int
) -> tuple[Image.Image, float, float]:
    width, height = image.size
    scale = min(1.0, long_edge / max(width, height))
    if scale < 1.0:
        proxy = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    else:
        proxy = image.copy()
    return proxy, width / proxy.width, height / proxy.height


def _map_box(
    box: Box, scale_x: float, scale_y: float, width: int, height: int
) -> Box:
    x0, y0, x1, y1 = box
    return _clip_box(
        (
            round(x0 * scale_x),
            round(y0 * scale_y),
            round(x1 * scale_x),
            round(y1 * scale_y),
        ),
        width,
        height,
    )


def _clip_box(box: Box, width: int, height: int) -> Box:
    x0, y0, x1, y1 = (int(value) for value in box)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def _box_size(box: Box) -> tuple[int, int]:
    return box[2] - box[0], box[3] - box[1]


def _box_area(box: Box) -> int:
    width, height = _box_size(box)
    return max(0, width) * max(0, height)


def _box_center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _best_head_for_person(heads: list[Detection], person_box: Box) -> int | None:
    matches: list[tuple[float, int]] = []
    for index, (head_box, _label, score) in enumerate(heads):
        cx, cy = _box_center(head_box)
        if person_box[0] <= cx <= person_box[2] and person_box[1] <= cy <= person_box[3]:
            matches.append((float(score), index))
    return max(matches)[1] if matches else None


def _primary_person(
    people: list[Detection], source_size: tuple[int, int]
) -> tuple[int, bool]:
    width, height = source_size
    image_area = max(1, width * height)
    image_cx, image_cy = width / 2, height / 2
    ranked: list[tuple[float, float, int]] = []
    for index, (box, _label, score) in enumerate(people):
        area_fraction = _box_area(box) / image_area
        cx, cy = _box_center(box)
        nx = abs(cx - image_cx) / max(1.0, width / 2)
        ny = abs(cy - image_cy) / max(1.0, height / 2)
        center_factor = max(0.55, 1.0 - 0.25 * min(1.0, (nx + ny) / 2))
        rank = area_fraction * center_factor * (0.75 + 0.25 * float(score))
        ranked.append((rank, area_fraction, index))
    ranked.sort(reverse=True)
    best_rank, best_area, best_index = ranked[0]
    if len(ranked) == 1:
        return best_index, False
    second_rank, second_area, _ = ranked[1]
    dominant = (
        best_area >= 0.35
        or best_area >= second_area * 1.35
        or best_rank >= second_rank * 1.45
    )
    return best_index, not dominant


def _infer_person_from_head(head_box: Box, width: int, height: int) -> Box:
    hx0, hy0, hx1, hy1 = head_box
    head_width, head_height = _box_size(head_box)
    cx = (hx0 + hx1) / 2
    inferred_height = max(head_height * 4.5, head_width * 4.0)
    inferred_width = max(head_width * 2.5, inferred_height * 0.45)
    top = hy0 - head_height * 0.35
    return _clip_box(
        (
            round(cx - inferred_width / 2),
            round(top),
            round(cx + inferred_width / 2),
            round(top + inferred_height),
        ),
        width,
        height,
    )
