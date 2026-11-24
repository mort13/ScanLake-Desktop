"""FrameRecognizer end to end with the real ONNX models on a synthetic HUD.

Anchors are pasted at their reference positions and real training strips/word images into the
ROI rectangles, then the frame goes through anchors -> ROIs -> filters -> batch -> decode ->
column validation. Strips the model misreads on their own are skipped while picking, so this
tests the plumbing, not the model.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from scanlake_core.decode import Floats, ctc_greedy, softmax
from scanlake_core.domain.deposit_materials import allowed_materials, pick_material
from scanlake_core.fields import field_spec
from scanlake_core.preprocess import STRIP_V1, STRIP_V2, preprocess_strip, to_input
from scanlake_core.profile.model import HudProfile
from scanlake_core.recognition import DEPOSIT_CONFIDENT, FrameRecognizer
from scanlake_core.schema_map import field_slots
from scanlake_core.vision.anchors import load_templates
from scanlake_core.vision.filters import apply_filters
from scanlake_core.vision.roi import crop
from scanlake_inference.metadata import ModelFiles
from scanlake_inference.ort_backend import OrtEngine
from tests.conftest import ASSETS
from tests.test_inference import STRIPS, WORDS, _read_gray

pytestmark = pytest.mark.skipif(
    not (STRIPS.exists() and WORDS.exists()), reason="Rock Capture CNN training data not found"
)

PATTERNS = {
    "mass": r"^\d{4,6}$",
    "resistance": r"^\d{1,2}%$",
    "instability": r"^\d{1,2}\.\d{2}$",
    "volume": r"^\d{2,3}\.\d{2}$",
    "amount": r"^\d{2}\.\d{2}%$",
    "quality": r"^\d{3}$",
}


class _DigitsOnV2:
    """The real engine, except that its digit model says it was trained with strip_v2."""

    digit_preprocess = STRIP_V2

    def __init__(self, inner: OrtEngine) -> None:
        self.inner = inner
        self.fed: list[Floats] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    def digits(self, batch: Floats) -> Floats:
        self.fed.append(batch)
        return self.inner.digits(batch)


@pytest.fixture(scope="module")
def engine(word_models: Path) -> OrtEngine:
    """Picks the word images (by the word classifier's reads) and reads them back."""
    return OrtEngine(ModelFiles(word_models), "cpu")


def _pick_strips(engine: OrtEngine, pattern: str, n: int, rng: random.Random) -> list[np.ndarray]:
    import re

    files = [p for p in STRIPS.glob("*.png") if re.match(pattern, p.stem.rsplit("_", 1)[0])]
    rng.shuffle(files)
    picked: list[np.ndarray] = []
    for path in files:
        gray, label = _read_gray(path), path.stem.rsplit("_", 1)[0]
        canvas = preprocess_strip(gray, version=engine.digit_preprocess)
        if canvas is None:
            continue
        [result] = ctc_greedy(engine.digits(to_input([canvas])), engine.digit_classes)
        if result.text == label and result.confidence > 0.9:
            picked.append(gray)
            if len(picked) == n:
                break
    return picked


def _pick_word(engine: OrtEngine, cls: str, rng: random.Random) -> np.ndarray:
    files = sorted((WORDS / cls).glob("*.png")) + sorted((WORDS / cls).glob("*.jpg"))
    rng.shuffle(files)
    for path in files:
        gray = _read_gray(path)
        canvas = preprocess_strip(gray, version=engine.word_preprocess)
        if canvas is None:
            continue
        p = softmax(engine.words(to_input([canvas])))[0]
        if engine.word_classes[int(p.argmax())] == cls and p.max() > 0.9:
            return gray
    raise AssertionError(f"no readable {cls} image")


def test_recognizer_end_to_end(engine: OrtEngine, profiles: dict[str, HudProfile]) -> None:
    rng = random.Random(7)
    hud = profiles["mole"]
    capture = hud.main
    templates = load_templates(capture.anchors, ASSETS / "anchors")
    region = capture.search_region
    frame = np.full((region.h + 40, region.w + 40, 3), 8, dtype=np.uint8)
    for anchor in capture.anchors:
        t = templates[anchor.name]
        x, y = round(20 + anchor.ref_x), round(20 + anchor.ref_y)
        frame[y : y + t.shape[0], x : x + t.shape[1]] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

    columns = {s.roi: s.column for s in field_slots(hud)}
    # A Bexalite rock: popped in ROI order material1-3, deposit, material4-5. Every material is
    # one the deposit can hold, so deposit_materials.json never overrides the model's own pick.
    words = ["inert_materials", "borase", "bexalite", "gold", "bexalite", "bexalite"]
    for roi in capture.rois:
        if roi.recognition_mode == "word_cnn":
            image = _pick_word(engine, words.pop(), rng)
        else:
            image = _pick_strips(engine, PATTERNS[columns[roi.name]], 1, rng)[0]
        x, y = round(20 + roi.ref_x), round(20 + roi.ref_y)
        resized = cv2.resize(image, (roi.width, roi.height), interpolation=cv2.INTER_AREA)
        frame[y : y + roi.height, x : x + roi.width] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    recognizer = FrameRecognizer(hud, templates, engine)
    result = recognizer.process(frame)
    assert result.locked and result.fix.scale == pytest.approx(1.0, rel=0.01)
    assert set(result.readings) == {r.name for r in capture.rois}

    # Each reading equals that ROI's crop run through the model on its own; material slots then
    # read the best class the (confidently read) Bexalite deposit can hold
    assert result.readings["deposit"].text == "Bexalite"
    assert result.readings["deposit"].confidence >= DEPOSIT_CONFIDENT
    bexalite = allowed_materials("Bexalite")
    assert bexalite is not None
    type_spec = field_spec("type")
    materials = [type_spec.normalize(c) or "" for c in engine.word_classes]
    blanks = 0
    for roi in capture.rois:
        reading = result.readings[roi.name]
        rect = reading.rect
        assert abs(rect.x - (20 + roi.ref_x)) <= 1 and abs(rect.y - (20 + roi.ref_y)) <= 1
        image = crop(frame, rect)
        if roi.recognition_mode == "digit_crnn":
            image = apply_filters(image, roi.filters)
        digits = roi.recognition_mode == "digit_crnn"
        version = engine.digit_preprocess if digits else engine.word_preprocess
        canvas = preprocess_strip(image, version=version)
        spec = field_spec(columns[roi.name])
        if canvas is None:
            expected: str | None = ""
            blanks += 1
        elif roi.recognition_mode == "word_cnn":
            p = softmax(engine.words(to_input([canvas])))[0]
            expected = spec.normalize(engine.word_classes[int(p.argmax())])
            if spec.column == "type":
                expected = materials[pick_material(p, materials, bexalite)[0]]
                assert reading.raw_text == engine.word_classes[int(p.argmax())]
        else:
            [r] = ctc_greedy(engine.digits(to_input([canvas])), engine.digit_classes)
            expected = spec.normalize(r.text) if r.text else None
        assert reading.text == expected, roi.name
    assert result.inferred == len(capture.rois) - blanks  # blank slots skip the model
    assert sum(r.text not in ("", None) for r in result.readings.values()) >= 15

    again = recognizer.process(frame)
    assert again.inferred == 0  # unchanged inputs come from the cache
    assert {n: r.text for n, r in again.readings.items()} == {
        n: r.text for n, r in result.readings.items()
    }

    # Each model is fed canvases made its own way: here digits strip_v2, words still strip_v1
    v2 = _DigitsOnV2(engine)
    other = FrameRecognizer(hud, templates, v2).process(frame)  # type: ignore[arg-type]
    digit_rois = [r for r in capture.rois if r.recognition_mode == "digit_crnn"]
    crops = [
        apply_filters(crop(frame, result.readings[r.name].rect), r.filters) for r in digit_rois
    ]

    def canvases(version: str) -> list[np.ndarray]:
        made = [preprocess_strip(c, version=version) for c in crops]
        return [c for c in made if c is not None]

    [batch] = v2.fed
    assert np.array_equal(batch, to_input(canvases(STRIP_V2)))
    assert not np.array_equal(batch, to_input(canvases(STRIP_V1)))  # the versions differ here
    words = [r.name for r in capture.rois if r.recognition_mode == "word_cnn"]
    assert {n: other.readings[n].text for n in words} == {n: result.readings[n].text for n in words}
