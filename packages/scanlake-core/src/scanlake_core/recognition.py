"""One frame in, one reading per ROI out.

anchors -> transform -> ROI crops -> filters (digit ROIs only) -> 32x256 inputs -> one batched
model run per model -> decode -> per-column validation.

The models sit behind the `Engine` protocol (implemented in scanlake-inference), so this module
needs neither ONNX Runtime nor PyTorch. A ROI whose model input is unchanged since the last frame
reuses its previous reading, so a HUD that does not change costs no inference at all.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from scanlake_core.decode import Floats, ctc_greedy, softmax
from scanlake_core.fields import FieldSpec, field_spec
from scanlake_core.geometry import Rect, apply
from scanlake_core.preprocess import Gray, preprocess_strip, to_gray, to_input
from scanlake_core.profile.model import DIGIT_CRNN, WORD_CNN, CaptureProfile, HudProfile
from scanlake_core.schema_map import field_slots
from scanlake_core.vision.anchors import AnchorFix, AnchorLocator
from scanlake_core.vision.filters import apply_filters
from scanlake_core.vision.roi import crop, place_rois


class Engine(Protocol):
    """Batched model runner (see scanlake_inference)."""

    @property
    def digit_classes(self) -> str: ...
    @property
    def word_classes(self) -> tuple[str, ...]: ...
    @property
    def blank_index(self) -> int: ...
    def digits(self, batch: Floats) -> Floats:
        """[N, 1, 32, 256] -> [T, N, C] log-probabilities."""
        ...

    def words(self, batch: Floats) -> Floats:
        """[N, 1, 32, 256] -> [N, K] logits."""
        ...


@dataclass(frozen=True)
class RoiReading:
    roi: str
    mode: str
    rect: Rect
    text: str | None  # normalized value; "" = blank slot; None = unreadable (no vote)
    confidence: float
    raw_text: str = ""  # decoder output before validation
    probs: Floats | None = None  # word CNN: class probabilities
    canvas: Gray | None = None  # model input, 32x256 (kept with keep_images)
    image: np.ndarray | None = None  # the crop the canvas was made from (kept with keep_images)


@dataclass(frozen=True)
class FrameResult:
    fix: AnchorFix
    readings: dict[str, RoiReading]
    timings: dict[str, float] = field(default_factory=dict)
    inferred: int = 0  # ROIs that actually ran through a model this frame

    @property
    def locked(self) -> bool:
        return self.fix.found


@dataclass(frozen=True)
class _Pending:
    """A ROI waiting for the model in this frame's batch."""

    name: str
    rect: Rect
    canvas: Gray
    key: bytes
    image: np.ndarray


@dataclass
class FrameRecognizer:
    hud: HudProfile
    templates: Mapping[str, Gray]
    engine: Engine
    keep_images: bool = False  # debug view and training crops
    _locator: AnchorLocator = field(init=False)
    _specs: dict[str, FieldSpec] = field(init=False)
    _cache: dict[str, tuple[bytes, RoiReading]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._locator = AnchorLocator(self.capture.anchors, self.templates)
        columns = {slot.roi: slot.column for slot in field_slots(self.hud)}
        self._specs = {
            roi.name: field_spec(columns.get(roi.name, roi.name), recognition=roi.recognition_mode)
            for roi in self.capture.rois
        }

    @property
    def capture(self) -> CaptureProfile:
        return self.hud.main

    def reset(self) -> None:
        self._locator.reset()
        self._cache.clear()

    def process(self, frame: np.ndarray, expected_scale: float | None = None) -> FrameResult:
        t0 = time.perf_counter()
        gray = to_gray(frame)
        fix = self._locator.locate(gray, expected_scale)
        t1 = time.perf_counter()
        if fix.transform is None:
            return FrameResult(fix, {}, {"anchors": t1 - t0})

        subs = self._sub_anchor_positions(gray, fix)
        placed = place_rois(self.capture, fix.transform, fix.scale, subs)

        readings: dict[str, RoiReading] = {}
        pending: dict[str, list[_Pending]] = {DIGIT_CRNN: [], WORD_CNN: []}
        for p in placed:
            roi = p.roi
            image = crop(frame, p.rect)
            if roi.recognition_mode == DIGIT_CRNN and image.size:
                image = apply_filters(image, roi.filters)
            canvas = preprocess_strip(image) if image.size else None
            if canvas is None:
                kept = image if self.keep_images and image.size else None
                readings[roi.name] = RoiReading(
                    roi.name, roi.recognition_mode, p.rect, "", 1.0, image=kept
                )
                self._cache.pop(roi.name, None)
                continue
            key = hashlib.blake2b(canvas.tobytes(), digest_size=16).digest()
            cached = self._cache.get(roi.name)
            if cached is not None and cached[0] == key:
                readings[roi.name] = _moved(cached[1], p.rect)
                continue
            pending[roi.recognition_mode].append(_Pending(roi.name, p.rect, canvas, key, image))
        t2 = time.perf_counter()

        self._run_digits(pending[DIGIT_CRNN], readings)
        self._run_words(pending[WORD_CNN], readings)
        t3 = time.perf_counter()
        inferred = len(pending[DIGIT_CRNN]) + len(pending[WORD_CNN])
        timings = {"anchors": t1 - t0, "rois": t2 - t1, "inference": t3 - t2}
        return FrameResult(fix, readings, timings, inferred)

    def _sub_anchor_positions(self, gray: Gray, fix: AnchorFix) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        assert fix.transform is not None
        for sub in self.capture.sub_anchors:
            if sub.name not in self.templates:
                continue
            ex, ey = apply(fix.transform, sub.ref_x, sub.ref_y)
            region = None
            if sub.search_region is not None:
                r = sub.search_region
                x0, y0 = apply(fix.transform, r.x, r.y)
                x1, y1 = apply(fix.transform, r.right, r.bottom)
                region = Rect(round(min(x0, x1)), round(min(y0, y1)),
                              round(abs(x1 - x0)), round(abs(y1 - y0)))  # fmt: skip
            hit = self._locator.match_near(gray, sub, ex, ey, fix.scale, region)
            positions[sub.name] = (hit.x, hit.y) if hit.found else (ex, ey)
        return positions

    def _run_digits(self, items: list[_Pending], out: dict[str, RoiReading]) -> None:
        if not items:
            return
        log_probs = self.engine.digits(to_input([i.canvas for i in items]))
        decoded = ctc_greedy(log_probs, self.engine.digit_classes, self.engine.blank_index)
        for item, result in zip(items, decoded, strict=False):
            text = self._specs[item.name].normalize(result.text) if result.text else None
            reading = RoiReading(
                item.name, DIGIT_CRNN, item.rect, text,
                result.confidence if text is not None else 0.0, raw_text=result.text,
                **self._images(item),
            )  # fmt: skip
            out[item.name] = reading
            self._cache[item.name] = (item.key, reading)

    def _run_words(self, items: list[_Pending], out: dict[str, RoiReading]) -> None:
        if not items:
            return
        probs = softmax(self.engine.words(to_input([i.canvas for i in items])))
        classes = self.engine.word_classes
        for item, p in zip(items, probs, strict=False):
            best = int(p.argmax())
            label = classes[best] if best < len(classes) else ""
            reading = RoiReading(
                item.name, WORD_CNN, item.rect, self._specs[item.name].normalize(label),
                float(p[best]), raw_text=label, probs=p, **self._images(item),
            )  # fmt: skip
            out[item.name] = reading
            self._cache[item.name] = (item.key, reading)

    def _images(self, item: _Pending) -> dict[str, np.ndarray | None]:
        if not self.keep_images:
            return {"canvas": None, "image": None}
        return {"canvas": item.canvas, "image": item.image}


def _moved(reading: RoiReading, rect: Rect) -> RoiReading:
    return reading if reading.rect == rect else replace(reading, rect=rect)
