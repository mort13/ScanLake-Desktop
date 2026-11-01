"""Find a HUD's anchors in a frame and fit the reference -> frame transform.

Anchors are template images (cut at the 2560x1440 reference resolution) whose top-left corner
sits at (ref_x, ref_y) in the profile's reference frame. Each anchor is matched with
TM_CCOEFF_NORMED over a set of template scales; once found, later frames only search a window
around the last position at the last scale, which keeps a locked HUD cheap to track.

From 2+ matches a similarity transform (translation, rotation, uniform scale) is fitted with
RANSAC, so one wrong match out of three is rejected instead of skewing every ROI. The fit must
agree with the template scale the anchors matched at and be (nearly) unrotated: two chance
matches anywhere on a desktop always fit *some* similarity transform, but not a plausible one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from scanlake_core.geometry import Affine, Rect, uniform_scale
from scanlake_core.preprocess import Gray
from scanlake_core.profile.model import AnchorPoint

# Scales tried on a cold start, after the expected one (H / 1440): the web app's candidates,
# then Rock Capture CNN's sweep.
WEB_SCALES = (1.0, 0.75, 1.5, 0.5, 0.625, 0.8, 1.25, 2.0)
SWEEP_SCALES = tuple(float(s) for s in np.linspace(0.5, 2.0, 17))
REFINE_STEPS = (-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04)
TRACK_PADDING = 30  # px at scale 1, around the last match
RANSAC_THRESHOLD = 3.0  # px at scale 1
SCALE_TOLERANCE = 0.15  # fitted scale vs the matched template scale
MAX_ROTATION_DEG = 5.0


def load_template(path: Path) -> Gray:
    """Grayscale template, decoded like Rock Capture CNN did (color decode, then BGR2GRAY).

    np.fromfile + imdecode instead of imread, which fails on non-ASCII Windows paths.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read anchor template {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def load_templates(anchors: Iterable[AnchorPoint], directory: Path) -> dict[str, Gray]:
    return {a.name: load_template(directory / a.template_path) for a in anchors}


@dataclass(frozen=True)
class AnchorMatch:
    name: str
    found: bool
    x: float = 0.0  # top-left of the matched template, frame coordinates
    y: float = 0.0
    w: int = 0
    h: int = 0
    confidence: float = 0.0
    scale: float = 1.0  # template scale that matched


@dataclass(frozen=True)
class AnchorFix:
    matches: dict[str, AnchorMatch]
    transform: Affine | None = None  # reference -> frame
    scale: float = 1.0
    inliers: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return self.transform is not None


@dataclass
class _Track:
    scale: float
    x: float
    y: float


@dataclass
class AnchorLocator:
    anchors: tuple[AnchorPoint, ...]
    templates: Mapping[str, Gray]
    _tracks: dict[str, _Track] = field(default_factory=dict, init=False)
    _scaled: dict[tuple[str, float], Gray] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self._tracks.clear()

    def locate(
        self,
        gray: Gray,
        expected_scale: float | None = None,
        only_scales: Sequence[float] | None = None,
    ) -> AnchorFix:
        """Finds the anchors and fits the transform.

        `only_scales` limits a cold search to those template scales (plus refinement), which
        keeps a search over a whole monitor affordable.
        """
        matches = {a.name: self._find(gray, a, expected_scale, only_scales) for a in self.anchors}
        found = [a for a in self.anchors if matches[a.name].found]
        if len(found) < 2:
            return AnchorFix(matches=matches)

        ref = np.array([[a.ref_x, a.ref_y] for a in found], dtype=np.float32)
        det = np.array([[matches[a.name].x, matches[a.name].y] for a in found], dtype=np.float32)
        scale_guess = float(np.median([matches[a.name].scale for a in found]))
        transform, mask = cv2.estimateAffinePartial2D(
            ref, det, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD * scale_guess
        )
        if transform is None:
            return AnchorFix(matches=matches)
        inliers = tuple(a.name for a, ok in zip(found, mask.ravel(), strict=True) if ok)
        if len(inliers) < 2:
            return AnchorFix(matches=matches)
        affine = np.asarray(transform, dtype=np.float64)
        scale = uniform_scale(affine)
        rotation = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
        if abs(scale / scale_guess - 1) > SCALE_TOLERANCE or abs(rotation) > MAX_ROTATION_DEG:
            for a in found:  # chance matches: do not track them
                self._tracks.pop(a.name, None)
            return AnchorFix(matches=matches)
        for name in matches:
            if name not in inliers:
                self._tracks.pop(name, None)
        return AnchorFix(matches=matches, transform=affine, scale=scale, inliers=inliers)

    def match_near(
        self, gray: Gray, anchor: AnchorPoint, x: float, y: float, scale: float, region: Rect | None
    ) -> AnchorMatch:
        """Sub-anchor refinement: search at `scale` near (x, y) or inside `region`."""
        template = self._template(anchor.name, scale)
        th, tw = template.shape
        pad = max(40, round(40 * scale))
        window = region or Rect(round(x) - pad, round(y) - pad, tw + 2 * pad, th + 2 * pad)
        return self._match(gray, anchor, template, scale, window)

    # -- matching ---------------------------------------------------------

    def _find(
        self,
        gray: Gray,
        anchor: AnchorPoint,
        expected: float | None,
        only_scales: Sequence[float] | None = None,
    ) -> AnchorMatch:
        track = self._tracks.get(anchor.name)
        if track is not None:
            template = self._template(anchor.name, track.scale)
            th, tw = template.shape
            pad = round(TRACK_PADDING * track.scale)
            window = Rect(round(track.x) - pad, round(track.y) - pad, tw + 2 * pad, th + 2 * pad)
            hit = self._match(gray, anchor, template, track.scale, window)
            if not hit.found:  # moved further than the window: whole frame, same scale
                hit = self._match(gray, anchor, template, track.scale, None)
            if hit.found:
                self._tracks[anchor.name] = _Track(track.scale, hit.x, hit.y)
                return hit
            del self._tracks[anchor.name]

        hit = self._sweep(gray, anchor, expected, only_scales)
        if hit.found:
            self._tracks[anchor.name] = _Track(hit.scale, hit.x, hit.y)
        return hit

    def _sweep(
        self,
        gray: Gray,
        anchor: AnchorPoint,
        expected: float | None,
        only_scales: Sequence[float] | None = None,
    ) -> AnchorMatch:
        scales: list[float] = []
        candidates = tuple(only_scales) if only_scales else WEB_SCALES + SWEEP_SCALES
        for s in ((expected,) if expected else ()) + candidates:
            if s is not None and all(abs(s - t) > 1e-6 for t in scales):
                scales.append(round(s, 4))

        best = AnchorMatch(anchor.name, found=False)
        for s in scales:
            hit = self._match(gray, anchor, self._template(anchor.name, s), s, None)
            if hit.confidence > best.confidence:
                best = hit
        if best.confidence <= 0:
            return best
        for step in REFINE_STEPS:  # the coarse sets are ~9 % apart: refine to 1 %
            s = round(best.scale * (1 + step), 4)
            hit = self._match(gray, anchor, self._template(anchor.name, s), s, None)
            if hit.confidence > best.confidence:
                best = hit
        return best

    def _template(self, name: str, scale: float) -> Gray:
        key = (name, round(scale, 4))
        cached = self._scaled.get(key)
        if cached is None:
            base = self.templates[name]
            th, tw = base.shape
            size = (max(1, round(tw * scale)), max(1, round(th * scale)))
            cached = (
                base if size == (tw, th) else cv2.resize(base, size, interpolation=cv2.INTER_LINEAR)
            )
            if len(self._scaled) > 512:
                self._scaled.clear()
            self._scaled[key] = cached
        return cached

    def _match(
        self, gray: Gray, anchor: AnchorPoint, template: Gray, scale: float, window: Rect | None
    ) -> AnchorMatch:
        fh, fw = gray.shape
        area = (window or Rect(0, 0, fw, fh)).clip(fw, fh)
        th, tw = template.shape
        if area.w < tw or area.h < th:
            return AnchorMatch(anchor.name, found=False)
        region = gray[area.y : area.bottom, area.x : area.right]
        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, (px, py) = cv2.minMaxLoc(result)
        if not np.isfinite(peak):
            return AnchorMatch(anchor.name, found=False)
        dx, dy = _subpixel(result, px, py)
        return AnchorMatch(
            name=anchor.name,
            found=peak >= anchor.match_threshold,
            x=area.x + px + dx,
            y=area.y + py + dy,
            w=tw,
            h=th,
            confidence=float(peak),
            scale=scale,
        )


def _subpixel(result: np.ndarray, px: int, py: int) -> tuple[float, float]:
    """Quadratic peak interpolation around the integer maximum (each axis in [-0.5, 0.5])."""

    def offset(a: float, b: float, c: float) -> float:
        denom = a - 2 * b + c
        return 0.0 if denom == 0 else float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))

    h, w = result.shape
    dx = offset(*result[py, px - 1 : px + 2]) if 0 < px < w - 1 else 0.0
    dy = offset(*result[py - 1 : py + 2, px]) if 0 < py < h - 1 else 0.0
    return dx, dy
