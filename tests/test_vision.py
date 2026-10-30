"""Preprocessing/filter parity with the training code, and anchor localisation on synthetic HUDs."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scanlake_core.geometry import apply
from scanlake_core.preprocess import preprocess_strip
from scanlake_core.profile.model import FilterSettings, HudProfile
from scanlake_core.vision.anchors import AnchorLocator, load_templates
from scanlake_core.vision.filters import apply_filters
from scanlake_core.vision.roi import place_rois
from tests.conftest import ASSETS
from tests.reference import legacy_preprocess as legacy

RNG = np.random.default_rng(1234)


def _strip(h: int, w: int, bright: int = 230) -> np.ndarray:
    """A HUD-like crop: dark noisy background with bright glyph-ish blocks."""
    img = RNG.integers(0, 40, size=(h, w, 3), dtype=np.uint8)
    x = 2
    while x < w - 6:
        gw = int(RNG.integers(3, 7))
        y0 = int(RNG.integers(1, max(2, h // 4)))
        img[y0 : h - y0, x : x + gw] = bright - RNG.integers(0, 40)
        x += gw + int(RNG.integers(2, 5))
    return img


CROPS = [_strip(h, w) for h, w in [(19, 41), (19, 118), (25, 60), (40, 300), (12, 20), (64, 64)]]
CROPS += [RNG.integers(0, 256, size=(23, 77, 3), dtype=np.uint8) for _ in range(20)]
CROPS += [np.full((19, 41, 3), 30, np.uint8), np.zeros((5, 5, 3), np.uint8)]


@pytest.mark.parametrize("index", range(len(CROPS)))
def test_preprocess_bit_exact(index: int) -> None:
    crop = CROPS[index]
    ours, theirs = preprocess_strip(crop), legacy.preprocess(crop)
    if theirs is None:
        assert ours is None
    else:
        assert ours is not None and np.array_equal(ours, theirs)


@pytest.mark.parametrize(
    "settings",
    [
        FilterSettings(),
        FilterSettings(contrast=3),
        FilterSettings(brightness=20, contrast=-10),
        FilterSettings(channel="red", grayscale=False),
        FilterSettings(threshold_enabled=True, threshold=90),
        FilterSettings(invert=True, channel="blue"),
    ],
)
def test_filters_bit_exact(settings: FilterSettings) -> None:
    for crop in CROPS[:8]:
        assert np.array_equal(
            apply_filters(crop, settings), legacy.ImageFilterPipeline.apply(crop, settings)
        )


def _paste_hud(
    hud: HudProfile, scale: float, offset: tuple[int, int], size: tuple[int, int]
) -> np.ndarray:
    capture = hud.main
    templates = load_templates(capture.anchors, ASSETS / "anchors")
    frame = RNG.integers(0, 50, size=(size[1], size[0], 3), dtype=np.uint8)
    for anchor in capture.anchors:
        t = templates[anchor.name]
        th, tw = t.shape
        scaled = cv2.resize(t, (round(tw * scale), round(th * scale)), interpolation=cv2.INTER_AREA)
        x = round(offset[0] + anchor.ref_x * scale)
        y = round(offset[1] + anchor.ref_y * scale)
        sh, sw = scaled.shape
        frame[y : y + sh, x : x + sw] = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    return frame


@pytest.mark.parametrize("ship", ["mole", "golem", "prospector"])
@pytest.mark.parametrize("scale", [0.75, 1.0, 1.5])
def test_anchors_on_synthetic_hud(profiles: dict[str, HudProfile], ship: str, scale: float) -> None:
    hud = profiles[ship]
    capture = hud.main
    region = capture.search_region
    offset = (37, 23)
    size = (round(region.w * scale) + 80, round(region.h * scale) + 60)
    frame = _paste_hud(hud, scale, offset, size)

    locator = AnchorLocator(capture.anchors, load_templates(capture.anchors, ASSETS / "anchors"))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fix = locator.locate(gray, expected_scale=None)
    assert fix.found, {k: round(m.confidence, 3) for k, m in fix.matches.items()}
    assert fix.scale == pytest.approx(scale, rel=0.01)
    assert fix.transform is not None
    for ref in [(0, 0), (region.w / 2, region.h / 2), (200.0, 400.0)]:
        x, y = apply(fix.transform, *ref)
        assert x == pytest.approx(offset[0] + ref[0] * scale, abs=1.5)
        assert y == pytest.approx(offset[1] + ref[1] * scale, abs=1.5)

    # Tracking: the next frame reuses the scale and the search window
    again = locator.locate(gray)
    assert again.found and again.scale == pytest.approx(fix.scale, rel=1e-3)

    rois = place_rois(capture, fix.transform, fix.scale)
    mass = next(p for p in rois if p.roi.name == "mass")
    assert mass.rect.w == round(mass.roi.width * fix.scale)


def test_one_wrong_anchor_is_rejected(profiles: dict[str, HudProfile]) -> None:
    capture = profiles["mole"].main
    templates = load_templates(capture.anchors, ASSETS / "anchors")
    frame = _paste_hud(profiles["mole"], 1.0, (20, 20), (400, 600))
    # Paste a second copy of the "auto" anchor far away and blank the real one
    auto = next(a for a in capture.anchors if a.name == "auto")
    t = templates["auto"]
    th, tw = t.shape
    x, y = round(20 + auto.ref_x), round(20 + auto.ref_y)
    frame[y : y + th, x : x + tw] = 10
    frame[300 : 300 + th, 300 : 300 + tw] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

    fix = AnchorLocator(capture.anchors, templates).locate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert fix.found and "auto" not in fix.inliers
    assert fix.scale == pytest.approx(1.0, rel=0.01)
