"""Place a profile's ROIs in the frame through the anchor transform, and crop them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from scanlake_core.geometry import Affine, Rect, apply
from scanlake_core.profile.model import CaptureProfile, RoiDef


@dataclass(frozen=True)
class PlacedRoi:
    roi: RoiDef
    rect: Rect  # frame coordinates, not clipped


def place_rois(
    profile: CaptureProfile,
    transform: Affine,
    scale: float,
    sub_anchor_positions: Mapping[str, tuple[float, float]] | None = None,
) -> list[PlacedRoi]:
    """Each ROI at its transformed reference position, sized by the transform's scale.

    A ROI tied to a sub-anchor is positioned relative to where that sub-anchor was found.
    """
    placed: list[PlacedRoi] = []
    subs = sub_anchor_positions or {}
    for roi in profile.rois:
        w = max(1, round(roi.width * scale))
        h = max(1, round(roi.height * scale))
        sub = profile.sub_anchor(roi.sub_anchor) if roi.sub_anchor else None
        if sub is not None and sub.name in subs:
            sx, sy = subs[sub.name]
            x, y = sx + (roi.ref_x - sub.ref_x) * scale, sy + (roi.ref_y - sub.ref_y) * scale
        else:
            x, y = apply(transform, roi.ref_x, roi.ref_y)
        placed.append(PlacedRoi(roi, Rect(round(x), round(y), w, h)))
    return placed


def crop(frame: np.ndarray, rect: Rect) -> np.ndarray:
    """The part of `rect` inside the frame (possibly empty)."""
    area = rect.clip(frame.shape[1], frame.shape[0])
    return frame[area.y : area.bottom, area.x : area.right]
