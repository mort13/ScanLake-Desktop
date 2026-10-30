"""Rectangles and 2x3 affine transforms (OpenCV convention: x right, y down)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# Profiles are authored on a 2560x1440 screen; ROI and region positions are in that frame.
REFERENCE_HEIGHT = 1440

Affine = npt.NDArray[Any]  # float64, shape (2, 3)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def clip(self, width: int, height: int) -> Rect:
        """This rect intersected with (0, 0, width, height); may come out empty."""
        x0, y0 = max(self.x, 0), max(self.y, 0)
        x1, y1 = min(self.right, width), min(self.bottom, height)
        return Rect(x0, y0, max(x1 - x0, 0), max(y1 - y0, 0))

    def pad(self, fraction: float) -> Rect:
        """Grown by `fraction` of its size on every side."""
        dx, dy = round(self.w * fraction), round(self.h * fraction)
        return Rect(self.x - dx, self.y - dy, self.w + 2 * dx, self.h + 2 * dy)

    def scaled(self, factor: float) -> Rect:
        return Rect(
            round(self.x * factor), round(self.y * factor),
            round(self.w * factor), round(self.h * factor),
        )  # fmt: skip


def identity() -> Affine:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def apply(m: Affine, x: float, y: float) -> tuple[float, float]:
    return (
        float(m[0, 0] * x + m[0, 1] * y + m[0, 2]),
        float(m[1, 0] * x + m[1, 1] * y + m[1, 2]),
    )


def uniform_scale(m: Affine) -> float:
    """The average axis scale of a similarity/affine transform."""
    sx = float(np.hypot(m[0, 0], m[1, 0]))
    sy = float(np.hypot(m[0, 1], m[1, 1]))
    return (sx + sy) / 2.0
