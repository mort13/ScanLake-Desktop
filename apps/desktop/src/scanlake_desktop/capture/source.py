"""Frame sources: the screen (mss) or image files (tests, replays, dev mode)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from scanlake_core.geometry import REFERENCE_HEIGHT, Rect

REFERENCE_WIDTH = 2560
BLACK_LEVEL = 6  # max pixel value below which a frame counts as black (fullscreen-exclusive)


@dataclass(frozen=True)
class Monitor:
    index: int  # mss numbering, from 1 (not necessarily the primary monitor)
    left: int
    top: int
    width: int
    height: int
    primary: bool = False

    @property
    def rect(self) -> Rect:
        return Rect(self.left, self.top, self.width, self.height)

    @property
    def scale(self) -> float:
        """HUD scale relative to the 2560x1440 reference (the HUD scales with screen height)."""
        return self.height / REFERENCE_HEIGHT

    def label(self) -> str:
        primary = " (primary)" if self.primary else ""
        return f"Monitor {self.index}: {self.width}x{self.height}{primary}"


def list_monitors() -> list[Monitor]:
    """The attached monitors ([] when the screen cannot be read, e.g. headless)."""
    import mss

    try:
        sct = mss.MSS()
    except Exception:
        return []
    with sct:
        monitors = [
            Monitor(
                i, m["left"], m["top"], m["width"], m["height"], bool(m.get("is_primary", False))
            )
            for i, m in enumerate(sct.monitors)
            if i > 0
        ]
    if monitors and not any(m.primary for m in monitors):
        # mss without primary info: Windows puts the primary monitor's origin at (0, 0)
        origin = next((m for m in monitors if (m.left, m.top) == (0, 0)), monitors[0])
        monitors = [replace(m, primary=m is origin) for m in monitors]
    return monitors


def primary_monitor(monitors: Sequence[Monitor]) -> Monitor:
    return next((m for m in monitors if m.primary), monitors[0])


def auto_region(search_region: Rect, monitor: Monitor, pad: float = 0.25) -> Rect:
    """Where the profile's search region lands on `monitor`, padded and clipped.

    Positions are scaled with the monitor height around the screen centre, which is plain
    scaling on 16:9 screens; on other aspect ratios the full-monitor search corrects it.
    """
    s = monitor.scale
    cx = monitor.width / 2 + (search_region.x - REFERENCE_WIDTH / 2) * s
    y = search_region.y * s
    local = Rect(round(cx), round(y), round(search_region.w * s), round(search_region.h * s))
    local = local.pad(pad).clip(monitor.width, monitor.height)
    return Rect(local.x + monitor.left, local.y + monitor.top, local.w, local.h)


def is_black(frame: np.ndarray) -> bool:
    return frame.size == 0 or int(frame.max()) < BLACK_LEVEL


class CaptureSource(Protocol):
    def grab(self, rect: Rect) -> np.ndarray:
        """BGR uint8 pixels of `rect` (virtual-desktop coordinates)."""
        ...

    def close(self) -> None: ...


class MssSource:
    """Screen capture with mss. mss handles are thread-bound: create it in the thread that grabs."""

    def __init__(self) -> None:
        import mss

        self._sct = mss.MSS()

    def grab(self, rect: Rect) -> np.ndarray:
        shot = self._sct.grab({"left": rect.x, "top": rect.y, "width": rect.w, "height": rect.h})
        # mss returns BGRA; dropping alpha gives correct BGR (the old app swapped R and B)
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])

    def close(self) -> None:
        self._sct.close()


class FileSource:
    """Replays image files in a loop; `rect` is ignored (the files are already the region)."""

    def __init__(self, files: Sequence[Path]) -> None:
        if not files:
            raise ValueError("no frames to replay")
        self._frames = [read_image(f) for f in files]
        self._next = 0

    def grab(self, rect: Rect) -> np.ndarray:
        frame = self._frames[self._next % len(self._frames)]
        self._next += 1
        return frame

    def close(self) -> None:
        pass


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return np.asarray(image)


def write_image(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"cannot encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(str(path))
