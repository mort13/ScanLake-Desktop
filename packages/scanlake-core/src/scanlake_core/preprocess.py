"""ROI crop -> 32x256 model input. One implementation for training and inference.

Bit-exact with Rock Capture CNN's `_autocrop_text` + `resize_pad` (see
tests/reference/legacy_preprocess.py), which the current models were trained with:

1. grayscale (OpenCV BGR2GRAY)
2. reject low-contrast crops (max - min < 100): blank slot or rock texture, not HUD text
3. crop to the pixels brighter than max(8, round(35% of peak))
4. INTER_AREA resize to fit 32x256 keeping aspect ratio, left-aligned,
   vertically centred with floor((32 - h) / 2), black padding
5. divide by 255 (no mean/std normalization)
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

INPUT_H, INPUT_W = 32, 256
MIN_CONTRAST = 100
AUTOCROP_FRACTION = 0.35
PREPROCESS_VERSION = "strip_v1"

Gray = npt.NDArray[Any]  # uint8, 2-D (OpenCV's stubs do not carry dtypes)


def to_gray(image: np.ndarray) -> Gray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def has_contrast(gray: Gray) -> bool:
    return int(gray.max()) - int(gray.min()) >= MIN_CONTRAST


def autocrop_text(gray: Gray) -> Gray:
    peak = int(gray.max())
    if peak < 10:
        return gray
    bright = max(8, round(peak * AUTOCROP_FRACTION))
    cols = np.flatnonzero(gray.max(axis=0) > bright)
    rows = np.flatnonzero(gray.max(axis=1) > bright)
    if cols.size == 0 or rows.size == 0:
        return gray
    return gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def resize_pad(gray: Gray, target_w: int = INPUT_W, target_h: int = INPUT_H) -> Gray:
    """Autocrop, then fit into target_w x target_h (left-aligned, vertically centred)."""
    img = autocrop_text(gray)
    h, w = img.shape[:2]
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, 0:new_w] = resized
    return canvas


def preprocess_strip(image: np.ndarray) -> Gray | None:
    """The uint8 32x256 canvas for a BGR or gray crop, or None for a blank/low-contrast crop."""
    if image.size == 0:
        return None
    gray = to_gray(image)
    if not has_contrast(gray):
        return None
    return resize_pad(gray)


def to_input(canvases: list[Gray], batch: int | None = None) -> npt.NDArray[np.float32]:
    """Stack canvases into a float32 [N, 1, 32, 256] batch in 0..1 (zero-padded to `batch`)."""
    n = max(len(canvases), batch or 0)
    out = np.zeros((n, 1, INPUT_H, INPUT_W), dtype=np.float32)
    for i, canvas in enumerate(canvases):
        out[i, 0] = canvas.astype(np.float32) / 255.0
    return out
