"""ROI crop -> 32x256 model input. One implementation for training and inference.

Every model's `.json` names the preprocessing it was trained with ("preprocess"); a model must be
fed with exactly that one. Models without the field are `strip_v1`.

1. grayscale (OpenCV BGR2GRAY)
2. reject low-contrast crops (max - min < 100): blank slot or rock texture, not HUD text
3. crop to the text:
   strip_v1  pixels brighter than max(8, round(35 % of peak)). Bit-exact with Rock Capture CNN's
             `_autocrop_text` + `resize_pad` (see tests/reference/legacy_preprocess.py). On grey
             or bright rock the background itself passes the threshold, so nothing is cropped and
             the text fills only part of the canvas.
   strip_v2  pixels brighter than bg + 35 % of (peak - bg), integer arithmetic, where bg is the
             20th-percentile pixel (the (n // 5)-th smallest of n, counting from 0). On dark
             rock about the same crop as v1; on grey rock the text fills the canvas height.
4. INTER_AREA resize to fit 32x256 keeping aspect ratio, left-aligned,
   vertically centred with floor((32 - h) / 2), black padding
5. divide by 255 (no mean/std normalization)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

INPUT_H, INPUT_W = 32, 256
MIN_CONTRAST = 100
AUTOCROP_FRACTION = 0.35
AUTOCROP_PERCENT = 35  # strip_v2: the same fraction, as an integer
BACKGROUND_PERCENTILE = 20  # strip_v2: share of the crop below the background level

STRIP_V1 = "strip_v1"
STRIP_V2 = "strip_v2"
PREPROCESS_VERSIONS = (STRIP_V1, STRIP_V2)
LEGACY_PREPROCESS = STRIP_V1  # models whose metadata names no preprocessing
PREPROCESS_VERSION = STRIP_V2  # what training uses and writes into new models' metadata

Gray = npt.NDArray[Any]  # uint8, 2-D (OpenCV's stubs do not carry dtypes)


def check_version(version: str) -> str:
    """`version` if this code can run it, else ValueError (a model newer than the app)."""
    if version not in PREPROCESS_VERSIONS:
        known = ", ".join(PREPROCESS_VERSIONS)
        raise ValueError(f"unknown preprocessing {version!r} (this version knows {known})")
    return version


def preprocess_of(meta: Mapping[str, Any]) -> str:
    """The preprocessing named in a model's metadata (`<model>.json`), checked."""
    return check_version(str(meta.get("preprocess", LEGACY_PREPROCESS)))


def to_gray(image: np.ndarray) -> Gray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def has_contrast(gray: Gray) -> bool:
    return int(gray.max()) - int(gray.min()) >= MIN_CONTRAST


def background_level(gray: Gray) -> int:
    """strip_v2's background: the (n * 20 // 100)-th smallest pixel value of the n."""
    flat = gray.ravel()
    k = flat.size * BACKGROUND_PERCENTILE // 100
    return int(np.partition(flat, k)[k])


def text_threshold(gray: Gray, version: str) -> int:
    """Pixels brighter than this are text (the crop keeps the rows and columns that have one)."""
    peak = int(gray.max())
    if check_version(version) == STRIP_V1:
        return max(8, round(peak * AUTOCROP_FRACTION))
    bg = background_level(gray)
    return max(8, bg + (peak - bg) * AUTOCROP_PERCENT // 100)


def autocrop_text(gray: Gray, *, version: str) -> Gray:
    if int(gray.max()) < 10:
        return gray
    bright = text_threshold(gray, version)
    cols = np.flatnonzero(gray.max(axis=0) > bright)
    rows = np.flatnonzero(gray.max(axis=1) > bright)
    if cols.size == 0 or rows.size == 0:
        return gray
    return gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def resize_pad(
    gray: Gray, *, version: str, target_w: int = INPUT_W, target_h: int = INPUT_H
) -> Gray:
    """Autocrop, then fit into target_w x target_h (left-aligned, vertically centred)."""
    img = autocrop_text(gray, version=version)
    h, w = img.shape[:2]
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, 0:new_w] = resized
    return canvas


def preprocess_strip(image: np.ndarray, *, version: str) -> Gray | None:
    """The uint8 32x256 canvas for a BGR or gray crop, or None for a blank/low-contrast crop.

    `version` is the model's (its metadata's "preprocess"): models only read their own."""
    check_version(version)
    if image.size == 0:
        return None
    gray = to_gray(image)
    if not has_contrast(gray):
        return None
    return resize_pad(gray, version=version)


def to_input(canvases: list[Gray], batch: int | None = None) -> npt.NDArray[np.float32]:
    """Stack canvases into a float32 [N, 1, 32, 256] batch in 0..1 (zero-padded to `batch`)."""
    n = max(len(canvases), batch or 0)
    out = np.zeros((n, 1, INPUT_H, INPUT_W), dtype=np.float32)
    for i, canvas in enumerate(canvases):
        out[i, 0] = canvas.astype(np.float32) / 255.0
    return out
