"""Frozen copy of Rock Capture CNN's preprocessing (commit ab4f06a), the training reference.

Copied verbatim from `digit_crnn/dataset.py` (`_autocrop_text`, `resize_pad`),
`digit_crnn/predictor.py` (`CRNNPredictor._preprocess`) and `core/filters.py`
(`ImageFilterPipeline`), with only the imports adapted. Do not edit: the parity tests compare
the new implementation against this, bit for bit.
"""
# ruff: noqa
# pyright: basic

import cv2
import numpy as np

INPUT_H, INPUT_W = 32, 256


def _autocrop_text(img: np.ndarray) -> np.ndarray:
    """Crop to the bounding box of bright text content (35% of peak brightness)."""
    peak = int(img.max())
    if peak < 10:
        return img
    bright = max(8, round(peak * 0.35))
    col_max = img.max(axis=0)
    row_max = img.max(axis=1)
    cols = np.where(col_max > bright)[0]
    rows = np.where(row_max > bright)[0]
    if cols.size == 0 or rows.size == 0:
        return img
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    return img[y0:y1, x0:x1]


def resize_pad(img: np.ndarray, target_w: int = INPUT_W, target_h: int = INPUT_H) -> np.ndarray:
    img = _autocrop_text(img)
    h, w = img.shape[:2]
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, 0:new_w] = resized
    return canvas


def preprocess(image: np.ndarray) -> np.ndarray | None:
    """CRNNPredictor._preprocess / WordPredictor.predict_all (identical)."""
    if image is None or image.size == 0:
        return None
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if int(gray.max()) - int(gray.min()) < 100:
        return None
    return resize_pad(gray, INPUT_W, INPUT_H)


class ImageFilterPipeline:
    @staticmethod
    def apply(image, settings):
        img = image.copy()
        img = ImageFilterPipeline._apply_brightness(img, settings.brightness)
        img = ImageFilterPipeline._apply_contrast(img, settings.contrast)
        if settings.channel != "none":
            img = ImageFilterPipeline._isolate_channel(img, settings.channel)
        if settings.grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if settings.threshold_enabled:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, settings.threshold, 255, cv2.THRESH_BINARY)
            img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        if settings.invert:
            img = cv2.bitwise_not(img)
        return img

    @staticmethod
    def _apply_brightness(img, value):
        if value == 0:
            return img
        return cv2.convertScaleAbs(img, alpha=1, beta=value)

    @staticmethod
    def _apply_contrast(img, value):
        if value == 0:
            return img
        alpha = 1.0 + (value / 100.0)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    @staticmethod
    def _isolate_channel(img, channel):
        result = img.copy()
        channel_map = {"blue": 0, "green": 1, "red": 2}
        keep = channel_map.get(channel)
        if keep is not None:
            for i in range(3):
                if i != keep:
                    result[:, :, i] = 0
        return result
