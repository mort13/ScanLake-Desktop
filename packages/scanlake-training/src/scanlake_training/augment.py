"""Random augmentation of 32x256 training canvases (from Rock Capture CNN's datasets).

Applied after `resize_pad`, so the model sees the same framing as at inference. The strip and
word presets keep the old ranges (words were shifted and scaled a little more).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np

from scanlake_core.preprocess import Gray


@dataclass(frozen=True)
class Preset:
    shift: float  # max translation as a fraction of width/height
    scale_x: tuple[float, float]
    scale_y: tuple[float, float] = (0.95, 1.05)


STRIP = Preset(shift=0.05, scale_x=(0.93, 1.07))
WORD = Preset(shift=0.08, scale_x=(0.92, 1.08))


def augment(img: Gray, rng: random.Random, preset: Preset = STRIP) -> Gray:
    h, w = img.shape[:2]
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.randint(-30, 30))

    if rng.random() < 0.5:
        sigma = rng.uniform(3, 15)
        noise = np.random.default_rng(rng.getrandbits(32)).normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.4:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if rng.random() < 0.3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = (cv2.erode if rng.random() < 0.5 else cv2.dilate)(img, kernel, iterations=1)

    if rng.random() < 0.6:
        m = np.array(
            [
                [rng.uniform(*preset.scale_x), 0, rng.uniform(-preset.shift, preset.shift) * w],
                [0, rng.uniform(*preset.scale_y), rng.uniform(-preset.shift, preset.shift) * h],
            ],
            dtype=np.float32,
        )
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return np.asarray(img)
