"""Per-ROI image filters (the canonical Python semantics the models were trained with).

Order: brightness, contrast, channel isolation, grayscale, threshold, invert. Input and output
are BGR uint8, like Rock Capture CNN's `ImageFilterPipeline`.
"""

from __future__ import annotations

import cv2
import numpy as np

from scanlake_core.profile.model import FilterSettings

Image = np.ndarray

_CHANNELS = {"blue": 0, "green": 1, "red": 2}


def apply_filters(image: Image, settings: FilterSettings) -> Image:
    img = image
    if settings.brightness:
        img = cv2.convertScaleAbs(img, alpha=1, beta=settings.brightness)
    if settings.contrast:
        img = cv2.convertScaleAbs(img, alpha=1.0 + settings.contrast / 100.0, beta=0)

    keep = _CHANNELS.get(settings.channel)
    if keep is not None:
        img = img.copy()
        for i in range(3):
            if i != keep:
                img[:, :, i] = 0

    if settings.grayscale:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if settings.threshold_enabled:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, settings.threshold, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    if settings.invert:
        img = cv2.bitwise_not(img)
    return img if img is not image else image.copy()
