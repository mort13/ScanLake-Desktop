"""Synthetic training data (ported from Rock Capture CNN's generate_synth_*.py).

Strips: digit glyphs composited side by side on HUD background crops, in the formats the HUD
shows (12.34, 56%, 123456, 12.34%, and 25.26m for volumes under 1 SCU). Words: a resource/deposit
name template (sometimes followed by "ore"/"raw") on a background crop; the `empty` class is
background only.

The glyphs and templates in `resources/` were cut from Rock Capture CNN captures, which swapped
the red and blue channels; that only shifts the grayscale slightly and is kept for parity.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

from scanlake_core.preprocess import Gray

STRIP_H = 32
GLYPH_FILES = {
    **{str(d): f"{d}.png" for d in range(10)},
    ".": "dot.png",
    "%": "percent.png",
    "m": "m.png",
}
LOWERCASE = frozenset("m")  # x-height letters: scaled to the digits, on their baseline
X_HEIGHT = 8 / 11  # the "m" vs the digits' height, measured on the HUD ("25.26m SCU")


def resource_dir() -> Path:
    return Path(str(resources.files("scanlake_training").joinpath("resources")))


def _read(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return np.asarray(image)


def _write(path: Path, image: Gray) -> None:
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"cannot encode {path}")
    data.tofile(str(path))


def _bgra(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def load_glyph(path: Path) -> np.ndarray:
    """A glyph as white BGRA with its shape in alpha, cropped to the ink.

    Glyphs with a soft alpha (the original ones) are used as they are. A glyph that is just a
    rectangle cut from a screenshot (alpha only 0/255, background included, possibly on a
    larger transparent canvas) gets its alpha from the brightness above that background.
    """
    image = _bgra(_read(path))
    alpha = image[:, :, 3]
    if set(np.unique(alpha).tolist()) <= {0, 255}:
        ys, xs = np.nonzero(alpha)
        box = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        gray = cv2.cvtColor(box[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        background, peak = float(np.percentile(gray, 20)), float(gray.max())
        soft = np.clip((gray - background) / max(peak - background, 1.0), 0, 1) * 255
        rows = np.nonzero(soft.max(axis=1) >= 0.35 * 255)[0]
        cols = np.nonzero(soft.max(axis=0) >= 0.35 * 255)[0]
        soft = soft[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        image = np.zeros((*soft.shape, 4), np.uint8)
        image[:, :, :3] = 255
        image[:, :, 3] = soft.round().astype(np.uint8)
    return image


def load_backgrounds(root: Path) -> list[np.ndarray]:
    folder = root / "backgrounds"
    images = [_read(p, cv2.IMREAD_COLOR) for p in sorted(folder.iterdir()) if p.is_file()]
    if not images:
        raise FileNotFoundError(f"no background images in {folder}")
    return images


def _crop(image: np.ndarray, h: int, w: int, rng: random.Random) -> np.ndarray:
    ih, iw = image.shape[:2]
    if ih < h or iw < w:
        image = np.tile(image, (h // ih + 2, w // iw + 2, 1))
        ih, iw = image.shape[:2]
    y, x = rng.randint(0, ih - h), rng.randint(0, iw - w)
    return image[y : y + h, x : x + w].copy()


def photometric(img: Gray, rng: random.Random) -> Gray:
    """The generators' own augmentation: contrast/brightness, noise, blur."""
    noise_rng = np.random.default_rng(rng.getrandbits(32))
    out = np.clip(rng.uniform(0.75, 1.30) * img.astype(np.float32) + rng.randint(-25, 25), 0, 255)
    if rng.random() < 0.5:
        out = np.clip(out + noise_rng.normal(0.0, rng.uniform(2.0, 10.0), out.shape), 0, 255)
    out = out.astype(np.uint8)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    return np.asarray(out)


# -- strips ---------------------------------------------------------------------


def _decimal(rng: random.Random) -> str:
    return f"{rng.randint(0, 999)}.{rng.randint(0, 99):02d}"


def _percent(rng: random.Random) -> str:
    return f"{rng.randint(0, 99)}%"


def _integer(rng: random.Random) -> str:
    n = rng.randint(1, 6)
    if n == 1:
        return str(rng.randint(0, 9))
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


def _decimal_percent(rng: random.Random) -> str:
    return f"{rng.randint(0, 99)}.{rng.randint(0, 99):02d}%"


def _milli(rng: random.Random) -> str:
    return f"{rng.randint(0, 999)}.{rng.randint(0, 99):02d}m"


STRIP_FORMATS: dict[str, Callable[[random.Random], str]] = {
    "decimal": _decimal,
    "percent": _percent,
    "integer": _integer,
    "decimal_percent": _decimal_percent,
    "milli": _milli,
}


def compose_strip(
    text: str, glyphs: dict[str, np.ndarray], background: np.ndarray, rng: random.Random
) -> Gray:
    """Alpha-composites the glyphs of `text` on a background crop; the dot and lowercase letters
    sit on the baseline."""
    glyph_h = max(4, int(STRIP_H * rng.uniform(0.65, 0.95)))
    # The digit glyphs carry a faint row or two around the ink: measure the ink of "0"
    digit = glyphs["0"]
    ink = np.nonzero(digit[:, :, 3].max(axis=1) >= 0.35 * digit[:, :, 3].max())[0]
    digit_scale = glyph_h / digit.shape[0]
    baseline = (STRIP_H - glyph_h) // 2 + round((ink.max() + 1) * digit_scale)
    x_height = (ink.max() - ink.min() + 1) * digit_scale * X_HEIGHT
    # ... and are not fully opaque: letters cut from a frame get the digits' typical opacity
    opacity = float(np.median([glyphs[str(d)][:, :, 3].max() for d in range(10)])) / 255
    placed: list[tuple[np.ndarray, int]] = []  # (glyph, top y)
    for ch in text:
        src = glyphs[ch]
        sh, sw = src.shape[:2]
        if ch in LOWERCASE:  # tight glyphs (see load_glyph): their height is the x-height
            factor = x_height / sh
            img = cv2.resize(
                src,
                (max(1, round(sw * factor)), max(1, round(sh * factor))),
                interpolation=cv2.INTER_AREA,
            )
            img[:, :, 3] = (img[:, :, 3] * opacity).round().astype(np.uint8)
            top = baseline - img.shape[0]
        elif ch == ".":
            dot_h = max(2, round(glyph_h * 0.4))
            img = cv2.resize(
                src, (max(1, round(sw * dot_h / sh)), dot_h), interpolation=cv2.INTER_AREA
            )
            top = STRIP_H - glyph_h + max(0, glyph_h - dot_h - round(glyph_h * 0.1))
        else:
            img = cv2.resize(
                src, (max(1, round(sw * glyph_h / sh)), glyph_h), interpolation=cv2.INTER_AREA
            )
            top = (STRIP_H - glyph_h) // 2
        placed.append((img, top))

    gaps = [rng.randint(0, 2) for _ in range(len(placed) - 1)]
    width = max(sum(g.shape[1] for g, _ in placed) + sum(gaps) + 2, 16)
    canvas = _crop(background, STRIP_H, width, rng).astype(np.float32)
    x = 0
    for i, (glyph, top) in enumerate(placed):
        gh, gw = glyph.shape[:2]
        y0, y1 = max(0, top), min(STRIP_H, top + gh)
        g0 = y0 - top
        x1 = min(width, x + gw)
        alpha = glyph[g0 : g0 + (y1 - y0), : x1 - x, 3:4].astype(np.float32) / 255.0
        color = glyph[g0 : g0 + (y1 - y0), : x1 - x, :3].astype(np.float32)
        canvas[y0:y1, x:x1] = alpha * color + (1 - alpha) * canvas[y0:y1, x:x1]
        x += gw + (gaps[i] if i < len(gaps) else 0)
    return np.asarray(cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY))


def generate_strips(
    out: Path, per_format: int, seed: int | None = None, root: Path | None = None
) -> int:
    """Writes `per_format` strips per format to `out` as `<label>_<n>.png`; returns the count."""
    root = root or resource_dir()
    rng = random.Random(seed)
    glyphs = {ch: load_glyph(root / "glyphs" / name) for ch, name in GLYPH_FILES.items()}
    backgrounds = load_backgrounds(root)
    out.mkdir(parents=True, exist_ok=True)
    start = _next_index(out)
    n = 0
    for make in STRIP_FORMATS.values():
        for _ in range(per_format):
            text = make(rng)
            strip = photometric(compose_strip(text, glyphs, rng.choice(backgrounds), rng), rng)
            _write(out / f"{text}_{start + n:06d}.png", strip)
            n += 1
    return n


def _next_index(out: Path) -> int:
    indices = []
    for p in out.glob("*_*.png"):
        tail = p.stem.rpartition("_")[2]
        if tail.isdigit():
            indices.append(int(tail))
    return max(indices, default=-1) + 1


# -- words ----------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    gray: np.ndarray
    alpha: np.ndarray  # 0..1


def _template(path: Path) -> Template:
    img = _read(path)
    if img.ndim == 3 and img.shape[2] == 4:
        return Template(
            cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY), img[:, :, 3].astype(np.float32) / 255.0
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return Template(gray, gray.astype(np.float32) / 255.0)  # brightness as alpha


def word_templates(root: Path | None = None) -> dict[str, Template]:
    root = root or resource_dir()
    return {p.stem: _template(p) for p in sorted((root / "words").glob("*.png"))}


def compose_word(
    parts: list[Template],
    background: np.ndarray,
    rng: random.Random,
    gap: int = 6,
    pad_v: int = 4,
    pad_h: int = 6,
) -> Gray:
    """`parts` side by side (a word, optionally a suffix), scaled alike, on a background crop."""
    scale = rng.uniform(2.0, 3.5)
    scaled = []
    for t in parts:
        h, w = t.gray.shape
        size = (max(4, round(w * scale)), max(4, round(h * scale)))
        scaled.append(
            (
                cv2.resize(t.gray, size, interpolation=cv2.INTER_LINEAR).astype(np.float32),
                cv2.resize(t.alpha, size, interpolation=cv2.INTER_LINEAR),
            )
        )
    height = max(g.shape[0] for g, _ in scaled) + 2 * pad_v
    width = sum(g.shape[1] for g, _ in scaled) + gap * (len(scaled) - 1) + 2 * pad_h
    canvas = cv2.cvtColor(_crop(background, height, width, rng), cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    x = pad_h
    for gray, alpha in scaled:
        h, w = gray.shape
        y = pad_v + (height - 2 * pad_v - h) // 2
        canvas[y : y + h, x : x + w] = alpha * gray + (1 - alpha) * canvas[y : y + h, x : x + w]
        x += w + gap
    return np.asarray(np.clip(canvas, 0, 255).astype(np.uint8))


def generate_words(
    out: Path, per_class: int, seed: int | None = None, root: Path | None = None
) -> dict[str, int]:
    """Writes `per_class` images per word class into `out/<class>/`; returns counts per class.

    Words listed in suffix/suffix_mapping.json get a suffix ("ore"/"raw") on half their images.
    """
    root = root or resource_dir()
    rng = random.Random(seed)
    templates = word_templates(root)
    suffixes = {p.stem: _template(p) for p in (root / "suffix").glob("*.png")}
    mapping_file = root / "suffix" / "suffix_mapping.json"
    mapping = json.loads(mapping_file.read_text(encoding="utf-8")) if mapping_file.exists() else {}
    backgrounds = load_backgrounds(root)
    counts: dict[str, int] = {}
    for name, template in templates.items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        start = _next_index(folder)
        options = [s for s in mapping.get(name, []) if s in suffixes]
        for i in range(per_class):
            background = rng.choice(backgrounds)
            if name == "empty":
                image = cv2.cvtColor(_crop(background, 40, 200, rng), cv2.COLOR_BGR2GRAY)
            elif options and i % 2 == 1:
                image = compose_word([template, suffixes[rng.choice(options)]], background, rng)
            else:
                image = compose_word([template], background, rng)
            _write(folder / f"synth_{start + i:06d}.png", photometric(np.asarray(image), rng))
        counts[name] = per_class
    return counts
