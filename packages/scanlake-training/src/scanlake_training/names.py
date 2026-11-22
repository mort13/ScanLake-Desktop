"""The letter model for deposit/material names: training samples, synthetic strips, task.

The model is a DigitCRNN over `scanlake_core.names.NAME_CHARS`; recognition matches its read
against the known names (scanlake_core/names.py). Samples are strips named `<text>_<n>.png`
like digit strips ("COPPER (ORE)_000123.png"); `blank_<n>.png` is an empty row.

Synthetic strips are composed from letter glyph sets: folders of `<stem>_<anything>.png`, text
bright on black, one character each (stems A-Z, "lparen", "rparen", "space"; a set without a
space glyph gets a gap). Glyphs cut from real HUD text keep the game's fonts; glyphs rendered
from other fonts teach letters the HUD has not shown yet. A strip is a form from the lexicon
(random spelling, suffix, prefix) or random letters, drawn with glow on a background crop, and
sometimes clipped like the HUD's scrolling window: its label then holds only the characters
that are mostly visible.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch import nn

from scanlake_core.names import NAME_CHARS, lexicon
from scanlake_models import checkpoint
from scanlake_training.data import IMAGE_SUFFIXES, Sample, load_gray
from scanlake_training.tasks import CrnnTask

BLANK = "blank"  # file-name label of an empty row
STEMS = {"(": "lparen", ")": "rparen", " ": "space"}
CHAR_OF = {v: k for k, v in STEMS.items()}

Glyphs = dict[str, list[np.ndarray]]


def name_samples(folder: Path) -> list[Sample]:
    """`<text>_<n>.png` samples whose text is in NAME_CHARS; `blank_<n>.png` has text ""."""
    out = []
    if not folder.exists():
        return out
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label, sep, _ = path.stem.rpartition("_")
        if not sep:
            continue
        if label == BLANK:
            out.append(Sample(path, ""))
        elif label and all(c in NAME_CHARS for c in label):
            out.append(Sample(path, label))
    return out


class NameTask(CrnnTask):
    """CTC over NAME_CHARS; starts from any CRNN checkpoint's backbone (e.g. the digit model)."""

    name = "names"

    def __init__(self) -> None:
        super().__init__(NAME_CHARS)

    def init_from(self, model: nn.Module, stem: Path) -> None:
        old, _ = checkpoint.load(stem)
        state = old.state_dict()
        if state["fc.weight"].shape != model.state_dict()["fc.weight"].shape:
            state = {k: v for k, v in state.items() if not k.startswith("fc.")}  # other classes
        model.load_state_dict(state, strict=False)

    def meta(self, metrics: dict[str, float]) -> dict[str, Any]:
        meta = super().meta(metrics)
        meta.pop("formats", None)
        meta["modelType"] = "name_crnn"
        return meta


# -- synthetic strips ----------------------------------------------------------------------


def load_glyphs(folder: Path) -> Glyphs:
    glyphs: Glyphs = {}
    for path in sorted(folder.glob("*.png")):
        stem = path.stem.split("_")[0]
        ch = CHAR_OF.get(stem, stem)
        if ch in NAME_CHARS:
            glyphs.setdefault(ch, []).append(load_gray(path))
    return glyphs


def load_backgrounds(folders: Sequence[Path]) -> list[np.ndarray]:
    out = []
    for folder in folders:
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                out.append(load_gray(path))
    return out


def random_text(rng: random.Random, chars: str) -> str:
    """A lexicon form (spelling with or without its suffix/prefix) or, sometimes, random letters
    (so the model reads letters, not just the known words)."""
    letters = "".join(c for c in chars if c.isalpha())
    if rng.random() < 0.3 and letters:
        words = [
            "".join(rng.choice(letters) for _ in range(rng.randint(2, 9)))
            for _ in range(rng.choice([1, 1, 2]))
        ]
        text = " ".join(words)
        if rng.random() < 0.4 and "(" in chars and ")" in chars:
            text += " (" + "".join(rng.choice(letters) for _ in range(3)) + ")"
        return text
    lex = lexicon()
    for _ in range(50):
        i = rng.randrange(len(lex.forms))
        form, (a, b) = lex.forms[i], lex.cores[i]
        start = 0 if rng.random() < 0.15 else a  # the prefix ("RAW ") is rare on screen
        end = len(form) if rng.random() < 0.75 else b
        text = form[start:end]
        if all(c in chars for c in text):
            return text
    return "".join(rng.choice(letters) for _ in range(rng.randint(3, 8)))


def _ink(text: str, glyphs: Glyphs, height: int, rng: random.Random) -> tuple[np.ndarray, list]:
    """Text as an ink map (0..1, height rows) and each character's column span."""
    parts: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    tracking = rng.choice([-1, 0, 0, 0, 1])  # extra px between characters
    x = 0
    for ch in text:
        if ch in glyphs:
            g = rng.choice(glyphs[ch]).astype(np.float32)
            w = max(1, round(g.shape[1] * height / g.shape[0]))
            g = cv2.resize(g, (w, height), interpolation=cv2.INTER_AREA)
            g = g / max(float(g.max()), 1.0)
        else:  # a space without a glyph
            g = np.zeros((height, max(2, round(height * 0.3))), np.float32)
        parts.append(g)
        spans.append((x, x + g.shape[1]))
        x += g.shape[1] + tracking
    width = max(1, x - tracking)
    ink = np.zeros((height, width + 2), np.float32)
    for g, (a, _) in zip(parts, spans, strict=True):
        a = max(0, a)
        w = min(g.shape[1], ink.shape[1] - a)
        ink[:, a : a + w] = np.maximum(ink[:, a : a + w], g[:, :w])
    return ink[:, :width], spans


def _background(backgrounds: list[np.ndarray], h: int, w: int, rng: random.Random) -> np.ndarray:
    bg = rng.choice(backgrounds)
    if bg.shape[0] < h or bg.shape[1] < w:
        scale = max(h / bg.shape[0], w / bg.shape[1]) * rng.uniform(1.0, 1.5)
        bg = cv2.resize(bg, (round(bg.shape[1] * scale) + 1, round(bg.shape[0] * scale) + 1))
    y = rng.randint(0, bg.shape[0] - h)
    x = rng.randint(0, bg.shape[1] - w)
    out = bg[y : y + h, x : x + w].astype(np.float32)
    if rng.random() < 0.5:  # the HUD's translucent panel behind the rows
        out = out * rng.uniform(0.25, 0.8)
    return out


def compose_name(
    text: str, glyphs: Glyphs, backgrounds: list[np.ndarray], rng: random.Random
) -> tuple[np.ndarray, str]:
    """A ROI-like strip showing `text` and its label (the mostly visible characters)."""
    height = rng.randint(12, 18)  # glyph rows (cap height + margins), as at HUD scales ~0.8-1.3
    ink, spans = _ink(text, glyphs, height, rng)
    label = text.strip()
    if rng.random() < 0.2 and len(text) >= 6:  # the scrolling window of a long name
        window = rng.randint(max(4 * height // 3, ink.shape[1] // 2), ink.shape[1] - 2)
        for _ in range(10):
            left = rng.randint(0, ink.shape[1] - window)
            seen = [(min(b, left + window) - max(a, left)) / max(1, b - a) for a, b in spans]
            visible = "".join(c for c, s in zip(text, seen, strict=True) if s >= 0.75).strip()
            # every character clearly in or out, and still a few letters to read
            if all(s <= 0.35 or s >= 0.75 for s in seen) and sum(c.isalpha() for c in visible) >= 3:
                label = visible
                ink = ink[:, left : left + window]
                break
    pad_top, pad_bottom = rng.randint(0, 3), rng.randint(0, 3)
    pad_left = rng.randint(0, 6)
    pad_right = rng.randint(0, 6) if rng.random() < 0.5 else rng.randint(6, 40)
    h = height + pad_top + pad_bottom
    w = ink.shape[1] + pad_left + pad_right
    canvas_ink = np.zeros((h, w), np.float32)
    canvas_ink[pad_top : pad_top + height, pad_left : pad_left + ink.shape[1]] = ink
    if rng.random() < 0.7:  # glow
        sigma = rng.uniform(0.8, 2.5)
        glow = cv2.GaussianBlur(canvas_ink, (0, 0), sigma) * rng.uniform(0.2, 0.8)
        canvas_ink = np.clip(np.maximum(canvas_ink, glow), 0, 1)
    if rng.random() < 0.3:
        canvas_ink = cv2.GaussianBlur(canvas_ink, (0, 0), rng.uniform(0.3, 0.7))
    bg = _background(backgrounds, h, w, rng)
    level = rng.uniform(110, 255)  # white, the purple deposit, the red Quantainium
    # the HUD's text always stands out: darken a background the text would drown in
    behind = float(np.percentile(bg, 60))
    contrast = rng.uniform(60, 120)
    if level - behind < contrast:
        bg = bg * max(0.0, level - contrast) / max(behind, 1.0)
    if rng.random() < 0.3:  # a soft dark shadow under the letters (bright snow)
        shadow = cv2.GaussianBlur(canvas_ink, (0, 0), rng.uniform(0.8, 1.6))
        bg = bg * (1 - np.clip(shadow * rng.uniform(0.4, 0.9), 0, 1))
    strip = bg * (1 - canvas_ink) + level * canvas_ink
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(0, 4), strip.shape)
    return np.clip(strip + noise, 0, 255).round().astype(np.uint8), label


def empty_strip(backgrounds: list[np.ndarray], rng: random.Random) -> np.ndarray:
    h, w = rng.randint(14, 22), rng.randint(60, 150)
    return np.clip(_background(backgrounds, h, w, rng), 0, 255).astype(np.uint8)


def generate_names(
    out: Path,
    count: int,
    glyph_sets: Sequence[tuple[Glyphs, float]],
    backgrounds: list[np.ndarray],
    seed: int | None = None,
    empty_share: float = 0.04,
) -> int:
    """Writes `count` strips into `out` (`<label>_<n>.png`, `blank_<n>.png`); glyph sets are
    drawn by weight, each writing only text it has every letter of."""
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    sets = [g for g, _ in glyph_sets]
    weights = [w for _, w in glyph_sets]
    for i in range(count):
        if rng.random() < empty_share:
            image, label = empty_strip(backgrounds, rng), ""
        else:
            glyphs = rng.choices(sets, weights)[0]
            chars = "".join(c for c in NAME_CHARS if c in glyphs) + " "
            image, label = compose_name(random_text(rng, chars), glyphs, backgrounds, rng)
        name = label if label else BLANK
        cv2.imwrite(str(out / f"{name}_{i:06d}.png"), image)
    return count
