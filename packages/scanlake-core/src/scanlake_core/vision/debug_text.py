"""Reads the game's debug text (r_DisplayInfo) from the top-right corner of the screen.

The text is drawn in a fixed-width bitmap font, plain white with a dark shadow, right-aligned at
the top right; it never shakes, scales or glows. That makes an exact reader simple:

- a `DebugFont` holds one template per glyph, cut from real screenshots (`scanlake_training.
  debug_font` builds them), with the font's advance (cell pitch);
- the line prefixes "CamDir:" and "Zone:" are found by template matching, which gives each
  line's position and grid phase (with a fractional advance such as 7.5 px the cells step
  7, 8, 7, 8 ... or 8, 7, 8, 7 ... px, depending on where the right-aligned line starts);
- every cell after a prefix is classified by normalised correlation against all glyphs;
- inside numbers, the look-alikes O, I and l are read as 0 and 1.

Only CamDir and Zone lines are read; the rest of the debug text (session, server, ...) is
never looked at. Images are compared by "whiteness" (the darkest of B, G, R), so the coloured
HUD behind the text counts as dark.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import cv2
import numpy as np

FONT_DIR = Path(__file__).resolve().parent.parent / "resources" / "display_info_font"
PREFIXES = ("CamDir:", "Zone:")
UNKNOWN = "?"  # a cell with ink that matches no glyph

CORNER_W = 0.5  # the searched corner: this share of the screen width, from the right
CORNER_H = 0.25  # ... and of its height, from the top
INK = 170  # whiteness a glyph pixel reaches at least ...
INK_CONTRAST = 80  # ... or how far it stands above its cell (a thin "_" may stay dim)
PREFIX_MIN = 0.75  # correlation that makes a prefix match
GLYPH_MIN = 0.6  # below this a cell is UNKNOWN
MAX_BLANK = 3  # blank cells in a row end a line ("CamDir: -155  45 -179" has two)
MAX_UNKNOWN = 3  # unknown cells in a row end a line (we ran into the HUD)
COARSE = 2  # prefixes are first looked for at 1/COARSE scale (cheap), then verified
COARSE_MIN = 0.45  # correlation that makes a coarse candidate (true: 0.5-0.85; verified after)


@dataclass(frozen=True)
class TextLine:
    text: str
    score: float  # lowest glyph correlation of the line, 0..1
    x: int  # top-left of the line's first cell in the read image
    y: int


def corner_rect(width: int, height: int) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the screen corner the debug text sits in, for a screen of that size."""
    w, h = int(width * CORNER_W), int(height * CORNER_H)
    return width - w, 0, w, h


def corner(image: np.ndarray) -> np.ndarray:
    """The debug-text corner of a whole-screen image."""
    x, y, w, h = corner_rect(image.shape[1], image.shape[0])
    return image[y : y + h, x : x + w]


def whiteness(image: np.ndarray) -> np.ndarray:
    """uint8 per pixel: the darkest channel (white text stays bright, coloured HUD does not)."""
    if image.ndim == 2:
        return image
    b, g, r = cv2.split(image[:, :, :3])
    return cv2.min(cv2.min(b, g), r)


@dataclass(frozen=True)
class DebugFont:
    """Glyph templates of one size of the debug font.

    Stored as `<name>.json` (chars, advance, cell size, where it was cut) plus `<name>.png`:
    the glyphs' whiteness side by side, `cell_w` pixels each, in `chars` order.
    """

    name: str
    chars: str
    advance: float  # px from one cell to the next
    cell_w: int
    cell_h: int
    glyphs: np.ndarray  # (len(chars), cell_h, cell_w) uint8 whiteness
    source: str = ""  # e.g. "2560x1440": where the glyphs were cut

    def __post_init__(self) -> None:
        if self.glyphs.shape != (len(self.chars), self.cell_h, self.cell_w):
            raise ValueError(f"font {self.name}: glyph array does not match chars/cell size")

    @classmethod
    def load(cls, json_path: Path) -> DebugFont:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        strip = cv2.imread(str(json_path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        if strip is None:
            raise ValueError(f"font {json_path.stem}: {json_path.with_suffix('.png')} missing")
        chars, w, h = meta["chars"], int(meta["cell_w"]), int(meta["cell_h"])
        glyphs = np.stack([strip[:h, i * w : (i + 1) * w] for i in range(len(chars))])
        return cls(json_path.stem, chars, float(meta["advance"]), w, h, glyphs,
                   meta.get("source", ""))  # fmt: skip

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.name}.json"
        meta = {"chars": self.chars, "advance": self.advance, "cell_w": self.cell_w,
                "cell_h": self.cell_h, "source": self.source}  # fmt: skip
        path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        cv2.imwrite(str(path.with_suffix(".png")), np.hstack(list(self.glyphs)))
        return path

    def glyph(self, char: str) -> np.ndarray:
        return self.glyphs[self.chars.index(char)]

    def cell_x(self, i: int, phase: float = 0.0) -> int:
        """Left edge of cell `i` of a line whose first cell starts `phase` px into a pixel."""
        return math.floor(phase + i * self.advance)

    def phases(self) -> tuple[float, ...]:
        return phases(self.advance)

    def render(self, text: str, background: int = 0, phase: float = 0.0) -> np.ndarray:
        """`text` in this font (whiteness); unknown characters and spaces stay background."""
        width = self.cell_x(len(text) - 1, phase) + self.cell_w if text else 0
        out = np.full((self.cell_h, width), background, np.uint8)
        for i, char in enumerate(text):
            if char in self.chars and char != " ":
                x = self.cell_x(i, phase)
                out[:, x : x + self.cell_w] = self.glyph(char)
        return out


def phases(advance: float) -> tuple[float, ...]:
    """The sub-pixel starts that give a line of cells a different pattern of whole-pixel steps:
    only 0 for a whole-pixel advance; 0 and 0.5 for 7.5 px; four for quarter pixels."""
    found: dict[tuple[int, ...], float] = {}
    for phase in (0.0, 0.25, 0.5, 0.75):
        found.setdefault(tuple(math.floor(phase + i * advance) for i in range(8)), phase)
    return tuple(found.values())


_AS_DIGIT = {"O": "0", "I": "1", "l": "1"}
_NUMERIC_LEFT = set("0123456789.-")
_NUMERIC_RIGHT = set("0123456789.")


def digits_in_numbers(text: str) -> str:
    """O, I and l next to a digit (or after a minus or point) become 0 and 1: at this size the
    font draws them almost alike, and the debug text is mostly numbers. Names keep their letters
    ("OOC_Stanton" stays as it is)."""
    chars = list(text)
    changed = True
    while changed:
        changed = False
        for i, char in enumerate(chars):
            if char not in _AS_DIGIT:
                continue
            left = chars[i - 1] if i else ""
            right = chars[i + 1] if i + 1 < len(chars) else ""
            if left in _NUMERIC_LEFT or right in _NUMERIC_RIGHT:
                chars[i] = _AS_DIGIT[char]
                changed = True
    return "".join(chars)


def load_fonts(folder: Path = FONT_DIR) -> list[DebugFont]:
    return [DebugFont.load(p) for p in sorted(folder.glob("*.json"))]


@cache
def shipped_fonts() -> tuple[DebugFont, ...]:
    return tuple(load_fonts())


def _unit_rows(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows made zero-mean and unit-length (for correlation by dot product), and their norms."""
    flat = patches.reshape(len(patches), -1).astype(np.float32)
    flat -= flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1)
    return flat / np.maximum(norms, 1e-6)[:, None], norms


@dataclass
class _Prepared:
    font: DebugFont
    unit: np.ndarray  # (n, cell_h*cell_w) glyphs, zero-mean unit rows
    # per prefix: (phase, float32 whiteness) for every grid phase of the font
    prefixes: dict[str, list[tuple[float, np.ndarray]]] = field(default_factory=dict)
    coarse: dict[str, np.ndarray] = field(default_factory=dict)  # phase 0 at 1/COARSE


def _shrink(image: np.ndarray) -> np.ndarray:
    h, w = max(1, image.shape[0] // COARSE), max(1, image.shape[1] // COARSE)
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


class DebugTextReader:
    """Reads CamDir and Zone lines; tries each font, the last one that matched first."""

    def __init__(self, fonts: Sequence[DebugFont] | None = None) -> None:
        fonts = list(shipped_fonts() if fonts is None else fonts)
        self._fonts: list[_Prepared] = []
        for font in fonts:
            if not all(c in font.chars for p in PREFIXES for c in p):
                continue  # cannot even find the lines
            unit, _ = _unit_rows(font.glyphs)
            prefixes = {
                p: [(ph, font.render(p, phase=ph).astype(np.float32)) for ph in font.phases()]
                for p in PREFIXES
            }
            coarse = {p: _shrink(t[0][1]) for p, t in prefixes.items()}
            self._fonts.append(_Prepared(font, unit, prefixes, coarse))

    @property
    def usable(self) -> bool:
        return bool(self._fonts)

    def read(self, image: np.ndarray) -> list[TextLine]:
        """The CamDir and Zone lines in `image` (the debug-text corner), top first."""
        if not self._fonts or image.size == 0:
            return []
        white = whiteness(image)
        if int(white.max()) < INK:
            return []
        small = _shrink(white)
        for i, prepared in enumerate(self._fonts):
            lines = self._read_with(prepared, white, small)
            if lines:
                if i:
                    self._fonts.insert(0, self._fonts.pop(i))
                return lines
        return []

    def _read_with(self, p: _Prepared, white: np.ndarray, small: np.ndarray) -> list[TextLine]:
        hits: list[tuple[int, int, float, str]] = []
        for prefix in PREFIXES:
            found = _find(white, small, p.prefixes[prefix], p.coarse[prefix], p.font.cell_h)
            hits += [(y, x, phase, prefix) for x, y, phase in found]
        hits.sort()
        lines: list[TextLine] = []
        for y, x, phase, prefix in hits:
            if lines and abs(lines[-1].y - y) < p.font.cell_h // 2:
                continue  # "CamDir:" and "Zone:" on the same row: keep the first
            rest, score = self._line(p, white, x, y, len(prefix), phase)
            lines.append(TextLine(digits_in_numbers(prefix + rest), score, x, y))
        return lines

    def _line(
        self, p: _Prepared, white: np.ndarray, x0: int, y0: int, start: int, phase: float
    ) -> tuple[str, float]:
        """Text of the cells after a prefix, and the lowest glyph correlation among them.

        Every cell up to the image edge is classified at once (one matrix product), each at
        -1, 0 and +1 px to absorb the rounding of a fractional advance; the line then ends at
        a run of blank or unknown cells.
        """
        font = p.font
        h, w = font.cell_h, font.cell_w
        band = white[y0 : y0 + h]
        if band.shape[0] < h:
            return "", 0.0
        xs: list[int] = []
        i = start
        while x0 + font.cell_x(i, phase) + w <= band.shape[1]:
            xs.append(x0 + font.cell_x(i, phase))
            i += 1
        if not xs:
            return "", 0.0
        n = len(xs)
        # A whole-pixel advance puts every cell exactly; a fractional one may round either way
        offsets = (0,) if float(font.advance).is_integer() else (-1, 0, 1)
        padded = np.pad(band, ((0, 0), (1, 1)), mode="edge")  # room for the +-1 px offsets
        ink = [_has_ink(band[:, x : x + w]) for x in xs]
        patches = np.stack([padded[:, x + 1 + d : x + 1 + d + w] for x in xs for d in offsets])
        unit, norms = _unit_rows(patches)
        # einsum, not `@`: a multi-threaded BLAS spends ~50 ms waking threads for this tiny
        # product when the game keeps every core busy
        scores = np.einsum("gk,ck->gc", p.unit, unit).reshape(len(p.unit), n, len(offsets))
        scores[:, (norms < 1e-3).reshape(n, len(offsets))] = -1.0
        per_glyph = scores.max(axis=2)  # (glyphs, cells)
        best_glyph = per_glyph.argmax(axis=0)
        best = per_glyph[best_glyph, np.arange(n)]

        chars: list[str] = []
        lowest = 1.0
        blanks = unknowns = 0
        for k in range(n):
            if not ink[k]:
                chars.append(" ")
                blanks, unknowns = blanks + 1, 0
                if blanks >= MAX_BLANK:
                    break
                continue
            blanks = 0
            score = float(best[k])
            known = math.isfinite(score) and score >= GLYPH_MIN
            chars.append(font.chars[int(best_glyph[k])] if known else UNKNOWN)
            unknowns = 0 if known else unknowns + 1
            if unknowns >= MAX_UNKNOWN:
                del chars[-unknowns:]
                break
            lowest = min(lowest, score if math.isfinite(score) else 0.0)
        return "".join(chars).rstrip(), max(0.0, round(lowest, 4))


def _has_ink(cell: np.ndarray) -> bool:
    top = int(cell.max())
    return top >= INK or top - int(np.median(cell)) >= INK_CONTRAST


def _find(
    white: np.ndarray,
    small: np.ndarray,
    templates: list[tuple[float, np.ndarray]],
    coarse: np.ndarray,
    cell_h: int,
) -> list[tuple[int, int, float]]:
    """(x, y, phase) of the prefix's matches in `white`: candidates found at 1/COARSE scale,
    each verified and placed exactly at full scale with the grid phase that fits best."""
    th = templates[0][1].shape[0]
    tw = max(t.shape[1] for _, t in templates)
    margin = 2 * COARSE
    found: list[tuple[int, int, float]] = []
    for cx, cy in _peaks(small, coarse, max(1, cell_h // COARSE), COARSE_MIN):
        x0, y0 = max(0, cx * COARSE - margin), max(0, cy * COARSE - margin)
        window = white[y0 : y0 + th + 2 * margin, x0 : x0 + tw + 2 * margin]
        if window.shape[0] < th or window.shape[1] < tw:
            continue
        window = window.astype(np.float32)
        best = (-1.0, 0, 0, 0.0)
        for phase, template in templates:
            result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            result[~np.isfinite(result)] = -1.0
            _, score, _, (dx, dy) = cv2.minMaxLoc(result)
            if score > best[0]:
                best = (score, dx, dy, phase)
        score, dx, dy, phase = best
        if score >= PREFIX_MIN and all((x0 + dx, y0 + dy) != (x, y) for x, y, _ in found):
            found.append((x0 + dx, y0 + dy, phase))
    return found


def _peaks(
    image: np.ndarray, template: np.ndarray, cell_h: int, minimum: float
) -> list[tuple[int, int]]:
    """(x, y) of the template's local best matches above `minimum`, at most one per line."""
    th, tw = template.shape
    if image.shape[0] < th or image.shape[1] < tw:
        return []
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    result[~np.isfinite(result)] = -1.0
    local_max = cv2.dilate(result, np.ones((max(3, cell_h), max(3, tw)), np.uint8))
    ys, xs = np.nonzero((result >= minimum) & (result >= local_max))
    order = np.argsort(-result[ys, xs])
    kept: list[tuple[int, int]] = []
    for k in order:
        x, y = int(xs[k]), int(ys[k])
        if all(abs(y - ky) >= max(1, cell_h // 2) or abs(x - kx) >= tw for kx, ky in kept):
            kept.append((x, y))
    return kept
