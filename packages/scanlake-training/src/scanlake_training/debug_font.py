"""Builds the debug-text font (`scanlake_core.vision.debug_text.DebugFont`) from screenshots.

    scanlake-train font cut SHOT.png [SHOT2.png ...] [--name NAME] [--advance PX] [--out DIR]
    scanlake-train font read SHOT.png [--font DIR]

`cut` takes lossless whole-screen PNGs with the game's debug text on (r_DisplayInfo). Next to
each one, `SHOT.txt` holds the transcript: the text lines in the top-right corner exactly as the
game shows them, top first, one per line, as many as you like (lower debug lines such as
"Current player location : ..." add letters the CamDir and Zone lines lack). A line starting
with `#` is skipped (e.g. where the HUD overlaps the text) but still counts as a line.

For every transcript line the matching text band is found in the corner, the font's advance is
fitted from all lines, each line's grid position (x and y: a band's top depends on its tallest
glyph) is settled by agreement between its cells and the glyphs averaged so far, and every
glyph is averaged over all its occurrences in all screenshots. The result is checked by reading
the screenshots back.

`read` prints what the reader finds in a screenshot and the position parsed from it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scanlake_core.domain.position import parse_position
from scanlake_core.vision.debug_text import (
    FONT_DIR,
    INK,
    DebugFont,
    DebugTextReader,
    corner,
    load_fonts,
    phases,
    whiteness,
)

SKIP = "#"
PHASE_SEARCH = 2  # px either way a line's grid may move while settling (x and y)


class CutError(ValueError):
    """A screenshot and its transcript do not fit together."""


@dataclass
class _Line:
    text: str
    white: np.ndarray  # whiteness of the corner
    top: int  # first row of the line's ink
    left: int  # first column of its ink
    right: int  # last column of its ink
    screen: str  # "WxH" of the screenshot
    x0: int = 0  # settled left edge of the first cell ...
    y0: int = 0  # ... and top of the cells ...
    phase: float = 0.0  # ... and grid phase (see DebugFont.cell_x)


def text_bands(white: np.ndarray, min_ink: int = 2) -> list[tuple[int, int]]:
    """(first, last) row of every band of rows holding glyph ink, top first."""
    rows = (white >= INK).sum(axis=1) >= min_ink
    bands: list[tuple[int, int]] = []
    start = None
    for y, ink in enumerate(rows):
        if ink and start is None:
            start = y
        elif not ink and start is not None:
            bands.append((start, y - 1))
            start = None
    if start is not None:
        bands.append((start, len(rows) - 1))
    return bands


def _lines(shot: Path) -> list[_Line]:
    transcript = shot.with_suffix(".txt")
    if not transcript.exists():
        raise CutError(f"{shot.name}: no transcript {transcript.name}")
    texts = [t.rstrip("\n") for t in transcript.read_text(encoding="utf-8").splitlines()]
    texts = [t for t in texts if t.strip()]
    import cv2

    image = cv2.imread(str(shot), cv2.IMREAD_COLOR)
    if image is None:
        raise CutError(f"{shot.name}: not an image")
    white = whiteness(corner(image))
    bands = text_bands(white)
    if len(bands) < len(texts):
        raise CutError(f"{shot.name}: {len(texts)} transcript lines but {len(bands)} text bands")
    lines = []
    for text, (top, bottom) in zip(texts, bands, strict=False):
        if text.startswith(SKIP):
            continue
        cols = np.nonzero((white[top : bottom + 1] >= INK).any(axis=0))[0]
        screen = f"{image.shape[1]}x{image.shape[0]}"
        lines.append(_Line(text, white, top, int(cols[0]), int(cols[-1]), screen))
    return lines


def fit_advance(lines: Sequence[_Line]) -> float:
    """The cell pitch from ink extents: width ~ advance * characters, over lines of different
    lengths (the bearings of the first and last glyph only shift the intercept)."""
    n = np.array([len(line.text) for line in lines], dtype=np.float64)
    width = np.array([line.right - line.left + 1 for line in lines], dtype=np.float64)
    if len(set(n.tolist())) < 2:
        raise CutError("need transcript lines of at least two different lengths (or --advance)")
    slope, _ = np.polyfit(n, width, 1)
    return float(round(slope * 4) / 4)  # quarter pixels: rendering positions are rounded


def cut(shots: Sequence[Path], name: str | None = None, advance: float | None = None) -> DebugFont:
    lines = [line for shot in shots for line in _lines(shot)]
    if not lines:
        raise CutError("no transcript lines")
    adv = advance or fit_advance(lines)
    cell_w = math.floor(adv) if not float(adv).is_integer() else int(adv)
    tops = sorted({line.top for line in lines})
    pitches = [b - a for a, b in zip(tops, tops[1:], strict=False) if b - a < 3 * cell_w]
    cell_h = int(np.median(pitches)) if pitches else cell_w * 2
    for line in lines:  # first guess: the ink starts one pixel into the first cell
        line.x0, line.y0 = line.left - 1, line.top - 1

    def cells(line: _Line, x0: int, y0: int, phase: float) -> list[tuple[str, np.ndarray]]:
        out = []
        for i, char in enumerate(line.text):
            x = x0 + math.floor(phase + i * adv)
            if char == " " or x < 0 or y0 < 0:
                continue
            cell = line.white[y0 : y0 + cell_h, x : x + cell_w]
            if cell.shape == (cell_h, cell_w):
                out.append((char, cell.astype(np.float32)))
        return out

    def averages() -> dict[str, np.ndarray]:
        sums: dict[str, list[np.ndarray]] = defaultdict(list)
        for line in lines:
            for char, cell in cells(line, line.x0, line.y0, line.phase):
                sums[char].append(cell)
        return {c: np.mean(v, axis=0) for c, v in sums.items()}

    def agreement(line: _Line, at: tuple[int, int, float], glyphs: dict[str, np.ndarray]) -> float:
        total = 0.0
        for char, cell in cells(line, *at):
            a, b = cell - cell.mean(), glyphs[char] - glyphs[char].mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            total += float((a * b).sum()) / denom if denom > 1e-6 else 0.0
        return total

    grid = phases(adv)
    for _ in range(6):  # settle every line's position and phase against all lines' glyphs
        glyphs = averages()
        moved = False
        steps = range(-PHASE_SEARCH, PHASE_SEARCH + 1)
        for line in lines:
            best = max(
                (
                    (line.x0 + dx, line.y0 + dy, phase)
                    for dy in steps
                    for dx in steps
                    for phase in grid
                ),
                key=lambda at, line=line: agreement(line, at, glyphs),
            )
            moved = moved or best != (line.x0, line.y0, line.phase)
            line.x0, line.y0, line.phase = best
        if not moved:
            break

    glyphs = averages()
    chars = "".join(sorted(glyphs))
    stack = np.stack([np.clip(np.rint(glyphs[c]), 0, 255).astype(np.uint8) for c in chars])
    source = lines[0].screen
    return DebugFont(name or source, chars, adv, cell_w, cell_h, stack, source)


def check(font: DebugFont, shots: Sequence[Path]) -> list[tuple[str, str]]:
    """(expected, read) for every CamDir and Zone transcript line that reads differently. An
    underscore read as a space counts as read: the parser turns gaps in zone names back."""
    import cv2

    reader = DebugTextReader([font])
    wrong = []
    for shot in shots:
        image = cv2.imread(str(shot), cv2.IMREAD_COLOR)
        if image is None:
            raise CutError(f"{shot.name}: not an image")
        found = [line.text for line in reader.read(corner(image))]
        expected = [
            t.strip() for t in shot.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
            if t.strip().startswith(("CamDir:", "Zone:"))
        ]  # fmt: skip
        for i, text in enumerate(expected):
            got = found[i] if i < len(found) else ""
            if got.replace(" ", "_") != text.replace(" ", "_"):
                wrong.append((text, got))
    return wrong


def run_cut(shots: Sequence[str], name: str | None, advance: float | None, out: str | None) -> int:
    paths = [Path(s) for s in shots]
    try:
        font = cut(paths, name, advance)
    except CutError as err:
        print(f"error: {err}")
        return 2
    folder = Path(out) if out else FONT_DIR
    path = font.save(folder)
    print(f"{path}: {len(font.chars)} glyphs, advance {font.advance}, cell "
          f"{font.cell_w}x{font.cell_h}")  # fmt: skip
    missing = sorted(set("".join(chr(c) for c in range(33, 127))) - set(font.chars))
    if missing:
        print("not seen yet:", "".join(missing))
    wrong = check(font, paths)
    for expected, got in wrong:
        print(f"MISREAD\n  want {expected!r}\n  got  {got!r}")
    print("read back: all CamDir/Zone lines correct" if not wrong else f"{len(wrong)} misread")
    return 1 if wrong else 0


def run_read(shot: str, font_dir: str | None) -> int:
    import cv2

    image = cv2.imread(shot, cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: cannot read {shot}")
        return 2
    fonts = load_fonts(Path(font_dir)) if font_dir else None
    reader = DebugTextReader(fonts)
    if not reader.usable:
        print("error: no usable font (cut one first)")
        return 2
    lines = reader.read(corner(image))
    for line in lines:
        print(f"{line.score:.3f}  {line.text}")
    position = parse_position([(line.text, line.score) for line in lines])
    print(position.details() if position else "no position")
    return 0 if position else 1
