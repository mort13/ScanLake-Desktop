"""Builds the debug-text font (`scanlake_core.vision.debug_text.DebugFont`) from screenshots.

    scanlake-train font cut SHOT.png [SHOT2.png ...] [--name NAME] [--advance PX] [--out DIR]
    scanlake-train font recut FRAME.png [FRAME2.png ...] --names NAMES.txt [--font DIR] [--out DIR]
    scanlake-train font draft SHOT.jpg [SHOT2.jpg ...] [--font DIR]
    scanlake-train font learn SHOT.jpg [SHOT2.jpg ...] [--font DIR] [--out DIR]
    scanlake-train font read SHOT.png [--font DIR]

`cut` takes lossless whole-screen PNGs with the game's debug text on (r_DisplayInfo). Next to
each one, `SHOT.txt` holds the transcript: the text lines in the top-right corner exactly as the
game shows them, top first, one per line, as many as you like (lower debug lines such as
"Current player location : ..." add letters the CamDir and Zone lines lack). A line starting
with `#` is skipped (e.g. where the HUD overlaps the text) but still counts as a line.

For every transcript line the matching text band is found in the corner, the font's advance is
fitted from all lines, and each line's grid position (x and y: a band's top depends on its
tallest glyph) is settled by agreement between its cells and the glyphs so far.

`recut` improves an existing font from many lossless captures of the live debug text (the
corner as the app grabs it, or whole screens): the reader finds the lines, which works over
bright backgrounds where `cut`'s ink bands run together, and its own text is the transcript,
with each zone name replaced by the closest one in NAMES.txt (every zone name the frames show,
one per line, as the game spells it). Glyphs the frames do not show stay as they were.

`learn` adds glyphs to an existing font from screenshots (the game's JPEGs are fine) and
`SHOT.txt` transcripts of ANY of their debug lines, in any order: the lower lines ("Radius:
... ElevationUnderCam: ...", "Planet:... (Load Complete)", "XYPos: ... (+Y)") hold letters the
zone names rarely show. Each line is placed by the glyphs the font already knows; lines in
another font (some debug lines are about 12% narrower) or colour fit nowhere and are reported.
`draft` writes `SHOT.draft.txt`, every row of the font's size as read, as a start for the
transcripts: correct it, keep the lines you trust, save it as `SHOT.txt`.

Each glyph's template is the median of its occurrences on a dark background (a bright one
shows the text's shadow), and with a fractional advance (7.5 px) there are two per glyph:
every other glyph starts half a pixel into its cell, which the game draws differently. The
font records which glyphs came from lossless captures; lossy screenshots never replace those.
The result is checked by reading the screenshots or frames back.

`read` prints what the reader finds in a screenshot and the position parsed from it.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scanlake_core.domain.position import parse_position
from scanlake_core.vision.debug_text import (
    FONT_DIR,
    GLYPH_MIN,
    INK,
    INK_CONTRAST,
    MAX_BLANK,
    UNKNOWN,
    DebugFont,
    DebugTextReader,
    corner,
    digits_in_numbers,
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
    lossless = all(is_lossless(shot) for shot in shots)
    return _cut_lines(lines, name or lines[0].screen, adv, cell_w, cell_h, lossless=lossless)


_ZONE_TEXT = re.compile(r"^(Zone: )(.+?)( Pos: .*)$")


def recut(frames: Sequence[Path], names: Sequence[str], base: DebugFont) -> DebugFont:
    """`base` re-cut from lossless captures of the live debug text, e.g. the debug-text corner
    as the app grabs it.

    Lines are located by the reader (with `base`), which works over bright backgrounds where
    the ink bands of `cut` run together. Its text is the transcript, every zone name replaced
    by the one in `names` it is closest to (same length, few differences; a line whose name
    fits none is left out). Glyphs the frames do not show (on a dark background) stay `base`'s.
    """
    reader = DebugTextReader([base])
    lines = [line for frame in frames for line in _reader_lines(frame, names, reader)]
    if not lines:
        raise CutError("no CamDir or Zone line with a known zone name in the frames")
    lossless = all(is_lossless(frame) for frame in frames)
    return _cut_lines(lines, base.name, base.advance, base.cell_w, base.cell_h, base, lossless)


def _reader_lines(frame: Path, names: Sequence[str], reader: DebugTextReader) -> list[_Line]:
    import cv2

    image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
    if image is None:
        raise CutError(f"{frame.name}: not an image")
    if image.shape[1] / image.shape[0] < 2.5:  # a whole screen: its debug-text corner
        image = corner(image)
    white = whiteness(image)
    out = []
    for line in reader.read(image):
        text = line.text
        if found := _ZONE_TEXT.match(text):
            name = closest_name(found.group(2), names)
            if name is None:
                continue
            text = found.group(1) + name + found.group(3)
        elif not text.startswith("CamDir:"):
            continue
        if UNKNOWN in text:
            continue
        out.append(_Line(text, white, line.y, line.x, 0, frame.name, line.x, line.y))
    return out


def closest_name(read: str, names: Sequence[str]) -> str | None:
    """The known zone name a read one stands for: same length, the fewest differing characters,
    at most a fifth of them (a faint underscore read as a space counts as right)."""
    read = read.replace(" ", "_")
    best: tuple[int, str] | None = None
    for name in names:
        if len(name) != len(read):
            continue
        differ = sum(a != b for a, b in zip(name, read, strict=True))
        if differ <= max(2, len(name) // 5) and (best is None or differ < best[0]):
            best = (differ, name)
    return best[1] if best else None


# -- screenshots with transcripts of any debug line: learn, draft ------------------------------

PLACE_MIN = 0.8  # mean match of a line's known glyphs where it is placed (the font's size)
PROBE = 16  # glyphs from a line's right end that shortlist its row
ROW_MIN = 0.6  # mean best-glyph match of a row's right end that makes it a row to draft


def learn(
    shots: Sequence[Path], base: DebugFont, lossless: bool | None = None
) -> tuple[DebugFont, list[_Line], list[str]]:
    """`base` with glyphs learnt from screenshots and transcripts of any of their debug lines.

    Next to each shot, SHOT.txt holds lines of its debug-text corner exactly as the game shows
    them, in any order and as many as you like: every debug line, not only CamDir and Zone
    (lower lines such as "Radius: ... ElevationUnderCam: ..." hold letters zone names rarely
    do). A line starting with `#` is a comment. Each line is placed where the glyphs `base`
    already knows match best, right-aligned like all debug text; one that fits nowhere (a
    coloured line, or the narrower font some debug lines use) is left out and reported.
    Glyphs `base` cut from lossless captures stay unless the shots are lossless too
    (`lossless`; None: PNG and the like are, JPEG is not).
    Returns the font, the lines placed (settled) and those left out.
    """
    lines: list[_Line] = []
    left_out: list[str] = []
    for shot in shots:
        transcript = shot.with_suffix(".txt")
        if not transcript.exists():
            raise CutError(f"{shot.name}: no transcript {transcript.name}")
        texts = [t.rstrip() for t in transcript.read_text(encoding="utf-8").splitlines()]
        image = _corner_image(shot)
        white = whiteness(image)
        right = right_edge(image, base)
        taken: set[int] = set()
        for text in texts:
            if not text.strip() or text.startswith(SKIP):
                continue
            found = place(white, text, base, right)
            if found is None or found[3] < PLACE_MIN or found[1] in taken:
                left_out.append(f"{shot.name}: {text}")
                continue
            x0, y, phase, _ = found
            taken.add(y)
            lines.append(_Line(text, white, y, x0, 0, shot.name, x0, y, phase))
    if not lines:
        raise CutError("no transcript line fits the font")
    if lossless is None:
        lossless = all(is_lossless(shot) for shot in shots)
    font = _cut_lines(lines, base.name, base.advance, base.cell_w, base.cell_h, base, lossless)
    return font, lines, left_out


def draft(shot: Path, font: DebugFont) -> list[str]:
    """Every row of the corner in `font`'s size, top first, as `font` reads it (UNKNOWN where no
    glyph fits): a start for SHOT.txt. Rows in other fonts or colours read as garbage, if at
    all: leave them out of the transcript (`learn` would)."""
    image = _corner_image(shot)
    white = whiteness(image).astype(np.float32)
    right = right_edge(image, font)
    w, h, adv = font.cell_w, font.cell_h, font.advance
    unit = _unit(np.concatenate(font.variants()))
    count = white.shape[0] - h + 1
    best = np.zeros(count, np.float32)
    inked = np.zeros(count, np.float32)
    for j in range(10):  # the right end's cells, for every row at once
        x = math.floor(right - w - j * adv + 0.5)
        scores = np.full(count, -1.0, np.float32)
        for d in (-1, 0, 1):
            if 0 <= x + d and x + d + w <= white.shape[1]:
                windows = _windows(white, x + d, h, w)
                scores = np.maximum(scores, (unit @ windows.T).max(axis=0))
        ink = _column_ink(white, x, h, w)
        best += scores * ink
        inked += ink
    quality = np.where(inked >= 3, best / np.maximum(inked, 1), 0.0)
    rows: list[int] = []
    for y in np.argsort(-quality):
        if quality[y] < ROW_MIN:
            break
        if all(abs(int(y) - r) >= h - 3 for r in rows):
            rows.append(int(y))
    # The right end can place a row a pixel off, and a line of odd length sits half a pixel
    # off the grid of one of even length: each row reads where the whole line matches best
    return [
        max(
            (_read_left(white, row + dy, right, font, unit, shift)
             for dy in (-1, 0, 1) for shift in (0.0, 0.5)),
            key=lambda read: read[1],
        )[0]
        for row in sorted(rows)
    ]  # fmt: skip


def read_back(font: DebugFont, lines: Sequence[_Line]) -> Counter[tuple[str, str]]:
    """(meant, read) for every glyph of the settled lines that `font` reads differently."""
    unit = _unit(np.concatenate(font.variants()))
    wrong: Counter[tuple[str, str]] = Counter()
    for line in lines:
        white = line.white.astype(np.float32)
        for i, char in enumerate(line.text):
            if char == " ":
                continue
            x = line.x0 + math.floor(line.phase + i * font.advance)
            read, _ = _classify(white, line.y0, x, font, unit)
            if read != char and not (char == "_" and read == " "):
                wrong[(char, read)] += 1
    return wrong


def right_edge(image: np.ndarray, font: DebugFont) -> int:
    """Just past the last cell of the debug text, to which every line is right-aligned: from
    the CamDir and Zone lines the reader finds, else the rightmost ink near the top."""
    lines = DebugTextReader([font]).read(image)
    if lines:
        return max(
            line.x + font.cell_x(len(line.text) - 1, phase) + font.cell_w
            for line in lines
            for phase in font.phases()
        )
    cols = np.nonzero((whiteness(image)[: 6 * font.cell_h] >= INK).any(axis=0))[0]
    if not len(cols):
        raise CutError("no debug text in the corner")
    return int(cols[-1]) + 2


def place(
    white: np.ndarray, text: str, font: DebugFont, right: int
) -> tuple[int, int, float, float] | None:
    """(x0, y, phase, match) of a right-aligned debug line: where the glyphs of `text` that
    `font` knows match best (their mean correlation). None when it knows too few of them."""
    known = [(i, c) for i, c in enumerate(text) if c != " " and c in font.chars]
    if len(known) < 4:
        return None
    w, h, adv = font.cell_w, font.cell_h, font.advance
    image = white.astype(np.float32)
    glyphs = {c: _unit(np.stack([font.glyph(c), font.glyph(c, True)])) for _, c in known}
    nominal = right - w - (len(text) - 1) * adv
    best: tuple[int, int, float, float] | None = None
    for start in np.arange(nominal - 2.5, nominal + 2.75, 0.5):

        def at(x: int) -> bool:
            return 0 <= x and x + w <= image.shape[1]

        # the right end shortlists rows (two lines may end alike), the whole line decides
        probe = np.zeros(image.shape[0] - h + 1, np.float32)
        for i, c in known[-PROBE:]:
            x = math.floor(start + i * adv)
            if at(x):
                probe += (glyphs[c] @ _windows(image, x, h, w).T).max(axis=0)
        for y in np.argsort(-probe)[:3]:
            scores = []
            for i, c in known:
                x = math.floor(start + i * adv)
                if at(x):
                    cell = _unit(image[y : y + h, x : x + w][None])[0]
                    scores.append(float((glyphs[c] @ cell).max()))
            match = sum(scores) / len(known)
            if best is None or match > best[3]:
                x0 = math.floor(start)
                best = (x0, int(y), float(start - x0), match)
    return best


def _corner_image(shot: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(shot), cv2.IMREAD_COLOR)
    if image is None:
        raise CutError(f"{shot.name}: not an image")
    return corner(image) if image.shape[1] / image.shape[0] < 2.5 else image


def _unit(patches: np.ndarray) -> np.ndarray:
    """Rows made zero-mean and unit-length, so a dot product is a correlation."""
    flat = patches.reshape(len(patches), -1).astype(np.float32)
    flat -= flat.mean(axis=1, keepdims=True)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-6)


def _windows(white: np.ndarray, x: int, h: int, w: int) -> np.ndarray:
    """The cell at column `x` for every top row, as unit rows."""
    from numpy.lib.stride_tricks import sliding_window_view

    return _unit(sliding_window_view(white[:, x : x + w], (h, w))[:, 0])


def _column_ink(white: np.ndarray, x: int, h: int, w: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    if x < 0 or x + w > white.shape[1]:
        return np.zeros(white.shape[0] - h + 1, np.float32)
    cells = sliding_window_view(white[:, x : x + w], (h, w))[:, 0].reshape(-1, h * w)
    top, middle = cells.max(axis=1), np.median(cells, axis=1)
    return ((top >= INK) | (top - middle >= INK_CONTRAST)).astype(np.float32)


def _classify(
    white: np.ndarray, y: int, x: int, font: DebugFont, unit: np.ndarray
) -> tuple[str, float]:
    """The glyph in the cell at (x, y), tried at -1, 0 and +1 px, and its correlation;
    (" ", 1.0) without ink."""
    w, h = font.cell_w, font.cell_h
    cell = white[y : y + h, x : x + w] if y >= 0 and x >= 0 else white[:0, :0]
    if cell.shape != (h, w):
        return " ", 1.0
    top = float(cell.max())
    if top < INK and top - float(np.median(cell)) < INK_CONTRAST:
        return " ", 1.0
    best, char = -1.0, UNKNOWN
    for d in (-1, 0, 1):
        if x + d < 0:
            continue
        moved = white[y : y + h, x + d : x + d + w]
        if moved.shape != (h, w):
            continue
        scores = unit @ _unit(moved[None])[0]
        k = int(scores.argmax())
        if scores[k] > best:
            best, char = float(scores[k]), font.chars[k % len(font.chars)]
    return (char if best >= GLYPH_MIN else UNKNOWN), best


def _read_left(
    white: np.ndarray, y: int, right: int, font: DebugFont, unit: np.ndarray, shift: float
) -> tuple[str, float]:
    """A right-aligned row read from its end leftwards until a run of blank cells, and its mean
    glyph correlation; `shift` moves the cell grid by a fraction of a pixel."""
    chars: list[str] = []
    scores: list[float] = []
    blanks = j = 0
    while blanks < MAX_BLANK:
        x = math.floor(right - font.cell_w - j * font.advance + shift)
        if x < 1:
            break
        char, score = _classify(white, y, x, font, unit)
        chars.append(char)
        if char != " ":
            scores.append(score)
        blanks = blanks + 1 if char == " " else 0
        j += 1
    text = digits_in_numbers("".join(reversed(chars)).strip())
    return text, (sum(scores) / len(scores) if scores else 0.0)


_Key = tuple[str, bool]  # (char, starts half a pixel into its cell)


def _cut_lines(
    lines: list[_Line],
    name: str,
    adv: float,
    cell_w: int,
    cell_h: int,
    base: DebugFont | None = None,
    lossless: bool = True,
) -> DebugFont:
    """Settles every line's grid against the glyphs, then cuts a template per glyph and, with a
    fractional advance, per whole- and half-pixel start (the game draws them differently).

    `lossless`: the lines come from lossless captures. If not, `base`'s glyphs from lossless
    captures stay as they are; a glyph `base` lacks is cut even from a few samples.
    """
    fractional = not float(adv).is_integer()
    keep = "" if lossless or base is None else base.lossless

    def cells(line: _Line, x0: int, y0: int, phase: float) -> list[tuple[_Key, np.ndarray]]:
        out = []
        for i, char in enumerate(line.text):
            x = x0 + math.floor(phase + i * adv)
            if char == " " or x < 0 or y0 < 0:
                continue
            cell = line.white[y0 : y0 + cell_h, x : x + cell_w]
            if cell.shape == (cell_h, cell_w):
                half = fractional and 0.25 <= (phase + i * adv) % 1 < 0.75
                out.append(((char, half), cell.astype(np.float32)))
        return out

    def averages() -> dict[_Key, np.ndarray]:
        sums: dict[_Key, list[np.ndarray]] = defaultdict(list)
        for line in lines:
            for key, cell in cells(line, line.x0, line.y0, line.phase):
                sums[key].append(cell)
        return {k: np.mean(v, axis=0) for k, v in sums.items()}

    def reference(key: _Key, glyphs: dict[_Key, np.ndarray]) -> np.ndarray | None:
        found = glyphs.get(key)
        if found is None:
            found = glyphs.get((key[0], not key[1]))
        if found is None and base is not None and key[0] in base.chars:
            found = base.glyph(key[0], key[1]).astype(np.float32)
        return found

    def agreement(line: _Line, at: tuple[int, int, float], glyphs: dict[_Key, np.ndarray]) -> float:
        total = 0.0
        for key, cell in cells(line, *at):
            glyph = reference(key, glyphs)
            if glyph is None:
                continue
            a, b = cell - cell.mean(), glyph - glyph.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            total += float((a * b).sum()) / denom if denom > 1e-6 else 0.0
        return total

    grid = phases(adv)
    steps = range(-PHASE_SEARCH, PHASE_SEARCH + 1)
    # Settle every line's position and phase against the glyphs: the base font's first when
    # there is one (averages over unsettled lines blur whole- and half-pixel glyphs together)
    glyphs: dict[_Key, np.ndarray] = {} if base is not None else averages()
    for _ in range(6):
        moved = False
        for line in lines:
            best = max(
                ((line.x0 + dx, line.y0 + dy, phase) for dy in steps for dx in steps
                 for phase in grid),
                key=lambda at, line=line: agreement(line, at, glyphs),
            )  # fmt: skip
            moved = moved or best != (line.x0, line.y0, line.phase)
            line.x0, line.y0, line.phase = best
        glyphs = averages()
        if not moved:
            break

    samples: dict[_Key, list[np.ndarray]] = defaultdict(list)
    for line in lines:
        for key, cell in cells(line, line.x0, line.y0, line.phase):
            samples[key].append(cell)
    templates = {
        key: template
        for key, cells_ in samples.items()
        if key[0] not in keep
        and (template := _template(cells_, base is None or key[0] not in base.chars)) is not None
    }

    chars = "".join(sorted({c for c, _ in templates} | set(base.chars if base else "")))

    def pick(char: str, half: bool) -> np.ndarray:
        if (char, half) in templates:
            return templates[(char, half)]
        if base is not None and char in base.chars:
            return base.glyph(char, half)
        return templates[(char, not half)]  # only ever seen starting the other way

    whole = np.stack([pick(c, False) for c in chars])
    half = np.stack([pick(c, True) for c in chars]) if fractional else None
    learnt = {c for c, _ in templates} if lossless else set()
    return DebugFont(
        name, chars, adv, cell_w, cell_h, whole,
        lines[0].screen if base is None else base.source, half,
        "".join(sorted(learnt | set(base.lossless if base else ""))),
    )  # fmt: skip


def is_lossless(path: Path) -> bool:
    """PNG (as the app saves frames) keeps every pixel; the game's own screenshots are JPEG."""
    return path.suffix.lower() in (".png", ".bmp", ".tif", ".tiff")


DARK = 70  # a cell whose median whiteness is at most this has a dark background ...
MIN_DARK = 5  # ... and this many such cells make a template on their own


def _template(cells: list[np.ndarray], fallback_to_all: bool) -> np.ndarray | None:
    """The glyph from its cells: the median over those on a dark background (a bright one shows
    the text's shadow and leaks into the template). Too few of them: the median over all, or
    None (keep the base font's glyph)."""
    stack = np.stack(cells)
    background = np.median(stack.reshape(len(cells), -1), axis=1)
    dark = stack[background <= DARK]
    if len(dark) >= MIN_DARK:
        chosen = dark
    elif fallback_to_all:
        chosen = stack
    else:
        return None
    return np.clip(np.rint(np.median(chosen, axis=0)), 0, 255).astype(np.uint8)


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


def check_frames(font: DebugFont, frames: Sequence[Path], names: Sequence[str]) -> Counter[str]:
    """Zone names read from the frames that are not among `names`, and how often."""
    import cv2

    reader = DebugTextReader([font])
    wrong: Counter[str] = Counter()
    for frame in frames:
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            raise CutError(f"{frame.name}: not an image")
        if image.shape[1] / image.shape[0] < 2.5:
            image = corner(image)
        position = parse_position([(line.text, line.score) for line in reader.read(image)])
        for zone in position.zones if position else ():
            if zone.zone not in names:
                wrong[zone.zone] += 1
    return wrong


def run_recut(frames: Sequence[str], names_file: str, font_dir: str | None, out: str | None) -> int:
    paths = [Path(f) for f in frames]
    text = Path(names_file).read_text(encoding="utf-8")
    names = [n.strip() for n in text.splitlines() if n.strip()]
    fonts = load_fonts(Path(font_dir)) if font_dir else load_fonts()
    if not fonts:
        print("error: no font to start from")
        return 2
    try:
        font = recut(paths, names, fonts[0])
    except CutError as err:
        print(f"error: {err}")
        return 2
    path = font.save(Path(out) if out else FONT_DIR)
    print(f"{path}: {len(font.chars)} glyphs, advance {font.advance}, cell "
          f"{font.cell_w}x{font.cell_h}, half-pixel glyphs: {font.half is not None}")  # fmt: skip
    wrong = check_frames(font, paths, names)
    for name, count in wrong.most_common():
        print(f"MISREAD {count}x {name!r}")
    print("read back: every zone name right" if not wrong else f"{sum(wrong.values())} misread")
    return 1 if wrong else 0


def _base_font(font_dir: str | None) -> DebugFont | None:
    fonts = load_fonts(Path(font_dir)) if font_dir else load_fonts()
    if not fonts:
        print("error: no font to start from")
        return None
    return fonts[0]


def run_draft(shots: Sequence[str], font_dir: str | None) -> int:
    base = _base_font(font_dir)
    if base is None:
        return 2
    for shot in map(Path, shots):
        try:
            rows = draft(shot, base)
        except CutError as err:
            print(f"error: {err}")
            return 2
        out = shot.with_name(f"{shot.stem}.draft.txt")
        out.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{out}: {len(rows)} rows")
    print(f"Correct each draft (every {UNKNOWN} and look-alike), keep the lines you trust, and "
          "save it as SHOT.txt next to its screenshot; then run `font learn`.")  # fmt: skip
    return 0


def run_learn(shots: Sequence[str], font_dir: str | None, out: str | None) -> int:
    base = _base_font(font_dir)
    if base is None:
        return 2
    try:
        font, lines, left_out = learn([Path(s) for s in shots], base)
    except CutError as err:
        print(f"error: {err}")
        return 2
    for line in left_out:
        print(f"left out (fits no row of this font): {line}")
    path = font.save(Path(out) if out else FONT_DIR)
    added = "".join(c for c in font.chars if c not in base.chars)
    changed = "".join(
        c for c in base.chars
        if not np.array_equal(font.glyph(c), base.glyph(c))
        or not np.array_equal(font.glyph(c, True), base.glyph(c, True))
    )  # fmt: skip
    print(f"{path}: {len(lines)} lines placed; {len(font.chars)} glyphs")
    print(f"  new: {added or '-'}   re-cut: {changed or '-'}   kept (lossless): {base.lossless}")
    wrong = read_back(font, lines)
    for (meant, read), count in wrong.most_common(12):
        print(f"  read back {count}x {read!r} for {meant!r}")
    total = sum(len(line.text.replace(" ", "")) for line in lines)
    print(f"read back: {total - sum(wrong.values())}/{total} glyphs right")
    return 0


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
