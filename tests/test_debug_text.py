"""Ship positions from the game's debug text: parser, glyph reader, font cutter, worker, storage
and the optional `positions` upload file.

The reader is tested with a synthetic fixed-width font (OpenCV's Hershey glyphs in 9 px cells,
with a dim shadow); the shipped font is cut from real screenshots with `scanlake-train font cut`.
"""

from __future__ import annotations

import io
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
import respx

from scanlake_core.domain.position import Position, ZonePosition, parse_position
from scanlake_core.format import ParquetFormat, coverage
from scanlake_core.geometry import Rect
from scanlake_core.profile.model import HudProfile
from scanlake_core.schema_map import slot_layout
from scanlake_core.vision.debug_text import (
    DebugFont,
    DebugTextReader,
    corner,
    digits_in_numbers,
)
from scanlake_desktop.capture.source import FileSource, Monitor, write_image
from scanlake_desktop.db.database import Database
from scanlake_desktop.worker import pipeline_worker
from scanlake_desktop.worker.pipeline_worker import LiveState, PipelineWorker, _Loop
from scanlake_io.parquet import build_batch
from tests.test_capture_region import PRIMARY, _config
from tests.test_desktop_data import _mock_session, _mock_upload, _service, _uploader, _values
from tests.test_parquet import IDENTITY, ROW_COLUMNS, _capture

LINES = [
    "CamDir: -155  45 -179 FOV: 60 Focal: 3.00 FStop: 64.0",
    "Zone: MISC_Prospector_915731471510 Pos: -0.01m 9.72m 0.41m",
    "Zone: ab_mine_pyro_regionb_sml_001 Pos: -18.8743km -3354.15m 12.6396km",
    "Zone: SolarSystem_906804811333 Pos: -14427194.5960km 7825291.9814km -14901.0881km",
    "Zone: Root Pos: -14427194.5960km 7825291.9814km -14901.0881km",
]
OTHER_LINES = ["FPS 59.9 [46..60] - 16.7ms", "CamPos Planet Zone: -613612390.421000 1578334280.3"]
# Crops of real 2560x1440 screenshots (JPEG, as the game saves them): CamDir and Zone lines only
REAL = Path(__file__).parent / "fixtures" / "display_info"


def synthetic_font() -> DebugFont:
    # Hershey draws I, l, | and ! alike: I gets serifs, | and ! are left out
    chars = "".join(chr(c) for c in range(33, 127) if chr(c) not in "|!")
    w, h = 9, 15

    def draw(image: np.ndarray, char: str, dx: int, value: int) -> None:
        if char == "I":
            for (x0, y0), (x1, y1) in (((2, 3), (6, 3)), ((4, 3), (4, 10)), ((2, 10), (6, 10))):
                cv2.line(image, (x0 + dx, y0 + dx), (x1 + dx, y1 + dx), value, 1)
        else:
            cv2.putText(image, char, (1 + dx, 10 + dx), cv2.FONT_HERSHEY_PLAIN, 0.8, value, 1,
                        cv2.LINE_AA)  # fmt: skip

    glyphs = []
    for char in chars:
        shadow = np.zeros((h, w), np.uint8)
        draw(shadow, char, 1, 60)
        glyph = np.zeros((h, w), np.uint8)
        draw(glyph, char, 0, 255)
        glyphs.append(np.maximum(shadow // 4, glyph))
    return DebugFont("synthetic", chars, 9.0, w, h, np.stack(glyphs), "test")


@pytest.fixture(scope="module")
def font() -> DebugFont:
    return synthetic_font()


def synthetic_screen(
    font: DebugFont,
    lines: list[str],
    seed: int = 1,
    size: tuple[int, int] = (2560, 1440),
    phases: tuple[float, ...] = (),
) -> np.ndarray:
    """A dark, noisy screen with `lines` right-aligned in the top-right corner, each drawn at its
    grid phase from `phases` (0 for the rest)."""
    rng = np.random.default_rng(seed)
    width, height = size
    screen = rng.integers(0, 90, (height, width, 3)).astype(np.uint8)
    screen = cv2.GaussianBlur(screen, (9, 9), 0)
    for k, text in enumerate(lines):
        image = font.render(text, phase=phases[k] if k < len(phases) else 0.0)
        y, x = 4 + k * 16, width - 6 - image.shape[1]
        region = screen[y : y + font.cell_h, x : x + image.shape[1]]
        ink = image > 0
        for channel in range(3):
            region[..., channel][ink] = image[ink]
    return screen


# -- parser ------------------------------------------------------------------


def test_parse_position_in_metres() -> None:
    position = parse_position([*LINES, *OTHER_LINES])
    assert position is not None
    assert position.cam_dir == (-155.0, 45.0, -179.0)
    assert [z.zone for z in position.zones] == [
        "MISC_Prospector_915731471510", "ab_mine_pyro_regionb_sml_001",
        "SolarSystem_906804811333", "Root",
    ]  # fmt: skip
    cluster = position.zones[1]
    assert (cluster.x, cluster.y, cluster.z) == (-18874.3, -3354.15, 12639.6)
    assert position.zones[3].x == -14427194596.0  # km -> m, no float noise
    assert [z.index for z in position.zones] == [0, 1, 2, 3]
    assert position.local_zone.zone == "ab_mine_pyro_regionb_sml_001"
    assert position.summary() == "ab_mine_pyro_regionb_sml_001: -18.87 / -3.35 / 12.64 km"


def test_unreadable_numbers_drop_their_line_but_keep_the_indices() -> None:
    lines = [
        ("Zone: MISC_Prosp?ctor_1 Pos: -0.01m 9.72m 0.41m", 0.5),  # a name glyph: kept
        ("Zone: ab_mine Pos: -18.87?3km -3354.15m 12.6396km", 0.4),  # a digit: dropped
        ("Zone: Root Pos: 1km 2km 3km", 0.9),
    ]
    position = parse_position(lines)
    assert position is not None and position.cam_dir is None
    assert [(z.index, z.zone) for z in position.zones] == [(0, "MISC_Prosp?ctor_1"), (2, "Root")]
    assert position.confidence == 0.5
    assert position.local_zone.zone == "MISC_Prosp?ctor_1"


def test_no_zone_line_is_no_position() -> None:
    assert parse_position(["CamDir: 1 2 3", *OTHER_LINES]) is None
    assert parse_position([]) is None


def test_gaps_in_zone_names_are_faint_underscores() -> None:
    position = parse_position(
        ["Zone: MISC Prospector 1 Pos: 1m 2m 3m", "Zone: ab mine  x Pos: 1km 2km 3km"]
    )
    assert position is not None
    assert [z.zone for z in position.zones] == ["MISC_Prospector_1", "ab_mine__x"]


@pytest.mark.parametrize(
    ("read", "meant"),
    [
        ("Pos: -O.O1m 9.72m 4O3.9892km", "Pos: -0.01m 9.72m 403.9892km"),
        ("-54.153Ikm -54.153lkm", "-54.1531km -54.1531km"),
        ("SolarSystem_9O6775865828", "SolarSystem_906775865828"),
        ("ab_mine_sml_OO1", "ab_mine_sml_001"),  # settles from the digit inwards
        ("Zone: OOC_Stanton_1_Hurston Pos:", "Zone: OOC_Stanton_1_Hurston Pos:"),
        ("CamDir: 1 2 3 FOV: 55 Focal: 3.00 MISC", "CamDir: 1 2 3 FOV: 55 Focal: 3.00 MISC"),
    ],
)
def test_look_alikes_in_numbers_are_digits(read: str, meant: str) -> None:
    assert digits_in_numbers(read) == meant


# -- reader ------------------------------------------------------------------


def test_reader_reads_camdir_and_zone_lines_only(font: DebugFont) -> None:
    screen = synthetic_screen(font, [*LINES, *OTHER_LINES])
    lines = DebugTextReader([font]).read(corner(screen))
    texts = [line.text for line in lines]
    assert texts[:5] == LINES
    # "CamPos Planet Zone: ..." holds "Zone:" too; it reads, but parses to nothing
    assert all(t.startswith(("CamDir:", "Zone:")) for t in texts)
    assert min(line.score for line in lines[:5]) > 0.8
    position = parse_position([(line.text, line.score) for line in lines])
    assert position is not None and len(position.zones) == 4


@pytest.mark.parametrize(
    ("line", "angles"),
    [
        ("CamDir:   9  11 -130 FOV: 60 Focal: 3.00 FStop: 64.0", (9.0, 11.0, -130.0)),
        ("CamDir: -12   5    7 FOV: 60 Focal: 3.00 FStop: 64.0", (-12.0, 5.0, 7.0)),
        ("CamDir:   0   0    0 FOV: 60 Focal: 3.00 FStop: 64.0", (0.0, 0.0, 0.0)),
    ],
)
def test_reader_reads_camdir_with_one_digit_angles(
    font: DebugFont, line: str, angles: tuple[float, float, float]
) -> None:
    # The game pads the angles to columns of 3, 3 and 4: a one-digit angle leaves 3 blanks
    # before it, 4 in the last column; none of that may end the line
    screen = synthetic_screen(font, [line, *LINES[1:]])
    lines = DebugTextReader([font]).read(corner(screen))
    assert lines[0].text == line
    position = parse_position([(found.text, found.score) for found in lines])
    assert position is not None and position.cam_dir == angles


def test_reader_follows_each_lines_grid_phase(font: DebugFont, tmp_path: Path) -> None:
    from scanlake_training import debug_font

    # A fractional advance steps 9, 10, 9, 10 ... or 10, 9, 10, 9 ... px, depending on where the
    # right-aligned line starts; the real font is 7.5 px
    half = replace(font, name="half", advance=9.5)
    assert half.phases() == (0.0, 0.5)
    phases = (0.0, 0.5, 0.5, 0.0, 0.5)
    screen = synthetic_screen(half, LINES, phases=phases)
    assert [line.text for line in DebugTextReader([half]).read(corner(screen))] == LINES

    shot = tmp_path / "shot.png"
    cv2.imwrite(str(shot), synthetic_screen(half, [*LINES, *OTHER_LINES], seed=4, phases=phases))
    shot.with_suffix(".txt").write_text("\n".join([*LINES, *OTHER_LINES]) + "\n", "utf-8")
    cut = debug_font.cut([shot])
    assert cut.advance == 9.5
    assert debug_font.check(cut, [shot]) == []


@pytest.mark.parametrize(
    ("fixture", "cam_dir", "zones", "local"),
    [
        (
            "ship_over_pyro3.png",
            (103.0, -8.0, -15.0),
            ["MISC_Prospector_915731471510", "pyro3", "SolarSystem_906775865828", "Root"],
            ("pyro3", 174741.3, 557833.7, -5231.22),
        ),
        (
            "on_foot_bright_sky.png",
            (93.0, 26.0, 132.0),
            ["pyro3", "SolarSystem_906775865828", "Root"],
            ("pyro3", -71975.2, 403989.2, -54153.1),
        ),
        (
            "asteroid_field.png",
            (-155.0, 45.0, -179.0),
            ["MISC_Prospector_915731471510", "ab_mine_pyro_regionb_sml_001",
             "SolarSystem_906804011333", "Root"],
            ("ab_mine_pyro_regionb_sml_001", -18874.3, -3354.15, 12639.6),
        ),
        (  # one-digit angles pad CamDir with 3 blanks: once read as just "CamDir:"
            "jumppoint_single_digits.png",
            (9.0, 11.0, -130.0),
            ["MISC_Prospector_915731471510", "jumppoint_pyro_nyx",
             "SolarSystem_906804011333", "Root"],
            ("jumppoint_pyro_nyx", 11605.7, -14207.9, 15046.5),
        ),
        (
            "stanton_calliope.png",
            (-139.0, -43.0, -58.0),
            ["MISC_Prospector_915731471510", "OOC_Stanton_4a_Calliope",
             "SolarSystem_906804011331", "Root"],
            ("OOC_Stanton_4a_Calliope", 61544.7, -230062.2, -34480.1),
        ),
        # Lossless, as the app grabs it; a sunlit hull behind CamDir and the station's name
        # once hid both lines and turned "keeger_segment" into "keeoer_segmept"
        (
            "live_station_bright.png",
            (39.0, -8.0, -64.0),
            ["MISC_Prospector_915731471510", "social_001_keeger_segment_rckcrk_102",
             "SolarSystem_911715898307", "Root"],
            ("social_001_keeger_segment_rckcrk_102", 6851.86, 56478.7, -9654.91),
        ),
        (
            "live_station.png",
            (39.0, -8.0, -64.0),
            ["MISC_Prospector_915731471510", "social_001_keeger_segment_rckcrk_102",
             "SolarSystem_911715898307", "Root"],
            ("social_001_keeger_segment_rckcrk_102", 6851.9, 56478.6, -9654.97),
        ),
    ],
)  # fmt: skip
def test_shipped_font_reads_real_screenshots(
    fixture: str, cam_dir: tuple[float, ...], zones: list[str], local: tuple[object, ...]
) -> None:
    reader = DebugTextReader()  # the font in scanlake_core/resources/display_info_font
    assert reader.usable
    image = cv2.imread(str(REAL / fixture))
    assert image is not None
    lines = reader.read(image)
    position = parse_position([(line.text, line.score) for line in lines])
    assert position is not None
    assert position.cam_dir == cam_dir
    assert [z.zone for z in position.zones] == zones
    z = position.local_zone
    assert (z.zone, z.x, z.y, z.z) == local
    assert position.confidence > 0.7


def test_reader_finds_lines_the_coarse_search_missed(
    font: DebugFont, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scanlake_core.vision import debug_text

    # Over a bright background a prefix can score too low at the coarse scale: here the coarse
    # search loses every line but the second; the row grid brings them back
    screen = corner(synthetic_screen(font, [*LINES, *OTHER_LINES]))
    find = debug_text._find
    second = 4 + 16  # synthetic_screen puts line k at y = 4 + k * 16

    def lossy(*args: object, **kwargs: object) -> list[tuple[int, int, float]]:
        hits = find(*args, **kwargs)  # type: ignore[arg-type]
        return [hit for hit in hits if abs(hit[1] - second) <= 2]

    monkeypatch.setattr(debug_text, "_find", lossy)
    texts = [line.text for line in DebugTextReader([font]).read(screen)]
    assert texts[:5] == LINES


def test_half_pixel_glyphs(font: DebugFont, tmp_path: Path) -> None:
    # With a 9.5 px advance every other glyph starts half a pixel into its cell, where the
    # game draws it differently (smoothed across the pixel boundary)
    half = replace(font, name="half", advance=9.5, half=_half_drawn(font.glyphs))
    assert half.is_half(1) and not half.is_half(2) and half.is_half(0, phase=0.5)
    image = half.render("ZZ")
    assert np.array_equal(image[:, :9], half.glyph("Z"))
    assert np.array_equal(image[:, 9:18], half.glyph("Z", half=True))
    assert not np.array_equal(half.glyph("Z"), half.glyph("Z", half=True))

    replace(half, lossless="Zo").save(tmp_path)
    assert (tmp_path / "half.half.png").exists()
    loaded = DebugFont.load(tmp_path / "half.json")
    assert loaded.half is not None and half.half is not None
    assert np.array_equal(loaded.half, half.half)
    assert loaded.lossless == "Zo"
    assert DebugFont.load(_saved(font, tmp_path)).half is None  # a font without them

    screen = synthetic_screen(half, LINES, phases=(0.0, 0.5, 0.5, 0.0, 0.5))
    assert [line.text for line in DebugTextReader([half]).read(corner(screen))] == LINES


def _saved(font: DebugFont, folder: Path) -> Path:
    return font.save(folder / "plain")


def _half_drawn(glyphs: np.ndarray) -> np.ndarray:
    """Glyphs drawn a little differently, smoothed towards the right: a quarter of each column
    from its left neighbour (a pure one-pixel shift would be the whole glyph in a cell one
    pixel further on; half and half blurs this thin font's n into an m)."""
    shifted = np.roll(glyphs, 1, axis=2).astype(np.int32)
    return ((3 * glyphs.astype(np.int32) + shifted) // 4).astype(np.uint8)


def test_reader_finds_nothing_without_text_or_font(font: DebugFont) -> None:
    empty = synthetic_screen(font, [])
    assert DebugTextReader([font]).read(corner(empty)) == []
    assert DebugTextReader([font]).read(np.zeros((0, 0, 3), np.uint8)) == []
    no_font = DebugTextReader([])
    assert not no_font.usable
    assert no_font.read(corner(synthetic_screen(font, LINES))) == []


def test_font_round_trip(font: DebugFont, tmp_path: Path) -> None:
    font.save(tmp_path)
    loaded = DebugFont.load(tmp_path / "synthetic.json")
    assert loaded.chars == font.chars and loaded.advance == font.advance
    assert np.array_equal(loaded.glyphs, font.glyphs)


def test_cut_rebuilds_a_font_that_reads_its_screenshots(font: DebugFont, tmp_path: Path) -> None:
    from scanlake_training import debug_font

    shot = tmp_path / "shot.png"
    cv2.imwrite(str(shot), synthetic_screen(font, [*LINES, *OTHER_LINES], seed=3))
    shot.with_suffix(".txt").write_text("\n".join([*LINES, *OTHER_LINES]) + "\n", "utf-8")
    cut = debug_font.cut([shot])
    assert cut.advance == 9.0 and cut.source == "2560x1440"
    assert set("Zone:CamDir-.0123456789") <= set(cut.chars)
    assert debug_font.check(cut, [shot]) == []
    assert debug_font.run_read(str(shot), str(cut.save(tmp_path / "font").parent)) == 0


def test_recut_learns_half_pixel_glyphs_from_live_frames(font: DebugFont, tmp_path: Path) -> None:
    from scanlake_training import debug_font

    # The live text, drawn with distinct half-pixel glyphs; the base font knows only one kind
    truth = replace(font, name="truth", advance=9.5, half=_half_drawn(font.glyphs))
    base = replace(truth, name="base", half=None)
    names = ["MISC_Prospector_915731471510", "ab_mine_pyro_regionb_sml_001",
             "SolarSystem_906804811333", "Root"]  # fmt: skip
    frames = []
    for seed in range(8):
        frame = tmp_path / f"frame{seed}.png"
        phases = tuple(float(p) for p in np.random.default_rng(seed).integers(0, 2, 5) / 2)
        cv2.imwrite(str(frame), corner(synthetic_screen(truth, LINES, seed=seed, phases=phases)))
        frames.append(frame)

    def alike(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.astype(float) - a.mean(), b.astype(float) - b.mean()
        return float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))

    recut = debug_font.recut(frames, names, base)
    assert recut.half is not None and recut.chars == base.chars
    for char in "Zone:":  # in every line; each start learnt on its own (over a noisy background)
        whole, half = recut.glyph(char), recut.glyph(char, half=True)
        assert alike(whole, truth.glyph(char)) > alike(whole, truth.glyph(char, half=True))
        assert alike(half, truth.glyph(char, half=True)) > alike(half, truth.glyph(char))
        assert min(alike(whole, truth.glyph(char)), alike(half, truth.glyph(char, True))) > 0.85
    assert np.array_equal(recut.glyph("x"), base.glyph("x"))  # never shown: kept
    assert debug_font.check_frames(recut, frames, names) == {}
    assert debug_font.check_frames(recut, frames, names[1:]) == {names[0]: 8}  # not listed


def _without(font: DebugFont, gone: str, **changes: object) -> DebugFont:
    keep = [i for i, c in enumerate(font.chars) if c not in gone]
    half = font.half[keep] if font.half is not None else None
    chars = "".join(font.chars[i] for i in keep)
    return replace(font, chars=chars, glyphs=font.glyphs[keep], half=half, **changes)  # type: ignore[arg-type]


def test_learn_adds_glyphs_from_any_debug_line(font: DebugFont, tmp_path: Path) -> None:
    from scanlake_training import debug_font

    truth = replace(font, name="truth", advance=9.5, half=_half_drawn(font.glyphs))
    base = _without(truth, "xqz", name="base", lossless="Zo")  # Z and o: from lossless captures
    extra = ["Box: quiz 12, xq zz", "Quiz Pos: 0.5ms xqz 7"]
    screen = synthetic_screen(truth, [*LINES, *extra], phases=(0.0, 0.5) * 4)
    # A debug line in a narrower font (the game draws some that way): fits no row of ours
    narrow = replace(truth, advance=8.0, half=None).render("Net: ping quiz xq 26.4ms")
    y, x = 4 + 7 * 16, screen.shape[1] - 6 - narrow.shape[1]
    region = screen[y : y + truth.cell_h, x : x + narrow.shape[1]]
    for channel in range(3):
        region[..., channel][narrow > 0] = narrow[narrow > 0]
    shot = tmp_path / "shot.png"
    cv2.imwrite(str(shot), screen)
    draft = debug_font.draft(shot, base)
    assert draft[:5] == LINES  # rows top first, as the base reads them
    assert all(line not in draft for line in extra)  # x, q, z unknown yet
    told = [*LINES[1:], *extra]  # any lines, in any order
    shot.with_suffix(".txt").write_text(
        "\n".join(["# zone lines, the extra ones, the narrow one", *told,
                   "Net: ping quiz xq 26.4ms"]),
        encoding="utf-8",
    )  # fmt: skip

    learnt, lines, left_out = debug_font.learn([shot], base)
    assert set("xqz") <= set(learnt.chars) and [line.text for line in lines] == told
    assert left_out == ["shot.png: Net: ping quiz xq 26.4ms"]
    assert debug_font.read_back(learnt, lines) == {}
    assert set("xqz") <= set(learnt.lossless)  # cut from a lossless capture
    assert debug_font.draft(shot, learnt)[5:7] == extra

    lossy, _, _ = debug_font.learn([shot], base, lossless=False)
    for char in "Zo":  # what lossless captures gave stays; other glyphs are re-cut
        assert np.array_equal(lossy.glyph(char), base.glyph(char))
    assert not np.array_equal(lossy.glyph("e"), base.glyph("e"))
    assert lossy.lossless == "Zo" and set("xqz") <= set(lossy.chars)


@pytest.mark.parametrize(
    ("read", "meant"),
    [
        ("social 001 keeoer seomept rckcrk 102", "social_001_keeger_segment_rckcrk_102"),
        ("ab_mibe_pyro_regionb_sml_001", "ab_mine_pyro_regionb_sml_001"),
        ("Root", "Root"),
        ("Rxyz", None),  # 3 of 4 differ
        ("ab_mine_pyro_regionb_sml_0011", None),  # another length
    ],
)
def test_closest_name(read: str, meant: str | None) -> None:
    from scanlake_training.debug_font import closest_name

    names = ["social_001_keeger_segment_rckcrk_102", "ab_mine_pyro_regionb_sml_001", "Root"]
    assert closest_name(read, names) == meant


# -- capture -----------------------------------------------------------------


def test_debug_text_rect_is_the_top_right_corner() -> None:
    assert PRIMARY.debug_text_rect == Rect(1280, 0, 1280, 360)
    left = Monitor(1, -1920, 0, 1920, 1080)
    assert left.debug_text_rect == Rect(-960, 0, 960, 270)


def test_worker_reads_the_position_now_and_then(
    font: DebugFont, profiles: dict[str, HudProfile], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    frame_path = tmp_path / "screen.png"
    frame = synthetic_screen(font, LINES)
    write_image(frame_path, frame)
    loop = _Loop(PipelineWorker())
    loop.config = replace(_config(profiles["golem"], PRIMARY), replay=(frame_path,))
    loop.source = FileSource([frame_path])
    reader = DebugTextReader([font])
    reads: list[int] = []
    original = reader.read
    monkeypatch.setattr(reader, "read", lambda image: reads.append(1) or original(image))
    loop.position_reader = reader

    now = time.time()
    loop._read_position(now, frame)
    assert loop.position is not None and loop.position_t == now
    assert loop.position.zones[1].zone == "ab_mine_pyro_regionb_sml_001"
    assert [text for text, _ in loop.position_lines] == LINES  # what the live panel shows
    loop._read_position(now + 0.5, frame)  # too soon
    assert len(reads) == 1
    loop._read_position(now + pipeline_worker.POSITION_EVERY_S, frame)
    assert len(reads) == 2

    blank = synthetic_screen(font, [])  # debug text turned off: look less often, keep the last
    loop._read_position(now + 5, blank)
    assert len(reads) == 3 and loop.position_t == now + pipeline_worker.POSITION_EVERY_S
    assert loop.position_lines == ()
    loop._read_position(now + 5 + pipeline_worker.POSITION_EVERY_S, blank)
    assert len(reads) == 3
    loop._read_position(now + 5 + pipeline_worker.POSITION_IDLE_S, blank)
    assert len(reads) == 4


def test_worker_reads_faster_while_navigating(
    font: DebugFont, profiles: dict[str, HudProfile], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    frame_path = tmp_path / "screen.png"
    frame = synthetic_screen(font, LINES)
    write_image(frame_path, frame)
    loop = _Loop(PipelineWorker())
    loop.config = replace(
        _config(profiles["golem"], PRIMARY), replay=(frame_path,), position_every_s=0.25
    )
    loop.source = FileSource([frame_path])
    reader = DebugTextReader([font])
    reads: list[int] = []
    original = reader.read
    monkeypatch.setattr(reader, "read", lambda image: reads.append(1) or original(image))
    loop.position_reader = reader

    now = time.time()
    for k in range(5):
        loop._read_position(now + k * 0.25, frame)
    assert len(reads) == 5
    # no debug text: still only every POSITION_IDLE_S
    blank = synthetic_screen(font, [])
    later = now + 10
    loop._read_position(later, blank)
    loop._read_position(later + 0.25, blank)
    assert len(reads) == 6


def test_worker_reads_positions_on_their_own_thread(
    qtbot: object, font: DebugFont, profiles: dict[str, HudProfile],
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    # Live, a HUD search frame can take seconds while the game runs: the debug text is read by
    # a thread of its own and every position sent at once, whatever the loop is doing
    screen = corner(synthetic_screen(font, LINES))

    class Screen:
        def grab(self, rect: Rect) -> np.ndarray:
            return screen

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline_worker, "MssSource", Screen)
    worker = PipelineWorker()
    got: list[pipeline_worker.PositionRead] = []
    worker.position.connect(got.append)
    loop = _Loop(worker)  # its own frames never run here
    loop.config = replace(_config(profiles["golem"], PRIMARY), position_every_s=0.25)
    loop.position_reader = DebugTextReader([font])
    loop.positions = pipeline_worker._PositionThread(loop)
    loop.positions.start()
    try:
        qtbot.waitUntil(lambda: len(got) >= 4, timeout=5000)  # type: ignore[attr-defined]
    finally:
        loop.close()
    assert not loop.positions.is_alive()
    assert got[0].position.zones[1].zone == "ab_mine_pyro_regionb_sml_001"
    assert [text for text, _ in got[0].lines] == LINES
    gaps = [b.position_t - a.position_t for a, b in zip(got, got[1:], strict=False)]
    assert all(0.2 <= gap < 0.6 for gap in gaps), gaps


def test_live_panel_previews_the_position(qtbot) -> None:  # type: ignore[no-untyped-def]
    from scanlake_desktop.ui.live_panel import LOCAL, LivePanel

    panel = LivePanel()
    qtbot.addWidget(panel)
    base = LiveState(t=100.0, ship="mole", locked=False, scale=1.0, fps=2.0, backend="ort-cpu",
                     region=Rect(0, 0, 10, 10), black=False, votes={}, readings={})  # fmt: skip
    panel.update_state(base)
    assert panel.position_title() == "Ship position: not read"
    assert "r_DisplayInfo" in panel.position_body()

    # read without a scan HUD in sight: the preview works before aiming at a rock
    position = parse_position(LINES)
    lines = tuple((text, 0.9) for text in LINES)
    panel.update_state(replace(base, position=position, position_t=99.5, position_lines=lines))
    assert panel.position_title() == "Ship position · read 0.5 s ago · match 100%"
    body = panel.position_body().splitlines()
    assert body[0] == "CamDir: -155 45 -179"
    assert body[2] == f"ab_mine_pyro_regionb_sml_001: -18.87 / -3.35 / 12.64 km   {LOCAL}"

    panel.update_state(replace(base, t=120.0, position=position, position_t=99.5))
    assert "last read 20 s ago" in panel.position_title()  # F9 would not take it


# -- storage and upload --------------------------------------------------------

POSITION = Position(
    zones=(
        ZonePosition(0, "MISC_Prospector_1", -0.01, 9.72, 0.41, 0.93),
        ZonePosition(2, "Root", -14427194596.0, 7825291981.4, -14901088.1, 0.91),
    ),
    cam_dir=(-155.0, 45.0, -179.0),
)


def test_positions_are_saved_and_kept_on_edit(
    tmp_path: Path, fmt: ParquetFormat, profiles: dict[str, HudProfile]
) -> None:
    db = Database(tmp_path / "scanlake.db")
    layout = slot_layout(profiles["mole"])
    cov = coverage(fmt, layout)
    service = _service(db)
    assert service.save(*_values(), layout, cov, ship="mole", position=POSITION).saved
    assert service.save(*_values("9000"), layout, cov, ship="mole").saved
    with_position, without = Database(tmp_path / "scanlake.db").captures(
        service.session.session_id  # type: ignore[union-attr]
    )
    assert with_position.position == POSITION and without.position is None

    scan, rows = _values("11900")
    assert service.save(scan, rows, layout, cov, ship="mole", editing=with_position).saved
    edited = db.capture(with_position.capture_id)
    assert edited is not None and edited.scan["mass"].text == "11900"
    assert edited.position == POSITION  # an edit never drops (or replaces) the position


def _zones(*zones: tuple[str, float, float, float]) -> Position:
    return Position(zones=tuple(ZonePosition(i, *z) for i, z in enumerate(zones)))


SHIP_LINE = ("MISC_Prospector_915731471510", -0.01, 9.72, 0.41)
SYSTEM_LINE = ("SolarSystem_906804011333", -14427194596.0, 7825291981.4, -14901088.1)
ROOT_LINE = ("Root", -14427194596.0, 7825291981.4, -14901088.1)


@pytest.mark.parametrize(
    ("position", "regional"),
    [
        # in an asteroid field, at a planet, at a station: that zone
        (_zones(SHIP_LINE, ("ab_mine_pyro_regionb_sml_001", -18874.3, -3354.15, 12639.6),
                SYSTEM_LINE, ROOT_LINE), "ab_mine_pyro_regionb_sml_001"),
        (_zones(("pyro3", -71975.2, 403989.2, -54153.1), SYSTEM_LINE, ROOT_LINE), "pyro3"),
        (_zones(SHIP_LINE, ("social_001_keeger_segment_rckcrk_102", 3417.41, 41098.9, 14287.3),
                SYSTEM_LINE), "social_001_keeger_segment_rckcrk_102"),
        # in no zone (open space, a cluster outside any field): the solar system's
        (_zones(SHIP_LINE, SYSTEM_LINE, ROOT_LINE), SYSTEM_LINE[0]),
        (_zones(SHIP_LINE, ROOT_LINE), "Root"),  # the SolarSystem line was not read
        (_zones(SYSTEM_LINE, ROOT_LINE), SYSTEM_LINE[0]),
        (_zones(SHIP_LINE), None),  # only the ship's own frame: nothing to place the scan by
    ],
)  # fmt: skip
def test_regional_zone(position: Position, regional: str | None) -> None:
    found = position.regional_zone
    assert (found.zone if found else None) == regional


def test_positions_file(fmt_v3: ParquetFormat, profiles: dict[str, HudProfile]) -> None:
    cov = coverage(fmt_v3, slot_layout(profiles["mole"]))
    assert cov.complete  # the app fills every positions column itself
    in_open_space = replace(_capture(), position=POSITION)
    in_field = replace(
        _capture(),
        capture_id="c-3",
        position=_zones(SHIP_LINE, ("ab_mine_x", 1.0, 2.0, 3.0), SYSTEM_LINE, ROOT_LINE),
    )
    captures = [in_open_space, replace(_capture(), capture_id="c-2"), in_field]
    files = {
        f.kind: f
        for f in build_batch(fmt_v3, cov, captures, IDENTITY, ROW_COLUMNS, session_id="s-1",
                             batch_number=1)
    }  # fmt: skip
    assert files["positions"].filename == "u-1_positions_s-1_batch1.parquet"
    table = pq.read_table(io.BytesIO(files["positions"].data))
    # one row per scan with a position: the regional zone, else the solar system's
    rows = table.to_pylist()
    assert rows[0] == {
        "capture_id": "c-1", "zone_index": 2, "zone": "Root", "pos_x": -14427194596.0,
        "pos_y": 7825291981.4, "pos_z": -14901088.1, "cam_dir_x": -155.0, "cam_dir_y": 45.0,
        "cam_dir_z": -179.0, "confidence": 0.91,
    }  # fmt: skip
    assert rows[1] == {
        "capture_id": "c-3", "zone_index": 1, "zone": "ab_mine_x", "pos_x": 1.0, "pos_y": 2.0,
        "pos_z": 3.0, "cam_dir_x": None, "cam_dir_y": None, "cam_dir_z": None, "confidence": 1.0,
    }  # fmt: skip
    assert len(rows) == 2
    assert str(table.schema.field("zone_index").type) == "int32"

    plain = build_batch(fmt_v3, cov, [_capture()], IDENTITY, ROW_COLUMNS, session_id="s-1",
                        batch_number=2)  # fmt: skip
    assert {f.kind for f in plain} == {"scans", "compositions", "confidences"}


@pytest.mark.parametrize("opted_in", [False, True])
@respx.mock
def test_positions_upload_only_when_opted_in(
    opted_in: bool, tmp_path: Path, fmt_v3: ParquetFormat, profiles: dict[str, HudProfile]
) -> None:
    db = Database(tmp_path / "scanlake.db")
    layout = slot_layout(profiles["mole"])
    service = _service(db)
    service.save(*_values(), layout, coverage(fmt_v3, layout), ship="mole", position=POSITION)
    service.queue_upload()
    _mock_session(respx.mock)
    sent = _mock_upload(respx.mock)
    uploader = _uploader(db, fmt_v3, tmp_path)
    uploader.upload_positions = lambda: opted_in

    [outcome] = uploader.run_due()
    assert outcome.ok, outcome.message
    assert ("positions" in sent[0]) is opted_in
    assert {"scans", "compositions", "confidences"} <= set(sent[0])
    exported = sorted(p.name.split("_")[1] for p in (tmp_path / "export").iterdir())
    assert exported == ["compositions", "confidences", "positions", "scans"]  # always kept


@respx.mock
def test_scans_without_positions_send_no_positions_file(
    tmp_path: Path, fmt_v3: ParquetFormat, profiles: dict[str, HudProfile]
) -> None:
    db = Database(tmp_path / "scanlake.db")
    layout = slot_layout(profiles["mole"])
    service = _service(db)
    service.save(*_values(), layout, coverage(fmt_v3, layout), ship="mole")
    service.queue_upload()
    _mock_session(respx.mock)
    sent = _mock_upload(respx.mock)
    uploader = _uploader(db, fmt_v3, tmp_path)
    uploader.upload_positions = lambda: True
    assert uploader.run_due()[0].ok
    assert len(sent) == 1 and "positions" not in sent[0]
