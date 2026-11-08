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


def test_positions_file(fmt_v3: ParquetFormat, profiles: dict[str, HudProfile]) -> None:
    cov = coverage(fmt_v3, slot_layout(profiles["mole"]))
    assert cov.complete  # the app fills every positions column itself
    with_position = replace(_capture(), position=POSITION)
    files = {
        f.kind: f
        for f in build_batch(fmt_v3, cov, [with_position, replace(_capture(), capture_id="c-2")],
                             IDENTITY, ROW_COLUMNS, session_id="s-1", batch_number=1)
    }  # fmt: skip
    assert files["positions"].filename == "u-1_positions_s-1_batch1.parquet"
    table = pq.read_table(io.BytesIO(files["positions"].data))
    assert table.num_rows == 2  # one per Zone line, none for the capture without a position
    rows = table.to_pylist()
    assert rows[1] == {
        "capture_id": "c-1", "zone_index": 2, "zone": "Root", "pos_x": -14427194596.0,
        "pos_y": 7825291981.4, "pos_z": -14901088.1, "cam_dir_x": -155.0, "cam_dir_y": 45.0,
        "cam_dir_z": -179.0, "confidence": 0.91,
    }  # fmt: skip
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
