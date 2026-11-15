"""Cluster summaries, the live offset to a cluster, the compass and the CamDir calibration."""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scanlake_core.domain import navigation as nav
from scanlake_core.domain.capture import Capture, FieldValue
from scanlake_core.domain.navigation import (
    GAME_CONVENTION,
    HOLD_S,
    CalibrationSample,
    CameraCheck,
    CameraFrames,
    ClusterSummary,
    Convention,
    NavFix,
    NavTracker,
    all_conventions,
    camera_axes,
    compass,
    fit_convention,
    motion_samples,
    nav_frame,
    nav_frames,
    summarize_clusters,
)
from scanlake_core.domain.position import Position, ZonePosition

SHIP = "MISC_Prospector_915731471510"
FIELD = "ab_mine_pyro_regionb_sml_001"
OTHER_FIELD = "ab_mine_pyro_regiona_med_002"
PLANET = "pyro3"
SYSTEM = "SolarSystem_906804011333"

# The engine's Ang3 order: Rz(z) · Ry(y) · Rx(x), angles shown x y z
ENGINE = Convention(axes=(0, 1, 2), order=(2, 1, 0), signs=(1, 1, 1))


def _position(
    *zones: tuple[str, float, float, float], cam_dir: tuple[float, float, float] | None = None
) -> Position:
    return Position(
        zones=tuple(ZonePosition(i, name, x, y, z) for i, (name, x, y, z) in enumerate(zones)),
        cam_dir=cam_dir,
    )


def _in_field(x: float, y: float, z: float, cam_dir: tuple[float, float, float] | None = None):
    """A ship in the asteroid field; the field sits at (1e9, 2e9, -1e7) in the system."""
    return _position(
        (SHIP, -0.01, 9.72, 0.41),
        (FIELD, x, y, z),
        (SYSTEM, 1e9 + x, 2e9 + y, -1e7 + z),
        ("Root", 1e9 + x, 2e9 + y, -1e7 + z),
        cam_dir=cam_dir,
    )


FIELD_ORIGIN = (1e9, 2e9, -1e7)
TURN = 85.0  # a Pyro asteroid field's frame, measured: turned 85° about z against the system's


def _rz(v: tuple[float, float, float], degrees: float) -> tuple[float, float, float]:
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


def _turned_field(
    offset: tuple[float, float, float], cam_dir: tuple[float, float, float] | None = None
) -> Position:
    """A ship in a field whose frame is turned TURN° against the system's; `offset` is from the
    field's origin in the system's axes."""
    system = (FIELD_ORIGIN[0] + offset[0], FIELD_ORIGIN[1] + offset[1], FIELD_ORIGIN[2] + offset[2])
    return _position(
        (SHIP, -0.01, 9.72, 0.41),
        (FIELD, *_rz(offset, -TURN)),
        (SYSTEM, *system),
        ("Root", *system),
        cam_dir=cam_dir,
    )


def _capture(
    cluster: str, *, deposit: str = "Quartz", volume: str = "10", position: Position | None
) -> Capture:
    return Capture(
        capture_id=f"{cluster}-{deposit}-{volume}-{id(position)}",
        session_id="s",
        cluster_id=cluster,
        timestamp="2026-11-08T12:00:00.000Z",
        scan={"deposit": FieldValue(deposit), "volume": FieldValue(volume)},
        position=position,
    )


# -- frames and summaries -------------------------------------------------------------------


def test_nav_frames_skip_the_vehicle() -> None:
    assert [z.zone for z in nav_frames(_in_field(1, 2, 3))] == [FIELD, SYSTEM, "Root"]
    open_space = _position((SHIP, 0.0, 9.7, 0.4), (SYSTEM, 5e9, 1e9, 0.0), ("Root", 5e9, 1e9, 0.0))
    assert [z.zone for z in nav_frames(open_space)] == [SYSTEM, "Root"]
    on_foot = _position((PLANET, -71975.2, 403989.2, -54153.1), (SYSTEM, 1e9, 2e9, -54153.1))
    assert [z.zone for z in nav_frames(on_foot)] == [PLANET, SYSTEM]


def test_summarize_clusters() -> None:
    captures = [
        _capture("a", volume="40", position=_in_field(1000, 2000, 3000)),
        _capture("a", volume="2.5", position=_in_field(3000, 4000, 5000)),
        _capture("a", volume="0.02526", position=None),  # a milli-SCU rock without position
        _capture("b", deposit="Gold", volume="7,5", position=None),
    ]
    summaries = summarize_clusters(captures, ["a", "empty", "b"])
    assert [s.cluster_id for s in summaries] == ["a", "b"]
    a, b = summaries
    assert (a.number, a.deposit, a.rocks) == (1, "Quartz", 3)
    assert a.total_scu == pytest.approx(42.52526)
    assert a.local_zone == FIELD
    assert a.centres[FIELD] == pytest.approx((2000, 3000, 4000))
    assert a.centres[SYSTEM] == pytest.approx((1e9 + 2000, 2e9 + 3000, -1e7 + 4000))
    assert SHIP not in a.centres
    assert a.centre is not None and a.centre.zone == FIELD
    assert (b.number, b.deposit, b.total_scu, b.has_position) == (3, "Gold", 7.5, False)
    assert b.centre is None


# -- live offset ----------------------------------------------------------------------------


def _target(*captures: Capture) -> ClusterSummary:
    return summarize_clusters(captures, [captures[0].cluster_id])[0]


def test_nav_frame_in_the_same_field() -> None:
    target = _target(_capture("a", position=_in_field(1000, 0, 0)))
    frame = nav_frame(_in_field(1000, 300, 400), target)
    assert frame is not None
    assert (frame.zone, frame.approximate) == (FIELD, False)
    assert frame.distance == pytest.approx(500)
    assert frame.delta == pytest.approx((0, -300, -400))


def test_nav_frame_from_another_field_uses_the_system() -> None:
    target = _target(_capture("a", position=_in_field(1000, 0, 0)))
    live = _position(
        (SHIP, 0.0, 9.7, 0.4),
        (OTHER_FIELD, 5.0, 5.0, 5.0),
        (SYSTEM, 1e9 + 1000, 2e9 + 50_000, -1e7),
    )
    frame = nav_frame(live, target)
    assert frame is not None
    assert (frame.zone, frame.approximate) == (SYSTEM, True)
    assert frame.distance == pytest.approx(50_000)


def test_nav_frame_none_without_a_shared_zone() -> None:
    target = _target(_capture("a", position=_in_field(1000, 0, 0)))
    elsewhere = _position((SHIP, 0.0, 9.7, 0.4), ("SolarSystem_1", 0.0, 0.0, 0.0))
    assert nav_frame(elsewhere, target) is None


def test_tracker_closing_speed_and_eta() -> None:
    target = _target(_capture("a", position=_in_field(0, 10_000, 0)))
    tracker = NavTracker(target)
    fix = None
    for k in range(9):  # 2 s at 4 Hz, closing at 100 m/s
        t = 100.0 + k * 0.25
        fix = tracker.update(_in_field(0, k * 25.0, 0), t, now=t + 0.1)
    assert fix is not None
    assert fix.distance == pytest.approx(9800)
    assert fix.closing_speed == pytest.approx(100)
    assert fix.eta == pytest.approx(98)
    assert fix.age == pytest.approx(0.1)
    assert fix.compass is None  # no convention yet
    # the same reading again only ages
    again = tracker.update(_in_field(0, 999.0, 0), 102.0, now=103.0)
    assert again.distance == pytest.approx(9800)
    assert again.age == pytest.approx(1.0)


def test_tracker_holds_the_frame_through_a_missed_line() -> None:
    # Scanned on another server: only Root is shared, and the reader misses its line now and then
    scanned = _position(
        (SHIP, 0.0, 9.7, 0.4), ("SolarSystem_1", 5e9, 1e9, 0.0), ("Root", 5e9, 1e9, 0.0)
    )
    tracker = NavTracker(_target(_capture("a", position=scanned)))

    def live(y: float, root: bool = True) -> Position:
        zones = [(SHIP, 0.0, 9.7, 0.4), ("SolarSystem_2", 5e9, 1e9 + y, 0.0)]
        return _position(*zones, *([("Root", 5e9, 1e9 + y, 0.0)] if root else []))

    fix = tracker.update(live(1000), 1.0, now=1.0)
    assert fix.frame is not None and fix.frame.zone == "Root"
    assert fix.distance == pytest.approx(1000)
    fix = tracker.update(live(900, root=False), 1.3, now=1.3)
    assert fix.distance == pytest.approx(1000) and fix.age == pytest.approx(0.3)  # held
    fix = tracker.update(live(800), 1.6, now=1.6)
    assert fix.distance == pytest.approx(800) and fix.age == pytest.approx(0)
    # A line gone for good counts once HOLD_S has passed
    for t in (2.0, 3.0, 1.7 + HOLD_S):
        fix = tracker.update(live(700, root=False), t, now=t)
    assert fix.frame is None and fix.age == pytest.approx(0)


def test_tracker_keeps_the_inner_frame_through_a_missed_line() -> None:
    tracker = NavTracker(_target(_capture("a", position=_in_field(1000, 0, 0))))

    def outside(y: float) -> Position:  # no field line: as read outside the field
        system = (1e9 + 1000, 2e9 + y, -1e7)
        return _position((SHIP, 0.0, 9.7, 0.4), (SYSTEM, *system), ("Root", *system))

    def zone(fix: NavFix) -> str | None:
        return fix.frame.zone if fix.frame else None

    assert zone(tracker.update(_in_field(1000, 500, 0), 1.0, now=1.0)) == FIELD
    assert zone(tracker.update(outside(400), 1.3, now=1.3)) == FIELD  # held
    fix = tracker.update(outside(300), 1.1 + HOLD_S, now=1.1 + HOLD_S)  # left the field
    assert zone(fix) == SYSTEM and fix.distance == pytest.approx(300)
    assert zone(tracker.update(_in_field(1000, 200, 0), 4.0, now=4.0)) == FIELD  # at once


def test_tracker_course_from_two_positions() -> None:
    tracker = NavTracker(_target(_capture("a", position=_in_field(0, 10_000, 0))))  # no compass
    fix = tracker.update(_in_field(0, 0, 0), 1.0, now=1.0)
    assert fix.course_off is None  # one position: no motion yet
    t = 1.0
    for k in range(1, 5):  # 1 s straight at the cluster
        t = 1.0 + k * 0.25
        fix = tracker.update(_in_field(0, k * 25.0, 0), t, now=t)
    assert fix.course_off == pytest.approx(0, abs=1e-6)
    for k in range(1, 6):  # then 1 s sideways
        t += 0.25
        fix = tracker.update(_in_field(k * 25.0, 100.0, 0), t, now=t)
    assert fix.course_off == pytest.approx(90, abs=1)
    for _ in range(5):  # hovering
        t += 0.25
        fix = tracker.update(_in_field(125.0, 100.0, 0), t, now=t)
    assert fix.course_off is None


def test_tracker_without_position() -> None:
    tracker = NavTracker(_target(_capture("a", position=_in_field(0, 0, 0))))
    fix = tracker.update(None, 0.0, now=5.0)
    assert (fix.frame, fix.distance, fix.age, fix.closing_speed, fix.eta) == (
        None, None, None, None, None,
    )  # fmt: skip


# -- compass ----------------------------------------------------------------------------------


def test_compass_ahead_right_and_behind() -> None:
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))  # right, forward, up
    ahead = compass((0.0, 500.0, 0.0), axes)
    assert ahead.dot == pytest.approx((0, 0)) and not ahead.behind and ahead.off_angle == 0
    right = compass((500.0, 0.0, 0.0), axes)
    assert right.dot == pytest.approx((1, 0)) and right.off_angle == pytest.approx(90)
    up_behind = compass((0.0, -100.0, 100.0), axes)
    assert up_behind.behind
    assert up_behind.dot == pytest.approx((0, math.sqrt(0.5)))
    assert up_behind.off_angle == pytest.approx(135)


def test_tracker_compass_from_cam_dir() -> None:
    target = _target(_capture("a", position=_in_field(0, 1000, 0)))
    tracker = NavTracker(target, ENGINE)
    # Engine camera at zero angles looks along +y: the target dead ahead
    fix = tracker.update(_in_field(0, 0, 0, cam_dir=(0, 0, 0)), 1.0, now=1.0)
    assert fix.compass is not None and fix.compass.off_angle == pytest.approx(0, abs=1e-6)
    # Yawed 90° left (+z): +y now lies to the right
    fix = tracker.update(_in_field(0, 0, 0, cam_dir=(0, 0, 90)), 2.0, now=2.0)
    assert fix.compass is not None
    assert fix.compass.dot == pytest.approx((1, 0), abs=1e-6)
    # A read that missed the CamDir line keeps the last one for a moment
    fix = tracker.update(_in_field(0, 10, 0), 2.3, now=2.3)
    assert fix.compass is not None
    assert fix.compass.dot == pytest.approx((1, 0), abs=1e-6)
    late = tracker.update(_in_field(0, 20, 0), 2.1 + HOLD_S, now=2.1 + HOLD_S)
    assert late.compass is None and late.cam_dir is None and late.calibrated


CAM = (20.0, 35.0, 40.0)  # pitched and rolled: no camera axis along the field's turn axis


def _leg(
    cam_dir: tuple[float, float, float] = CAM,
    *,
    axis: int = 1,
    sign: float = 1.0,
    camera_in_field: bool = False,
    t0: float = 0.0,
) -> list[tuple[float, Position]]:
    """3 s at 4 Hz and 50 m/s along one camera axis (0 right, 1 forward, 2 up)."""
    direction = _scaled(camera_axes(cam_dir, ENGINE)[axis], sign)
    if camera_in_field:  # the angles are in the field's frame: turn the motion into the system's
        direction = _rz(direction, TURN)
    return [
        (t0 + k * 0.25, _turned_field(_scaled(direction, 12.5 * k), cam_dir)) for k in range(13)
    ]


def _scaled(v: tuple[float, float, float], k: float) -> tuple[float, float, float]:
    return (v[0] * k, v[1] * k, v[2] * k)


def _learn(*legs: list[tuple[float, Position]]) -> CameraFrames:
    cameras = CameraFrames()
    for leg in legs:
        for t, position in leg:
            cameras.add(position, t, ENGINE)
    return cameras


def test_camera_frames_find_camdir_in_the_system_frame() -> None:
    assert CameraFrames().check(FIELD) == CameraCheck(nav.SYSTEM, False)
    assert _learn(_leg()).check(FIELD) == CameraCheck(nav.SYSTEM, True)


def test_camera_frames_find_camdir_in_the_local_frame_when_reversing() -> None:
    cameras = _learn(_leg(sign=-1.0, camera_in_field=True))
    assert cameras.check(FIELD) == CameraCheck(nav.LOCAL, True)


def test_camera_frames_skip_spans_along_axes_of_both_frames() -> None:
    # Level flight in a field turned ~90° about the vertical: ahead in one frame is nearly
    # sideways in the other, so the leg cannot tell them apart
    assert not _learn(_leg((0.0, 0.0, 30.0))).check(FIELD).checked
    # Frames lined up (no turn): either is right
    aligned = CameraFrames()
    forward = camera_axes(CAM, ENGINE)[1]
    for k in range(13):
        t = k * 0.25
        x, y, z = (50.0 * t * v for v in forward)
        aligned.add(_in_field(x, y, z, cam_dir=CAM), t, ENGINE)
    assert aligned.check(FIELD) == CameraCheck(nav.SYSTEM, True)


def test_tracker_compass_uses_the_frame_camdir_is_in() -> None:
    # The cluster lies 1 km dead ahead in the system's axes; in the turned field's own axes the
    # same offset points 85° away
    target = _target(
        _capture("a", position=_turned_field(_scaled(camera_axes(CAM, ENGINE)[1], 1e3)))
    )
    cameras = CameraFrames()
    tracker = NavTracker(target, ENGINE, cameras)
    fix = tracker.update(_turned_field((0.0, 0.0, 0.0), CAM), 1.0, now=1.0)
    assert fix.frame is not None and fix.frame.zone == FIELD  # the distance: innermost frame
    assert fix.distance == pytest.approx(1000)
    assert fix.compass is not None and fix.compass.off_angle == pytest.approx(0, abs=1e-6)
    assert fix.camera == CameraCheck(nav.SYSTEM, False)
    # what it is made of, for the debug readout
    assert fix.cam_dir == CAM and fix.calibrated
    assert fix.compass_frame is not None and fix.compass_frame.zone == SYSTEM
    for t, position in _leg(t0=2.0):  # straight flight confirms it
        cameras.add(position, t, ENGINE)
        fix = tracker.update(position, t, now=t)
    assert fix.camera == CameraCheck(nav.SYSTEM, True)
    assert fix.compass is not None and fix.compass.off_angle == pytest.approx(0, abs=1e-6)


def test_the_games_convention_is_built_in() -> None:
    assert str(GAME_CONVENTION) == "xyz/210/+++/N" and GAME_CONVENTION == ENGINE
    calibrated = Convention.parse("xyz/012/---/T")  # how the 2026-11-08 flight spelt it
    assert calibrated is not None and calibrated.same_rotation(GAME_CONVENTION)
    other = Convention((2, 0, 1), (0, 1, 2), (1, -1, 1), True)
    assert not other.same_rotation(GAME_CONVENTION)


def test_a_real_calibration_flight_fits_the_built_in_convention() -> None:
    # The flight that settled it (Pyro, open space, 2026-11-08): straight legs and strafes
    path = Path(__file__).parent / "fixtures" / "navigation" / "calibration-flight.csv"
    with path.open(encoding="utf-8") as f:
        samples = [
            CalibrationSample(
                float(row["t"]),
                (float(row["cam_dir_x"]), float(row["cam_dir_y"]), float(row["cam_dir_z"])),
                row["zone"],
                (float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])),
            )
            for row in csv.DictReader(f)
        ]
    result = fit_convention(samples)
    assert result.convention is not None, result.message
    assert result.convention.same_rotation(GAME_CONVENTION)
    assert result.median_error is not None and result.median_error < 1.5


def test_convention_round_trip() -> None:
    assert len(all_conventions()) == 576
    for conv in all_conventions()[::37]:
        assert Convention.parse(str(conv)) == conv
    assert str(ENGINE) == "xyz/210/+++/N"
    for bad in ("", "xyz/210/+++", "xxz/210/+++/N", "xyz/211/+++/N", "xyz/210/+*+/N"):
        assert Convention.parse(bad) is None


# -- calibration ------------------------------------------------------------------------------


def _flight(
    conv: Convention,
    rng: np.random.Generator,
    legs: int = 5,
    ups: int = 2,
    honest: bool = True,
) -> list[CalibrationSample]:
    """Straight legs at 50 m/s, then strafes up at 15 m/s, 4 Hz, with a turn between legs.

    The game shows whole degrees, so the camera angles are the true ones rounded.
    """
    samples = []
    t, pos = 0.0, np.zeros(3)
    for leg in range(legs + ups):
        true = (rng.uniform(-180, 180), rng.uniform(-80, 80), rng.uniform(-180, 180))
        _right, forward, up = camera_axes(true, conv)
        axis = np.array(forward if leg < legs else up)
        if not honest:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
        speed = 50.0 if leg < legs else 15.0
        shown = (float(round(true[0])), float(round(true[1])), float(round(true[2])))
        for _ in range(20):
            t += 0.25
            pos = pos + axis * speed * 0.25 + rng.normal(0, 0.03, 3)
            samples.append(CalibrationSample(t, shown, FIELD, (pos[0], pos[1], pos[2])))
        t += 3.0  # turning: the reader sees no steady span here
    return samples


def _same_rotation(a: Convention, b: Convention, rng: np.random.Generator) -> bool:
    for _ in range(20):
        angles = (rng.uniform(-180, 180), rng.uniform(-80, 80), rng.uniform(-180, 180))
        if not np.allclose(camera_axes(angles, a), camera_axes(angles, b), atol=1e-6):
            return False
    return True


@pytest.mark.parametrize("truth", [ENGINE, Convention((2, 0, 1), (0, 1, 2), (1, -1, 1), True)])
def test_fit_recovers_the_convention(truth: Convention) -> None:
    rng = np.random.default_rng(7)
    result = fit_convention(_flight(truth, rng))
    assert result.convention is not None, result.message
    assert _same_rotation(result.convention, truth, rng)
    assert result.forward >= 10 and result.up >= 4 and result.directions >= 3
    assert result.median_error is not None and result.median_error < 2


def test_fit_finds_the_frame_camdir_is_in() -> None:
    # Calibrated inside a field turned against the solar system: every read in both frames
    rng = np.random.default_rng(7)
    flight = _flight(ENGINE, rng)  # along the camera's axes: the solar system's frame here
    system = [replace(s, zone=SYSTEM) for s in flight]
    field = [replace(s, zone=FIELD, pos=_rz(s.pos, -TURN)) for s in flight]
    result = fit_convention(field + system)
    assert result.convention is not None, result.message
    assert result.frame == SYSTEM and "solar system" in result.message
    assert _same_rotation(result.convention, ENGINE, rng)


def test_fit_needs_up_legs() -> None:
    result = fit_convention(_flight(ENGINE, np.random.default_rng(1), ups=0))
    assert result.convention is None
    assert "up" in result.message


def test_fit_needs_several_directions() -> None:
    result = fit_convention(_flight(ENGINE, np.random.default_rng(2), legs=2))
    assert result.convention is None


def test_fit_rejects_unrelated_angles() -> None:
    result = fit_convention(_flight(ENGINE, np.random.default_rng(3), honest=False))
    assert result.convention is None


def test_motion_samples_average_the_camera_sway() -> None:
    sway = [(10, 179, 0), (11, -179, 1), (10, 180, 0), (11, 179, 1), (10, -179, 0)]

    def flight(cams: list[tuple[int, int, int]]) -> list[CalibrationSample]:
        return [
            CalibrationSample(k * 0.25, c, FIELD, (0.0, 50.0 * k * 0.25, 0.0))
            for k, c in enumerate(cams)
        ]

    spans = motion_samples(flight(sway))
    assert spans and spans[0].direction == pytest.approx((0, 1, 0))
    assert spans[0].cam_dir == pytest.approx((10.5, 179.75, 0.5))  # across ±180
    # One reading straying mid-span spoils it, even with steady ends
    sway[1] = (15, -179, 1)
    assert motion_samples(flight(sway)) == []


def test_fit_without_flight() -> None:
    hover = [CalibrationSample(t * 0.25, (0, 0, 0), FIELD, (0.0, 0.0, 0.0)) for t in range(40)]
    assert motion_samples(hover) == []
    assert fit_convention(hover).convention is None
