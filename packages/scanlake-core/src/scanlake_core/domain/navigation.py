"""Finding the way back to a cluster of the session.

Three parts, all pure:

- `summarize_clusters`: one `ClusterSummary` per cluster (deposit, rocks, total SCU and the
  mean ship position of its scans in every zone frame from the local zone outward).
- `NavTracker`: the live offset to one cluster, in the innermost zone frame the live position
  and the cluster share (the Refinery's rule), plus closing speed and ETA.
- The compass: where the cluster lies relative to the camera, from the game's `CamDir`
  angles. Which axis each angle turns about, and in what order, is not documented, so
  `fit_convention` learns it from a calibration flight: flying straight ahead moves the ship
  along the camera's forward axis, strafing up along its up axis. Camera axes follow the
  engine: X right, Y forward, Z up (the pilot's seat sits at +y / +z in the ship's frame).
  The angles are measured in the solar system's frame in open space and in asteroid fields,
  whose own frames are turned against it; `CameraFrames` checks this per zone from straight
  flight, and the compass works in that frame.
- The course: the angle between the way the ship moves (two positions a second apart) and
  the way to the cluster, in one frame; it needs no CamDir.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from scanlake_core.domain.capture import Capture
from scanlake_core.domain.position import Position, ZonePosition, is_system_zone

Vec = tuple[float, float, float]

# The only zone inside the solar system this close to its origin is the vehicle's own frame
# (the pilot sits ~10 m from the ship's origin); planets and asteroid fields are km across.
VEHICLE_FRAME_M = 1000.0


def nav_frames(position: Position) -> tuple[ZonePosition, ...]:
    """The local zone and every zone outside it; the vehicle's own frame is never included.

    With a ship in open space the innermost line (the ship) is also the only one inside the
    solar system, so `local_zone` names it: it is dropped when it sits next to its origin.
    """
    zones = position.zones
    local = position.local_zone
    frames = zones[zones.index(local) :]
    if (
        not is_system_zone(local.zone)
        and math.dist((local.x, local.y, local.z), (0.0, 0.0, 0.0)) < VEHICLE_FRAME_M
        and sum(not is_system_zone(z.zone) for z in zones) == 1
    ):
        frames = frames[1:]
    return frames


# -- cluster summaries --------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: str
    number: int  # "#N" in the session bar and the scans table
    deposit: str
    rocks: int  # saved scans in the cluster
    total_scu: float
    local_zone: str  # the frame most of its scans were in; "" without positions
    centres: Mapping[str, Vec] = field(default_factory=dict)  # zone -> mean position, metres
    first_scan: str = ""  # ISO 8601 UTC of its oldest and newest scan
    last_scan: str = ""
    owner: str = ""  # who scanned it, in a shared session; "" for your own

    @property
    def has_position(self) -> bool:
        return bool(self.centres)

    @property
    def centre(self) -> ZonePosition | None:
        """The mean position in the cluster's local zone, for display."""
        found = self.centres.get(self.local_zone)
        return None if found is None else ZonePosition(0, self.local_zone, *found)


def _number(text: str) -> float:
    try:
        value = float(text.replace(",", "."))
    except ValueError:
        return 0.0
    return value if math.isfinite(value) else 0.0


def summarize_clusters(
    captures: Iterable[Capture], cluster_ids: Sequence[str]
) -> list[ClusterSummary]:
    """One summary per cluster that has scans, in session order (`cluster_ids`)."""
    by_cluster: dict[str, list[Capture]] = {}
    for capture in captures:
        by_cluster.setdefault(capture.cluster_id, []).append(capture)
    summaries = []
    for number, cluster_id in enumerate(cluster_ids, start=1):
        scans = by_cluster.get(cluster_id)
        if not scans:
            continue
        deposits = Counter(d for c in scans if (d := c.value("deposit")))
        sums: dict[str, list[float]] = {}
        locals_: Counter[str] = Counter()
        for capture in scans:
            if capture.position is None:
                continue
            frames = nav_frames(capture.position)
            if not frames:
                continue
            locals_[frames[0].zone] += 1
            for zone in frames:
                acc = sums.setdefault(zone.zone, [0.0, 0.0, 0.0, 0.0])
                acc[0] += zone.x
                acc[1] += zone.y
                acc[2] += zone.z
                acc[3] += 1
        centres = {name: (a[0] / a[3], a[1] / a[3], a[2] / a[3]) for name, a in sums.items()}
        summaries.append(
            ClusterSummary(
                cluster_id=cluster_id,
                number=number,
                deposit=deposits.most_common(1)[0][0] if deposits else "",
                rocks=len(scans),
                total_scu=sum(_number(c.value("volume")) for c in scans),
                local_zone=locals_.most_common(1)[0][0] if locals_ else "",
                centres=centres,
                first_scan=min(c.timestamp for c in scans),
                last_scan=max(c.timestamp for c in scans),
            )
        )
    return summaries


# -- live offset ----------------------------------------------------------------------------


@dataclass(frozen=True)
class NavFrame:
    zone: str
    live: Vec
    target: Vec
    # Compared in the solar system's frame while the cluster sits in a local zone: right for
    # asteroid fields, which do not move in it, but planets turn under their deposits.
    approximate: bool = False

    @property
    def delta(self) -> Vec:
        return (
            self.target[0] - self.live[0],
            self.target[1] - self.live[1],
            self.target[2] - self.live[2],
        )

    @property
    def distance(self) -> float:
        return math.dist(self.live, self.target)


def _shared(zones: Iterable[ZonePosition], target: ClusterSummary) -> NavFrame | None:
    for zone in zones:
        centre = target.centres.get(zone.zone)
        if centre is not None:
            approximate = is_system_zone(zone.zone) and not is_system_zone(target.local_zone)
            return NavFrame(zone.zone, (zone.x, zone.y, zone.z), centre, approximate)
    return None


def nav_frame(live: Position, target: ClusterSummary) -> NavFrame | None:
    """The innermost frame both share, walking out from the live local zone."""
    return _shared(nav_frames(live), target)


# The two kinds of frame CamDir can be measured in
SYSTEM = "system"  # the solar system's (SolarSystem_…, else Root: the same axes)
LOCAL = "local"  # the local zone's (asteroid field, planet, station), which may be turned


def frame_of_kind(live: Position, target: ClusterSummary, kind: str) -> NavFrame | None:
    """The frame of one kind both share (the solar system's falls back to Root)."""
    zones = [z for z in nav_frames(live) if is_system_zone(z.zone) == (kind == SYSTEM)]
    return _shared(zones, target)


@dataclass(frozen=True)
class Compass:
    """Where the target lies seen from the camera.

    `dot` is the target direction's (right, up) part: (0, 0) dead ahead or dead behind, on
    the unit circle at 90° off. `off_angle` is the angle from straight ahead, in degrees.
    """

    dot: tuple[float, float]
    behind: bool
    off_angle: float


@dataclass(frozen=True)
class CameraCheck:
    """The frame CamDir is taken to be in, and whether straight flight confirmed it."""

    kind: str = SYSTEM
    checked: bool = False


@dataclass(frozen=True)
class NavFix:
    target: ClusterSummary
    frame: NavFrame | None = None
    closing_speed: float | None = None  # m/s, positive while the distance shrinks
    age: float | None = None  # seconds since the position was read; None: never read
    compass: Compass | None = None
    camera: CameraCheck = CameraCheck()
    # Angle between the way the ship moves (from its positions) and the way to the cluster;
    # needs no CamDir. None while hardly moving.
    course_off: float | None = None
    # What the compass is made of (for a debug readout): the CamDir in use (None: not read
    # lately), the way to the cluster in CamDir's frame, and whether the angles' convention is
    # known (calibrated). The compass needs all three.
    cam_dir: Vec | None = None
    compass_frame: NavFrame | None = None
    calibrated: bool = False

    @property
    def distance(self) -> float | None:
        return None if self.frame is None else self.frame.distance

    @property
    def eta(self) -> float | None:
        """Seconds to arrival at the current closing speed; None unless closing."""
        if self.frame is None or self.closing_speed is None or self.closing_speed < 0.5:
            return None
        return self.frame.distance / self.closing_speed


CLOSING_WINDOW_S = 2.0
# The reader misses a line now and then (a bright background behind it). A read without the
# line of the frame in use, or without CamDir, is passed over this long before it counts.
HOLD_S = 2.0


def _depth(zone: str) -> int:
    """0 for a local zone, 1 for the solar system, 2 for Root: the lowest shared one is used."""
    if not is_system_zone(zone):
        return 0
    return 2 if zone.startswith("Root") else 1


COURSE_WINDOW_S = 1.0
MIN_COURSE_SPEED = 5.0  # m/s; slower, the course is mostly reading noise


class NavTracker:
    """The live offset to one cluster; fed every live update, uses each position once.

    The distance comes from the innermost frame both share. The compass needs the way to the
    cluster in the frame CamDir is measured in, which `cameras` learns per zone (the solar
    system's until straight flight there says otherwise); an asteroid field's own frame is
    turned against it.
    """

    def __init__(
        self,
        target: ClusterSummary,
        convention: Convention | None = None,
        cameras: CameraFrames | None = None,
    ) -> None:
        self.target = target
        self.convention = convention
        self.cameras = cameras
        self._seen = -1.0  # the last position_t looked at
        self._t = -1.0  # the position_t the frame comes from
        self._frame: NavFrame | None = None
        self._cam_dir: Vec | None = None
        self._cam_t = -1.0
        self._camera = CameraCheck()
        self._camera_t = -1.0
        self._cam_frame: NavFrame | None = None  # the way to the cluster in CamDir's frame
        self._cam_frame_t = -1.0
        self._used_cam: Vec | None = None  # what the compass was last made of
        self._used_frame: NavFrame | None = None
        self._compass: Compass | None = None
        self._history: deque[tuple[float, float, Vec]] = deque()  # (t, distance, live)
        self._history_zone = ""

    def update(self, position: Position | None, position_t: float, now: float) -> NavFix:
        if position is not None and position_t > 0 and position_t != self._seen:
            self._seen = position_t
            if position.cam_dir is not None:
                self._cam_dir, self._cam_t = position.cam_dir, position_t
            frame = nav_frame(position, self.target)
            if not self._lost_line(frame, position_t):
                self._use(frame, position_t)
            self._check_camera(position, position_t)
            self._compass = self._point(position_t)
        age = now - self._t if self._t > 0 else None
        return NavFix(
            target=self.target,
            frame=self._frame,
            closing_speed=self._closing_speed(),
            age=age,
            compass=self._compass,
            camera=self._camera,
            course_off=self._course(),
            cam_dir=self._used_cam,
            compass_frame=self._used_frame,
            calibrated=self.convention is not None,
        )

    def _lost_line(self, frame: NavFrame | None, position_t: float) -> bool:
        """The read lacks the frame in use (none shared, or only an outer one), lately read."""
        held = self._frame
        if held is None or position_t - self._t >= HOLD_S:
            return False
        return frame is None or _depth(frame.zone) > _depth(held.zone)

    def _use(self, frame: NavFrame | None, position_t: float) -> None:
        self._t = position_t
        self._frame = frame
        if frame is None:
            return
        if frame.zone != self._history_zone:
            self._history.clear()
            self._history_zone = frame.zone
        self._history.append((position_t, frame.distance, frame.live))
        while self._history and position_t - self._history[0][0] > CLOSING_WINDOW_S:
            self._history.popleft()

    def _check_camera(self, position: Position, position_t: float) -> None:
        """Which frame CamDir is in where the ship is, and the way to the cluster in it."""
        frames = nav_frames(position)
        check = (
            self.cameras.check(frames[0].zone)
            if self.cameras is not None and frames
            else CameraCheck()
        )
        # A misread zone name has no check of its own: keep a recent checked one meanwhile
        recent = self._camera.checked and position_t - self._camera_t < HOLD_S
        if check.checked or not recent:
            if check.kind != self._camera.kind:
                self._cam_frame = None
            self._camera, self._camera_t = check, position_t
        found = frame_of_kind(position, self.target, self._camera.kind)
        if found is not None:
            self._cam_frame, self._cam_frame_t = found, position_t

    def _point(self, position_t: float) -> Compass | None:
        """The compass from the latest CamDir and way (recent ones when this read missed them)."""
        cam_dir, frame = self._cam_dir, self._cam_frame
        recent = cam_dir is not None and position_t - self._cam_t < HOLD_S
        self._used_cam = cam_dir if recent else None
        fresh = frame is not None and position_t - self._cam_frame_t < HOLD_S
        self._used_frame = frame if fresh else None
        if self._used_frame is None or self._used_cam is None or self.convention is None:
            return None
        return compass(self._used_frame.delta, camera_axes(self._used_cam, self.convention))

    def _course(self) -> float | None:
        """Degrees between the ship's motion over the last second and the way to the cluster."""
        if self._frame is None or len(self._history) < 2:
            return None
        t1, _, p1 = self._history[-1]
        t0, _, p0 = next(p for p in self._history if t1 - p[0] <= COURSE_WINDOW_S)
        if t1 - t0 < 0.4:
            return None
        motion = np.subtract(p1, p0)
        if float(np.linalg.norm(motion)) / (t1 - t0) < MIN_COURSE_SPEED:
            return None
        way = np.subtract(self._frame.target, p1)
        if float(np.linalg.norm(way)) < 1e-6:
            return None
        return _angle(motion, way)

    def _closing_speed(self) -> float | None:
        points = self._history
        if len(points) < 2 or points[-1][0] - points[0][0] < 0.4:
            return None
        t0 = points[0][0]
        ts = [t - t0 for t, _, _ in points]
        ds = [d for _, d, _ in points]
        mt, md = sum(ts) / len(ts), sum(ds) / len(ds)
        var = sum((t - mt) ** 2 for t in ts)
        slope = sum((t - mt) * (d - md) for t, d in zip(ts, ds, strict=True)) / var
        return -slope


# -- camera orientation ---------------------------------------------------------------------

_AXES = "xyz"


@dataclass(frozen=True)
class Convention:
    """How the three CamDir angles (as shown, left to right) build the camera's rotation.

    Angle `i` turns about axis `axes[i]` (0 x, 1 y, 2 z) by `signs[i]` × its value; the
    rotations are multiplied in `order` (leftmost outermost), and `transpose` inverts the
    result. The matrix's columns are the camera's right, forward and up in the zone frame.
    """

    axes: tuple[int, int, int]
    order: tuple[int, int, int]
    signs: tuple[int, int, int]
    transpose: bool = False

    def __str__(self) -> str:
        axes = "".join(_AXES[a] for a in self.axes)
        order = "".join(str(i) for i in self.order)
        signs = "".join("+" if s > 0 else "-" for s in self.signs)
        return f"{axes}/{order}/{signs}/{'T' if self.transpose else 'N'}"

    @classmethod
    def parse(cls, text: str) -> Convention | None:
        """The convention a `str()` wrote; None for "" or anything malformed."""
        parts = text.strip().split("/")
        if len(parts) != 4:
            return None
        axes_text, order_text, signs_text, t = parts
        if (
            sorted(axes_text) != list(_AXES)
            or sorted(order_text) != ["0", "1", "2"]
            or len(signs_text) != 3
            or any(s not in "+-" for s in signs_text)
            or t not in ("T", "N")
        ):
            return None
        a, o, s = [_AXES.index(c) for c in axes_text], [int(c) for c in order_text], signs_text
        return cls(
            axes=(a[0], a[1], a[2]),
            order=(o[0], o[1], o[2]),
            signs=tuple(1 if c == "+" else -1 for c in s),  # type: ignore[arg-type]
            transpose=t == "T",
        )


@cache
def all_conventions() -> tuple[Convention, ...]:
    """Every axis assignment, order, sign and transpose: 576 (many pairs are the same matrix)."""
    return tuple(
        Convention(axes, order, signs, transpose)  # type: ignore[arg-type]
        for axes in itertools.permutations(range(3))
        for order in itertools.permutations(range(3))
        for signs in itertools.product((1, -1), repeat=3)
        for transpose in (False, True)
    )


def _rotations(axis: int, degrees: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotations about one axis."""
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    out = np.zeros((len(t), 3, 3))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    out[:, axis, axis] = 1.0
    out[:, i, i] = c
    out[:, j, j] = c
    out[:, i, j] = -s
    out[:, j, i] = s
    return out


def _matrices(cam_dirs: np.ndarray, conv: Convention) -> np.ndarray:
    parts = [_rotations(conv.axes[k], conv.signs[k] * cam_dirs[:, k]) for k in range(3)]
    m = parts[conv.order[0]] @ parts[conv.order[1]] @ parts[conv.order[2]]
    return np.transpose(m, (0, 2, 1)) if conv.transpose else m


def camera_axes(cam_dir: Vec, conv: Convention) -> tuple[Vec, Vec, Vec]:
    """The camera's right, forward and up as unit vectors in the zone frame."""
    m = _matrices(np.array([cam_dir], dtype=float), conv)[0]
    cols = [tuple(float(v) for v in m[:, k]) for k in range(3)]
    return cols[0], cols[1], cols[2]  # type: ignore[return-value]


def compass(delta: Vec, axes: tuple[Vec, Vec, Vec]) -> Compass:
    right, forward, up = (np.array(a) for a in axes)
    d = np.array(delta, dtype=float)
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return Compass((0.0, 0.0), False, 0.0)
    d /= norm
    ahead = float(d @ forward)
    return Compass(
        dot=(float(d @ right), float(d @ up)),
        behind=ahead < 0,
        off_angle=math.degrees(math.acos(max(-1.0, min(1.0, ahead)))),
    )


# -- calibration ------------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSample:
    t: float  # seconds (position_t)
    cam_dir: Vec  # degrees as shown
    zone: str  # the local zone the position is in
    pos: Vec  # metres in that zone


@dataclass(frozen=True)
class MotionSample:
    cam_dir: Vec
    direction: Vec  # unit vector of the ship's motion in the zone frame


MOTION_SPAN_S = 1.0
MIN_SPEED = 10.0  # m/s
STEADY_DEG = 5.0  # course change allowed within one span
STEADY_CAM_DEG = 3.0  # camera angle change allowed within one span
MATCH_DEG = 8.0  # a sample counts for a convention when its motion lies this close to an axis
DIRECTION_SEPARATION_DEG = 30.0
MIN_FORWARD = 10
MIN_UP = 4
MIN_DIRECTIONS = 3
MAX_MEDIAN_DEG = 4.0
MIN_EXPLAINED = 0.6  # share of motion samples the convention must explain


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _max_angle(a: np.ndarray, b: np.ndarray) -> float:
    """The largest angle between matching rows of two (N, 3) arrays of unit vectors."""
    cos = np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos.min())))


def _by_zone(samples: Iterable[CalibrationSample]) -> dict[str, list[CalibrationSample]]:
    groups: dict[str, list[CalibrationSample]] = {}
    for s in samples:
        groups.setdefault(s.zone, []).append(s)
    return groups


def _steady_span(span: Sequence[CalibrationSample]) -> MotionSample | None:
    """One span (its readings in time order, one zone) as a motion sample, if it is steady.

    The camera sways a little even on a steady course, and the game shows whole degrees, so a
    span's camera angles are the mean of all its readings (none may stray STEADY_CAM_DEG).
    """
    if len(span) < 3:
        return None
    a, mid, b = span[0], span[len(span) // 2], span[-1]
    dt = b.t - a.t
    if not MOTION_SPAN_S * 0.7 <= dt <= MOTION_SPAN_S * 1.5:
        return None
    pa, pm, pb = (np.array(s.pos) for s in (a, mid, b))
    travel = pb - pa
    if float(np.linalg.norm(travel)) / dt < MIN_SPEED:
        return None
    if _angle(pm - pa, pb - pm) > STEADY_DEG:
        return None
    cams = np.array([s.cam_dir for s in span], dtype=float)
    offsets = (cams - cams[0] + 180.0) % 360.0 - 180.0  # wrapped, from the first reading
    if float(np.ptp(offsets, axis=0).max()) > STEADY_CAM_DEG:
        return None
    mean = (cams[0] + offsets.mean(axis=0) + 180.0) % 360.0 - 180.0
    unit = travel / np.linalg.norm(travel)
    return MotionSample(
        (float(mean[0]), float(mean[1]), float(mean[2])),
        (float(unit[0]), float(unit[1]), float(unit[2])),
    )


def motion_samples(samples: Sequence[CalibrationSample]) -> list[MotionSample]:
    """Spans of about a second with steady course, speed and camera angles, in each zone."""
    out = []
    for group in _by_zone(samples).values():
        ordered = sorted(group, key=lambda s: s.t)
        j = 0
        for i, a in enumerate(ordered):
            j = max(j, i + 1)
            while j < len(ordered) and ordered[j].t - a.t < MOTION_SPAN_S * 0.7:
                j += 1
            if j >= len(ordered):
                break
            found = _steady_span(ordered[i : j + 1])
            if found is not None:
                out.append(found)
    return out


def nearest_axis(direction: Vec, axes: tuple[Vec, Vec, Vec]) -> float:
    """Degrees from `direction` to the closest of the six camera axis directions (±)."""
    d = np.array(direction, dtype=float)
    return min(min(_angle(d, np.array(a)), _angle(d, -np.array(a))) for a in axes)


# -- which frame CamDir is in ----------------------------------------------------------------

CHECK_SPANS = 3  # straight spans along a camera axis that settle a zone's camera frame


@dataclass(frozen=True)
class _Read:
    t: float
    cam_dir: Vec
    key: str  # the ship's local zone (the solar system's in open space)
    zones: Mapping[str, Vec]  # every frame from the local zone outward


class CameraFrames:
    """Which frame CamDir is measured in, per zone, learned from straight flight.

    Flying straight along a camera axis (ahead, reversing or strafing) moves the ship along
    that axis in the frame the angles are measured in, and generally not in a frame turned
    against it. In open space and asteroid fields the game measures them in the solar system's
    frame (a field's own frame is turned against it); near planets and stations they may follow
    the local frame, so each zone is checked on its own. An unchecked zone assumes the solar
    system's. Fed every new position; only a few reads a second make spans.
    """

    def __init__(self) -> None:
        self._reads: deque[_Read] = deque()
        self._votes: dict[str, Counter[str]] = {}

    def check(self, zone: str) -> CameraCheck:
        votes = self._votes.get(zone)
        if not votes:
            return CameraCheck()
        kind = SYSTEM if votes[SYSTEM] >= votes[LOCAL] else LOCAL
        return CameraCheck(kind, votes[kind] >= CHECK_SPANS)

    def add(self, position: Position, position_t: float, convention: Convention) -> None:
        frames = nav_frames(position)
        if position.cam_dir is None or not frames:
            return
        if self._reads and position_t <= self._reads[-1].t:
            return
        zones = {z.zone: (z.x, z.y, z.z) for z in frames}
        read = _Read(position_t, position.cam_dir, frames[0].zone, zones)
        self._reads.append(read)
        while position_t - self._reads[0].t > MOTION_SPAN_S * 1.5:
            self._reads.popleft()
        start = next(
            (r for r in reversed(self._reads) if position_t - r.t >= MOTION_SPAN_S * 0.7), None
        )
        if start is None:
            return
        span = [r for r in self._reads if r.t >= start.t]
        if any(r.key != read.key for r in span):
            return
        matched: dict[str, Vec] = {}  # kind -> the motion that lies along a camera axis
        for zone in read.zones.keys() & start.zones.keys():
            samples = [
                CalibrationSample(r.t, r.cam_dir, zone, r.zones[zone])
                for r in span
                if zone in r.zones
            ]
            motion = _steady_span(samples)
            if motion is None:
                continue
            if nearest_axis(motion.direction, camera_axes(motion.cam_dir, convention)) < MATCH_DEG:
                matched[SYSTEM if is_system_zone(zone) else LOCAL] = motion.direction
        if len(matched) == 2:
            # Along an axis in both frames: they are lined up (either gives the same compass),
            # or turned so one axis lands on another (level flight in a field turned 90°),
            # which tells nothing
            if _angle(np.array(matched[SYSTEM]), np.array(matched[LOCAL])) >= MATCH_DEG:
                return
        votes = self._votes.setdefault(read.key, Counter())
        for kind in matched:
            votes[kind] += 1


def distinct_directions(directions: Iterable[Vec]) -> int:
    """How many directions lie at least DIRECTION_SEPARATION_DEG from each other (greedy)."""
    kept: list[np.ndarray] = []
    for d in directions:
        v = np.array(d)
        if all(_angle(v, k) >= DIRECTION_SEPARATION_DEG for k in kept):
            kept.append(v)
    return len(kept)


@dataclass(frozen=True)
class CalibrationResult:
    convention: Convention | None  # None: the fit failed (see `message`)
    message: str
    motion: int = 0  # steady motion spans found
    forward: int = 0  # explained by the forward axis
    up: int = 0  # explained by the up axis
    directions: int = 0  # distinct forward directions
    median_error: float | None = None  # degrees, over the explained spans
    frame: str = ""  # the zone whose frame the angles were found in


def frame_name(zone: str) -> str:
    return "the solar system's frame" if is_system_zone(zone) else f"the frame of {zone}"


@dataclass(frozen=True)
class _Score:
    conv: Convention
    forward_mask: np.ndarray
    up_mask: np.ndarray
    errors: np.ndarray
    forward_axes: np.ndarray
    up_axes: np.ndarray

    @property
    def explained(self) -> int:
        return int(self.forward_mask.sum() + self.up_mask.sum())

    @property
    def median(self) -> float:
        mask = self.forward_mask | self.up_mask
        return float(np.median(self.errors[mask])) if mask.any() else 180.0


def _score(cam_dirs: np.ndarray, motion: np.ndarray, conv: Convention) -> _Score:
    m = _matrices(cam_dirs, conv)
    fwd, up = m[:, :, 1], m[:, :, 2]
    ef = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", fwd, motion), -1, 1)))
    eu = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", up, motion), -1, 1)))
    forward_mask = (ef < MATCH_DEG) & (ef <= eu)
    up_mask = (eu < MATCH_DEG) & (eu < ef)
    return _Score(conv, forward_mask, up_mask, np.minimum(ef, eu), fwd, up)


def fit_convention(samples: Sequence[CalibrationSample]) -> CalibrationResult:
    """The CamDir convention that best explains the flight, or why none does.

    Samples may hold each read in several frames (the local zone's and the solar system's):
    each frame is fitted on its own, and the one the camera angles are measured in explains
    the flight; a frame turned against it does not. So calibrating in an asteroid field, whose
    frame is turned against the solar system's, cannot go wrong.
    """
    results = [_fit_frame(zone, group) for zone, group in _by_zone(samples).items()]
    if not results:
        return CalibrationResult(None, "No steady flight found: fly straight ahead faster.")
    return max(
        results,
        key=lambda r: (
            r.convention is not None,
            r.forward + r.up,
            -(r.median_error if r.median_error is not None else 180.0),
        ),
    )


def _fit_frame(zone: str, samples: Sequence[CalibrationSample]) -> CalibrationResult:
    spans = motion_samples(samples)
    if not spans:
        return CalibrationResult(
            None, "No steady flight found: fly straight ahead faster.", frame=zone
        )
    cam_dirs = np.array([s.cam_dir for s in spans], dtype=float)
    motion = np.array([s.direction for s in spans], dtype=float)
    scores = sorted(
        (_score(cam_dirs, motion, c) for c in all_conventions()),
        key=lambda s: (-s.explained, s.median),
    )
    best = scores[0]
    forward = int(best.forward_mask.sum())
    up = int(best.up_mask.sum())
    directions = distinct_directions(tuple(map(tuple, motion[best.forward_mask])))  # type: ignore[arg-type]

    def result(conv: Convention | None, message: str) -> CalibrationResult:
        return CalibrationResult(
            conv, message, len(spans), forward, up, directions,
            best.median if best.explained else None, zone,
        )  # fmt: skip

    if best.explained < MIN_EXPLAINED * len(spans):
        return result(None, "The camera angles do not match the flight: no convention fits.")
    if forward < MIN_FORWARD or directions < MIN_DIRECTIONS:
        return result(None, "Fly more straight legs in clearly different directions.")
    if up < MIN_UP:
        return result(None, "Strafe straight up for a few seconds, twice.")
    if best.median > MAX_MEDIAN_DEG:
        return result(None, f"The fit is too loose ({best.median:.1f}°): fly steadier legs.")
    # A different camera orientation that fits nearly as well means the flight cannot tell
    # them apart (conventions giving the same matrices tie exactly and are harmless).
    for other in scores[1:]:
        if other.explained < 0.9 * best.explained:
            break
        differs = max(
            _max_angle(best.forward_axes, other.forward_axes),
            _max_angle(best.up_axes, other.up_axes),
        )
        if differs > 2.0:
            return result(None, "Ambiguous: fly more varied legs (pitch and roll between them).")
    return result(best.conv, f"Calibrated in {frame_name(zone)} (median {best.median:.1f}°).")
