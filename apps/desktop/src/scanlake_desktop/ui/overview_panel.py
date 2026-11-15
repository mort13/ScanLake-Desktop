"""The Overview tab: the session's clusters as cards, and the live way to the one you pick.

In a shared session the cards include the other members' clusters. They are ordered need
help, on the way, open, done, then newest scan first. Clicking a card makes it the navigation
target (clicking it again, or Stop, ends that); the box below shows the distance, offset,
closing speed and ETA from the live position. Each card can be marked done or need help.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from scanlake_core.domain import crew
from scanlake_core.domain.crew import SharedCluster
from scanlake_core.domain.navigation import (
    SYSTEM,
    CameraCheck,
    ClusterSummary,
    NavFix,
    Vec,
    nav_frame,
)
from scanlake_core.domain.position import Position
from scanlake_desktop.ui.cluster_card import ClusterCard, format_scu
from scanlake_desktop.ui.flow_layout import FlowArea
from scanlake_desktop.ui.style import set_state

AGE_REFRESH_MS = 30_000  # how often the cards' "12 min ago" is brought up to date
STALE_S = 3.0  # a position older than this no longer counts as live
TURN_ON = "turn on the game's debug info (r_DisplayInfo)"
NO_POSITION = f"No position: {TURN_ON}."


def format_distance(metres: float) -> str:
    if metres < 1000:
        return f"{metres:.0f} m"
    km = metres / 1000
    if km < 10:
        return f"{km:.2f} km"
    if km < 1000:
        return f"{km:.1f} km"
    return f"{km:,.0f} km"


def format_speed(closing: float | None) -> str:
    if closing is None:
        return "—"
    if abs(closing) < 0.5:
        return "holding"
    arrow = "▼" if closing > 0 else "▲"
    word = "closing" if closing > 0 else "opening"
    return f"{arrow} {abs(closing):,.0f} m/s {word}"


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = round(seconds)
    if s >= 3600:
        return f"{s // 3600} h {s % 3600 // 60:02d} min"
    return f"{s // 60}:{s % 60:02d}"


def format_course(degrees: float | None) -> str:
    if degrees is None:
        return "— (fly to measure it)"
    away = "  (moving away)" if degrees > 90 else ""
    return f"{degrees:.0f}° off the cluster{away}"


def camera_text(check: CameraCheck) -> str:
    """How far the compass can be trusted in the zone the ship is in."""
    frame = "the solar system's" if check.kind == SYSTEM else "this zone's own"
    if check.checked:
        return f"checked here: CamDir is in {frame} frame"
    return "not checked in this zone yet: fly straight for 2 s (ahead, reversing or strafing)"


def xyz(v: Vec) -> str:
    """x / y / z in m, or in km once any of them reaches 10 km."""
    if all(abs(c) < 10_000 for c in v):
        return " / ".join(f"{c:.1f}" for c in v) + " m"
    return " / ".join(f"{c / 1000:,.3f}" for c in v) + " km"


DEBUG_ROWS = 9  # the most lines `debug_lines` gives


def mono_font(pixels: int = 12) -> QFont:
    """A fixed-pitch font for the debug readout (its columns line up)."""
    font = QFont("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    font.setPixelSize(pixels)
    return font


def debug_lines(fix: NavFix) -> list[str]:
    """What the navigation works from, for a debug readout (in a monospace font)."""
    lines = [
        f"Read    {fix.age:.1f} s ago" if fix.age is not None else "Read    no position yet",
        "CamDir  " + (" ".join(f"{a:g}" for a in fix.cam_dir) if fix.cam_dir
                      else "not read in the last 2 s"),
    ]  # fmt: skip
    frame = fix.frame
    if frame is None:
        lines.append("Frame   none shared with the cluster")
    else:
        lines += [
            f"Frame   {frame.zone}" + ("  (approximate)" if frame.approximate else ""),
            f"Ship    {xyz(frame.live)}",
            f"Target  {xyz(frame.target)}",
            f"Offset  {xyz(frame.delta)}  = {format_distance(frame.distance)}",
        ]
    kind = "solar system" if fix.camera.kind == SYSTEM else "local zone"
    state = "checked" if fix.camera.checked else "unchecked"
    way = fix.compass_frame
    lines.append(f"Compass {way.zone if way else '-'} ({kind} frame, {state})")
    if way is not None and frame is not None and way.zone != frame.zone:
        lines.append(f"        offset {xyz(way.delta)}")
    found = fix.compass
    if found is not None:
        ahead = math.cos(math.radians(found.off_angle))
        right, up = found.dot
        lines.append(f"Camera  ahead {ahead:+.2f} right {right:+.2f} up {up:+.2f}"
                     f"  {found.off_angle:.0f}° off")  # fmt: skip
    else:
        why = (
            "not calibrated" if not fix.calibrated
            else "no CamDir" if fix.cam_dir is None
            else "the cluster has no position in that frame"
        )  # fmt: skip
        lines.append(f"Camera  no dot: {why}")
    return lines


def target_label(summary: ClusterSummary) -> str:
    """ "#3 · Gold", with the owner's name in front for another member's cluster."""
    return crew.label(summary)


__all__ = ["OverviewPanel", "format_scu", "target_label"]


class OverviewPanel(QWidget):
    target_selected = Signal(str)  # cluster_id
    target_cleared = Signal()
    status_changed = Signal(str, str)  # cluster_id, open | done | need_help
    calibrate_clicked = Signal()
    debug_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.title = QLabel("Clusters this session")
        self.title.setProperty("role", "title")
        self.hint = QLabel()
        self.hint.setProperty("role", "hint")
        self.hint.setTextFormat(Qt.TextFormat.PlainText)
        self.area = FlowArea()
        self.card_scroll = QScrollArea()
        self.card_scroll.setWidgetResizable(True)
        self.card_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.card_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.card_scroll.setWidget(self.area)
        self.cards: dict[str, ClusterCard] = {}  # by cluster id
        self.shown: list[SharedCluster] = []  # in display order
        self.offline = False  # the shared session is unreachable: other members' cards dim
        self.position: Position | None = None
        self.ages = QTimer(self)
        self.ages.setInterval(AGE_REFRESH_MS)
        self.ages.timeout.connect(self._refresh_ages)
        self.ages.start()

        self.target = QLabel("No target")
        self.target.setTextFormat(Qt.TextFormat.PlainText)
        self.target.setProperty("role", "title")
        self.distance = QLabel("—")
        self.distance.setProperty("role", "readout")
        self.offset = QLabel("—")
        self.closing = QLabel("—")
        self.eta = QLabel("—")
        self.course = QLabel("—")
        self.course.setToolTip(
            "From your positions a second apart: the angle between where you are flying and"
            " where the cluster is. Needs no compass calibration."
        )
        self.camera = QLabel("—")
        self.camera.setWordWrap(True)
        self.frame = QLabel("—")
        self.frame.setWordWrap(True)
        self.position_age = QLabel("—")
        self.position_age.setWordWrap(True)
        self.compass_status = QLabel()
        self.compass_status.setWordWrap(True)
        self.calibrate = QPushButton("Calibrate compass…")
        self.calibrate.setToolTip(
            "Not needed: the game's camera angles are built in. Fly a calibration only if the "
            "dot keeps pointing the wrong way."
        )
        self.calibrate.clicked.connect(self.calibrate_clicked.emit)
        self.stop = QPushButton("Stop")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.clear_target)
        self.debug = QCheckBox("Debug readout (also under the compass in game)")
        self.debug.setToolTip(
            "Shows what the navigation works from: the CamDir angles, your position, the "
            "cluster's, the offset, the frames used and why the compass shows a dot or not."
        )
        self.debug.toggled.connect(self._debug_toggled)
        self.debug_text = QLabel()
        self.debug_text.setFont(mono_font(13))
        self.debug_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.debug_text.setVisible(False)

        form = QFormLayout()
        form.addRow("Distance", self.distance)
        form.addRow("Offset", self.offset)
        form.addRow("Speed", self.closing)
        form.addRow("ETA", self.eta)
        form.addRow("Course", self.course)
        form.addRow("Compass", self.camera)
        form.addRow("Frame", self.frame)
        form.addRow("Position", self.position_age)
        compass_row = QHBoxLayout()
        compass_row.addWidget(self.compass_status, 1)
        compass_row.addWidget(self.calibrate)
        heading = QHBoxLayout()
        heading.addWidget(self.target, 1)
        heading.addWidget(self.stop)
        nav = QGroupBox("Navigation")
        nav_box = QVBoxLayout(nav)
        nav_box.addLayout(heading)
        nav_box.addLayout(form)
        nav_box.addLayout(compass_row)
        nav_box.addWidget(self.debug)
        nav_box.addWidget(self.debug_text)

        box = QVBoxLayout(self)
        box.addWidget(self.title)
        box.addWidget(self.hint)
        box.addWidget(self.card_scroll, 1)
        box.addWidget(nav)

        self.target_id: str | None = None
        self.set_compass_status()
        self.show_clusters([])

    # -- clusters ---------------------------------------------------------------------------

    @property
    def summaries(self) -> list[ClusterSummary]:
        """The clusters shown, in display order."""
        return [card.summary for card in self.shown]

    def show_clusters(self, clusters: Sequence[SharedCluster | ClusterSummary]) -> None:
        """Shows the cards in the order given (see `crew.build_cards`; bare summaries are
        taken as own, open clusters). A target that is no longer among them is cleared.

        Cards are made and dropped only for clusters that appear or vanish; the others are
        updated in place and moved."""
        self.shown = [
            c if isinstance(c, SharedCluster) else SharedCluster(summary=c) for c in clusters
        ]
        now = time.time()
        wanted = {card.cluster_id for card in self.shown}
        for cluster_id in [c for c in self.cards if c not in wanted]:
            widget = self.cards.pop(cluster_id)
            self.area.flow.removeWidget(widget)
            widget.deleteLater()
        for card in self.shown:
            widget = self.cards.get(card.cluster_id)
            if widget is None:
                widget = ClusterCard(card)
                widget.clicked.connect(self.click)
                widget.status_clicked.connect(self.status_changed.emit)
                self.cards[card.cluster_id] = widget
                self.area.flow.addWidget(widget)
                widget.show()
            widget.show_card(card, now)
            widget.set_stale(self.offline and not card.mine)
        self.area.flow.set_order([self.cards[card.cluster_id] for card in self.shown])
        self.area.fit()
        if self.target_id is not None and self._summary(self.target_id) is None:
            self.clear_target()
        else:
            self._select(self.target_id)
        self.update_position(self.position)
        if not self.shown:
            self.hint.setText("No clusters yet: start a session and save scans.")
        elif not any(s.has_position for s in self.summaries):
            self.hint.setText(f"No cluster has a position yet: {TURN_ON} before saving scans.")
        else:
            self.hint.setText("Click a cluster to navigate to it; click it again to stop.")

    def set_shared(self, note: str, offline: bool = False) -> None:
        """The title line's note about the shared session ("" when the session is not shared);
        `offline` dims the other members' cards, whose state can no longer be trusted."""
        self.title.setText("Clusters this session" + (f" · {note}" if note else ""))
        if offline != self.offline:
            self.offline = offline
            for card in self.shown:
                self.cards[card.cluster_id].set_stale(offline and not card.mine)

    def update_position(self, position: Position | None) -> None:
        """Fills the cards' distances from the live position (nothing else is touched)."""
        self.position = position
        for card in self.shown:
            frame = nav_frame(position, card.summary) if position is not None else None
            self.cards[card.cluster_id].set_distance(
                format_distance(frame.distance) if frame else "—"
            )

    def _refresh_ages(self) -> None:
        now = time.time()
        for card in self.shown:
            self.cards[card.cluster_id].show_card(card, now)

    # -- target -----------------------------------------------------------------------------

    def _summary(self, cluster_id: str | None) -> ClusterSummary | None:
        return next((s for s in self.summaries if s.cluster_id == cluster_id), None)

    def click(self, cluster_id: str) -> None:
        """A card was clicked: its cluster becomes the target, or stops being it."""
        summary = self._summary(cluster_id)
        if summary is None:
            return
        if summary.cluster_id == self.target_id:
            self.clear_target()
        elif not summary.has_position:
            self.hint.setText(f"{target_label(summary)} has no position to navigate to.")
        else:
            self.set_target(summary.cluster_id)
            self.target_selected.emit(summary.cluster_id)

    def set_target(self, cluster_id: str | None) -> None:
        summary = self._summary(cluster_id)
        self.target_id = summary.cluster_id if summary else None
        self._select(self.target_id)
        self.stop.setEnabled(summary is not None)
        self.target.setText(target_label(summary) if summary else "No target")
        if summary is None:
            self.show_fix(None)

    def clear_target(self) -> None:
        had = self.target_id is not None
        self.set_target(None)
        if had:
            self.target_cleared.emit()

    def _select(self, cluster_id: str | None) -> None:
        for card_id, widget in self.cards.items():
            widget.set_target(card_id == cluster_id)

    # -- live -------------------------------------------------------------------------------

    def _debug_toggled(self, on: bool) -> None:
        self.debug_text.setVisible(on)
        self.debug_toggled.emit(on)

    def set_debug(self, on: bool) -> None:
        """Ticks the debug readout box without telling anyone (the setting it came from)."""
        self.debug.blockSignals(True)
        self.debug.setChecked(on)
        self.debug.blockSignals(False)
        self.debug_text.setVisible(on)

    def show_fix(self, fix: NavFix | None) -> None:
        live = (self.distance, self.offset, self.closing, self.eta, self.course, self.frame)
        self.debug_text.setText("No target" if fix is None else "\n".join(debug_lines(fix)))
        if fix is None:
            for label in (*live, self.camera):
                label.setText("—")
            self.position_age.setText("—")
            set_state(self.position_age, "")
            return
        self.camera.setText(camera_text(fix.camera))
        stale = fix.age is None or fix.age > STALE_S
        if fix.age is None:
            self.position_age.setText(NO_POSITION)
        elif stale:
            self.position_age.setText(f"Last read {fix.age:.0f} s ago. " + NO_POSITION)
        else:
            self.position_age.setText(f"Read {fix.age:.1f} s ago")
        frame = fix.frame
        if frame is None:
            for label in live[:-1]:
                label.setText("—")
            self.frame.setText(
                "—" if fix.age is None else "No zone in common with the cluster: fly to its system."
            )
            return
        self.distance.setText(format_distance(frame.distance))
        dx, dy, dz = frame.delta
        self.offset.setText(f"x {dx / 1000:+.2f} / y {dy / 1000:+.2f} / z {dz / 1000:+.2f} km")
        self.closing.setText(format_speed(fix.closing_speed))
        self.eta.setText(format_eta(fix.eta))
        self.course.setText(format_course(fix.course_off))
        note = "  (approximate: a planet turns in this frame)" if frame.approximate else ""
        self.frame.setText(frame.zone + note)

    def set_compass_status(self, own: str | None = None) -> None:
        """`own`: a calibration flown here that differs from the game's built-in one."""
        if own is None:
            text = "Compass ready: how the game's camera angles work is built in."
        else:
            text = f"Compass uses your own calibration ({own}), not the built-in one."
        self.compass_status.setText(text)
