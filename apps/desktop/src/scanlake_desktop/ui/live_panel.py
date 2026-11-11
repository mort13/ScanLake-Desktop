"""What the capture loop sees right now: HUD lock, speed, the voted value of every field, and the
ship position read from the game's debug text."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scanlake_core.domain.position import Position
from scanlake_core.schema_map import SlotLayout
from scanlake_core.voting import Stability
from scanlake_desktop.ui.style import ERROR, MUTED, OK, WARN
from scanlake_desktop.worker.pipeline_worker import POSITION_FRESH_S, LiveState
from scanlake_inference.registry import LABELS

DOT = "●"
STABILITY_COLOR = {Stability.STABLE: OK, Stability.UNSTABLE: WARN, Stability.EMPTY: MUTED}
# The theme has one hue: the shape says it too
STABILITY_DOT = {Stability.STABLE: DOT, Stability.UNSTABLE: "○", Stability.EMPTY: "·"}
LOCAL = "◂ local"


def position_text(position: Position) -> str:
    """CamDir and every Zone line, the local zone (the one a map places the scan in) marked."""
    lines = []
    if position.cam_dir is not None:
        lines.append("CamDir: " + " ".join(f"{v:g}" for v in position.cam_dir))
    local = position.local_zone
    for zone in position.zones:
        mark = f"   {LOCAL}" if zone is local else ""
        lines.append(f"{zone.zone}: {zone.readable()}{mark}")
    return "\n".join(lines)


class LivePanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._status = QLabel("Starting…")
        self._status.setProperty("role", "title")
        self._detail = QLabel()
        self._detail.setProperty("role", "hint")
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Field", "Live value", ""])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._rois: list[str] = []
        self._position_title = QLabel()
        self._position_title.setProperty("role", "heading")
        self._position = QLabel()
        self._position.setWordWrap(True)
        self._position.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        box = QVBoxLayout(self)
        box.addWidget(self._status)
        box.addWidget(self._detail)
        box.addWidget(self._table, 1)
        box.addWidget(self._position_title)
        box.addWidget(self._position)
        self._set_position_title("Ship position: not read", MUTED)

    def set_layout(self, layout: SlotLayout) -> None:
        rows: list[tuple[str, str]] = [(s.roi, s.column) for s in layout.scan]
        for i, row in enumerate(layout.rows):
            rows += [(s.roi, f"{s.column} {i + 1}") for s in row]
        self._rois = [roi for roi, _ in rows]
        self._table.setRowCount(len(rows))
        for r, (_, label) in enumerate(rows):
            self._table.setItem(r, 0, QTableWidgetItem(label))
            self._table.setItem(r, 1, QTableWidgetItem(""))
            dot = QTableWidgetItem(DOT)
            dot.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(r, 2, dot)

    def show_message(self, text: str, color: str = MUTED) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}")

    def update_state(self, state: LiveState) -> None:
        if state.black:
            self.show_message("Black screen: run the game in borderless window mode", ERROR)
        elif state.locked:
            stable = sum(v.stable for v in state.votes.values())
            self.show_message(f"HUD locked · {stable}/{len(state.votes)} stable", OK)
        else:
            self.show_message("Looking for the scan HUD…", WARN)
        backend = LABELS.get(state.backend, state.backend)
        ms = sum(state.timings.values()) * 1000
        self._detail.setText(
            f"{backend} · {state.fps:.0f} fps · {ms:.0f} ms/frame · scale {state.scale:.2f}"
        )
        for r, roi in enumerate(self._rois):
            vote = state.votes.get(roi)
            value = self._table.item(r, 1)
            dot = self._table.item(r, 2)
            if vote is None or value is None or dot is None:
                continue
            value.setText(vote.text if vote.text else "")
            tip = f"confidence {vote.confidence:.0%}, {vote.count} votes"
            if len(vote.candidates) > 1:
                tip += " · also: " + ", ".join(
                    f"{t or '∅'} {s:.0%}" for t, s in vote.candidates[1:]
                )
            value.setToolTip(tip)
            dot.setText(STABILITY_DOT[vote.stability])
            dot.setForeground(QBrush(QColor(STABILITY_COLOR[vote.stability])))
            dot.setToolTip(vote.stability.value)
        self._show_position(state)

    def _show_position(self, state: LiveState) -> None:
        """The last position read from the debug text, and whether F9 would take it now."""
        seen = "\n".join(f"{score:.2f}  {text}" for text, score in state.position_lines)
        self._position.setToolTip(f"Last read (match score, line):\n{seen}" if seen else "")
        position = state.position
        if position is None:
            self._set_position_title("Ship position: not read", MUTED)
            self._position.setText(
                "Turn on the game's debug info (r_DisplayInfo) and each scan also records "
                "where the ship was."
            )
            return
        age = max(0.0, state.t - state.position_t)
        if age <= POSITION_FRESH_S:
            title = f"Ship position · read {age:.1f} s ago · match {position.confidence:.0%}"
            self._set_position_title(title, OK)
        else:
            title = f"Ship position · last read {age:.0f} s ago: a scan staged now gets none"
            self._set_position_title(title, WARN)
        self._position.setText(position_text(position))

    def _set_position_title(self, text: str, color: str) -> None:
        self._position_title.setText(text)
        self._position_title.setStyleSheet(f"color: {color}")

    def position_title(self) -> str:
        return self._position_title.text()

    def position_body(self) -> str:
        return self._position.text()
