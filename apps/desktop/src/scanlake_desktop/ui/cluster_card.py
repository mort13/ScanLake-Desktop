"""One cluster of the Overview as a card: whose it is, what it holds, how far away it is, its
status (done / need help) and who is on the way to it.

Names and deposits of other members come from the network: every label is plain text.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from scanlake_core.domain.crew import DONE, NEED_HELP, OPEN, SharedCluster
from scanlake_desktop.ui.style import BG_RAISED, ERROR, MONO_BOLD, MUTED, OK, TINT, WARN

CARD_WIDTH = 250


def format_age(timestamp: str, now: float) -> str:
    """ "12 min ago" for an ISO 8601 UTC time; "" when it cannot be read."""
    try:
        then = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    seconds = max(0.0, now - then.timestamp())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60:.0f} min ago"
    if seconds < 86400:
        return f"{seconds // 3600:.0f} h {seconds % 3600 // 60:02.0f} min ago"
    return f"{seconds // 86400:.0f} d ago"


def format_scu(scu: float) -> str:
    return f"{scu:.3f}" if scu < 1 else f"{scu:,.2f}"


def card_state(card: SharedCluster) -> str:
    """need_help | enroute | done | open: what the card's frame shows."""
    if card.status == NEED_HELP:
        return NEED_HELP
    if card.status == DONE:
        return DONE
    return "enroute" if card.enroute else OPEN


FRAME_COLOURS = {NEED_HELP: ERROR, "enroute": WARN, DONE: MUTED, OPEN: MUTED}


def _plain(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


class ClusterCard(QFrame):
    clicked = Signal(str)  # cluster_id
    status_clicked = Signal(str, str)  # cluster_id, the status it should get

    def __init__(self, card: SharedCluster) -> None:
        super().__init__()
        self.setObjectName("card")
        self.setFixedWidth(CARD_WIDTH)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.card = card
        self.is_target = False
        self.stale = False  # another member's, while the shared session is unreachable
        self._style = ""

        self.title = _plain()
        self.title.setProperty("role", "heading")
        self.badge = _plain()
        self.content = _plain()
        self.zone = _plain()
        self.distance = _plain("—")
        self.distance.setProperty("role", "readout")
        self.distance.setMinimumWidth(110)  # a changing distance never re-lays the card out
        self.age = _plain()
        self.age.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.enroute = _plain()
        self.enroute.setWordWrap(True)
        self.done = QPushButton("Done")
        self.help = QPushButton("Need help")
        for button in (self.done, self.help):
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.ArrowCursor)
        self.done.clicked.connect(lambda: self._status(DONE))
        self.help.clicked.connect(lambda: self._status(NEED_HELP))

        head = QHBoxLayout()
        head.addWidget(self.title, 1)
        head.addWidget(self.badge)
        middle = QHBoxLayout()
        middle.addWidget(self.distance, 1)
        middle.addWidget(self.age)
        buttons = QHBoxLayout()
        buttons.addWidget(self.done)
        buttons.addWidget(self.help)
        box = QVBoxLayout(self)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(3)
        box.addLayout(head)
        box.addWidget(self.content)
        box.addWidget(self.zone)
        box.addLayout(middle)
        box.addWidget(self.enroute)
        box.addLayout(buttons)
        self.show_card(card, 0.0)

    @property
    def cluster_id(self) -> str:
        return self.card.cluster_id

    def show_card(self, card: SharedCluster, now: float) -> None:
        self.card = card
        s = card.summary
        deposit = s.deposit or "unknown deposit"
        owner = f"{s.owner} · " if s.owner else ""
        self._set(self.title, self._elided(f"{owner}#{s.number} · {deposit}", 30))
        self.title.setToolTip(html.escape(f"{owner}#{s.number} · {deposit}"))
        rocks = "1 rock" if s.rocks == 1 else f"{s.rocks} rocks"
        self._set(self.content, f"{rocks} · {format_scu(s.total_scu)} SCU")
        centre = s.centre
        self._set(self.zone, self._elided(s.local_zone, 34) if centre else "No position")
        self.zone.setToolTip(
            html.escape(f"{centre.zone}: {centre.readable()}")
            if centre
            else "No ship position was read for this cluster's scans."
        )
        self._set(self.age, format_age(s.last_scan, now) if now else "")
        self._set(self.enroute, "→ " + ", ".join(card.enroute) if card.enroute else "")
        self.enroute.setVisible(bool(card.enroute))
        self.done.setChecked(card.status == DONE)
        self.help.setChecked(card.status == NEED_HELP)
        self._restyle()

    def set_target(self, on: bool) -> None:
        if on != self.is_target:
            self.is_target = on
            self._restyle()

    def set_stale(self, on: bool) -> None:
        if on != self.stale:
            self.stale = on
            self._restyle()

    def set_distance(self, text: str) -> None:
        self._set(self.distance, text)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.cluster_id)
        super().mousePressEvent(event)

    def _status(self, status: str) -> None:
        """A toggle: clicking the status a card has reopens it."""
        wanted = OPEN if self.card.status == status else status
        self.done.setChecked(self.card.status == DONE)  # until the new status is shown
        self.help.setChecked(self.card.status == NEED_HELP)
        self.status_clicked.emit(self.cluster_id, wanted)

    def _restyle(self) -> None:
        state = card_state(self.card)
        frame = OK if self.is_target and state in (OPEN, "enroute") else FRAME_COLOURS[state]
        width = 2 if self.is_target or state in (NEED_HELP, "enroute") else 1
        muted = state == DONE or self.stale
        badge = {NEED_HELP: "NEED HELP", DONE: "DONE", "enroute": "ON THE WAY"}.get(state, "")
        if self.is_target:
            badge = "TARGET" if not badge or state == "enroute" else badge
        self._set(self.badge, badge)
        self.badge.setStyleSheet(f'color: {frame}; font-family: "{MONO_BOLD}"; font-size: 8pt')
        for button in (self.done, self.help):
            button.setEnabled(not self.stale)
        style = (
            f"#card {{ border: {width}px solid {frame};"
            f" background-color: {TINT if self.is_target else BG_RAISED}; }}"
            + (f" #card QLabel {{ color: {MUTED}; }}" if muted else "")
        )
        if style != self._style:
            self._style = style
            self.setStyleSheet(style)

    @staticmethod
    def _set(label: QLabel, text: str) -> None:
        if label.text() != text:
            label.setText(text)

    @staticmethod
    def _elided(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[: limit - 1] + "…"
