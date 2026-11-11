"""Join a shared session: paste the invite link, see whose session it is, join."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from scanlake_desktop.tasks import run_in_background
from scanlake_io.crew_client import RoomInfo, parse_invite

SHARING_NOTE = (
    "Members of a shared session see each other's clusters with their positions, whatever "
    "“Upload ship positions” is set to. Your scans still upload under your own ID."
)


class JoinDialog(QDialog):
    def __init__(
        self,
        lookup: Callable[[str], RoomInfo],
        closing_note: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Join a shared session")
        self.setMinimumWidth(520)
        self._lookup = lookup
        self._closing_note = closing_note
        self.code: str | None = None  # the code the preview belongs to
        self.info: RoomInfo | None = None
        self._asked: str | None = None

        self.link = QLineEdit()
        self.link.setPlaceholderText("Paste the invite link")
        self.link.setClearButtonEnabled(True)
        self.preview = QLabel()
        self.preview.setTextFormat(Qt.TextFormat.PlainText)
        self.preview.setWordWrap(True)
        note = QLabel(SHARING_NOTE)
        note.setProperty("role", "hint")
        note.setWordWrap(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.join = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.join.setText("Join")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        box = QVBoxLayout(self)
        box.addWidget(QLabel("Invite link"))
        box.addWidget(self.link)
        box.addWidget(self.preview)
        box.addWidget(note)
        box.addWidget(self.buttons)

        self.link.textChanged.connect(self._changed)
        clipboard = QGuiApplication.clipboard().text()
        if parse_invite(clipboard):
            self.link.setText(clipboard.strip())
        else:
            self._changed("")

    def _changed(self, text: str) -> None:
        code = parse_invite(text)
        self.info, self.code = None, None
        self.join.setEnabled(False)
        if code is None:
            self._asked = None
            self.preview.setText("" if not text.strip() else "This is not a ScanLake invite link.")
            return
        if code == self._asked:
            return
        self._asked = code
        self.preview.setText("Looking the session up…")
        run_in_background(lambda: self._lookup(code), lambda result: self._found(code, result))

    def _found(self, code: str, result: object) -> None:
        if code != self._asked:
            return  # the link was changed meanwhile
        if not isinstance(result, RoomInfo):
            self._asked = None
            self.preview.setText(str(result) or "The session could not be looked up.")
            return
        self.code, self.info = code, result
        where = " / ".join(p for p in (result.system, result.gravity_well, result.region) if p)
        members = "1 member" if result.members == 1 else f"{result.members} members"
        text = f"{result.host_name or 'Someone'}'s session · {members}"
        if where:
            text += f" · {where}"
        if self._closing_note:
            text += "\n" + self._closing_note
        self.preview.setText(text)
        self.join.setEnabled(True)
