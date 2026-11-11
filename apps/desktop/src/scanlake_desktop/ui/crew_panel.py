"""The Crew tab: who is in the shared session, what they do, fly and where they are heading.

Everyone sets their own roles, ship and crew here; members are listed grouped by crew. Names
and texts of other members come from the network and are shown as plain text only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scanlake_core.domain.crew import NAME_MAX, ROLES, Member
from scanlake_desktop.ui.style import MUTED

COLUMNS = ["Name", "Roles", "Ship", "Status", "Heading to"]
NO_CREW = "No crew"


def group_members(members: Sequence[Member]) -> list[tuple[str, list[Member]]]:
    """Members by crew, crews by name, those without one last; online members first."""
    groups: dict[str, list[Member]] = {}
    for member in members:
        groups.setdefault(member.crew, []).append(member)
    named = sorted((name for name in groups if name), key=str.lower)
    ordered = named + ([""] if "" in groups else [])
    return [
        (name or NO_CREW, sorted(groups[name], key=lambda m: (not m.online, m.name.lower())))
        for name in ordered
    ]


class CrewPanel(QWidget):
    profile_changed = Signal(list, str, str)  # roles, ship, crew
    invite_clicked = Signal()
    join_clicked = Signal()
    leave_clicked = Signal()

    def __init__(self, ships: Sequence[str] = ()) -> None:
        super().__init__()
        title = QLabel("Shared session")
        title.setProperty("role", "title")
        self.state = QLabel()
        self.state.setProperty("role", "hint")
        self.state.setTextFormat(Qt.TextFormat.PlainText)
        self.state.setWordWrap(True)
        self.link = QLineEdit()
        self.link.setReadOnly(True)
        self.link.setPlaceholderText("Invite creates a link the others paste under Share → Join…")
        self.invite = QPushButton("Invite")
        self.join = QPushButton("Join…")
        self.leave = QPushButton("Leave")
        self.invite.clicked.connect(self.invite_clicked.emit)
        self.join.clicked.connect(self.join_clicked.emit)
        self.leave.clicked.connect(self.leave_clicked.emit)
        share = QHBoxLayout()
        share.addWidget(self.link, 1)
        share.addWidget(self.invite)
        share.addWidget(self.join)
        share.addWidget(self.leave)

        self.roles: dict[str, QPushButton] = {}
        roles = QHBoxLayout()
        for key, text in ROLES.items():
            button = QPushButton(text)
            button.setCheckable(True)
            button.clicked.connect(self._emit_profile)
            self.roles[key] = button
            roles.addWidget(button)
        roles.addStretch(1)
        self.ship = QComboBox()
        self.ship.setEditable(True)
        self.ship.addItems(list(ships))
        self.ship.setCurrentText("")
        self.crew = QComboBox()
        self.crew.setEditable(True)
        for combo, placeholder in ((self.ship, ""), (self.crew, "e.g. Mole crew")):
            edit = combo.lineEdit()
            assert edit is not None  # editable
            edit.setMaxLength(NAME_MAX)
            edit.setPlaceholderText(placeholder)
            edit.editingFinished.connect(self._emit_profile)
            combo.activated.connect(self._emit_profile)
        you = QGroupBox("You")
        form = QFormLayout(you)
        form.addRow("Roles", roles)
        form.addRow("Ship", self.ship)
        form.addRow("Crew", self.crew)

        self.members = QTreeWidget()
        self.members.setColumnCount(len(COLUMNS))
        self.members.setHeaderLabels(COLUMNS)
        self.members.setRootIsDecorated(False)
        self.members.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        header = self.members.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COLUMNS.index("Heading to"), QHeaderView.ResizeMode.Stretch)

        box = QVBoxLayout(self)
        box.addWidget(title)
        box.addWidget(self.state)
        box.addLayout(share)
        box.addWidget(you)
        box.addWidget(self.members, 1)
        self._last: tuple[list[str], str, str] | None = None
        self.show_room([], "", "", "", {}, shared=False, can_invite=False, enabled=True)

    # -- own profile ------------------------------------------------------------------------

    def set_profile(self, roles: Sequence[str], ship: str, crew: str) -> None:
        """Shows the stored profile without announcing a change."""
        for key, button in self.roles.items():
            button.setChecked(key in roles)
        self.ship.setCurrentText(ship)
        self.crew.setCurrentText(crew)
        self._last = self.profile()

    def profile(self) -> tuple[list[str], str, str]:
        roles = [key for key, button in self.roles.items() if button.isChecked()]
        return roles, self.ship.currentText().strip(), self.crew.currentText().strip()

    def _emit_profile(self, *_: object) -> None:
        current = self.profile()
        if current != self._last:
            self._last = current
            self.profile_changed.emit(*current)

    # -- the room ---------------------------------------------------------------------------

    def show_room(
        self,
        members: Sequence[Member],
        you: str,
        link: str,
        state: str,
        heading: Mapping[str, str],
        *,
        shared: bool,
        can_invite: bool,
        enabled: bool,
    ) -> None:
        """`heading`: user id -> the cluster that member is on the way to."""
        self.state.setText(state)
        if self.link.text() != link:
            self.link.setText(link)
        self.invite.setText("Copy link" if shared else "Invite")
        self.invite.setEnabled(enabled and (shared or can_invite))
        self.join.setEnabled(enabled)
        self.leave.setEnabled(shared)

        typed = self.crew.currentText()
        names = sorted({m.crew for m in members if m.crew}, key=str.lower)
        if [self.crew.itemText(i) for i in range(self.crew.count())] != names:
            self.crew.blockSignals(True)
            self.crew.clear()
            self.crew.addItems(names)
            self.crew.setCurrentText(typed)
            self.crew.blockSignals(False)

        self.members.clear()
        for crew_name, group in group_members(members):
            head = QTreeWidgetItem([f"{crew_name} ({len(group)})"])
            font = head.font(0)
            font.setBold(True)
            head.setFont(0, font)
            head.setFirstColumnSpanned(True)
            self.members.addTopLevelItem(head)
            for member in group:
                name = member.name + (" (you)" if member.user_id == you else "")
                item = QTreeWidgetItem(
                    [
                        name,
                        member.role_names or "—",
                        member.ship or "—",
                        "online" if member.online else "offline",
                        heading.get(member.user_id, ""),
                    ]
                )
                if not member.online:
                    for column in range(len(COLUMNS)):
                        item.setForeground(column, QColor(MUTED))
                head.addChild(item)
            head.setExpanded(True)

    def rows(self) -> list[list[str]]:
        """The member list as text, crew headings included (for tests)."""
        out = []
        for i in range(self.members.topLevelItemCount()):
            head = self.members.topLevelItem(i)
            assert head is not None
            out.append([head.text(0)])
            for j in range(head.childCount()):
                child = head.child(j)
                assert child is not None
                out.append([child.text(c) for c in range(len(COLUMNS))])
        return out
