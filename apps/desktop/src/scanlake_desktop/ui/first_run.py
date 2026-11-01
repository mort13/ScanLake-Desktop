"""First start: who you are, which ship, which screen. Everything can be changed in Settings."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from scanlake_desktop.capture.source import Monitor, primary_monitor
from scanlake_desktop.settings import Settings

WELCOME = (
    "ScanLake Desktop reads the mining scan panel while you play and uploads the scans to "
    "ScanLake. Press F9 when a scan is on screen, check the values, and press F10 to save. "
    "Run Star Citizen in borderless window mode so the screen can be read."
)


class FirstRunDialog(QDialog):
    def __init__(
        self,
        settings: Settings,
        ships: list[tuple[str, str]],
        monitors: list[Monitor],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to ScanLake Desktop")
        self.setMinimumWidth(460)
        intro = QLabel(WELCOME)
        intro.setWordWrap(True)
        self.name = QLineEdit(settings.user_name)
        self.name.setPlaceholderText("shown with your scans (optional)")
        self.org = QLineEdit(settings.org)
        self.org.setPlaceholderText("optional")
        self.ship = QComboBox()
        for key, label in ships:
            self.ship.addItem(label, key)
        self.ship.setCurrentIndex(max(0, self.ship.findData(settings.ship)))
        self.monitor = QComboBox()
        for m in monitors:
            self.monitor.addItem(m.label(), m.index)
        if monitors:  # the game usually runs on the primary monitor (mss may list it second)
            self.monitor.setCurrentIndex(self.monitor.findData(primary_monitor(monitors).index))
        hint = QLabel(
            "Optional: log in with Discord (top right) to keep your ScanLake ID across PCs."
        )
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        form = QFormLayout(self)
        form.addRow(intro)
        form.addRow("Name", self.name)
        form.addRow("Organisation", self.org)
        form.addRow("Ship", self.ship)
        form.addRow("Game screen", self.monitor)
        form.addRow(hint)
        form.addRow(buttons)

    def apply(self, settings: Settings) -> None:
        settings.user_name = self.name.text().strip()
        settings.org = self.org.text().strip()
        settings.ship = str(self.ship.currentData())
        if self.monitor.count():
            settings.monitor = int(self.monitor.currentData())
