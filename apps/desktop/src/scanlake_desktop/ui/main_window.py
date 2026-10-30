"""The main window: session bar, live readings, review form, saved scans, status bar."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from scanlake_desktop.ui.live_panel import LivePanel
from scanlake_desktop.ui.review_form import ReviewForm
from scanlake_desktop.ui.scans_table import ScansTable
from scanlake_desktop.ui.session_bar import SessionBar


class MainWindow(QMainWindow):
    stage_clicked = Signal()
    save_clicked = Signal()
    clear_clicked = Signal()
    upload_clicked = Signal()
    settings_clicked = Signal()
    closing = Signal()

    def __init__(self, ships: list[tuple[str, str]], version: str) -> None:
        super().__init__()
        self.setWindowTitle(f"ScanLake Desktop {version}")
        self.resize(1180, 820)

        self.session_bar = SessionBar(ships)
        self.live = LivePanel()
        self.form = ReviewForm()
        self.scans = ScansTable()

        self.stage_button = QPushButton("Stage (F9)")
        self.save_button = QPushButton("Save (F10)")
        self.save_button.setDefault(True)
        clear = QPushButton("Clear")
        self.stage_button.clicked.connect(self.stage_clicked.emit)
        self.save_button.clicked.connect(self.save_clicked.emit)
        clear.clicked.connect(self.clear_clicked.emit)
        buttons = QHBoxLayout()
        buttons.addWidget(self.stage_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(clear)
        buttons.addStretch(1)

        right = QWidget()
        right_box = QVBoxLayout(right)
        right_box.addWidget(self.form, 1)
        right_box.addLayout(buttons)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.live)
        top.addWidget(right)
        top.setStretchFactor(0, 2)
        top.setStretchFactor(1, 3)
        body = QSplitter(Qt.Orientation.Vertical)
        body.addWidget(top)
        body.addWidget(self.scans)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 2)

        central = QWidget()
        box = QVBoxLayout(central)
        box.setContentsMargins(4, 0, 4, 0)
        box.addWidget(self.session_bar)
        box.addWidget(body, 1)
        self.setCentralWidget(central)

        self.upload_status = QLabel()
        self.format_status = QLabel()
        self.user_status = QLabel()
        status = self.statusBar()
        status.addWidget(self.upload_status, 1)
        status.addPermanentWidget(self.format_status)
        status.addPermanentWidget(self.user_status)

        self._build_menu()

    def _build_menu(self) -> None:
        menu = self.menuBar()
        app_menu = menu.addMenu("&ScanLake")
        settings = QAction("Settings…", self)
        settings.triggered.connect(self.settings_clicked.emit)
        upload = QAction("Upload now", self)
        upload.triggered.connect(self.upload_clicked.emit)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        app_menu.addAction(settings)
        app_menu.addAction(upload)
        app_menu.addSeparator()
        app_menu.addAction(quit_action)
        self.dev_menu: QMenu = menu.addMenu("&Dev")
        self.dev_menu.menuAction().setVisible(False)

    def set_dev_mode(self, on: bool) -> None:
        self.dev_menu.menuAction().setVisible(on)

    def flash(self, text: str, ms: int = 5000) -> None:
        self.statusBar().showMessage(text, ms)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.closing.emit()
        super().closeEvent(event)
