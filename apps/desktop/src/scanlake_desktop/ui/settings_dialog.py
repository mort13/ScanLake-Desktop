"""Settings: player, capture, recognition, hotkeys, upload, advanced."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from scanlake_desktop import paths
from scanlake_desktop.capture.source import Monitor
from scanlake_desktop.settings import Hotkeys, Region, Settings
from scanlake_desktop.ui.region_overlay import RegionOverlay
from scanlake_inference.registry import AUTO, BackendInfo


class KeyEdit(QLineEdit):
    """Press a key to bind it (F1-F12, letters, digits); Backspace clears."""

    def __init__(self, name: str) -> None:
        super().__init__(name.upper())
        self.setReadOnly(True)
        self.setPlaceholderText("press a key")
        self.key_name = name

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.key_name = ""
        elif Qt.Key.Key_F1 <= key <= Qt.Key.Key_F24:
            self.key_name = f"f{key - Qt.Key.Key_F1 + 1}"
        elif event.text() and event.text().isalnum() and len(event.text()) == 1:
            self.key_name = event.text().lower()
        else:
            return
        self.setText(self.key_name.upper())


class SettingsDialog(QDialog):
    benchmark_requested = Signal()

    def __init__(
        self,
        settings: Settings,
        monitors: list[Monitor],
        backends: list[BackendInfo],
        active_backend: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)
        self._settings = settings
        self._monitors = monitors
        self._manual_region = settings.manual_region

        tabs = QTabWidget()
        tabs.addTab(self._player_tab(), "Player")
        tabs.addTab(self._capture_tab(), "Capture")
        tabs.addTab(self._recognition_tab(backends, active_backend), "Recognition")
        tabs.addTab(self._hotkeys_tab(), "Hotkeys")
        tabs.addTab(self._upload_tab(), "Upload")
        tabs.addTab(self._advanced_tab(), "Advanced")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box = QVBoxLayout(self)
        box.addWidget(tabs)
        box.addWidget(buttons)

    # -- tabs ---------------------------------------------------------------

    def _player_tab(self) -> QWidget:
        s = self._settings
        self.user_name = QLineEdit(s.user_name)
        self.org = QLineEdit(s.org)
        user_id = QLineEdit(s.user_id)
        user_id.setReadOnly(True)
        copy = QPushButton("Copy")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(s.user_id))
        id_row = QHBoxLayout()
        id_row.addWidget(user_id)
        id_row.addWidget(copy)
        form = QFormLayout()
        form.addRow("Name", self.user_name)
        form.addRow("Organisation", self.org)
        form.addRow("ScanLake ID", id_row)
        hint = QLabel("Your scans are uploaded under this anonymous ID.")
        hint.setProperty("role", "hint")
        form.addRow(hint)
        return _page(form)

    def _capture_tab(self) -> QWidget:
        s = self._settings
        self.monitor = QComboBox()
        for m in self._monitors:
            self.monitor.addItem(m.label(), m.index)
        self.monitor.setCurrentIndex(max(0, self.monitor.findData(s.monitor)))
        self.region_mode = QComboBox()
        self.region_mode.addItem("Automatic (from the ship profile)", "auto")
        self.region_mode.addItem("Manual", "manual")
        self.region_mode.setCurrentIndex(0 if s.region_mode == "auto" else 1)
        pick = QPushButton("Select region on screen…")
        pick.clicked.connect(self._pick_region)
        self.region_label = QLabel(self._region_text())
        self.region_label.setProperty("role", "hint")
        self.fps = QSpinBox()
        self.fps.setRange(1, 30)
        self.fps.setValue(s.fps)
        form = QFormLayout()
        form.addRow("Monitor", self.monitor)
        form.addRow("Capture region", self.region_mode)
        form.addRow("", pick)
        form.addRow("", self.region_label)
        form.addRow("Frames per second", self.fps)
        return _page(form)

    def _recognition_tab(self, backends: list[BackendInfo], active: str) -> QWidget:
        s = self._settings
        self.backend = QComboBox()
        self.backend.addItem("Automatic (fastest)", AUTO)
        for b in backends:
            label = b.label if b.available else f"{b.label} — {b.reason}"
            self.backend.addItem(label, b.id)
            if not b.available:
                item = self.backend.model().item(self.backend.count() - 1)  # type: ignore[attr-defined]
                item.setEnabled(False)
        self.backend.setCurrentIndex(max(0, self.backend.findData(s.backend)))
        running = QLabel(f"Running: {active or '—'}")
        running.setProperty("role", "hint")
        bench = QPushButton("Benchmark again")
        bench.clicked.connect(self.benchmark_requested.emit)
        self.vote_window = QDoubleSpinBox()
        self.vote_window.setRange(0.3, 10.0)
        self.vote_window.setSingleStep(0.1)
        self.vote_window.setSuffix(" s")
        self.vote_window.setValue(s.vote_window_s)
        form = QFormLayout()
        form.addRow("Engine", self.backend)
        form.addRow("", running)
        form.addRow("", bench)
        form.addRow("Voting window", self.vote_window)
        hint = QLabel("PyTorch engines take effect after a restart.")
        hint.setProperty("role", "hint")
        form.addRow(hint)
        return _page(form)

    def _hotkeys_tab(self) -> QWidget:
        h = self._settings.hotkeys
        self.key_stage, self.key_save, self.key_cluster = (
            KeyEdit(h.stage), KeyEdit(h.save), KeyEdit(h.new_cluster),
        )  # fmt: skip
        form = QFormLayout()
        form.addRow("Stage the current scan", self.key_stage)
        form.addRow("Save", self.key_save)
        form.addRow("New rock (cluster)", self.key_cluster)
        hint = QLabel("These work while the game has focus; the game still sees the key.")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        form.addRow(hint)
        return _page(form)

    def _upload_tab(self) -> QWidget:
        s = self._settings
        self.auto_upload = QCheckBox("Upload automatically")
        self.auto_upload.setChecked(s.auto_upload)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 500)
        self.batch_size.setValue(s.batch_size)
        self.export_dir = QLineEdit(s.export_dir)
        self.export_dir.setPlaceholderText("(off)")
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse_export)
        row = QHBoxLayout()
        row.addWidget(self.export_dir)
        row.addWidget(browse)
        form = QFormLayout()
        form.addRow(self.auto_upload)
        form.addRow(
            "Upload every", _with_suffix(self.batch_size, "scans (and when a session closes)")
        )
        form.addRow("Also save batches to", row)
        return _page(form)

    def _advanced_tab(self) -> QWidget:
        s = self._settings
        self.dev_mode = QCheckBox("Developer mode (profile editor, labelers, training)")
        self.dev_mode.setChecked(s.dev_mode)
        self.collect_crops = QCheckBox("Keep ROI crops of saved scans as training data")
        self.collect_crops.setChecked(s.collect_crops)
        self.collect_crops.setToolTip(
            "Dev mode: crops land in the data folder's training/inbox for the labeler"
        )
        self.beep = QCheckBox("Beep on stage / save / error")
        self.beep.setChecked(s.beep)
        reset_assets = QPushButton("Reset ship profiles and models to the installed ones")
        reset_assets.setToolTip("Removes downloaded asset updates (used from the next start)")
        reset_assets.clicked.connect(self._reset_assets)
        open_data = QPushButton("Open data folder")
        open_data.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.data_dir())))
        )
        form = QFormLayout()
        form.addRow(self.dev_mode)
        form.addRow(self.collect_crops)
        form.addRow(self.beep)
        form.addRow(reset_assets)
        form.addRow(open_data)
        return _page(form)

    # -- helpers ------------------------------------------------------------

    def _region_text(self) -> str:
        r = self._manual_region
        return f"Manual region: {r.w}x{r.h} at {r.x},{r.y}" if r else "No manual region yet"

    def _pick_region(self) -> None:
        index = self.monitor.currentData()
        monitor = next((m for m in self._monitors if m.index == index), None)
        if monitor is None:
            return
        overlay = RegionOverlay(monitor)

        def chosen(x: int, y: int, w: int, h: int) -> None:
            self._manual_region = Region(x=x, y=y, w=w, h=h)
            self.region_mode.setCurrentIndex(1)
            self.region_label.setText(self._region_text())

        overlay.selected.connect(chosen)
        overlay.showFullScreen()

    def _reset_assets(self) -> None:
        from scanlake_desktop import updates

        updates.reset(paths.updates_dir())
        QMessageBox.information(self, "Reset", "Restart ScanLake Desktop to use the installed set.")

    def _browse_export(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Save batches to", self.export_dir.text())
        if folder:
            self.export_dir.setText(folder)

    def result_settings(self) -> Settings:
        s = self._settings.model_copy(deep=True)
        s.user_name = self.user_name.text().strip()
        s.org = self.org.text().strip()
        s.monitor = int(self.monitor.currentData())
        s.region_mode = self.region_mode.currentData()
        s.manual_region = self._manual_region
        s.fps = self.fps.value()
        backend = str(self.backend.currentData())
        if backend != s.backend:
            s.auto_backend = None if backend == AUTO else s.auto_backend
        s.backend = backend
        s.vote_window_s = self.vote_window.value()
        s.hotkeys = Hotkeys(
            stage=self.key_stage.key_name,
            save=self.key_save.key_name,
            new_cluster=self.key_cluster.key_name,
        )
        s.auto_upload = self.auto_upload.isChecked()
        s.batch_size = self.batch_size.value()
        s.export_dir = self.export_dir.text().strip()
        s.dev_mode = self.dev_mode.isChecked()
        s.beep = self.beep.isChecked()
        s.collect_crops = self.collect_crops.isChecked()
        return s


def _page(form: QFormLayout) -> QWidget:
    page = QWidget()
    page.setLayout(form)
    return page


def _with_suffix(widget: QWidget, text: str) -> QWidget:
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(widget)
    row.addWidget(QLabel(text))
    row.addStretch(1)
    return holder
