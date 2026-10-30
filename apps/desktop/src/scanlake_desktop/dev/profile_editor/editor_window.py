"""Dev mode: edit a ship's HUD profile on a frozen frame.

Coordinates: profiles are authored in the reference frame (2560x1440 HUD). The anchors are matched
on the shown frame to get the reference -> frame transform; ROI boxes are drawn through it and
moves are mapped back. Without an anchor lock the frame is taken as reference-scale.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from scanlake_core.format import CONFIDENCES, Coverage, ParquetFormat, Source
from scanlake_core.geometry import Affine, apply, identity
from scanlake_core.preprocess import Gray, to_gray
from scanlake_core.profile.load import ProfileError
from scanlake_core.recognition import FrameRecognizer
from scanlake_core.schema_map import COMPOSITIONS, SCANS, slot_layout
from scanlake_core.vision.anchors import AnchorLocator, load_template
from scanlake_desktop.assets import AssetSet
from scanlake_desktop.capture.source import read_image, write_image
from scanlake_desktop.dev.profile_editor.canvas import FrameCanvas, Mode
from scanlake_desktop.dev.profile_editor.model import EditableProfile

log = logging.getLogger(__name__)

MODES = ("digit_crnn", "word_cnn")
CHANNELS = ("none", "red", "green", "blue")


class NewRoiDialog(QDialog):
    def __init__(self, columns: dict[str, list[str]], rows: int, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("New ROI")
        self.name = QLineEdit()
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.target = QComboBox()
        self.target.setEditable(True)
        for column in columns.get(SCANS, []):
            self.target.addItem(f"scan: {column}", (column, None))
        for r in range(rows):
            for column in columns.get(COMPOSITIONS, []):
                self.target.addItem(f"material {r + 1}: {column}", (column, r))
        hint = QLabel("Type “scan: <name>” for a column the format does not list yet.")
        hint.setProperty("role", "hint")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("Name", self.name)
        form.addRow("Model", self.mode)
        form.addRow("Fills", self.target)
        form.addRow(hint)
        form.addRow(buttons)

    def choice(self) -> tuple[str, str, str, int | None]:
        data = self.target.currentData()
        text = self.target.currentText()
        if data is None or self.target.itemText(self.target.currentIndex()) != text:
            column = text.split(":", 1)[-1].strip()
            data = (column, None)
        column, row = data
        return self.name.text().strip(), self.mode.currentText(), column, row


class ProfileEditor(QWidget):
    applied = Signal(str)  # ship key whose profile was saved and should be reloaded

    def __init__(
        self,
        assets: AssetSet,
        user_assets: Path,
        ships: list[tuple[str, str, str]],  # key, label, file
        current_format: Callable[[], ParquetFormat],
        live_frame: Callable[[], np.ndarray | None],
        make_engine: Callable[[], object],
    ) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("ScanLake · Profile editor")
        self.resize(1300, 820)
        self.assets = assets
        self.user_assets = user_assets
        self.ships = ships
        self.current_format = current_format
        self.live_frame = live_frame
        self.make_engine = make_engine
        self._engine: object | None = None
        self.profile: EditableProfile | None = None
        self.frame: np.ndarray | None = None
        self.transform: Affine = identity()
        self.scale = 1.0
        self.selected: str | None = None
        self._loading = False

        self.canvas = FrameCanvas()
        self.canvas.roi_selected.connect(self.select_roi)
        self.canvas.roi_moved.connect(self._roi_moved)
        self.canvas.rect_drawn.connect(self._rect_drawn)

        self.ship = QComboBox()
        for key, label, _ in ships:
            self.ship.addItem(label, key)
        self.ship.currentIndexChanged.connect(lambda _: self.load_ship())
        live = QPushButton("Use live frame")
        live.clicked.connect(self.use_live_frame)
        open_image = QPushButton("Open image…")
        open_image.clicked.connect(self.open_image)
        self.draw_button = QPushButton("Draw ROI")
        self.draw_button.setCheckable(True)
        self.draw_button.toggled.connect(lambda on: self._set_mode(Mode.DRAW_ROI if on else None))
        self.anchor_button = QPushButton("Cut anchor")
        self.anchor_button.setCheckable(True)
        self.anchor_button.toggled.connect(
            lambda on: self._set_mode(Mode.CUT_ANCHOR if on else None)
        )
        save = QPushButton("Save")
        save.clicked.connect(self.save)
        apply_button = QPushButton("Save and use in capture")
        apply_button.clicked.connect(self.save_and_apply)
        self.status = QLabel()
        self.status.setProperty("role", "hint")

        bar = QHBoxLayout()
        for w in (
            QLabel("Ship"),
            self.ship,
            live,
            open_image,
            self.draw_button,
            self.anchor_button,
            save,
            apply_button,
        ):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.status)

        tabs = QTabWidget()
        tabs.addTab(self._roi_tab(), "ROIs")
        tabs.addTab(self._anchor_tab(), "Anchors")
        tabs.addTab(self._coverage_tab(), "Format coverage")
        tabs.addTab(self._test_tab(), "Test")
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.canvas)
        split.addWidget(tabs)
        split.setSizes([800, 480])
        box = QVBoxLayout(self)
        box.addLayout(bar)
        box.addWidget(split, 1)
        self.load_ship()

    # -- tabs -----------------------------------------------------------------

    def _roi_tab(self) -> QWidget:
        self.roi_list = QListWidget()
        self.roi_list.currentTextChanged.connect(lambda t: self.select_roi(t.split("  →")[0]))
        self.f_name = QLineEdit()
        self.f_mode = QComboBox()
        self.f_mode.addItems(MODES)
        self.f_x, self.f_y = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.f_x, self.f_y):
            spin.setRange(-5000, 5000)
            spin.setDecimals(1)
        self.f_w, self.f_h = QSpinBox(), QSpinBox()
        for spin in (self.f_w, self.f_h):
            spin.setRange(1, 2000)
        self.f_column = QComboBox()
        self.f_column.setEditable(True)
        self.f_gray = QCheckBox("grayscale")
        self.f_invert = QCheckBox("invert")
        self.f_contrast, self.f_brightness, self.f_threshold = QSpinBox(), QSpinBox(), QSpinBox()
        self.f_contrast.setRange(-100, 100)
        self.f_brightness.setRange(-255, 255)
        self.f_threshold.setRange(0, 255)
        self.f_threshold_on = QCheckBox("threshold")
        self.f_channel = QComboBox()
        self.f_channel.addItems(CHANNELS)
        delete = QPushButton("Delete ROI")
        delete.clicked.connect(self.delete_roi)

        for w in (self.f_name, self.f_column.lineEdit()):
            w.editingFinished.connect(self._form_changed)  # type: ignore[union-attr]
        for w in (
            self.f_x,
            self.f_y,
            self.f_w,
            self.f_h,
            self.f_contrast,
            self.f_brightness,
            self.f_threshold,
        ):
            w.editingFinished.connect(self._form_changed)
        for w in (self.f_gray, self.f_invert, self.f_threshold_on):
            w.toggled.connect(self._form_changed)
        for w in (self.f_mode, self.f_channel):
            w.currentIndexChanged.connect(self._form_changed)
        self.f_column.activated.connect(self._form_changed)

        form = QFormLayout()
        form.addRow("Name", self.f_name)
        form.addRow("Model", self.f_mode)
        form.addRow("Fills column", self.f_column)
        row = QHBoxLayout()
        for label, w in (("x", self.f_x), ("y", self.f_y), ("w", self.f_w), ("h", self.f_h)):
            row.addWidget(QLabel(label))
            row.addWidget(w)
        form.addRow("Reference box", row)
        filters = QHBoxLayout()
        for w in (
            self.f_gray,
            self.f_invert,
            QLabel("contrast"),
            self.f_contrast,
            QLabel("brightness"),
            self.f_brightness,
        ):
            filters.addWidget(w)
        form.addRow("Filters", filters)
        thresholds = QHBoxLayout()
        for w in (self.f_threshold_on, self.f_threshold, QLabel("channel"), self.f_channel):
            thresholds.addWidget(w)
        form.addRow("", thresholds)
        hint = QLabel("Filters apply to digit ROIs only; word ROIs are read unfiltered.")
        hint.setProperty("role", "hint")
        form.addRow(hint)
        form.addRow(delete)
        page = QWidget()
        box = QVBoxLayout(page)
        box.addWidget(self.roi_list, 1)
        box.addLayout(form)
        return page

    def _anchor_tab(self) -> QWidget:
        self.anchor_table = QTableWidget(0, 5)
        self.anchor_table.setHorizontalHeaderLabels(
            ["Anchor", "Template", "ref x", "ref y", "Match"]
        )
        hint = QLabel(
            "Anchors fix the HUD position and scale. To replace one: press “Cut anchor”, drag a "
            "box around a distinctive, unchanging part of the HUD, then pick which anchor it is."
        )
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")
        page = QWidget()
        box = QVBoxLayout(page)
        box.addWidget(self.anchor_table, 1)
        box.addWidget(hint)
        return page

    def _coverage_tab(self) -> QWidget:
        self.coverage_table = QTableWidget(0, 4)
        self.coverage_table.setHorizontalHeaderLabels(["File", "Column", "Filled by", "ROIs"])
        self.coverage_note = QLabel()
        self.coverage_note.setWordWrap(True)
        page = QWidget()
        box = QVBoxLayout(page)
        box.addWidget(self.coverage_note)
        box.addWidget(self.coverage_table, 1)
        return page

    def _test_tab(self) -> QWidget:
        run = QPushButton("Read this frame with the edited profile")
        run.clicked.connect(self.run_test)
        self.test_table = QTableWidget(0, 4)
        self.test_table.setHorizontalHeaderLabels(["ROI", "Decoded", "Value", "Confidence"])
        page = QWidget()
        box = QVBoxLayout(page)
        box.addWidget(run)
        box.addWidget(self.test_table, 1)
        return page

    # -- loading --------------------------------------------------------------

    def load_ship(self) -> None:
        key = str(self.ship.currentData())
        file = next(f for k, _, f in self.ships if k == key)
        self.profile = EditableProfile.load(key, self.assets.profile_path(file))
        self.selected = None
        self._relocate()
        self.refresh()

    def use_live_frame(self) -> None:
        frame = self.live_frame()
        if frame is None:
            QMessageBox.information(self, "No frame", "No live frame yet: is the HUD on screen?")
            return
        self._show(frame)

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open frame", "", "Images (*.png *.jpg)")
        if path:
            self._show(read_image(Path(path)))

    def _show(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.canvas.show_frame(frame)
        self._relocate()
        self.refresh()

    def _templates(self) -> dict[str, Gray]:
        assert self.profile is not None
        out = {}
        for anchor in self.profile.anchors():
            try:
                out[anchor["name"]] = load_template(
                    self.assets.anchor_path(anchor["template_path"])
                )
            except (FileNotFoundError, OSError):
                log.warning("anchor template missing: %s", anchor["template_path"])
        return out

    def _relocate(self) -> None:
        """Match the anchors on the frame to get the reference -> frame transform."""
        self.transform, self.scale = identity(), 1.0
        self._matches: dict[str, tuple[QRectF, float]] = {}
        if self.frame is None or self.profile is None:
            return
        try:
            hud = self.profile.hud()
        except (ProfileError, KeyError, ValueError) as err:
            self.status.setText(f"Profile invalid: {err}")
            return
        fix = AnchorLocator(hud.main.anchors, self._templates()).locate(to_gray(self.frame))
        for name, m in fix.matches.items():
            self._matches[name] = (QRectF(m.x, m.y, m.w, m.h), m.confidence)
        if fix.transform is not None:
            self.transform, self.scale = fix.transform, fix.scale
            self.status.setText(f"HUD located · scale {fix.scale:.3f}")
        else:
            self.status.setText("Anchors not found: showing the frame at reference scale")

    # -- display --------------------------------------------------------------

    def refresh(self) -> None:
        if self.profile is None:
            return
        self._loading = True
        fmt = self.current_format()
        cov = self.profile.coverage(fmt)
        names = [r["name"] for r in self.profile.rois()]
        self.roi_list.clear()
        for name in names:
            self.roi_list.addItem(f"{name}  →  {self._target(name)}")
        self._fill_form()
        self._fill_anchors()
        self._fill_coverage(cov)
        if self.frame is not None:
            rois = {}
            for roi in self.profile.rois():
                x, y = apply(self.transform, roi["ref_x"], roi["ref_y"])
                rois[roi["name"]] = QRectF(
                    x, y, roi["width"] * self.scale, roi["height"] * self.scale
                )
            anchors = {n: r for n, (r, _) in self._matches.items()}
            self.canvas.set_overlay(rois, anchors, self.selected)
        dirty = " · unsaved changes" if self.profile.dirty else ""
        self.setWindowTitle(f"ScanLake · Profile editor · {self.profile.file}{dirty}")
        self._loading = False

    def _target(self, name: str) -> str:
        assert self.profile is not None
        bound = self.profile.binding(name)
        if bound is None:
            return "(unused)"
        _, column, row = bound
        return column if row is None else f"{column} (material {row + 1})"

    def _fill_form(self) -> None:
        assert self.profile is not None
        enabled = self.selected is not None
        for w in (
            self.f_name,
            self.f_mode,
            self.f_x,
            self.f_y,
            self.f_w,
            self.f_h,
            self.f_column,
            self.f_gray,
            self.f_invert,
            self.f_contrast,
            self.f_brightness,
            self.f_threshold,
            self.f_threshold_on,
            self.f_channel,
        ):
            w.setEnabled(enabled)
        if self.selected is None:
            return
        roi = self.profile.roi(self.selected)
        f = roi.get("filters", {})
        bound = self.profile.binding(self.selected)
        kind = bound[0] if bound else SCANS
        self.f_name.setText(roi["name"])
        self.f_mode.setCurrentText(roi.get("recognition_mode", "digit_crnn"))
        self.f_x.setValue(float(roi.get("ref_x", 0)))
        self.f_y.setValue(float(roi.get("ref_y", 0)))
        self.f_w.setValue(int(roi.get("width", 1)))
        self.f_h.setValue(int(roi.get("height", 1)))
        self.f_column.clear()
        spec = self.current_format().file(kind)
        options = [c.name for c in spec.columns] if spec else []
        self.f_column.addItems(options)
        self.f_column.setCurrentText(bound[1] if bound else "")
        self.f_gray.setChecked(bool(f.get("grayscale", True)))
        self.f_invert.setChecked(bool(f.get("invert", False)))
        self.f_contrast.setValue(int(f.get("contrast", 0)))
        self.f_brightness.setValue(int(f.get("brightness", 0)))
        self.f_threshold_on.setChecked(bool(f.get("threshold_enabled", False)))
        self.f_threshold.setValue(int(f.get("threshold", 127)))
        self.f_channel.setCurrentText(str(f.get("channel", "none")))

    def _fill_anchors(self) -> None:
        assert self.profile is not None
        anchors = self.profile.anchors()
        self.anchor_table.setRowCount(len(anchors))
        for r, a in enumerate(anchors):
            match = self._matches.get(a["name"])
            found = f"{match[1]:.2f}" if match else "—"
            for c, text in enumerate(
                [a["name"], a["template_path"], f"{a['ref_x']:.1f}", f"{a['ref_y']:.1f}", found]
            ):
                self.anchor_table.setItem(r, c, QTableWidgetItem(text))

    def _fill_coverage(self, cov: Coverage) -> None:
        self.coverage_table.setRowCount(len(cov.bindings))
        for r, b in enumerate(cov.bindings):
            source = b.source.value + (
                ""
                if b.source is not Source.UNBOUND
                else (" — REQUIRED" if b.required else " (left out)")
            )
            for c, text in enumerate([b.kind, b.column, source, ", ".join(b.rois)]):
                self.coverage_table.setItem(r, c, QTableWidgetItem(text))
        fmt_version = cov.format_version
        if cov.complete:
            self.coverage_note.setText(
                f"This profile fills every required column of format v{fmt_version}."
            )
        else:
            self.coverage_note.setText(
                f"Missing for format v{fmt_version}: {cov.describe_missing()}. "
                "Draw a ROI and let it fill that column."
            )

    # -- editing --------------------------------------------------------------

    def select_roi(self, name: str) -> None:
        if self._loading or not name or self.profile is None:
            return
        if name not in {r["name"] for r in self.profile.rois()}:
            return
        self.selected = name
        self._loading = True
        self._fill_form()
        self._loading = False

    def _form_changed(self, *_: object) -> None:
        if self._loading or self.profile is None or self.selected is None:
            return
        filters = {
            "grayscale": self.f_gray.isChecked(),
            "invert": self.f_invert.isChecked(),
            "contrast": self.f_contrast.value(),
            "brightness": self.f_brightness.value(),
            "threshold_enabled": self.f_threshold_on.isChecked(),
            "threshold": self.f_threshold.value(),
            "channel": self.f_channel.currentText(),
        }
        new_name = self.f_name.text().strip() or self.selected
        try:
            self.profile.update_roi(
                self.selected,
                name=new_name,
                recognition_mode=self.f_mode.currentText(),
                ref_x=self.f_x.value(),
                ref_y=self.f_y.value(),
                width=self.f_w.value(),
                height=self.f_h.value(),
                filters={**self.profile.roi(self.selected).get("filters", {}), **filters},
            )
        except ValueError as err:
            QMessageBox.warning(self, "Cannot rename", str(err))
            return
        self.selected = new_name
        column = self.f_column.currentText().strip()
        bound = self.profile.binding(new_name)
        if column and bound and column != bound[1]:
            self.profile.set_column(new_name, column)
        self.refresh()

    def _roi_moved(self, name: str, x: float, y: float) -> None:
        if self.profile is None:
            return
        inverse = cv2.invertAffineTransform(self.transform)
        ref_x, ref_y = apply(inverse, x, y)
        self.profile.update_roi(name, ref_x=round(ref_x, 1), ref_y=round(ref_y, 1))
        self.selected = name
        self.refresh()

    def _rect_drawn(self, mode: Mode, rect: QRectF) -> None:
        if self.profile is None:
            return
        inverse = cv2.invertAffineTransform(self.transform)
        ref_x, ref_y = apply(inverse, rect.x(), rect.y())
        ref_w, ref_h = rect.width() / self.scale, rect.height() / self.scale
        if mode is Mode.DRAW_ROI:
            self._new_roi((ref_x, ref_y, round(ref_w), round(ref_h)))
            self.draw_button.setChecked(False)
        elif mode is Mode.CUT_ANCHOR:
            self._cut_anchor(rect, (ref_x, ref_y))
            self.anchor_button.setChecked(False)

    def _new_roi(self, ref: tuple[float, float, int, int]) -> None:
        assert self.profile is not None
        fmt = self.current_format()
        columns = {
            kind: [c.name for c in spec.columns]
            for kind, spec in fmt.files.items()
            if kind != CONFIDENCES
        }
        rows = len(slot_layout(self.profile.hud()).rows)
        dialog = NewRoiDialog(columns, rows, self)
        if not dialog.exec():
            return
        name, mode, column, row = dialog.choice()
        if not name or not column:
            return
        try:
            self.profile.add_roi(
                name, mode, (round(ref[0], 1), round(ref[1], 1), ref[2], ref[3]), column, row
            )
        except ValueError as err:
            QMessageBox.warning(self, "Cannot add ROI", str(err))
            return
        self.selected = name
        self.refresh()

    def _cut_anchor(self, rect: QRectF, ref: tuple[float, float]) -> None:
        assert self.profile is not None and self.frame is not None
        names = [a["name"] for a in self.profile.anchors()]
        name, ok = QInputDialog.getItem(
            self, "Cut anchor", "Replace which anchor?", names, 0, False
        )
        if not ok:
            return
        x, y, w, h = round(rect.x()), round(rect.y()), round(rect.width()), round(rect.height())
        crop = self.frame[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0:
            return
        # Templates are kept at reference scale (the locator rescales them per screen)
        size = (
            max(1, round(crop.shape[1] / self.scale)),
            max(1, round(crop.shape[0] / self.scale)),
        )
        template = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        file = f"{self.profile.key}_{name}.png"
        write_image(self.user_assets / "anchors" / file, template)
        self.profile.update_anchor(
            name, template_path=file, ref_x=round(ref[0], 1), ref_y=round(ref[1], 1)
        )
        self._relocate()
        self.refresh()

    def delete_roi(self) -> None:
        if self.profile is None or self.selected is None:
            return
        self.profile.delete_roi(self.selected)
        self.selected = None
        self.refresh()

    def _set_mode(self, mode: Mode | None) -> None:
        if mode is Mode.DRAW_ROI:
            self.anchor_button.setChecked(False)
        elif mode is Mode.CUT_ANCHOR:
            self.draw_button.setChecked(False)
        self.canvas.set_mode(mode or Mode.SELECT)

    # -- test and save --------------------------------------------------------

    def run_test(self) -> None:
        if self.profile is None or self.frame is None:
            QMessageBox.information(self, "No frame", "Load a live frame or an image first.")
            return
        if self._engine is None:
            self._engine = self.make_engine()
        recognizer = FrameRecognizer(self.profile.hud(), self._templates(), self._engine)  # type: ignore[arg-type]
        result = recognizer.process(self.frame, expected_scale=self.scale)
        if not result.locked:
            self.status.setText("Test: anchors not found on this frame")
        rows = sorted(result.readings.items())
        self.test_table.setRowCount(len(rows))
        for r, (name, reading) in enumerate(rows):
            for c, text in enumerate(
                [
                    name,
                    reading.raw_text,
                    "∅" if reading.text is None else reading.text,
                    f"{reading.confidence:.2f}",
                ]
            ):
                self.test_table.setItem(r, c, QTableWidgetItem(text))

    def save(self) -> bool:
        if self.profile is None:
            return False
        try:
            path = self.profile.save(self.user_assets)
        except (ProfileError, KeyError, ValueError) as err:
            QMessageBox.warning(self, "Not saved", f"The profile is not valid: {err}")
            return False
        self.status.setText(f"Saved {path.name} (overrides the bundled profile)")
        self.refresh()
        return True

    def save_and_apply(self) -> None:
        if self.profile is not None and self.save():
            self.applied.emit(self.profile.key)
