"""Dev mode: what the pipeline sees (frame, anchors, ROIs, readings) and Save snapshot.

A snapshot folder is a golden test fixture in the making:

    <snapshots>/<ship>/<WxH>/<timestamp>/frame.png      the captured region
                                         capture.json   ship, region, monitor, scale, backend
                                         observed.json  what was read (per ROI: text, confidence)
                                         expected.json  copy of observed.json: correct it by hand

Move the folder to tests/fixtures/golden/ once expected.json is right.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from scanlake_desktop.capture.source import Monitor, write_image
from scanlake_desktop.worker.pipeline_worker import LiveState


def annotate(state: LiveState) -> np.ndarray | None:
    if state.frame is None:
        return None
    image = state.frame.copy()
    for name, (x, y, w, h, conf) in state.anchors.items():
        p = (round(x), round(y))
        cv2.rectangle(image, p, (p[0] + w, p[1] + h), (255, 160, 0), 1)
        cv2.putText(image, f"{name} {conf:.2f}", (p[0], p[1] - 3), cv2.FONT_HERSHEY_PLAIN, 0.8,
                    (255, 160, 0), 1)  # fmt: skip
    for name, reading in state.readings.items():
        r = reading.rect
        vote = state.votes.get(name)
        color = (90, 200, 90) if vote is not None and vote.stable else (0, 170, 255)
        cv2.rectangle(image, (r.x, r.y), (r.right, r.bottom), color, 1)
    return image


def to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


class DebugWindow(QWidget):
    snapshot_saved = Signal(str)

    def __init__(self, snapshots: Path, monitor: Monitor) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("ScanLake · Pipeline debug")
        self.resize(1000, 700)
        self._snapshots = snapshots
        self._monitor = monitor
        self._state: LiveState | None = None

        self._image = QLabel()
        self._image.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidget(self._image)
        scroll.setWidgetResizable(True)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["ROI", "Decoded", "Value", "Conf", "Vote"])
        self._timings = QLabel()
        snap = QPushButton("Save snapshot")
        snap.clicked.connect(self.save_snapshot)
        folder = QPushButton("Open snapshots")
        folder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._snapshots)))
        )
        buttons = QHBoxLayout()
        buttons.addWidget(snap)
        buttons.addWidget(folder)
        buttons.addStretch(1)
        buttons.addWidget(self._timings)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(self._table)
        split.setSizes([560, 440])
        box = QVBoxLayout(self)
        box.addWidget(split, 1)
        box.addLayout(buttons)

    def set_monitor(self, monitor: Monitor) -> None:
        self._monitor = monitor

    def update_state(self, state: LiveState) -> None:
        self._state = state
        image = annotate(state)
        if image is not None:
            self._image.setPixmap(to_pixmap(image))
        names = sorted(state.readings)
        self._table.setRowCount(len(names))
        for r, name in enumerate(names):
            reading = state.readings[name]
            vote = state.votes.get(name)
            cells = [name, reading.raw_text, "∅" if reading.text is None else reading.text,
                     f"{reading.confidence:.2f}",
                     f"{vote.text} ({vote.stability.value})" if vote else ""]  # fmt: skip
            for c, text in enumerate(cells):
                self._table.setItem(r, c, QTableWidgetItem(text))
        self._timings.setText(
            " · ".join(f"{k} {v * 1000:.1f} ms" for k, v in state.timings.items())
        )

    def save_snapshot(self) -> None:
        state = self._state
        if state is None or state.frame is None:
            return
        h, w = state.frame.shape[:2]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = (
            self._snapshots / state.ship / f"{self._monitor.width}x{self._monitor.height}" / stamp
        )
        folder.mkdir(parents=True, exist_ok=True)
        write_image(folder / "frame.png", state.frame)
        capture = {
            "ship": state.ship,
            "monitor": asdict(self._monitor),
            "region": asdict(state.region),
            "frame_size": [w, h],
            "scale": state.scale,
            "locked": state.locked,
            "backend": state.backend,
        }
        observed = {
            name: {"text": r.text, "raw": r.raw_text, "confidence": round(r.confidence, 4),
                   "vote": state.votes[name].text if name in state.votes else None}
            for name, r in sorted(state.readings.items())
        }  # fmt: skip
        expected = {name: v["vote"] if v["vote"] is not None else v["text"]
                    for name, v in observed.items()}  # fmt: skip
        for filename, data in (("capture.json", capture), ("observed.json", observed),
                               ("expected.json", expected)):  # fmt: skip
            (folder / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.snapshot_saved.emit(str(folder))
