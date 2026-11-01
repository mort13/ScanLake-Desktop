"""Dev mode: review the crops collected from real captures and turn them into training data.

Accepting moves a crop from `inbox/` to `real/` under the (possibly corrected) label; `real/`
is then the held-out validation set for `scanlake-train --val-data` (or extra training data).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from scanlake_core.preprocess import resize_pad
from scanlake_desktop.capture.source import read_image
from scanlake_desktop.dev.debug_window import to_pixmap

STRIPS, WORDS = "strips", "words"
DIGITS = set("0123456789.%m")


@dataclass(frozen=True)
class Item:
    path: Path
    label: str


class LabelQueue:
    """The unreviewed crops of one kind (strips or words) under `root/inbox`."""

    def __init__(self, root: Path, kind: str) -> None:
        self.root = root
        self.kind = kind
        self.items = self._scan()

    def _scan(self) -> list[Item]:
        inbox = self.root / "inbox" / self.kind
        if not inbox.exists():
            return []
        if self.kind == STRIPS:
            return [Item(p, p.stem.rpartition("_")[0]) for p in sorted(inbox.glob("*.png"))]
        return [Item(p, p.parent.name) for p in sorted(inbox.glob("*/*.png"))]

    def accept(self, item: Item, label: str) -> Path:
        if self.kind == STRIPS:
            if not label or not set(label) <= DIGITS:
                raise ValueError(f"{label!r} is not a digit label (0-9 . % m)")
            tail = item.path.stem.rpartition("_")[2]
            target = self.root / "real" / STRIPS / f"{label}_{tail}.png"
        else:
            target = self.root / "real" / WORDS / label / item.path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item.path), str(target))
        self.items.remove(item)
        return target

    def delete(self, item: Item) -> None:
        item.path.unlink(missing_ok=True)
        self.items.remove(item)


class LabelerWindow(QWidget):
    def __init__(self, root: Path, word_classes: tuple[str, ...]) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("ScanLake · Labeler")
        self.resize(760, 380)
        self.root = root
        self.word_classes = word_classes
        self.queue = LabelQueue(root, STRIPS)
        self.index = 0

        self.kind = QComboBox()
        self.kind.addItem("Digit crops", STRIPS)
        self.kind.addItem("Word crops", WORDS)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.count = QLabel()
        open_folder = QPushButton("Open training folder")
        open_folder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root)))
        )
        self.crop = QLabel()
        self.crop.setMinimumHeight(110)
        self.crop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_input = QLabel()
        self.model_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text = QLineEdit()
        self.word = QComboBox()
        self.word.addItems(list(word_classes))
        self.error = QLabel()
        self.error.setProperty("role", "issues")
        accept = QPushButton("Accept (Enter)")
        skip = QPushButton("Skip")
        delete = QPushButton("Delete (Del)")
        accept.clicked.connect(self.accept)
        skip.clicked.connect(self.skip)
        delete.clicked.connect(self.delete)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, self.accept)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self.delete)

        top = QHBoxLayout()
        top.addWidget(self.kind)
        top.addWidget(self.count, 1)
        top.addWidget(open_folder)
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label"))
        label_row.addWidget(self.text, 1)
        label_row.addWidget(self.word, 1)
        buttons = QHBoxLayout()
        for b in (accept, skip, delete):
            buttons.addWidget(b)
        buttons.addStretch(1)
        hint = QLabel(
            "Crops come from saved scans (Settings → Advanced → keep ROI crops). "
            "Accepted ones go to training/real, the validation set for training."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        box = QVBoxLayout(self)
        box.addLayout(top)
        box.addWidget(self.crop)
        box.addWidget(self.model_input)
        box.addLayout(label_row)
        box.addWidget(self.error)
        box.addLayout(buttons)
        box.addWidget(hint)
        self.show_current()

    def _kind_changed(self) -> None:
        self.queue = LabelQueue(self.root, str(self.kind.currentData()))
        self.index = 0
        self.show_current()

    def current(self) -> Item | None:
        if not self.queue.items:
            return None
        self.index %= len(self.queue.items)
        return self.queue.items[self.index]

    def show_current(self) -> None:
        words = self.queue.kind == WORDS
        self.text.setVisible(not words)
        self.word.setVisible(words)
        self.error.clear()
        item = self.current()
        self.count.setText(f"{len(self.queue.items)} to review")
        if item is None:
            self.crop.setText("Nothing to review")
            self.model_input.clear()
            self.text.clear()
            return
        image = read_image(item.path)
        h, w = image.shape[:2]
        zoom = max(1, min(6, 600 // max(w, 1)))
        self.crop.setPixmap(
            to_pixmap(cv2.resize(image, (w * zoom, h * zoom), interpolation=cv2.INTER_NEAREST))
        )
        canvas = cv2.cvtColor(
            resize_pad(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR
        )
        self.model_input.setPixmap(
            to_pixmap(cv2.resize(canvas, (512, 64), interpolation=cv2.INTER_NEAREST))
        )
        if words:
            self.word.setCurrentText(item.label)
        else:
            self.text.setText(item.label)
            self.text.selectAll()
            self.text.setFocus()

    def accept(self) -> None:
        item = self.current()
        if item is None:
            return
        label = self.word.currentText() if self.queue.kind == WORDS else self.text.text().strip()
        try:
            self.queue.accept(item, label)
        except ValueError as err:
            self.error.setText(str(err))
            return
        self.show_current()

    def skip(self) -> None:
        self.index += 1
        self.show_current()

    def delete(self) -> None:
        item = self.current()
        if item is not None:
            self.queue.delete(item)
            self.show_current()
