"""Dev mode: train a model, watch it learn, compare it with the current one, and use it.

Runs `scanlake-train` in a child process (the app itself with --train-cli, so it also works from
the packaged full build) and reads its JSON-lines progress. Needs PyTorch.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QProcess, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from scanlake_desktop.ui.style import OK, WARN

MODEL_NAMES = {"crnn": "digit_crnn", "word": "word_model2k"}
DATA_KIND = {"crnn": "strips", "word": "words"}


def app_command() -> list[str]:
    """How to start this app again (frozen exe, or `python -m scanlake_desktop` from source)."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "scanlake_desktop"]


class Chart(QWidget):
    """Training loss (scaled to its maximum) and validation accuracy (0..1) per epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(180)
        self.loss: list[float] = []
        self.accuracy: list[float] = []

    def clear(self) -> None:
        self.loss, self.accuracy = [], []
        self.update()

    def add(self, loss: float, accuracy: float) -> None:
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h, pad = self.width(), self.height(), 24
        p.setPen(QPen(QColor(120, 120, 130), 1))
        p.drawRect(pad, 8, w - 2 * pad, h - pad - 8)
        n = len(self.loss)
        if n == 0:
            p.setPen(QPen(self.palette().color(QPalette.ColorRole.WindowText)))
            p.drawText(pad + 8, 28, "Loss (amber) and validation accuracy (green) appear here")
            return
        top = max(max(self.loss), 1e-9)

        def line(values: list[float], scale: float, color: str) -> None:
            pts = [
                QPointF(
                    pad + (w - 2 * pad) * (i / max(n - 1, 1)), h - pad - (h - pad - 8) * (v / scale)
                )
                for i, v in enumerate(values)
            ]
            p.setPen(QPen(QColor(color), 2))
            p.drawPolyline(QPolygonF(pts))

        line(self.loss, top, WARN)
        line(self.accuracy, 1.0, OK)
        p.setPen(QPen(self.palette().color(QPalette.ColorRole.WindowText)))
        p.drawText(
            pad + 6,
            24,
            f"epoch {n} · val {self.accuracy[-1]:.3f} · best "
            f"{max(self.accuracy):.3f} · loss {self.loss[-1]:.4f}",
        )


class TrainingWindow(QWidget):
    model_installed = Signal(str)  # model name

    def __init__(self, root: Path, current_models: Path, user_models: Path, torch_ok: bool):
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("ScanLake · Training")
        self.resize(820, 640)
        self.root = root
        self.current_models = current_models
        self.user_models = user_models
        self.process: QProcess | None = None
        self.run_stem: Path | None = None
        self.export_stem: Path | None = None
        self._buffer = ""
        self._step = ""
        self._compare: list[dict[str, Any]] = []
        self._pending_eval: list[str] = []
        self._eval_folder = ""

        self.task = QComboBox()
        self.task.addItem("Digits (CRNN)", "crnn")
        self.task.addItem("Words (CNN)", "word")
        self.task.currentIndexChanged.connect(self._task_changed)
        self.data = QLineEdit()
        self.val = QLineEdit()
        self.val.setPlaceholderText("empty = hold out 15 % of the training data")
        self.epochs, self.patience = QSpinBox(), QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(40)
        self.patience.setRange(0, 100)
        self.patience.setValue(8)
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda"])
        synth = QPushButton("Generate synthetic data")
        synth.clicked.connect(self.generate)
        self.start_button = QPushButton("Start training")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.export_button = QPushButton("Export and compare")
        self.export_button.clicked.connect(self.export)
        self.use_button = QPushButton("Use this model")
        self.use_button.clicked.connect(self.use_model)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.chart = Chart()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)

        form = QFormLayout()
        form.addRow("Model", self.task)
        form.addRow("Training data", _with_browse(self.data, self))
        form.addRow("Validation data", _with_browse(self.val, self))
        row = QHBoxLayout()
        for label, w in (
            ("epochs", self.epochs),
            ("stop after flat epochs", self.patience),
            ("device", self.device),
        ):
            row.addWidget(QLabel(label))
            row.addWidget(w)
        row.addStretch(1)
        form.addRow("", row)
        buttons = QHBoxLayout()
        for b in (synth, self.start_button, self.stop_button, self.export_button, self.use_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        box = QVBoxLayout(self)
        box.addLayout(form)
        box.addLayout(buttons)
        box.addWidget(self.status)
        box.addWidget(self.chart)
        box.addWidget(self.log, 1)

        self._task_changed()
        self._update_buttons()
        if not torch_ok:
            for w in (synth, self.start_button):
                w.setEnabled(False)
            self.status.setText("Training needs PyTorch: use the full build or run from source.")

    # -- helpers --------------------------------------------------------------

    @property
    def task_id(self) -> str:
        return str(self.task.currentData())

    def _task_changed(self) -> None:
        kind = DATA_KIND[self.task_id]
        self.data.setText(str(self.root / "synth" / kind))
        real = self.root / "real" / kind
        has_real = real.exists() and any(real.rglob("*.png"))
        self.val.setText(str(real) if has_real else "")

    def _update_buttons(self) -> None:
        running = self.process is not None
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.export_button.setEnabled(
            not running
            and self.run_stem is not None
            and self.run_stem.with_suffix(".safetensors").exists()
        )
        self.use_button.setEnabled(not running and self.export_stem is not None)

    def _run(self, step: str, args: list[str]) -> None:
        command = app_command() + ["--train-cli", *args]
        self.log.appendPlainText(f"$ scanlake-train {' '.join(args)}")
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self._step = step
        self.process.start(command[0], command[1:])
        self._update_buttons()

    def _read(self) -> None:
        assert self.process is not None
        self._buffer += bytes(self.process.readAllStandardOutput().data()).decode(errors="replace")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self._handle(line.rstrip("\r"))

    def _handle(self, line: str) -> None:
        if line.startswith("{"):
            try:
                event: dict[str, Any] = json.loads(line)
            except ValueError:
                self.log.appendPlainText(line)
                return
            if event.get("event") == "start":
                self.status.setText(
                    f"Training on {event['device']}: {event['train']} samples, "
                    f"{event['val']} for validation"
                )
            elif event.get("event") == "epoch":
                self.chart.add(float(event["loss"]), float(event["accuracy"]))
            elif event.get("event") == "done":
                self.status.setText(
                    f"Best validation accuracy {event['best_accuracy']:.3f} "
                    f"(epoch {event['best_epoch']}, {event['stopped']})"
                )
            elif "accuracy" in event and "model" in event:
                self._compare.append(event)
            return
        if line.strip():
            self.log.appendPlainText(line)

    def _finished(self, code: int, _status: object) -> None:
        if self._buffer:
            self._handle(self._buffer)
            self._buffer = ""
        step, self.process = self._step, None
        if code != 0:
            self.status.setText(f"{step} failed (exit code {code}); see the log")
        elif step == "export":
            self._evaluate()
            return
        elif step == "evaluate" and self._pending_eval:
            self._run_next_eval()
            return
        elif step == "evaluate":
            self._show_comparison()
        self._update_buttons()

    # -- actions --------------------------------------------------------------

    def generate(self) -> None:
        kind = DATA_KIND[self.task_id]
        out = self.root / "synth" / kind
        count = ["--per-format", "1000"] if kind == "strips" else ["--per-class", "500"]
        self._run("synth", ["synth", kind, str(out), *count])

    def start(self) -> None:
        data = Path(self.data.text().strip())
        if not data.exists():
            self.status.setText(f"No training data at {data}: generate synthetic data first")
            return
        self.chart.clear()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_stem = self.root / "runs" / f"{self.task_id}-{stamp}" / MODEL_NAMES[self.task_id]
        self.export_stem = None
        args = [
            "train",
            self.task_id,
            "--data",
            str(data),
            "--out",
            str(self.run_stem),
            "--epochs",
            str(self.epochs.value()),
            "--patience",
            str(self.patience.value()),
            "--device",
            self.device.currentText(),
            "--json",
        ]
        if self.val.text().strip():
            args += ["--val-data", self.val.text().strip()]
        self._run("training", args)

    def stop(self) -> None:
        if self.process is not None:
            self.process.kill()
            self.status.setText("Stopped; the best epoch so far is kept")

    def export(self) -> None:
        assert self.run_stem is not None
        self.export_stem = self.run_stem.parent / "onnx" / self.run_stem.name
        self._run("export", ["export", str(self.run_stem), str(self.export_stem)])

    def _evaluate(self) -> None:
        folder = self.val.text().strip() or self.data.text().strip()
        name = MODEL_NAMES[self.task_id]
        assert self.export_stem is not None
        self._compare = []
        self._pending_eval = [str(self.export_stem), str(self.current_models / name)]
        self._eval_folder = folder
        self._run_next_eval()

    def _run_next_eval(self) -> None:
        stem = self._pending_eval.pop(0)
        self._run("evaluate", ["evaluate", stem, "--data", self._eval_folder, "--json"])

    def _show_comparison(self) -> None:
        if len(self._compare) == 2:
            new, old = self._compare
            self.status.setText(
                f"On {self._eval_folder}: new model {new['accuracy']:.3f} "
                f"({new['correct']}/{new['total']}), current model {old['accuracy']:.3f}"
            )

    def use_model(self) -> None:
        """Copies the export into the user models folder, which then overrides the bundled one."""
        assert self.export_stem is not None
        name = MODEL_NAMES[self.task_id]
        self.user_models.mkdir(parents=True, exist_ok=True)
        for other in MODEL_NAMES.values():  # the engine needs both models in one folder
            for suffix in (".onnx", ".onnx.data", ".json"):
                target = self.user_models / f"{other}{suffix}"
                if other == name:
                    shutil.copyfile(f"{self.export_stem}{suffix}", target)
                elif not target.exists():
                    shutil.copyfile(self.current_models / f"{other}{suffix}", target)
        self.status.setText(f"Installed as {self.user_models / name}.onnx; capture reloads it")
        self.model_installed.emit(name)


def _with_browse(edit: QLineEdit, parent: QWidget) -> QWidget:
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(edit, 1)
    browse = QPushButton("…")
    browse.setFixedWidth(30)

    def pick() -> None:
        folder = QFileDialog.getExistingDirectory(parent, "Folder", edit.text())
        if folder:
            edit.setText(folder)

    browse.clicked.connect(pick)
    row.addWidget(browse)
    return holder
