"""The saved scans of the current session; double-click edits, Delete removes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

from scanlake_core.domain.capture import Capture

UPLOAD_ICON = {"uploaded": "✓ uploaded", "pending": "… pending"}


class ScansTable(QTableWidget):
    edit_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__(0, 0)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.cellDoubleClicked.connect(self._double_clicked)
        self._ids: list[str] = []

    def show_captures(
        self,
        captures: Sequence[Capture],
        scan_columns: Sequence[str],
        row_columns: Sequence[str],
        upload_state: Mapping[str, str],
        cluster_numbers: Mapping[str, int],
    ) -> None:
        headers = ["Time", "Rock", *scan_columns, "Materials", "Upload"]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels([h.replace("_", " ").capitalize() for h in headers])
        self.setRowCount(len(captures))
        self._ids = []
        for r, capture in enumerate(reversed(captures)):
            self._ids.append(capture.capture_id)
            materials = "; ".join(
                " ".join(row[c].text for c in row_columns if c in row)
                for row in capture.complete_rows(tuple(row_columns))
            )
            cells = [
                capture.timestamp[11:19],
                f"#{cluster_numbers.get(capture.cluster_id, 0)}",
                *(capture.value(c) for c in scan_columns),
                materials,
                UPLOAD_ICON.get(upload_state.get(capture.capture_id, ""), ""),
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if 1 < c < len(cells) - 2:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.setItem(r, c, item)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(headers) - 2, QHeaderView.ResizeMode.Stretch)

    def _double_clicked(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._ids):
            self.edit_requested.emit(self._ids[row])

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt override)
        row = self.currentRow()
        if event.key() == Qt.Key.Key_Delete and 0 <= row < len(self._ids):
            self.delete_requested.emit(self._ids[row])
            return
        super().keyPressEvent(event)
