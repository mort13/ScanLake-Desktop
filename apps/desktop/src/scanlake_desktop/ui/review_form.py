"""The editable form a staged scan lands in (F9), built from the ship profile's columns.

Scan values on top, one line per material slot below. Values read by OCR keep their confidence
until edited; edited values count as checked by hand. Highlights: red = blocks saving,
amber = low confidence or not stable yet, blue = typed by hand.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from scanlake_core.domain.capture import FieldValue, Row
from scanlake_core.domain.tables import tables
from scanlake_core.domain.validation import Issue
from scanlake_core.fields import field_spec
from scanlake_core.schema_map import COMPOSITIONS, SCANS, SlotLayout
from scanlake_desktop.ui.style import set_state

FieldWidget = QLineEdit | QComboBox


def _vocabulary(column: str) -> Sequence[str]:
    t = tables()
    return {"deposit": t.deposits, "type": t.materials}.get(column, ())


def _label(column: str) -> str:
    return {"type": "Material", "amount": "Amount %", "instability": "Instability"}.get(
        column, column.replace("_", " ").capitalize()
    )


class ReviewForm(QWidget):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._scan_widgets: dict[str, FieldWidget] = {}
        self._row_widgets: list[dict[str, FieldWidget]] = []
        self._original: dict[tuple[int, str], FieldValue] = {}
        self._attention: set[tuple[int, str]] = set()
        self._states: dict[tuple[int, str], str] = {}  # highlight per cell, as last validated
        self._layout_key: tuple[object, ...] = ()

        self._title = QLabel("Nothing staged")
        self._title.setProperty("role", "title")
        self._scan_grid = QGridLayout()
        self._rows_grid = QGridLayout()
        self._issues = QLabel()
        self._issues.setProperty("role", "issues")
        self._issues.setWordWrap(True)

        box = QVBoxLayout(self)
        box.addWidget(self._title)
        box.addLayout(self._scan_grid)
        box.addSpacing(8)
        box.addLayout(self._rows_grid)
        box.addWidget(self._issues)
        box.addStretch(1)

    # -- building -----------------------------------------------------------

    def set_layout(self, layout: SlotLayout) -> None:
        key = (layout.scan_columns, layout.row_columns, len(layout.rows))
        if key == self._layout_key:
            return
        self._layout_key = key
        for grid in (self._scan_grid, self._rows_grid):
            while grid.count():
                item = grid.takeAt(0)
                widget = item.widget() if item is not None else None
                if widget is not None:
                    widget.deleteLater()
        self._scan_widgets, self._row_widgets = {}, []

        for i, column in enumerate(layout.scan_columns):
            widget = self._make(column)
            r, c = divmod(i, 2)
            self._scan_grid.addWidget(QLabel(_label(column)), r, c * 2)
            self._scan_grid.addWidget(widget, r, c * 2 + 1)
            self._scan_widgets[column] = widget

        for c, column in enumerate(layout.row_columns):
            header = QLabel(_label(column))
            header.setProperty("role", "hint")
            self._rows_grid.addWidget(header, 0, c + 1)
        for r in range(len(layout.rows)):
            self._rows_grid.addWidget(QLabel(f"{r + 1}"), r + 1, 0)
            widgets: dict[str, FieldWidget] = {}
            for c, column in enumerate(layout.row_columns):
                widget = self._make(column)
                self._rows_grid.addWidget(widget, r + 1, c + 1)
                widgets[column] = widget
            self._row_widgets.append(widgets)
        self._rows_grid.setColumnStretch(1, 3)

    def _make(self, column: str) -> FieldWidget:
        vocab = _vocabulary(column)
        if vocab:
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(["", *vocab])
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            combo.currentTextChanged.connect(self._edited)
            return combo
        edit = QLineEdit()
        edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        edit.textEdited.connect(self._edited)
        return edit

    # -- values ---------------------------------------------------------------

    def load(
        self,
        scan: Mapping[str, FieldValue],
        rows: Sequence[Row],
        title: str,
        attention: set[tuple[int, str]] | None = None,
    ) -> None:
        """Shows values; `attention` marks (row, column) to highlight (row -1 = scan value)."""
        self._original = {}
        self._attention = attention or set()
        for column, widget in self._scan_widgets.items():
            value = scan.get(column, FieldValue())
            self._original[(-1, column)] = value
            _set_text(widget, value.text)
        for r, widgets in enumerate(self._row_widgets):
            row = rows[r] if r < len(rows) else {}
            for column, widget in widgets.items():
                value = row.get(column, FieldValue())
                self._original[(r, column)] = value
                _set_text(widget, value.text)
        self._title.setText(title)
        self.show_issues([])

    def clear(self, keep: Mapping[str, str] | None = None) -> None:
        keep = keep or {}
        scan = {c: FieldValue(text=keep[c], source="manual") for c in keep}
        self.load(scan, [], "Nothing staged")

    def values(self) -> tuple[dict[str, FieldValue], list[Row]]:
        scan = {c: self._value(-1, c, w) for c, w in self._scan_widgets.items()}
        rows = [{c: self._value(r, c, w) for c, w in ws.items()} for r, ws in
                enumerate(self._row_widgets)]  # fmt: skip
        return scan, rows

    def _value(self, row: int, column: str, widget: FieldWidget) -> FieldValue:
        text = _text(widget).strip()
        normalized = field_spec(column).normalize(text)
        if normalized is not None:
            text = normalized
        original = self._original.get((row, column), FieldValue())
        if original.ocr_text is not None:
            return original.edited(text)
        if text == original.text:
            return original
        return FieldValue(text=text, source="manual")

    # -- highlights -----------------------------------------------------------

    def show_issues(self, issues: Sequence[Issue]) -> None:
        bad: set[tuple[int, str]] = set()
        for issue in issues:
            if issue.column is None:
                continue
            if issue.kind == SCANS:
                bad.add((-1, issue.column))
            elif issue.kind == COMPOSITIONS:
                rows = [issue.row] if issue.row is not None else range(len(self._row_widgets))
                bad.update((r, issue.column) for r in rows)
        for (row, column), widget in self._all_widgets():
            value = self._value(row, column, widget)
            if (row, column) in bad:
                state = "error"
            elif value.source == "manual" and not value.empty:
                state = "manual"
            elif (row, column) in self._attention and not value.empty:
                state = "warn"
            else:
                state = ""
            set_state(widget, state)
            self._states[(row, column)] = state
            original = self._original.get((row, column))
            if original is not None and original.confidence is not None and value.source == "ocr":
                widget.setToolTip(f"OCR confidence {original.confidence:.0%}")
            else:
                widget.setToolTip("")
        self._issues.setText("\n".join(f"• {i.message}" for i in issues))

    def cell_states(self) -> dict[tuple[int, str], tuple[FieldValue, str]]:
        """Each cell's (row, column) value and highlight: "error", "manual", "warn" or ""."""
        return {
            cell: (self._value(*cell, widget), self._states.get(cell, ""))
            for cell, widget in self._all_widgets()
        }

    def _all_widgets(self) -> list[tuple[tuple[int, str], FieldWidget]]:
        out: list[tuple[tuple[int, str], FieldWidget]] = [
            ((-1, c), w) for c, w in self._scan_widgets.items()
        ]
        for r, widgets in enumerate(self._row_widgets):
            out += [((r, c), w) for c, w in widgets.items()]
        return out

    def _edited(self, *_: object) -> None:
        self.changed.emit()


def _text(widget: FieldWidget) -> str:
    return widget.currentText() if isinstance(widget, QComboBox) else widget.text()


def _set_text(widget: FieldWidget, text: str) -> None:
    widget.blockSignals(True)
    if isinstance(widget, QComboBox):
        widget.setCurrentText(text)
    else:
        widget.setText(text)
    widget.blockSignals(False)
