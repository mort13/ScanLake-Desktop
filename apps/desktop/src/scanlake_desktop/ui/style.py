"""Shared colours and the app stylesheet. Widgets mark their state with a `state` property."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

OK = "#2e9e5b"
WARN = "#d39a1c"
ERROR = "#d9534f"
MANUAL = "#3a7bd5"
MUTED = "#8a8f98"

STYLESHEET = f"""
QLineEdit[state="error"], QComboBox[state="error"] {{ border: 2px solid {ERROR}; }}
QLineEdit[state="warn"], QComboBox[state="warn"] {{ border: 2px solid {WARN}; }}
QLineEdit[state="manual"], QComboBox[state="manual"] {{ border: 1px solid {MANUAL}; }}
QLabel[role="hint"] {{ color: {MUTED}; }}
QLabel[role="issues"] {{ color: {ERROR}; }}
QLabel[role="title"] {{ font-weight: 600; font-size: 13pt; }}
"""


def set_state(widget: QWidget, state: str) -> None:
    if widget.property("state") == state:
        return
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
