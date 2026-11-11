"""Shared colours. Widgets mark their state with a `state` property (see ui/theme.py).

The app's windows use the amber CRT palette of the ScanLake web pages (their theme.css).
What is drawn over the game keeps its own traffic-light colours (`HUD_*`): amber on the
game's own yellow and orange HUD text would be hard to tell apart.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

# The amber ramp, darkest to brightest (theme.css: --bg, --ink-1 … --amber-hi)
BG = "#0a0602"
BG_RAISED = "#120b03"
INK_1 = "#1f1405"
INK_2 = "#3d2808"
INK_3 = "#6b450c"
INK_4 = "#a86a10"
AMBER = "#ffb000"
AMBER_HI = "#ffd98a"
CYAN = "#b8f0ff"  # the one contrast colour
DANGER = "#ff4b2b"  # failures only
BORDER = "#4d3209"
TEXT_MUTED = "#d4921c"
TEXT_SUBTLE = INK_4
TINT = "rgba(255, 176, 0, 0.12)"  # a faint amber wash behind what is selected

# What the app's widgets say with colour: brightness instead of hue, as on the web pages
OK = AMBER_HI
WARN = AMBER
ERROR = DANGER
MANUAL = CYAN
MUTED = TEXT_SUBTLE

# The in-game overlays (values beside the HUD, the compass)
HUD_OK = "#2e9e5b"
HUD_WARN = "#d39a1c"
HUD_ERROR = "#d9534f"
HUD_MANUAL = "#3a7bd5"
HUD_MUTED = "#8a8f98"
HUD_TEXT = "#f0f0f0"
HUD_FONT = "Segoe UI"  # not the app's font: the overlays' sizes are measured for this one

DISPLAY_FONT = "VT323"  # headings, tabs
MONO_FONT = "IBM Plex Mono"
MONO_BOLD = "IBM Plex Mono SemiBold"


def set_state(widget: QWidget, state: str) -> None:
    if widget.property("state") == state:
        return
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
