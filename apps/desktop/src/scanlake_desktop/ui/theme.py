"""The amber CRT terminal theme of the ScanLake web pages, for the app's windows.

`apply_theme` sets it up once for the whole application: the two fonts of the web pages
(bundled in resources/fonts), a palette, a style sheet, the ◈ logo as the window icon and
title bars in the theme's colours. `CrtOverlay` lays the scanlines over a window.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from string import Template

from PySide6.QtCore import QEvent, QObject, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPalette,
    QPixmap,
    QRadialGradient,
    QWindow,
)
from PySide6.QtWidgets import QApplication, QWidget

from scanlake_desktop.ui import style

log = logging.getLogger(__name__)

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
FONT_FILES = ("VT323-Regular.ttf", "IBMPlexMono-Regular.ttf", "IBMPlexMono-SemiBold.ttf")
FONT_POINTS = 9.0
# favicon.svg of the web pages: ◈ on a 16 px grid
LOGO = (
    "M7 1h2v1h1v1h1v1h1v1h1v1h1v1h1v2h-1v1h-1v1h-1v1h-1v1h-1v1h-1v1H7v-1H6v-1H5v-1H4v-1H3v-1H2V9H1"
    "V7h1V6h1V5h1V4h1V3h1V2h1zM7 3h2v1h1v1h1v1h1v1h1v2h-1v1h-1v1h-1v1h-1v1H7v-1H6v-1H5v-1H4V9H3V7"
    "h1V6h1V5h1V4h1zM7 5h2v1h1v1h1v2h-1v1h-1v1H7v-1H6V9H5V7h1V6h1z"
)

STYLESHEET = Template("""
/* Windows gives menus, item views, headers and tooltips fonts of their own: one family for all */
* { font-family: "$mono"; }
QToolTip { color: $amber_hi; background-color: $ink_1; border: 1px solid $ink_4; padding: 3px 6px; }

/* ── Window furniture ─────────────────────────────────────────── */
QMenuBar { background-color: $bg; border-bottom: 1px solid $border; }
QMenuBar::item { background: transparent; color: $text_muted; padding: 4px 10px; }
QMenuBar::item:selected, QMenuBar::item:pressed { background-color: $amber; color: $bg; }
QMenu { background-color: $bg_raised; border: 1px solid $ink_4; padding: 2px; }
QMenu::item { padding: 4px 24px 4px 14px; }
QMenu::item:selected { background-color: $amber; color: $bg; }
QMenu::item:disabled { color: $ink_3; }
QMenu::separator { height: 1px; background: $border; margin: 3px 6px; }
QStatusBar { background-color: $bg; border-top: 1px solid $border; color: $text_muted; }
QStatusBar::item { border: none; }
QStatusBar QLabel { color: $text_muted; }
QToolBar { background-color: $bg; border: none; border-bottom: 1px solid $border; spacing: 4px; }
QSplitter::handle { background-color: $bg; }
QSplitter::handle:hover { background-color: $ink_3; }
QSplitter::handle:horizontal { width: 5px; }
QSplitter::handle:vertical { height: 5px; }

/* ── A terminal window: title sitting in the top border (.box) ── */
QGroupBox {
  background-color: $bg_raised; border: 1px solid $ink_4; margin-top: 13px; padding: 12px 6px 6px;
  font-family: "$display"; font-size: 15pt; text-transform: uppercase;
}
QGroupBox::title {
  subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 6px;
  background-color: $bg; color: $amber;
}
/* The title's font is the box's: give what is inside the app's font back */
QGroupBox * { font-family: "$mono"; font-size: ${points}pt; text-transform: none; }

/* ── Tabs: the pages' site nav ────────────────────────────────── */
QTabWidget::pane { border: none; border-top: 1px solid $border; }
QTabBar { font-family: "$display"; font-size: 16pt; text-transform: uppercase; }
QTabBar::tab {
  background: transparent; color: $text_muted; border: none; border-bottom: 2px solid transparent;
  padding: 1px 12px 2px; margin: 4px 4px 0 0;
}
QTabBar::tab:hover { background-color: $amber; color: $bg; }
QTabBar::tab:selected { color: $amber_hi; border-bottom: 2px solid $amber; }
QTabBar::tab:selected:hover { color: $bg; }
QTabBar::tab:disabled { color: $ink_3; }
QTabBar QToolButton { background-color: $bg; border: 1px solid $ink_3; padding: 0; }

/* ── Buttons: inverse video on hover ──────────────────────────── */
QPushButton, QToolButton {
  background-color: $ink_1; color: $amber; border: 1px solid $ink_3; border-radius: 0;
  padding: 4px 10px;
}
QPushButton:focus, QToolButton:focus { border-color: $amber_hi; }
QPushButton:default { background-color: $amber; color: $bg; border-color: $amber; }
QPushButton:hover, QToolButton:hover { background-color: $amber; color: $bg; border-color: $amber; }
QPushButton:default:hover { background-color: $amber_hi; border-color: $amber_hi; }
QPushButton:checked, QToolButton:checked {
  background-color: $amber; color: $bg; border-color: $amber;
}
QPushButton:pressed, QToolButton:pressed {
  background-color: $amber_hi; color: $bg; border-color: $amber_hi;
}
QPushButton:flat { background: transparent; color: $amber_hi; border-color: transparent; }
QPushButton:flat:hover { background-color: $amber; color: $bg; }
QPushButton:disabled, QToolButton:disabled {
  background-color: transparent; color: $ink_3; border-color: $ink_2;
}
QToolButton::menu-indicator {
  image: url("$images/arrow-down.png"); subcontrol-origin: padding;
  subcontrol-position: center right; right: 6px;
}
QToolButton::menu-indicator:hover { image: url("$images/arrow-down-inverse.png"); }
QToolButton::menu-indicator:disabled { image: url("$images/arrow-down-off.png"); }

/* ── Inputs ───────────────────────────────────────────────────── */
QLineEdit, QComboBox, QAbstractSpinBox, QPlainTextEdit, QTextEdit {
  background-color: $bg_raised; color: $amber; border: 1px solid $border; border-radius: 0;
  padding: 3px 6px; selection-background-color: $amber; selection-color: $bg;
}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
  border-color: $amber;
}
QLineEdit:read-only { color: $text_muted; }
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {
  color: $ink_3; border-color: $ink_2;
}
QComboBox QLineEdit, QAbstractSpinBox QLineEdit {
  background: transparent; border: none; padding: 0;
}
QComboBox::drop-down {
  subcontrol-origin: border; subcontrol-position: center right; width: 20px; border: none;
}
QComboBox::down-arrow { image: url("$images/arrow-down.png"); }
QComboBox::down-arrow:disabled { image: url("$images/arrow-down-off.png"); }
QComboBox QAbstractItemView {
  background-color: $ink_1; color: $amber; border: 1px solid $ink_4;
  selection-background-color: $amber; selection-color: $bg; outline: 0;
}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
  subcontrol-origin: border; width: 18px; border: none; background: transparent;
}
QAbstractSpinBox::up-button { subcontrol-position: top right; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background-color: $ink_2; }
QAbstractSpinBox::up-arrow { image: url("$images/arrow-up.png"); }
QAbstractSpinBox::down-arrow { image: url("$images/arrow-down.png"); }
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {
  image: url("$images/arrow-up-off.png");
}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {
  image: url("$images/arrow-down-off.png");
}

/* The review form's highlights; the padding keeps a wider border from moving the text */
QLineEdit[state="error"], QComboBox[state="error"] { border: 2px solid $danger; padding: 2px 5px; }
QLineEdit[state="warn"], QComboBox[state="warn"] { border: 2px dashed $amber_hi; padding: 2px 5px; }
QLineEdit[state="manual"], QComboBox[state="manual"] { border: 1px solid $cyan; }

QCheckBox, QRadioButton { spacing: 8px; }
QCheckBox::indicator, QRadioButton::indicator, QGroupBox::indicator,
QTreeView::indicator, QListView::indicator, QTableView::indicator {
  width: 11px; height: 11px; background-color: $bg_raised; border: 1px solid $ink_4;
}
QRadioButton::indicator { border-radius: 6px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover, QGroupBox::indicator:hover {
  border-color: $amber;
}
QCheckBox::indicator:checked, QGroupBox::indicator:checked, QTreeView::indicator:checked,
QListView::indicator:checked, QTableView::indicator:checked {
  image: url("$images/check.png"); border-color: $amber;
}
QRadioButton::indicator:checked { image: url("$images/radio.png"); border-color: $amber; }
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled, QGroupBox::indicator:disabled {
  border-color: $ink_2;
}
QCheckBox::indicator:checked:disabled, QGroupBox::indicator:checked:disabled {
  image: url("$images/check-off.png");
}
QRadioButton::indicator:checked:disabled { image: url("$images/radio-off.png"); }
QCheckBox:disabled, QRadioButton:disabled { color: $ink_3; }

/* ── Tables, trees, lists ─────────────────────────────────────── */
QAbstractItemView {
  background-color: $bg_raised; alternate-background-color: $ink_1; color: $amber;
  border: 1px solid $border; gridline-color: $ink_2;
  selection-background-color: $amber; selection-color: $bg; outline: 0;
}
QHeaderView { background-color: $bg; border: none; }
QHeaderView::section {
  background-color: $bg; color: $text_muted; border: none; border-bottom: 1px solid $border;
  padding: 4px 8px;
}
QTableCornerButton::section {
  background-color: $bg; border: none; border-bottom: 1px solid $border;
}

QScrollBar { background-color: $bg; border: none; margin: 0; }
QScrollBar:vertical { width: 12px; border-left: 1px solid $ink_2; }
QScrollBar:horizontal { height: 12px; border-top: 1px solid $ink_2; }
QScrollBar::handle { background-color: $ink_3; }
QScrollBar::handle:vertical { min-height: 24px; }
QScrollBar::handle:horizontal { min-width: 24px; }
QScrollBar::handle:hover, QScrollBar::handle:pressed { background-color: $amber; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
QAbstractScrollArea::corner { background-color: $bg; }

QProgressBar {
  background-color: $bg_raised; color: $amber_hi; border: 1px solid $ink_4; text-align: center;
}
QProgressBar::chunk { background-color: $ink_3; width: 8px; margin: 1px; }
QSlider::groove:horizontal { height: 2px; background-color: $ink_3; }
QSlider::handle:horizontal { width: 8px; margin: -7px 0; background-color: $amber; }
QSlider::handle:horizontal:hover { background-color: $amber_hi; }

/* ── Text roles ───────────────────────────────────────────────── */
QLabel[role="title"] { font-family: "$display"; font-size: 17pt; color: $amber_hi; }
QLabel[role="heading"] { font-family: "$mono_bold"; }
QLabel[role="readout"] { font-family: "$display"; font-size: 22pt; color: $amber_hi; }
QLabel[role="hint"] { color: $text_muted; }
QLabel[role="issues"] { color: $danger; }
""")


def stylesheet() -> str:
    return STYLESHEET.substitute(
        bg=style.BG,
        bg_raised=style.BG_RAISED,
        ink_1=style.INK_1,
        ink_2=style.INK_2,
        ink_3=style.INK_3,
        ink_4=style.INK_4,
        amber=style.AMBER,
        amber_hi=style.AMBER_HI,
        cyan=style.CYAN,
        danger=style.DANGER,
        border=style.BORDER,
        text_muted=style.TEXT_MUTED,
        display=style.DISPLAY_FONT,
        mono=style.MONO_FONT,
        mono_bold=style.MONO_BOLD,
        points=f"{FONT_POINTS:g}",
        images=(RESOURCES / "theme").as_posix(),
    )


def palette() -> QPalette:
    """The theme for everything the style sheet leaves to the style (Fusion)."""
    role = QPalette.ColorRole
    colours = {
        role.Window: style.BG,
        role.WindowText: style.AMBER,
        role.Base: style.BG_RAISED,
        role.AlternateBase: style.INK_1,
        role.Text: style.AMBER,
        role.Button: style.INK_1,
        role.ButtonText: style.AMBER,
        role.BrightText: style.AMBER_HI,
        role.Highlight: style.AMBER,
        role.HighlightedText: style.BG,
        role.Accent: style.AMBER,
        role.ToolTipBase: style.INK_1,
        role.ToolTipText: style.AMBER_HI,
        role.PlaceholderText: style.INK_4,
        role.Link: style.AMBER_HI,
        role.LinkVisited: style.TEXT_MUTED,
        role.Light: style.INK_3,
        role.Midlight: style.INK_2,
        role.Mid: style.INK_2,
        role.Dark: style.INK_2,
        role.Shadow: "#000000",
    }
    result = QPalette()
    for which, colour in colours.items():
        result.setColor(which, QColor(colour))
    disabled = QPalette.ColorGroup.Disabled
    for which in (role.WindowText, role.Text, role.ButtonText):
        result.setColor(disabled, which, QColor(style.INK_3))
    result.setColor(disabled, role.Highlight, QColor(style.INK_2))
    result.setColor(disabled, role.HighlightedText, QColor(style.INK_4))
    return result


def load_fonts() -> None:
    for name in FONT_FILES:
        if QFontDatabase.addApplicationFont(str(RESOURCES / "fonts" / name)) < 0:
            log.warning("theme font %s not loaded: the system's monospace font stands in", name)


def app_font() -> QFont:
    font = QFont([style.MONO_FONT, "Cascadia Mono", "Consolas"])
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSizeF(FONT_POINTS)
    return font


def logo_icon() -> QIcon:
    """The ◈ mark on its dark tile, kept on the pixel grid at every size."""
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    x = y = 0.0
    for command, numbers in re.findall(r"([MhvHVz])([-\d ]*)", LOGO):
        values = [float(v) for v in numbers.split()]
        if command == "M":
            x, y = values
            path.moveTo(x, y)
        elif command == "z":
            path.closeSubpath()
        else:
            if command in "hH":
                x = values[0] + (x if command == "h" else 0.0)
            else:
                y = values[0] + (y if command == "v" else 0.0)
            path.lineTo(x, y)
    icon = QIcon()
    for size in (16, 32, 48, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(style.BG))
        painter = QPainter(pixmap)
        painter.scale(size / 16, size / 16)
        painter.fillPath(path, QColor(style.AMBER_HI))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def tint_title_bar(window: QWindow | None) -> None:
    """Windows 11 draws the title bar in the theme's colours (older systems: just dark)."""
    if window is None or sys.platform != "win32":
        return
    if window.type() not in (Qt.WindowType.Window, Qt.WindowType.Dialog):
        return  # menus, tooltips, the in-game overlays
    if window.flags() & Qt.WindowType.FramelessWindowHint:
        return
    try:
        import ctypes

        set_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        # DWMWA_BORDER_COLOR, DWMWA_CAPTION_COLOR, DWMWA_TEXT_COLOR
        for attribute, colour in ((34, style.INK_3), (35, style.BG), (36, style.AMBER)):
            c = QColor(colour)
            value = ctypes.c_int(c.red() | c.green() << 8 | c.blue() << 16)  # a COLORREF
            set_attribute(
                ctypes.c_void_p(int(window.winId())),
                attribute,
                ctypes.byref(value),
                ctypes.sizeof(value),
            )
    except (AttributeError, OSError):
        log.debug("title bar not tinted", exc_info=True)


def apply_theme(app: QApplication) -> None:
    load_fonts()
    app.setStyle("Fusion")  # draws from the palette, the same on every Windows version
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    app.setPalette(palette())
    app.setFont(app_font())
    app.setStyleSheet(stylesheet())
    app.setWindowIcon(logo_icon())
    app.focusWindowChanged.connect(tint_title_bar)


class CrtOverlay(QWidget):
    """Scanlines and a vignette over a whole window (the pages' `body::after`); the mouse and
    the keyboard never notice it. Nothing flickers: the window is a tool, open for hours."""

    LINE_EVERY = 3  # px: one dark line in three
    LINE_ALPHA = 0.22
    VIGNETTE_FROM = 0.55  # of the way from the centre to a corner
    VIGNETTE_ALPHA = 0.35

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._window = window
        self._picture: QPixmap | None = None
        window.installEventFilter(self)
        self._cover()
        self.show()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.Resize, QEvent.Type.ChildAdded):
            self._cover()
        return False

    def _cover(self) -> None:
        self.setGeometry(self._window.rect())
        self.raise_()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        ratio = self.devicePixelRatioF()
        width, height = round(self.width() * ratio), round(self.height() * ratio)
        if width <= 0 or height <= 0:
            return
        picture = self._picture
        if picture is None or (picture.width(), picture.height()) != (width, height):
            picture = self._picture = self._draw(width, height, ratio)
        picture.setDevicePixelRatio(ratio)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, picture)
        painter.end()

    def _draw(self, width: int, height: int, ratio: float) -> QPixmap:
        """The overlay in device pixels, so the lines stay even on a scaled display."""
        picture = QPixmap(width, height)
        picture.fill(Qt.GlobalColor.transparent)
        painter = QPainter(picture)
        every = max(2, round(self.LINE_EVERY * ratio))
        thick = max(1, round(ratio))
        line = QColor(0, 0, 0, round(255 * self.LINE_ALPHA))
        for y in range(every - thick, height, every):
            painter.fillRect(0, y, width, thick, line)
        # An ellipse that reaches the corners: a unit circle, stretched
        painter.translate(width / 2, height / 2)
        painter.scale(width / 2 * 2**0.5, height / 2 * 2**0.5)
        vignette = QRadialGradient(0.0, 0.0, 1.0)
        vignette.setColorAt(0.0, QColor(0, 0, 0, 0))
        vignette.setColorAt(self.VIGNETTE_FROM, QColor(0, 0, 0, 0))
        vignette.setColorAt(1.0, QColor(0, 0, 0, round(255 * self.VIGNETTE_ALPHA)))
        painter.fillRect(QRectF(-1.0, -1.0, 2.0, 2.0), vignette)
        painter.end()
        return picture
