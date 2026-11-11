"""The app's theme: the amber CRT look of the ScanLake web pages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize
from PySide6.QtGui import QFontDatabase, QPalette
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget


def test_fonts_and_glyphs_are_bundled(qtbot: Any) -> None:
    from scanlake_desktop.ui import style, theme

    theme.load_fonts()
    families = QFontDatabase.families()
    assert style.DISPLAY_FONT in families and style.MONO_FONT in families
    sheet = theme.stylesheet()
    assert "$" not in sheet  # every placeholder filled in
    images = set(re.findall(r'url\("([^"]+)"\)', sheet))
    assert images and all(Path(image).is_file() for image in images)


def test_stylesheet_parses_and_colours_by_role_and_state(qtbot: Any, qtlog: Any) -> None:
    from scanlake_desktop.ui import style, theme

    window = QWidget()
    issues, field = QLabel("Mass does not match"), QLineEdit("7932")
    issues.setProperty("role", "issues")
    box = QVBoxLayout(window)
    for widget in (issues, field, QPushButton("Save")):
        box.addWidget(widget)
    window.setStyleSheet(theme.stylesheet())
    qtbot.addWidget(window)
    window.show()
    assert not [r for r in qtlog.records if "parse" in r.message.lower()]
    assert issues.palette().color(QPalette.ColorRole.WindowText).name() == style.DANGER
    before = field.sizeHint()
    style.set_state(field, "error")
    assert field.sizeHint() == before  # the wider border does not move the form


def test_the_palette_is_the_pages_amber_on_black() -> None:
    from scanlake_desktop.ui import style, theme

    palette = theme.palette()
    assert palette.color(QPalette.ColorRole.Window).name() == style.BG == "#0a0602"
    assert palette.color(QPalette.ColorRole.Text).name() == style.AMBER == "#ffb000"
    assert palette.color(QPalette.ColorRole.Highlight).name() == style.AMBER
    disabled = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
    assert disabled.name() == style.INK_3


def test_logo_is_the_pages_favicon(qtbot: Any) -> None:
    from scanlake_desktop.ui import style, theme

    image = theme.logo_icon().pixmap(QSize(16, 16), 1.0).toImage()
    assert image.pixelColor(0, 0).name() == style.BG
    assert image.pixelColor(7, 1).name() == style.AMBER_HI  # the outline's top
    assert image.pixelColor(7, 3).name() == style.BG  # the gap
    assert image.pixelColor(7, 7).name() == style.AMBER_HI  # the solid diamond inside


def test_scanlines_cover_the_window_and_let_the_mouse_through(qtbot: Any) -> None:
    from scanlake_desktop.ui import theme

    window = QWidget()
    window.resize(300, 200)
    button = QPushButton("Save", window)
    button.setGeometry(0, 0, 300, 200)
    overlay = theme.CrtOverlay(window)
    qtbot.addWidget(window)
    window.show()
    assert overlay.geometry() == window.rect()
    window.resize(420, 260)
    qtbot.waitUntil(lambda: overlay.geometry() == window.rect())
    assert window.childAt(50, 50) is button

    picture = overlay._draw(30, 30, 1.0).toImage()  # noqa: SLF001
    assert picture.pixelColor(15, 15).alpha() == 0  # between two lines, in the middle
    assert picture.pixelColor(15, 14).alpha() == round(255 * overlay.LINE_ALPHA)
    assert picture.pixelColor(0, 0).alpha() > 60  # the vignette, in a corner
    scaled = overlay._draw(60, 60, 2.0).toImage()  # noqa: SLF001  (200 % display scaling)
    assert [scaled.pixelColor(30, y).alpha() > 0 for y in (27, 28, 29, 30)] == [
        False, True, True, False,
    ]  # fmt: skip


def test_title_bar_tint_ignores_what_has_no_title_bar() -> None:
    from scanlake_desktop.ui import theme

    theme.tint_title_bar(None)  # focus left the app
