"""Training data from real captures (dev mode): each saved scan's ROI crops, labelled.

    <training>/inbox/strips/<label>_<capture><roi>.png     digit crops, label as shown on the HUD
    <training>/inbox/words/<class>/<capture>-<roi>.png     word crops, by model class

The labeler moves reviewed crops from inbox/ to real/; real/ is the honest validation set.
The crops are the exact images the models saw (filtered for digits, raw for words).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from scanlake_core.domain.capture import Capture, FieldValue
from scanlake_core.domain.fuzzy import fuzzy_match
from scanlake_core.fields import field_spec
from scanlake_core.profile.model import DIGIT_CRNN
from scanlake_core.recognition import RoiReading
from scanlake_core.schema_map import SlotLayout
from scanlake_desktop.capture.source import write_image

DIGIT_ALPHABET = set("0123456789.%")
PERCENT_COLUMNS = {"amount", "resistance"}


def hud_text(column: str, value: str, raw_text: str) -> str:
    """The digits as the HUD shows them: the model's text if it was right, else rebuilt."""
    if raw_text and field_spec(column).normalize(raw_text) == value:
        return raw_text
    return value + ("%" if column in PERCENT_COLUMNS or raw_text.endswith("%") else "")


def word_class(value: str, classes: Sequence[str]) -> str | None:
    if not value:
        return "empty" if "empty" in classes else None
    match = fuzzy_match(value.lower().replace(" ", "_"), classes)
    return match if match in classes else None


def collect(
    capture: Capture,
    layout: SlotLayout,
    readings: Mapping[str, RoiReading],
    word_classes: Sequence[str],
    root: Path,
) -> int:
    """Writes the labelled crops of a saved capture to `root/inbox`; returns how many."""
    tag = capture.capture_id[:8]
    written = 0
    slots = [*layout.scan, *(s for row in layout.rows for s in row)]
    for slot in slots:
        reading = readings.get(slot.roi)
        if reading is None or reading.image is None:
            continue
        if slot.row is None:
            value = capture.scan.get(slot.column, FieldValue())
        else:
            row = capture.rows[slot.row] if slot.row < len(capture.rows) else {}
            value = row.get(slot.column, FieldValue())
        if reading.mode == DIGIT_CRNN:
            if value.empty:
                continue  # a blank digit field teaches nothing
            label = hud_text(slot.column, value.text, reading.raw_text)
            if not set(label) <= DIGIT_ALPHABET:
                continue
            path = root / "inbox" / "strips" / f"{label}_{tag}{slot.roi}.png"
        else:
            cls = word_class(value.text, word_classes)
            if cls is None:
                continue
            path = root / "inbox" / "words" / cls / f"{tag}-{slot.roi}.png"
        write_image(path, reading.image)
        written += 1
    return written
