"""Turn the live votes into an editable draft (what F9 stages into the review form)."""

from __future__ import annotations

from collections.abc import Mapping

from scanlake_core.domain.capture import FieldValue, Row
from scanlake_core.schema_map import SlotLayout
from scanlake_core.voting import Vote

LOW_CONFIDENCE = 0.75


def draft_values(
    layout: SlotLayout, votes: Mapping[str, Vote]
) -> tuple[dict[str, FieldValue], list[Row]]:
    """Scan values and material rows from the votes of the profile's ROIs."""

    def value(roi: str) -> FieldValue:
        vote = votes.get(roi)
        if vote is None or vote.text == "":
            return FieldValue()
        return FieldValue.from_ocr(vote.text, vote.confidence)

    scan = {slot.column: value(slot.roi) for slot in layout.scan}
    rows = [{slot.column: value(slot.roi) for slot in row} for row in layout.rows]
    return scan, rows


def needs_attention(value: FieldValue, vote: Vote | None) -> bool:
    """OCR values worth a second look: low confidence or not stable when staged."""
    if value.source != "ocr" or value.empty:
        return False
    return (value.confidence or 0.0) < LOW_CONFIDENCE or (vote is not None and not vote.stable)
