"""A captured scan, keyed by format column, plus the values the app adds around it."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Literal

ValueSource = Literal["ocr", "manual"]


@dataclass(frozen=True)
class FieldValue:
    """One value as stored: normalized text plus where it came from.

    `confidence` is the voted OCR confidence of `ocr_text`. Once the user types a different
    value the source becomes "manual" and the saved confidence is 1.0 (human-verified),
    matching what the web app writes.
    """

    text: str = ""
    confidence: float | None = None
    source: ValueSource = "ocr"
    ocr_text: str | None = None

    @classmethod
    def from_ocr(cls, text: str, confidence: float) -> FieldValue:
        return cls(text=text, confidence=confidence, source="ocr", ocr_text=text)

    def edited(self, text: str) -> FieldValue:
        if text == self.ocr_text and self.ocr_text is not None:
            return replace(self, text=text, source="ocr")
        return replace(self, text=text, source="manual")

    @property
    def empty(self) -> bool:
        return self.text == ""

    @property
    def saved_confidence(self) -> float:
        if self.source == "manual" or self.confidence is None:
            return 1.0
        return self.confidence


Row = dict[str, FieldValue]


@dataclass
class Capture:
    capture_id: str
    session_id: str
    cluster_id: str
    timestamp: str  # ISO 8601 UTC with milliseconds, e.g. 2026-06-24T09:01:12.345Z
    context: dict[str, str] = field(default_factory=dict)  # system, gravity_well, region, ...
    scan: dict[str, FieldValue] = field(default_factory=dict)
    rows: list[Row] = field(default_factory=list)

    def value(self, column: str) -> str:
        found = self.scan.get(column)
        return found.text if found else ""

    def complete_rows(self, row_columns: tuple[str, ...]) -> list[Row]:
        return [r for r in self.rows if row_complete(r, row_columns)]


@dataclass(frozen=True)
class Identity:
    """Who uploads: taken from the current settings when a batch is built (like the web app)."""

    user_id: str
    user_name: str
    org: str
    client: str = ""


def row_filled(row: Row) -> bool:
    return any(not v.empty for v in row.values())


def row_complete(row: Row, row_columns: tuple[str, ...]) -> bool:
    return all(column in row and not row[column].empty for column in row_columns)


def utc_timestamp(now: datetime | None = None) -> str:
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"
