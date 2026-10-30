"""What a valid value looks like for each format column, for OCR text and typed input alike.

The CRNN emits strings like `11826`, `12%`, `0.42`, `42.09%`. A FieldSpec turns such a string
into the normalized text that is stored and later cast to the column's Parquet type, or rejects
it. This replaces the per-ROI `format_pattern` of the profiles, which only `amount1` used.

Columns without a dedicated spec (new auxiliary data) get a generic one from their format type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scanlake_core.domain.fuzzy import fuzzy_match
from scanlake_core.domain.tables import tables

INTEGER_TYPES = frozenset(
    {"INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64"}
)
FLOAT_TYPES = frozenset({"FLOAT", "FLOAT16", "DOUBLE", "DECIMAL"})


@dataclass(frozen=True)
class FieldSpec:
    """Validation and normalization of one column's text value."""

    column: str
    kind: str  # "int" | "decimal" | "word" | "text"
    pattern: re.Pattern[str] | None = None  # full match on the cleaned text
    minimum: float | None = None
    maximum: float | None = None
    decimals: int | None = None  # decimal: format to this many places
    vocabulary: tuple[str, ...] = ()  # word: snap to these names

    def normalize(self, text: str) -> str | None:
        """The normalized text, "" for empty input, or None when `text` is not a valid value."""
        cleaned = text.strip().replace(",", ".").replace(" ", "")
        if cleaned == "":
            return ""
        if self.kind == "word":
            if cleaned.lower() in ("empty", "none") and self.column != "deposit":
                return ""
            return fuzzy_match(text.strip(), self.vocabulary) if self.vocabulary else text.strip()
        if self.kind == "text":
            return text.strip()

        cleaned = cleaned.removesuffix("%")
        if self.pattern is not None and not self.pattern.fullmatch(cleaned):
            return None
        try:
            number = float(cleaned)
        except ValueError:
            return None
        if self.kind == "int" and not number.is_integer():
            return None
        if self.minimum is not None and number < self.minimum:
            return None
        if self.maximum is not None and number > self.maximum:
            return None
        if self.kind == "int":
            return str(int(number))
        if self.decimals is not None:
            return f"{number:.{self.decimals}f}"
        return cleaned


_INT = re.compile(r"\d{1,9}")
_DECIMAL = re.compile(r"\d{1,9}(\.\d{1,4})?")


def _specs() -> dict[str, FieldSpec]:
    t = tables()
    return {
        "deposit": FieldSpec("deposit", "word", vocabulary=t.deposits),
        "type": FieldSpec("type", "word", vocabulary=t.materials),
        "mass": FieldSpec("mass", "int", re.compile(r"\d{1,7}"), minimum=1),
        "resistance": FieldSpec("resistance", "int", re.compile(r"\d{1,3}"), 0, 100),
        "instability": FieldSpec("instability", "decimal", re.compile(r"\d{1,4}(\.\d{1,2})?"), 0),
        "volume": FieldSpec("volume", "decimal", re.compile(r"\d{1,6}(\.\d{1,2})?"), 0),
        "amount": FieldSpec("amount", "decimal", re.compile(r"\d{1,3}(\.\d{1,2})?"), 0, 100, 2),
        "quality": FieldSpec("quality", "int", re.compile(r"\d{1,4}"), 0, 1000),
    }


_SPECS: dict[str, FieldSpec] | None = None


def field_spec(column: str, column_type: str = "STRING", recognition: str = "") -> FieldSpec:
    """The spec for `column`; unknown columns get a generic one from type and model."""
    global _SPECS
    if _SPECS is None:
        _SPECS = _specs()
    spec = _SPECS.get(column)
    if spec is not None:
        return spec
    if column_type in INTEGER_TYPES:
        return FieldSpec(column, "int", _INT)
    if column_type in FLOAT_TYPES:
        return FieldSpec(column, "decimal", _DECIMAL)
    return FieldSpec(column, "word" if recognition == "word_cnn" else "text")
