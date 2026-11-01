"""Turn captures into the rows of each format file, typed per column.

The format decides which files and columns exist; `bindings` decides which source fills each
column. This module computes the values: context from the capture and the uploader identity,
OCR values from the capture, derived values from other values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from scanlake_core.domain.capture import Capture, Identity, Row
from scanlake_core.fields import FLOAT_TYPES, INTEGER_TYPES
from scanlake_core.format.bindings import CONF_SUFFIX, CONFIDENCES
from scanlake_core.format.format import ColumnSpec, FileSpec
from scanlake_core.schema_map import COMPOSITIONS, SCANS

Value = str | int | float | bool | None


class CoercionError(ValueError):
    """A stored text value cannot be cast to its column type."""


def coerce(text: str, column: ColumnSpec) -> Value:
    """Casts stored text to the Python value for the column's (first) type.

    A decimal in an integer column that also allows a float type (mass: INT32 or DOUBLE) stays
    a float; `write_type` then writes that batch's column as the float type.
    """
    if text == "":
        return None
    kind = column.type
    try:
        if kind in INTEGER_TYPES:
            number = float(text.replace(",", "."))
            if not number.is_integer() and any(t in FLOAT_TYPES for t in column.types):
                return number
            return round(number)  # the format has no float type for it: the nearest integer
        if kind in FLOAT_TYPES:
            return float(text.replace(",", "."))
    except ValueError as err:
        raise CoercionError(f'"{text}" is not a valid {column.name}') from err
    if kind == "BOOLEAN":
        return text.strip().lower() in ("1", "true", "yes")
    return text


def write_type(column: ColumnSpec, values: Iterable[Value]) -> str:
    """The type to write a batch's column as: the first listed one, unless the batch holds a
    decimal that only a later-listed float type can store."""
    if any(isinstance(v, float) and not v.is_integer() for v in values):
        return next((t for t in column.types if t in FLOAT_TYPES), column.type)
    return column.type


_SCAN_CONTEXT: dict[str, Callable[[Capture, Identity], str]] = {
    "session_id": lambda c, i: c.session_id,
    "user_id": lambda c, i: i.user_id,
    "user_name": lambda c, i: i.user_name,
    "org": lambda c, i: i.org,
    "timestamp": lambda c, i: c.timestamp,
    "capture_id": lambda c, i: c.capture_id,
    "cluster_id": lambda c, i: c.cluster_id,
    "system": lambda c, i: c.context.get("system", ""),
    "gravity_well": lambda c, i: c.context.get("gravity_well", ""),
    "region": lambda c, i: c.context.get("region") or "none",
    "place": lambda c, i: c.context.get("place") or "none",
    "ship": lambda c, i: c.context.get("ship", ""),
    "client": lambda c, i: i.client,
}


def _material_volume(capture: Capture, row: Row) -> float | None:
    volume, amount = capture.value("volume"), row.get("amount")
    if not volume or amount is None or amount.empty:
        return None
    # 5 places, not 2: deposits under 1 SCU (milli-SCU on the HUD) would round to 0.01 or 0
    return round(float(volume) * float(amount.text) / 100, 5)


def _min_confidence(row: Row) -> float:
    confidences = [v.saved_confidence for v in row.values() if not v.empty]
    return min(confidences, default=1.0)


def build_rows(
    spec: FileSpec,
    columns: Sequence[str],
    captures: Iterable[Capture],
    identity: Identity,
    row_columns: tuple[str, ...],
) -> list[dict[str, Value]]:
    """The rows of file `spec.kind` for `captures`, restricted to `columns`.

    `row_columns` are the composition columns a material row needs to count as complete;
    incomplete rows are not written (the review form refuses to save them anyway).
    """
    specs = {c.name: c for c in spec.columns}
    out: list[dict[str, Value]] = []
    for capture in captures:
        if spec.kind == COMPOSITIONS:
            for index, row in enumerate(capture.complete_rows(row_columns)):
                out.append({n: _composition(n, specs[n], capture, row, index) for n in columns})
        else:
            out.append(
                {n: _per_capture(spec.kind, n, specs[n], capture, identity) for n in columns}
            )
    return out


def _per_capture(
    kind: str, name: str, column: ColumnSpec, capture: Capture, identity: Identity
) -> Value:
    if kind == SCANS and name in _SCAN_CONTEXT:
        return coerce(_SCAN_CONTEXT[name](capture, identity), column)
    if kind == CONFIDENCES:
        if name == "capture_id":
            return capture.capture_id
        base = name.removesuffix(CONF_SUFFIX)
        if base in capture.scan:
            return capture.scan[base].saved_confidence
        return None
    found = capture.scan.get(name)
    return coerce(found.text, column) if found else None


def _composition(name: str, column: ColumnSpec, capture: Capture, row: Row, index: int) -> Any:
    if name == "capture_id":
        return capture.capture_id
    if name == "mat_index":
        return index
    if name == "material_volume":
        return _material_volume(capture, row)
    if name == "min_confidence":
        return _min_confidence(row)
    found = row.get(name)
    return coerce(found.text, column) if found else None
