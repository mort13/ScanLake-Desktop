"""Which source fills each column of the format, and whether the app can fill them all.

Every column is filled from one of three sources:

- context: the app knows it (session, user, location, ids, time)
- ocr:     a HUD value, bound through the ship profile's output schema (`schema_map`)
- derived: computed from other values (row index, material volume, confidences, ...)

A column none of them covers is unbound. Optional unbound columns are left out of the file;
a required unbound one means this profile or app version cannot produce the format.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from scanlake_core.format.format import ParquetFormat
from scanlake_core.schema_map import COMPOSITIONS, SCANS, SlotLayout

CONFIDENCES = "confidences"
CONF_SUFFIX = "_conf"

# Columns the app fills from its own state, per file kind
CONTEXT_COLUMNS: dict[str, frozenset[str]] = {
    SCANS: frozenset(
        {
            "session_id", "user_id", "user_name", "org", "timestamp", "capture_id",
            "cluster_id", "system", "gravity_well", "region", "place",
            # Not in format v1; available as soon as the format adds them
            "ship", "client",
        }
    ),
    COMPOSITIONS: frozenset({"capture_id"}),
    CONFIDENCES: frozenset({"capture_id"}),
}  # fmt: skip

# Columns computed from other values, per file kind (see `scanlake_core.domain.rows`)
DERIVED_COLUMNS: dict[str, frozenset[str]] = {
    COMPOSITIONS: frozenset({"mat_index", "material_volume", "min_confidence"}),
}


class Source(StrEnum):
    CONTEXT = "context"
    OCR = "ocr"
    DERIVED = "derived"
    UNBOUND = "unbound"


@dataclass(frozen=True)
class ColumnBinding:
    kind: str
    column: str
    source: Source
    required: bool
    rois: tuple[str, ...] = ()  # OCR: the ROIs that fill it


@dataclass(frozen=True)
class Coverage:
    format_version: int
    bindings: tuple[ColumnBinding, ...]

    @property
    def missing_required(self) -> list[ColumnBinding]:
        return [b for b in self.bindings if b.source is Source.UNBOUND and b.required]

    @property
    def complete(self) -> bool:
        return not self.missing_required

    def binding(self, kind: str, column: str) -> ColumnBinding | None:
        return next((b for b in self.bindings if b.kind == kind and b.column == column), None)

    def written_columns(self, kind: str) -> list[str]:
        """The columns a file of `kind` will contain: everything except unbound ones."""
        return [
            b.column for b in self.bindings if b.kind == kind and b.source is not Source.UNBOUND
        ]

    def describe_missing(self) -> str:
        return ", ".join(f"{b.kind}.{b.column}" for b in self.missing_required)


def coverage(fmt: ParquetFormat, layout: SlotLayout) -> Coverage:
    """How a ship profile fills `fmt` (OCR columns come from its output schema)."""
    scan_rois: dict[str, list[str]] = {}
    for slot in layout.scan:
        scan_rois.setdefault(slot.column, []).append(slot.roi)
    row_rois: dict[str, list[str]] = {}
    for row in layout.rows:
        for slot in row:
            row_rois.setdefault(slot.column, []).append(slot.roi)
    return _coverage(fmt, scan_rois, row_rois)


def coverage_of_columns(
    fmt: ParquetFormat, scan_columns: Iterable[str], row_columns: Iterable[str]
) -> Coverage:
    """How stored captures fill `fmt` (OCR columns are the ones the captures hold values for).

    Used when building upload batches, which may mix captures from different ship profiles.
    """
    return _coverage(fmt, {c: [] for c in scan_columns}, {c: [] for c in row_columns})


def _coverage(
    fmt: ParquetFormat, scan_rois: dict[str, list[str]], row_rois: dict[str, list[str]]
) -> Coverage:
    ocr_by_kind = {SCANS: scan_rois, COMPOSITIONS: row_rois}
    bindings: list[ColumnBinding] = []
    for kind, spec in fmt.files.items():
        for col in spec.columns:
            source = Source.UNBOUND
            rois: tuple[str, ...] = ()
            if col.name in CONTEXT_COLUMNS.get(kind, ()):
                source = Source.CONTEXT
            elif col.name in DERIVED_COLUMNS.get(kind, ()):
                source = Source.DERIVED
            elif col.name in ocr_by_kind.get(kind, {}):
                source = Source.OCR
                rois = tuple(ocr_by_kind[kind][col.name])
            elif kind == CONFIDENCES and col.name.endswith(CONF_SUFFIX):
                base = col.name.removesuffix(CONF_SUFFIX)
                if base in scan_rois:
                    source = Source.DERIVED
                    rois = tuple(scan_rois[base])
            bindings.append(ColumnBinding(kind, col.name, source, col.required, rois))
    return Coverage(format_version=fmt.version, bindings=tuple(bindings))
