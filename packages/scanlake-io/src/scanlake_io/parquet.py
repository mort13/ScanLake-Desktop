"""Format-driven Parquet files for ScanLake uploads.

Which files and columns exist and their types come from the published format; the values come
from `scanlake_core.domain.rows`. Every written file is read back and checked against the format
with the same rules as the Store, so a mismatch shows up here, not as a rejected upload.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scanlake_core.domain.capture import Capture, Identity
from scanlake_core.domain.rows import build_rows
from scanlake_core.format import Coverage, FileSpec, ParquetFormat, validate_columns

_ARROW_TYPES: dict[str, pa.DataType] = {
    "STRING": pa.string(),
    "BINARY": pa.binary(),
    "BOOLEAN": pa.bool_(),
    "INT8": pa.int8(),
    "INT16": pa.int16(),
    "INT32": pa.int32(),
    "INT64": pa.int64(),
    "UINT8": pa.uint8(),
    "UINT16": pa.uint16(),
    "UINT32": pa.uint32(),
    "UINT64": pa.uint64(),
    "FLOAT": pa.float32(),
    "DOUBLE": pa.float64(),
    "DATE": pa.date32(),
    "TIMESTAMP": pa.timestamp("ms", tz="UTC"),
}


class FormatMismatch(Exception):
    """A written file does not match the format (a bug or an unsupported column type)."""

    def __init__(self, kind: str, violations: list[str]) -> None:
        super().__init__(f"{kind}: " + "; ".join(violations))
        self.kind = kind
        self.violations = violations


def arrow_type(type_name: str) -> pa.DataType:
    try:
        return _ARROW_TYPES[type_name]
    except KeyError:
        raise NotImplementedError(f"writing {type_name} columns is not supported yet") from None


def column_types(parquet: bytes | Path) -> dict[str, str]:
    """Top-level column name -> format type name, as the Store's `columnType` reads them."""
    source = io.BytesIO(parquet) if isinstance(parquet, bytes) else parquet
    schema = pq.ParquetFile(source).metadata.schema
    leaves: dict[str, list[Any]] = {}
    for i in range(len(schema)):
        col = schema.column(i)
        leaves.setdefault(col.path.split(".")[0], []).append(col)

    types: dict[str, str] = {}
    for name, cols in leaves.items():
        col = cols[0]
        if len(cols) > 1 or "." in col.path or col.max_repetition_level > 0:
            types[name] = "NESTED"
            continue
        types[name] = _type_name(col)
    return types


def _type_name(col: Any) -> str:
    logical = json.loads(col.logical_type.to_json())
    kind = logical.get("Type", "None")
    if kind == "Int":
        return f"{'INT' if logical.get('isSigned') else 'UINT'}{logical.get('bitWidth')}"
    if kind not in ("None", None):
        return "NESTED" if kind in ("List", "Map") else str(kind).upper()
    converted = col.converted_type
    if converted and converted != "NONE":
        if converted == "UTF8":
            return "STRING"
        if converted.startswith(("INT_", "UINT_")):
            return converted.replace("_", "")
        if converted.startswith("TIMESTAMP_"):
            return "TIMESTAMP"
        if converted.startswith("TIME_"):
            return "TIME"
        return converted
    return "BINARY" if col.physical_type == "BYTE_ARRAY" else col.physical_type


def table_for(spec: FileSpec, columns: Sequence[str], rows: list[dict[str, Any]]) -> pa.Table:
    fields = [pa.field(name, arrow_type(spec.column(name).type)) for name in columns]  # type: ignore[union-attr]
    schema = pa.schema(fields)
    data = {name: [row[name] for row in rows] for name in columns}
    return pa.Table.from_pydict(data, schema=schema)


def to_parquet_bytes(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="snappy")
    return sink.getvalue()


def check(kind: str, data: bytes, spec: FileSpec) -> None:
    violations = validate_columns(column_types(data), spec)
    if violations:
        raise FormatMismatch(kind, violations)


@dataclass(frozen=True)
class BatchFile:
    kind: str
    filename: str
    data: bytes
    rows: int


def build_batch(
    fmt: ParquetFormat,
    cov: Coverage,
    captures: Sequence[Capture],
    identity: Identity,
    row_columns: tuple[str, ...],
    *,
    session_id: str,
    batch_number: int,
) -> list[BatchFile]:
    """One Parquet file per format file kind, named like the web app names them."""
    files: list[BatchFile] = []
    for kind, spec in fmt.files.items():
        columns = cov.written_columns(kind)
        rows = build_rows(spec, columns, captures, identity, row_columns)
        data = to_parquet_bytes(table_for(spec, columns, rows))
        check(kind, data, spec)
        filename = f"{identity.user_id}_{kind}_{session_id}_batch{batch_number}.parquet"
        files.append(BatchFile(kind=kind, filename=filename, data=data, rows=len(rows)))
    return files
