"""The ScanLake upload format (format.json from the ScanLake-Format repository).

Mirrors `parseFormat` / `validateParquet` in ScanLake Store (`src/services/format.ts`): the
Store rejects every upload whose Parquet columns deviate from this file, so the desktop
app parses it with the same rules and checks its own files before uploading.

Column types are compared by name (STRING, INT32, DOUBLE, ...). Reading those names from a
Parquet file is pyarrow's job and lives in `scanlake_io.parquet`; this module stays free of
pyarrow so the core package has no I/O dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# The file kinds an upload consists of (the Store's FILE_KINDS).
UPLOAD_KINDS: tuple[str, ...] = ("scans", "compositions", "confidences")

# Type names a column spec may use (the Store's COLUMN_TYPES).
COLUMN_TYPES: frozenset[str] = frozenset(
    {
        "STRING", "BINARY", "BOOLEAN",
        "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64", "INT96",
        "FLOAT", "FLOAT16", "DOUBLE", "DECIMAL",
        "DATE", "TIME", "TIMESTAMP", "INTERVAL",
        "JSON", "BSON", "UUID", "ENUM", "FIXED_LEN_BYTE_ARRAY", "NESTED",
    }
)  # fmt: skip


class FormatError(ValueError):
    """format.json is structurally invalid (the Store would refuse to use it as well)."""


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    types: tuple[str, ...]
    required: bool = True
    description: str = ""

    @property
    def type(self) -> str:
        """The preferred type: the first listed one, used when writing."""
        return self.types[0]


@dataclass(frozen=True)
class FileSpec:
    kind: str
    columns: tuple[ColumnSpec, ...]
    allow_extra_columns: bool = False
    description: str = ""

    def column(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


@dataclass(frozen=True)
class ParquetFormat:
    version: int
    files: Mapping[str, FileSpec]
    # Kinds present in format.json that this app does not upload (strict=False only)
    ignored_kinds: tuple[str, ...] = field(default=())

    def file(self, kind: str) -> FileSpec | None:
        return self.files.get(kind)


def parse_format(data: Any, *, strict: bool = True) -> ParquetFormat:
    """Validates and parses a decoded format.json.

    With `strict=True` this rejects exactly what the Store rejects. With `strict=False`
    file kinds the app does not know are skipped instead of failing, so an older app keeps
    working when the format grows a new kind.
    """
    if not isinstance(data, dict):
        raise FormatError("format must be a JSON object")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int | float):
        raise FormatError("version must be a number")
    files_raw = data.get("files")
    if not isinstance(files_raw, dict):
        raise FormatError("files must be an object")

    files: dict[str, FileSpec] = {}
    ignored: list[str] = []
    for kind, spec in files_raw.items():
        if kind not in UPLOAD_KINDS:
            if strict:
                raise FormatError(
                    f"files.{kind}: unknown file kind (expected {', '.join(UPLOAD_KINDS)})"
                )
            ignored.append(kind)
            continue
        files[kind] = _parse_file(kind, spec)
    return ParquetFormat(version=int(version), files=files, ignored_kinds=tuple(ignored))


def _parse_file(kind: str, spec: Any) -> FileSpec:
    if not isinstance(spec, dict) or not isinstance(spec.get("columns"), list):
        raise FormatError(f"files.{kind}.columns must be an array")
    allow_extra = spec.get("allowExtraColumns", False)
    if not isinstance(allow_extra, bool):
        raise FormatError(f"files.{kind}.allowExtraColumns must be a boolean")

    columns: list[ColumnSpec] = []
    names: set[str] = set()
    for col in spec["columns"]:
        name = col.get("name") if isinstance(col, dict) else None
        if not isinstance(name, str) or name == "":
            raise FormatError(f"files.{kind}: every column needs a name")
        if name in names:
            raise FormatError(f'files.{kind}: duplicate column "{name}"')
        names.add(name)

        raw_type = col.get("type")
        types = raw_type if isinstance(raw_type, list) else [raw_type]
        if not types or not all(isinstance(t, str) and t in COLUMN_TYPES for t in types):
            raise FormatError(
                f"files.{kind}.{name}: type must be one of {', '.join(sorted(COLUMN_TYPES))}"
            )
        required = col.get("required", True)
        if not isinstance(required, bool):
            raise FormatError(f"files.{kind}.{name}: required must be a boolean")
        description = col.get("description", "")
        columns.append(
            ColumnSpec(
                name=name,
                types=tuple(str(t) for t in types),
                required=required,
                description=description if isinstance(description, str) else "",
            )
        )

    description = spec.get("description", "")
    return FileSpec(
        kind=kind,
        columns=tuple(columns),
        allow_extra_columns=allow_extra,
        description=description if isinstance(description, str) else "",
    )


def validate_columns(actual: Mapping[str, str], spec: FileSpec) -> list[str]:
    """Every way a file with columns `actual` (name -> type name) deviates from `spec`.

    Same messages as the Store's `validateParquet`, so local and server errors read alike.
    """
    violations: list[str] = []
    for col in spec.columns:
        found = actual.get(col.name)
        if found is None:
            if col.required:
                violations.append(f'missing column "{col.name}"')
            continue
        if found not in col.types:
            violations.append(f'column "{col.name}" is {found}, expected {" or ".join(col.types)}')

    if not spec.allow_extra_columns:
        known = set(spec.column_names)
        violations.extend(f'unexpected column "{name}"' for name in actual if name not in known)
    return violations
