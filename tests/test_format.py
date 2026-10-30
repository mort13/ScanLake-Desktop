"""format.json parsing and column validation, mirroring ScanLake Store's services/format.ts."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from scanlake_core.format import FormatError, ParquetFormat, parse_format, validate_columns
from scanlake_core.format.format import ColumnSpec, FileSpec


def test_live_v1_format_parses(fmt: ParquetFormat) -> None:
    assert fmt.version >= 1
    assert set(fmt.files) == {"scans", "compositions", "confidences"}
    scans = fmt.files["scans"]
    assert scans.column("mass") is not None and scans.column("mass").types == ("INT32",)  # type: ignore[union-attr]
    assert scans.column("deposit").type == "STRING"  # type: ignore[union-attr]
    assert not scans.allow_extra_columns


def _broken(base: dict[str, Any], mutate: Any) -> dict[str, Any]:
    data = copy.deepcopy(base)
    mutate(data)
    return data


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(version="1"), "version must be a number"),
        (lambda d: d.update(files=[]), "files must be an object"),
        (lambda d: d["files"].update(extra={"columns": []}), "unknown file kind"),
        (lambda d: d["files"]["scans"].update(columns={}), "columns must be an array"),
        (lambda d: d["files"]["scans"].update(allowExtraColumns="no"), "must be a boolean"),
        (lambda d: d["files"]["scans"]["columns"].append({"type": "STRING"}), "needs a name"),
        (
            lambda d: d["files"]["scans"]["columns"].append({"name": "mass", "type": "INT32"}),
            'duplicate column "mass"',
        ),
        (
            lambda d: d["files"]["scans"]["columns"].append({"name": "x", "type": "TEXT"}),
            "type must be one of",
        ),
        (
            lambda d: d["files"]["scans"]["columns"].append({"name": "x", "type": []}),
            "type must be one of",
        ),
        (
            lambda d: d["files"]["scans"]["columns"].append(
                {"name": "x", "type": "STRING", "required": "yes"}
            ),
            "required must be a boolean",
        ),
    ],
)
def test_rejects_what_the_store_rejects(
    format_json: dict[str, Any], mutate: Any, message: str
) -> None:
    with pytest.raises(FormatError, match=message):
        parse_format(_broken(format_json, mutate))


def test_non_strict_skips_unknown_kinds(format_json: dict[str, Any]) -> None:
    data = _broken(format_json, lambda d: d["files"].update(signals={"columns": []}))
    parsed = parse_format(data, strict=False)
    assert parsed.ignored_kinds == ("signals",)
    assert "signals" not in parsed.files


def test_optional_and_multi_type_columns(format_json: dict[str, Any]) -> None:
    data = _broken(
        format_json,
        lambda d: d["files"]["scans"]["columns"].extend(
            [
                {"name": "note", "type": "STRING", "required": False},
                {"name": "count", "type": ["INT32", "INT64"]},
            ]
        ),
    )
    scans = parse_format(data).files["scans"]
    assert scans.column("note").required is False  # type: ignore[union-attr]
    assert scans.column("count").types == ("INT32", "INT64")  # type: ignore[union-attr]


SPEC = FileSpec(
    kind="scans",
    columns=(
        ColumnSpec("mass", ("INT32",)),
        ColumnSpec("volume", ("DOUBLE",)),
        ColumnSpec("note", ("STRING",), required=False),
        ColumnSpec("count", ("INT32", "INT64")),
    ),
)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"mass": "INT32", "volume": "DOUBLE", "count": "INT64"}, []),
        ({"mass": "INT32", "volume": "DOUBLE", "note": "STRING", "count": "INT32"}, []),
        (
            {"mass": "INT64", "volume": "DOUBLE", "count": "INT32"},
            ['column "mass" is INT64, expected INT32'],
        ),
        ({"mass": "INT32", "count": "INT32"}, ['missing column "volume"']),
        (
            {"mass": "INT32", "volume": "DOUBLE", "count": "INT32", "notes": "STRING"},
            ['unexpected column "notes"'],
        ),
        (
            {"mass": "INT32", "volume": "DOUBLE", "count": "DOUBLE"},
            ['column "count" is DOUBLE, expected INT32 or INT64'],
        ),
    ],
)
def test_validate_columns(actual: dict[str, str], expected: list[str]) -> None:
    assert validate_columns(actual, SPEC) == expected


def test_extra_columns_allowed_when_the_format_says_so() -> None:
    spec = FileSpec("scans", (ColumnSpec("mass", ("INT32",)),), allow_extra_columns=True)
    assert validate_columns({"mass": "INT32", "anything": "STRING"}, spec) == []
