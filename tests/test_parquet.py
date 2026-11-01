"""Parquet files built from the format: names, types and values, read back with pyarrow."""

from __future__ import annotations

import copy
import io
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scanlake_core.domain.capture import Capture, FieldValue, Identity
from scanlake_core.format import ParquetFormat, coverage, parse_format
from scanlake_core.profile.model import HudProfile
from scanlake_core.schema_map import slot_layout
from scanlake_io.parquet import build_batch, column_types

ROW_COLUMNS = ("type", "amount", "quality")
IDENTITY = Identity(user_id="u-1", user_name="Mort13", org="RoseCorp", client="test")


def _capture() -> Capture:
    ocr = FieldValue.from_ocr
    return Capture(
        capture_id="c-1",
        session_id="s-1",
        cluster_id="k-1",
        timestamp="2026-10-29T12:00:00.000Z",
        context={"system": "Stanton", "gravity_well": "Calliope", "region": ""},
        scan={
            "deposit": ocr("Quartz", 0.98),
            "mass": ocr("11826", 0.99),
            "resistance": ocr("12", 0.97),
            "instability": ocr("4.58", 0.95),
            "volume": ocr("42.10", 0.9).edited("42.15"),
        },
        rows=[
            {"type": ocr("Quartz", 0.99), "amount": ocr("58.40", 0.96), "quality": ocr("512", 0.9)},
            {"type": ocr("Inert Materials", 0.8), "amount": ocr("41.60", 0.92),
             "quality": ocr("0", 0.85)},
            {"type": FieldValue(), "amount": FieldValue(), "quality": FieldValue()},
        ],
    )  # fmt: skip


def _read(data: bytes) -> pa.Table:
    return pq.read_table(io.BytesIO(data))


def test_v1_batch(fmt: ParquetFormat, profiles: dict[str, HudProfile]) -> None:
    cov = coverage(fmt, slot_layout(profiles["mole"]))
    files = {
        f.kind: f
        for f in build_batch(fmt, cov, [_capture()], IDENTITY, ROW_COLUMNS, session_id="s-1",
                             batch_number=3)
    }  # fmt: skip
    assert set(files) == {"scans", "compositions", "confidences"}
    assert files["scans"].filename == "u-1_scans_s-1_batch3.parquet"

    for kind, f in files.items():
        expected = {c.name: c.type for c in fmt.files[kind].columns}
        assert column_types(f.data) == expected

    scans = _read(files["scans"].data).to_pylist()[0]
    assert scans["mass"] == 11826 and scans["volume"] == 42.15
    assert scans["region"] == "none" and scans["place"] == "none"
    assert scans["user_name"] == "Mort13"

    comps = _read(files["compositions"].data).to_pylist()
    assert [c["mat_index"] for c in comps] == [0, 1]  # the empty slot is not written
    assert comps[0]["material_volume"] == round(42.15 * 58.40 / 100, 5)
    assert comps[1]["min_confidence"] == pytest.approx(0.8)

    conf = _read(files["confidences"].data).to_pylist()[0]
    assert conf["mass_conf"] == pytest.approx(0.99)
    assert conf["volume_conf"] == 1.0  # edited by hand


def test_new_optional_column_flows_through(
    format_json: dict[str, Any], profiles: dict[str, HudProfile]
) -> None:
    data = copy.deepcopy(format_json)
    data["files"]["scans"]["columns"].append({"name": "ship", "type": "STRING", "required": False})
    fmt = parse_format(data)
    capture = _capture()
    capture.context["ship"] = "mole"
    cov = coverage(fmt, slot_layout(profiles["mole"]))
    scans = next(
        f
        for f in build_batch(fmt, cov, [capture], IDENTITY, ROW_COLUMNS, session_id="s",
                             batch_number=1)
        if f.kind == "scans"
    )  # fmt: skip
    assert _read(scans.data).to_pylist()[0]["ship"] == "mole"


def test_decimal_mass_is_written_as_double_only_when_needed(
    format_json: dict[str, Any], profiles: dict[str, HudProfile]
) -> None:
    data = copy.deepcopy(format_json)
    mass = next(c for c in data["files"]["scans"]["columns"] if c["name"] == "mass")
    mass["type"] = ["INT32", "DOUBLE"]  # deposits under 1 SCU show masses like 12.34
    fmt = parse_format(data)
    cov = coverage(fmt, slot_layout(profiles["mole"]))

    def scans(*captures: Capture) -> bytes:
        batch = build_batch(fmt, cov, list(captures), IDENTITY, ROW_COLUMNS, session_id="s",
                            batch_number=1)  # fmt: skip
        return next(f for f in batch if f.kind == "scans").data

    whole = scans(_capture())
    assert column_types(whole)["mass"] == "INT32"  # what older clients write as well
    small = _capture()
    small.capture_id = "c-2"
    small.scan["mass"] = FieldValue.from_ocr("12.34", 0.97)
    small.scan["volume"] = FieldValue.from_ocr("0.02526", 0.97)
    mixed = scans(_capture(), small)
    assert column_types(mixed)["mass"] == "DOUBLE"
    assert [r["mass"] for r in _read(mixed).to_pylist()] == [11826.0, 12.34]


def test_column_types_detects_int64() -> None:
    """pandas writes INT64 by default; the Store rejects that for INT32 columns."""
    sink = io.BytesIO()
    pq.write_table(pa.table({"mass": pa.array([1], pa.int64()), "n": pa.array(["a"])}), sink)
    assert column_types(sink.getvalue()) == {"mass": "INT64", "n": "STRING"}
