"""Every format column must be fillable for every published ship profile."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from scanlake_core.format import ParquetFormat, Source, coverage, parse_format
from scanlake_core.profile.load import parse_hud_profile
from scanlake_core.profile.model import HudProfile
from scanlake_core.schema_map import slot_layout


@pytest.mark.parametrize("ship", ["mole", "golem", "prospector"])
def test_v1_fully_covered(fmt: ParquetFormat, profiles: dict[str, HudProfile], ship: str) -> None:
    cov = coverage(fmt, slot_layout(profiles[ship]))
    assert cov.complete, cov.describe_missing()
    assert cov.binding("scans", "mass").source is Source.OCR  # type: ignore[union-attr]
    assert cov.binding("scans", "deposit").rois == ("deposit",)  # type: ignore[union-attr]
    assert cov.binding("scans", "user_id").source is Source.CONTEXT  # type: ignore[union-attr]
    assert cov.binding("compositions", "type").rois == tuple(f"material{i}" for i in range(1, 6))  # type: ignore[union-attr]
    assert cov.binding("compositions", "material_volume").source is Source.DERIVED  # type: ignore[union-attr]
    assert cov.binding("confidences", "volume_conf").source is Source.DERIVED  # type: ignore[union-attr]


def test_layout_matches_hud(profiles: dict[str, HudProfile]) -> None:
    layout = slot_layout(profiles["mole"])
    assert layout.scan_columns == ("deposit", "mass", "resistance", "instability", "volume")
    assert len(layout.rows) == 5
    assert layout.row_columns == ("type", "amount", "quality")


def _with_column(format_json: dict[str, Any], column: dict[str, Any]) -> ParquetFormat:
    data = copy.deepcopy(format_json)
    data["files"]["scans"]["columns"].append(column)
    return parse_format(data)


def test_optional_unbound_column_is_left_out(
    format_json: dict[str, Any], profiles: dict[str, HudProfile]
) -> None:
    fmt = _with_column(format_json, {"name": "signal", "type": "INT32", "required": False})
    cov = coverage(fmt, slot_layout(profiles["mole"]))
    assert cov.complete
    assert "signal" not in cov.written_columns("scans")


def test_required_unbound_column_blocks(
    format_json: dict[str, Any], profiles: dict[str, HudProfile]
) -> None:
    fmt = _with_column(format_json, {"name": "signal", "type": "INT32"})
    cov = coverage(fmt, slot_layout(profiles["mole"]))
    assert not cov.complete
    assert cov.describe_missing() == "scans.signal"


def test_profile_column_attribute_binds_new_column(
    format_json: dict[str, Any], profiles: dict[str, HudProfile]
) -> None:
    """Adding a ROI bound to a new column is all it takes: no code change."""
    fmt = _with_column(format_json, {"name": "signal", "type": "INT32"})
    source = copy.deepcopy(profiles["mole"].source)
    source["profiles"]["scan_results"]["rois"].append(
        {"name": "sig", "recognition_mode": "digit_crnn", "ref_x": 10, "ref_y": 10, "width": 30,
         "height": 19}
    )  # fmt: skip
    source["output_schema"][0]["children"].append(
        {"profile": "scan_results", "roi": "sig", "key": "signal_strength", "column": "signal"}
    )
    cov = coverage(fmt, slot_layout(parse_hud_profile(source)))
    assert cov.complete
    assert cov.binding("scans", "signal").rois == ("sig",)  # type: ignore[union-attr]
