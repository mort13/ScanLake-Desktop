"""Dev mode profile editing: the JSON model, user overrides, and the M6 acceptance case.

Acceptance (plan M6): add an optional column to a copy of format.json, bind a new ROI to it in
the editor model, and the column is filled in the written Parquet without code changes.
"""

from __future__ import annotations

import copy
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from scanlake_core.domain.capture import Capture, FieldValue, Identity
from scanlake_core.format import Source, parse_format
from scanlake_core.schema_map import slot_layout
from scanlake_desktop.assets import AssetSet
from scanlake_desktop.dev.profile_editor.model import EditableProfile
from scanlake_io.parquet import build_batch
from tests.conftest import ASSETS

MOLE = ASSETS / "profiles" / "mole_relative_anchors_crnn.json"


def test_round_trip_keeps_unknown_fields() -> None:
    original = json.loads(MOLE.read_text(encoding="utf-8"))
    profile = EditableProfile.load("mole", MOLE)
    assert json.loads(profile.to_json()) == original  # nothing lost or reordered semantically
    assert profile.roi("mass")["seg_mode"] == "fixed_width"  # web-only field preserved


def test_rename_and_delete_follow_the_schema() -> None:
    profile = EditableProfile.load("mole", MOLE)
    profile.update_roi("mass", name="mass2", width=50)
    assert profile.binding("mass2") == ("scans", "mass", None)
    assert profile.roi("mass2")["width"] == 50
    profile.delete_roi("qual5")
    assert profile.binding("qual5") is None
    assert "qual5" not in {r["name"] for r in profile.rois()}
    profile.hud()  # still a loadable profile


def test_add_roi_into_composition_row() -> None:
    profile = EditableProfile.load("mole", MOLE)
    profile.add_roi("purity1", "digit_crnn", (300, 234, 30, 19), "purity", row=0)
    assert profile.binding("purity1") == ("compositions", "purity", 0)
    layout = slot_layout(profile.hud())
    assert "purity" in layout.row_columns


def test_user_override_replaces_bundled_profile(tmp_path: Path) -> None:
    profile = EditableProfile.load("mole", MOLE)
    profile.update_roi("mass", ref_x=130.0)
    profile.save(tmp_path)
    ships = AssetSet(ASSETS, tmp_path).ships()
    assert ships["mole"].hud.main.roi("mass").ref_x == 130.0  # type: ignore[union-attr]
    assert AssetSet(ASSETS).ships()["mole"].hud.main.roi("mass").ref_x == 123.0  # type: ignore[union-attr]

    # An edited anchor template in the overrides folder wins as well
    (tmp_path / "anchors").mkdir()
    shutil.copy(ASSETS / "anchors" / "cargo.png", tmp_path / "anchors" / "scan.png")
    assert AssetSet(ASSETS, tmp_path).anchor_path("scan.png").parent == tmp_path / "anchors"


def test_new_column_needs_no_code_change(format_json: dict[str, Any]) -> None:
    data = copy.deepcopy(format_json)
    data["files"]["scans"]["columns"].append({"name": "signal", "type": "INT32", "required": False})
    fmt = parse_format(data)

    profile = EditableProfile.load("mole", MOLE)
    before = profile.coverage(fmt)
    assert before.binding("scans", "signal").source is Source.UNBOUND  # type: ignore[union-attr]

    profile.add_roi("signal", "digit_crnn", (123, 300, 41, 19), "signal")
    cov = profile.coverage(fmt)
    assert cov.binding("scans", "signal").rois == ("signal",)  # type: ignore[union-attr]

    layout = slot_layout(profile.hud())
    capture = Capture(
        capture_id="c",
        session_id="s",
        cluster_id="k",
        timestamp="2026-10-29T12:00:00.000Z",
        context={"system": "Stanton", "gravity_well": "Calliope"},
        scan={
            c: FieldValue.from_ocr(v, 0.9)
            for c, v in zip(
                layout.scan_columns, ["Quartz", "11826", "12", "4.58", "42.10", "1234"], strict=True
            )
        },
        rows=[
            {
                "type": FieldValue.from_ocr("Quartz", 0.9),
                "amount": FieldValue.from_ocr("100.00", 0.9),
                "quality": FieldValue.from_ocr("500", 0.9),
            }
        ],
    )
    files = build_batch(
        fmt,
        cov,
        [capture],
        Identity("u", "n", "o"),
        layout.row_columns,
        session_id="s",
        batch_number=1,
    )
    scans = next(f for f in files if f.kind == "scans")
    row = pq.read_table(io.BytesIO(scans.data)).to_pylist()[0]
    assert row["signal"] == 1234
    conf = next(f for f in files if f.kind == "confidences")
    assert "signal_conf" not in pq.read_table(io.BytesIO(conf.data)).column_names  # not in format
