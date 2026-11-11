"""Our Parquet files and our validator against ScanLake Store's own validator.

Runs the Store's `src/services/format.ts` with Node on files this app writes plus deliberately
broken ones, and requires (a) the Store accepts our files and (b) both validators report the
same violations. Skipped when Node or a Store checkout with `node_modules` is not around, or
when that checkout predates the pinned format (it cannot parse a format with `positions`); point
`SCANLAKE_STORE_REPO` at the checkout if it is not the sibling folder.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scanlake_core.format import ParquetFormat, coverage, validate_columns
from scanlake_core.profile.model import HudProfile
from scanlake_core.schema_map import slot_layout
from scanlake_io.parquet import build_batch, column_types
from tests.conftest import ASSETS
from tests.test_parquet import IDENTITY, ROW_COLUMNS, _capture

HERE = Path(__file__).resolve().parent
STORE_REPO = Path(os.environ.get("SCANLAKE_STORE_REPO", HERE.parents[2] / "ScanLake Store"))


def _store_available() -> bool:
    return (
        shutil.which("node") is not None
        and (STORE_REPO / "node_modules" / "hyparquet" / "package.json").exists()
    )


def _store_knows_positions() -> bool:
    source = STORE_REPO / "src" / "services" / "format.ts"
    return source.exists() and "'positions'" in source.read_text(encoding="utf-8")


pytestmark = [
    pytest.mark.skipif(not _store_available(), reason="no Node or ScanLake Store checkout"),
    pytest.mark.skipif(
        not _store_knows_positions(), reason="the Store checkout predates positions"
    ),
]


def _parquet(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def test_store_agrees(tmp_path: Path, fmt: ParquetFormat, profiles: dict[str, HudProfile]) -> None:
    cov = coverage(fmt, slot_layout(profiles["mole"]))
    files: dict[Path, tuple[str, bytes]] = {}
    for f in build_batch(fmt, cov, [_capture()], IDENTITY, ROW_COLUMNS, session_id="s",
                         batch_number=1):  # fmt: skip
        files[tmp_path / f.filename] = (f.kind, f.data)

    good_scans = pq.read_table(io.BytesIO(files[tmp_path / "u-1_scans_s_batch1.parquet"][1]))
    broken = {
        "int64_mass": good_scans.set_column(
            good_scans.schema.get_field_index("mass"), "mass", pa.array([11826], pa.int64())
        ),
        "missing_volume": good_scans.drop_columns(["volume"]),
        "extra_column": good_scans.append_column("notes", pa.array(["x"])),
    }
    for name, table in broken.items():
        files[tmp_path / f"{name}.parquet"] = ("scans", _parquet(table))

    for path, (_, data) in files.items():
        path.write_bytes(data)
    args = [str(HERE / "store_check.mjs"), str(STORE_REPO), str(ASSETS / "format" / "format.json")]
    args += [f"{kind}={path}" for path, (kind, _) in files.items()]
    out = subprocess.run(["node", *args], capture_output=True, text=True, check=True, timeout=60)
    store = {Path(k): v for k, v in json.loads(out.stdout).items()}

    for path, (kind, data) in files.items():
        ours = validate_columns(column_types(data), fmt.files[kind])
        assert store[path] == ours, path.name
        if not path.name.startswith(("int64", "missing", "extra")):
            assert ours == [], f"{path.name} would be rejected: {ours}"
    assert store[tmp_path / "int64_mass.parquet"] == [
        'column "mass" is INT64, expected INT32 or DOUBLE'
    ]


FORMAT_V3 = HERE.parent / "fixtures" / "format" / "format_v3.json"


def test_store_accepts_format_v3_with_positions(
    tmp_path: Path, fmt_v3: ParquetFormat, profiles: dict[str, HudProfile]
) -> None:
    from dataclasses import replace

    from tests.test_debug_text import POSITION

    cov = coverage(fmt_v3, slot_layout(profiles["mole"]))
    capture = replace(_capture(), position=POSITION)
    files: dict[Path, str] = {}
    for f in build_batch(fmt_v3, cov, [capture], IDENTITY, ROW_COLUMNS, session_id="s",
                         batch_number=1):  # fmt: skip
        path = tmp_path / f.filename
        path.write_bytes(f.data)
        files[path] = f.kind
    assert sorted(files.values()) == ["compositions", "confidences", "positions", "scans"]

    args = [str(HERE / "store_check.mjs"), str(STORE_REPO), str(FORMAT_V3)]
    args += [f"{kind}={path}" for path, kind in files.items()]
    out = subprocess.run(["node", *args], capture_output=True, text=True, check=True, timeout=60)
    assert all(v == [] for v in json.loads(out.stdout).values()), out.stdout
