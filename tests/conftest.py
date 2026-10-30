from __future__ import annotations

import json
from pathlib import Path

import pytest

from scanlake_core.format import ParquetFormat, parse_format
from scanlake_core.profile.load import load_hud_profile, load_manifest
from scanlake_core.profile.model import HudProfile

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _require_assets() -> None:
    if not (ASSETS / "format" / "format.json").exists():
        pytest.skip("assets not fetched: run `python packaging/fetch_assets.py`")


@pytest.fixture(scope="session")
def format_json() -> dict[str, object]:
    _require_assets()
    return json.loads((ASSETS / "format" / "format.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fmt(format_json: dict[str, object]) -> ParquetFormat:
    return parse_format(format_json)


@pytest.fixture(scope="session")
def profiles() -> dict[str, HudProfile]:
    _require_assets()
    manifest = load_manifest(ASSETS / "profiles" / "manifest.json")
    return {e.key: load_hud_profile(ASSETS / "profiles" / e.file) for e in manifest}
