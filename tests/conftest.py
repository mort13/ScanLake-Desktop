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


@pytest.fixture(autouse=True)
def offline_services(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only `network` tests reach production: an app under test uploads closed sessions."""
    if request.node.get_closest_marker("network"):
        return
    for name in ("STORE", "AUTH", "GATEWAY"):
        monkeypatch.setenv(f"SCANLAKE_{name}_URL", f"https://{name.lower()}.invalid")


@pytest.fixture(autouse=True)
def memory_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    """Tests never touch the real credential store: it may hold the developer's real login."""
    store: dict[tuple[str, str], str] = {}
    try:
        import keyring
    except ImportError:
        return store

    def set_password(service: str, user: str, secret: str) -> None:
        store[service, user] = secret

    monkeypatch.setattr(keyring, "get_password", lambda service, user: store.get((service, user)))
    monkeypatch.setattr(keyring, "set_password", set_password)
    monkeypatch.setattr(
        keyring, "delete_password", lambda service, user: store.pop((service, user))
    )
    return store


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
