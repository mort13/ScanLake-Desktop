"""Asset releases and updates: manifest, the updater against a mocked Gateway, release check."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest
import respx

from scanlake_desktop import updates
from scanlake_desktop.assets import AssetSet
from scanlake_desktop.dev.profile_editor.model import EditableProfile
from scanlake_desktop.dev.publish import prepare_release
from scanlake_io.manifest import ManifestError, build_manifest, parse_manifest, sha256_file
from tests.conftest import ASSETS

GATEWAY = "https://gateway.test"


def _copy_assets(target: Path) -> Path:
    for kind in ("profiles", "anchors", "models"):
        shutil.copytree(ASSETS / kind, target / kind)
    return target


def test_manifest_round_trip(tmp_path: Path) -> None:
    source = _copy_assets(tmp_path / "set")
    manifest = build_manifest(source, "2026.10.29.1", "0.1.0")
    assert any(f.path == "models/digit_crnn.onnx.data" for f in manifest.files)
    again = parse_manifest(json.loads(manifest.to_json()))
    assert again == manifest
    with pytest.raises(ManifestError, match="unsafe"):
        parse_manifest(
            {
                "schema": 1,
                "version": "x",
                "files": [{"path": "../evil.dll", "sha256": "0" * 64, "size": 1}],
            }
        )
    with pytest.raises(ManifestError, match="sha256"):
        parse_manifest(
            {
                "schema": 1,
                "version": "x",
                "files": [{"path": "profiles/a.json", "sha256": "zz", "size": 1}],
            }
        )


def _published(tmp_path: Path, change: float = 140.0) -> tuple[Path, dict[str, object]]:
    """A release with one changed profile (mass ref_x); returns its folder and manifest JSON."""
    release = _copy_assets(tmp_path / "release")
    profile = EditableProfile.load("mole", release / "profiles" / "mole_relative_anchors_crnn.json")
    profile.update_roi("mass", ref_x=change)
    (release / "profiles" / profile.file).write_text(profile.to_json(), encoding="utf-8")
    manifest = build_manifest(release, f"v-{change}", "0.1.0")
    return release, json.loads(manifest.to_json())


def _serve(router: respx.Router, release: Path, manifest: dict[str, object]) -> None:
    router.get(f"{GATEWAY}/public/desktop/manifest.json").mock(
        return_value=httpx.Response(200, json=manifest)
    )
    for f in manifest["files"]:  # type: ignore[union-attr]
        router.get(f"{GATEWAY}/public/desktop/files/{f['sha256']}").mock(
            return_value=httpx.Response(200, content=(release / f["path"]).read_bytes())
        )


def test_update_downloads_only_changed_files(tmp_path: Path) -> None:
    release, manifest = _published(tmp_path)
    updates_dir = tmp_path / "updates"
    with respx.mock as router:
        _serve(router, release, manifest)
        updater = updates.AssetUpdater(GATEWAY, updates_dir, ASSETS, "0.1.0")
        result = updater.check()
        assert result.status == "updated" and result.downloaded == 1, result
        assert updater.check().status == "current"

    active = updates.active_assets(updates_dir, ASSETS)
    assert active == updates_dir / "active"
    assert AssetSet(active).ships()["mole"].hud.main.roi("mass").ref_x == 140.0  # type: ignore[union-attr]
    updates.reset(updates_dir)
    assert updates.active_assets(updates_dir, ASSETS) == ASSETS


def test_corrupt_download_is_not_activated(tmp_path: Path) -> None:
    release, manifest = _published(tmp_path)
    changed = next(
        f
        for f in manifest["files"]  # type: ignore[union-attr]
        if f["path"].endswith("mole_relative_anchors_crnn.json")
    )
    with respx.mock as router:
        _serve(router, release, manifest)
        router.get(f"{GATEWAY}/public/desktop/files/{changed['sha256']}").mock(
            return_value=httpx.Response(200, content=b"{}")
        )
        result = updates.AssetUpdater(GATEWAY, tmp_path / "updates", ASSETS, "0.1.0").check()
    assert result.status == "failed" and "does not match" in result.message
    assert not (tmp_path / "updates" / "active").exists()
    assert not list((tmp_path / "updates").glob("staging-*"))


def test_too_old_app_and_missing_manifest(tmp_path: Path) -> None:
    release, manifest = _published(tmp_path)
    manifest["min_app_version"] = "9.0.0"
    with respx.mock as router:
        _serve(router, release, manifest)
        result = updates.AssetUpdater(GATEWAY, tmp_path / "u", ASSETS, "0.1.0").check()
    assert result.status == "too_old"
    with respx.mock as router:
        router.get(f"{GATEWAY}/public/desktop/manifest.json").mock(return_value=httpx.Response(404))
        assert (
            updates.AssetUpdater(GATEWAY, tmp_path / "u", ASSETS, "0.1.0").check().status
            == "current"
        )


def test_second_update_replaces_the_active_set(tmp_path: Path) -> None:
    updates_dir = tmp_path / "updates"
    for value in (140.0, 150.0):
        release, manifest = _published(tmp_path / str(value), value)
        with respx.mock as router:
            _serve(router, release, manifest)
            assert (
                updates.AssetUpdater(GATEWAY, updates_dir, ASSETS, "0.1.0").check().status
                == "updated"
            )
    active = AssetSet(updates.active_assets(updates_dir, ASSETS))
    assert active.ships()["mole"].hud.main.roi("mass").ref_x == 150.0  # type: ignore[union-attr]
    assert not list(updates_dir.glob("old-*"))


def test_prepare_release_includes_dev_edits(tmp_path: Path) -> None:
    overrides = tmp_path / "user"
    profile = EditableProfile.load("mole", ASSETS / "profiles" / "mole_relative_anchors_crnn.json")
    profile.update_roi("mass", ref_x=133.0)
    profile.save(overrides)
    folder, manifest, commands = prepare_release(
        AssetSet(ASSETS, overrides), tmp_path / "publish", "2026.10.29.9", "0.1.0"
    )
    edited = folder / "profiles" / "mole_relative_anchors_crnn.json"
    entry = next(f for f in manifest.files if f.path == "profiles/mole_relative_anchors_crnn.json")
    assert entry.sha256 == sha256_file(edited)
    assert json.loads(edited.read_text(encoding="utf-8"))["profiles"]["scan_results"]["rois"]
    assert commands[-1].startswith(
        'npx wrangler r2 object put "scanlake-data/public/desktop/manifest.json"'
    )
    assert (folder / "manifest.json").exists() and (folder / "upload-commands.txt").exists()


@respx.mock
def test_release_check() -> None:
    respx.get(updates.RELEASES_API).mock(
        side_effect=[
            httpx.Response(200, json={"tag_name": "v0.2.0", "html_url": "https://x/rel"}),
            httpx.Response(200, json={"tag_name": "v0.1.0"}),
            httpx.Response(404),
        ]
    )
    assert updates.latest_release("0.1.0") == ("0.2.0", "https://x/rel")
    assert updates.latest_release("0.1.0") is None
    assert updates.latest_release("0.1.0") is None
    assert updates.latest_release("dev") is None
