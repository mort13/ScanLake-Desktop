"""Asset updates (profiles, anchors, models) from the Gateway, and the app release check.

Assets: `GET {gateway}/public/desktop/manifest.json`; files the app does not already have (same
hash in the active or bundled set) come from `/public/desktop/files/<sha256>`. Everything is
verified in a staging folder, then swapped in whole as `updates/active`, which the app uses
from the next start. "Reset to bundled" deletes it.

Releases: the latest GitHub release of the app; the user is only told, nothing is installed.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from scanlake_desktop.assets import AssetSet
from scanlake_io.manifest import (
    FILES_PREFIX,
    MANIFEST_KEY,
    Manifest,
    ManifestError,
    parse_manifest,
    sha256_file,
    version_tuple,
)

log = logging.getLogger(__name__)

RELEASES_API = "https://api.github.com/repos/mort13/ScanLake-Desktop/releases/latest"
VERSION_FILE = "asset-version.txt"


@dataclass(frozen=True)
class UpdateResult:
    status: str  # "current" | "updated" | "too_old" | "failed"
    version: str = ""
    downloaded: int = 0
    message: str = ""


def active_dir(updates: Path) -> Path:
    return updates / "active"


def active_assets(updates: Path, bundled: Path) -> Path:
    """The asset folder to use: the last verified update, else the bundled assets."""
    active = active_dir(updates)
    if (active / "profiles" / "manifest.json").exists() and (active / VERSION_FILE).exists():
        return active
    return bundled


def reset(updates: Path) -> None:
    shutil.rmtree(active_dir(updates), ignore_errors=True)


def current_version(updates: Path) -> str:
    try:
        return (active_dir(updates) / VERSION_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


class AssetUpdater:
    def __init__(
        self,
        gateway: str,
        updates: Path,
        bundled: Path,
        app_version: str,
        http: httpx.Client | None = None,
    ) -> None:
        self.gateway = gateway.rstrip("/")
        self.updates = updates
        self.bundled = bundled
        self.app_version = app_version
        self.http = http or httpx.Client(timeout=30.0, follow_redirects=True)

    def check(self) -> UpdateResult:
        """Downloads and activates a newer asset set if the Gateway has one (blocking)."""
        try:
            response = self.http.get(f"{self.gateway}/{MANIFEST_KEY}", timeout=5.0)
            if response.status_code == 404:
                return UpdateResult("current", message="no asset manifest published")
            response.raise_for_status()
            manifest = parse_manifest(response.json())
        except (httpx.HTTPError, ValueError, ManifestError) as err:
            return UpdateResult("failed", message=f"cannot read the asset manifest: {err}")

        if manifest.version == current_version(self.updates):
            return UpdateResult("current", manifest.version)
        if self.app_version != "dev" and version_tuple(self.app_version) < version_tuple(
            manifest.min_app_version
        ):
            return UpdateResult(
                "too_old",
                manifest.version,
                message=f"needs ScanLake Desktop {manifest.min_app_version} or newer",
            )
        try:
            downloaded = self._install(manifest)
        except (httpx.HTTPError, OSError, ValueError) as err:
            log.warning("asset update %s failed: %s", manifest.version, err)
            return UpdateResult("failed", manifest.version, message=str(err))
        return UpdateResult("updated", manifest.version, downloaded)

    def _local_copies(self) -> dict[str, Path]:
        """sha256 -> an existing local file with that content (active set, then bundled)."""
        known: dict[str, Path] = {}
        for root in (active_dir(self.updates), self.bundled):
            for folder in ("profiles", "anchors", "models"):
                for path in (root / folder).rglob("*") if (root / folder).exists() else ():
                    if path.is_file():
                        known.setdefault(sha256_file(path), path)
        return known

    def _install(self, manifest: Manifest) -> int:
        staging = self.updates / f"staging-{uuid.uuid4().hex[:8]}"
        local = self._local_copies()
        downloaded = 0
        try:
            for f in manifest.files:
                target = staging / f.path
                target.parent.mkdir(parents=True, exist_ok=True)
                if f.sha256 in local:
                    shutil.copyfile(local[f.sha256], target)
                else:
                    self._download(f"{self.gateway}/{FILES_PREFIX}{f.sha256}", target)
                    downloaded += 1
                if target.stat().st_size != f.size or sha256_file(target) != f.sha256:
                    raise ValueError(f"{f.path} does not match the manifest")
            AssetSet(staging).ships()  # must load, or the set is not activated
            (staging / VERSION_FILE).write_text(manifest.version, encoding="utf-8")
            self._activate(staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return downloaded

    def _download(self, url: str, target: Path) -> None:
        with self.http.stream("GET", url) as response:
            response.raise_for_status()
            with target.open("wb") as fh:
                for chunk in response.iter_bytes(1 << 20):
                    fh.write(chunk)

    def _activate(self, staging: Path) -> None:
        active = active_dir(self.updates)
        old = self.updates / f"old-{uuid.uuid4().hex[:8]}"
        if active.exists():
            active.rename(old)  # Windows cannot rename over an existing folder
        staging.rename(active)
        shutil.rmtree(old, ignore_errors=True)


def latest_release(app_version: str, http: httpx.Client | None = None) -> tuple[str, str] | None:
    """(version, page URL) of a newer GitHub release, or None."""
    if app_version == "dev":
        return None
    client = http or httpx.Client(timeout=5.0, headers={"Accept": "application/vnd.github+json"})
    try:
        response = client.get(RELEASES_API)
        if response.status_code != 200:
            return None
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    tag = str(data.get("tag_name", "")).lstrip("v")
    if tag and version_tuple(tag) > version_tuple(app_version):
        return tag, str(data.get("html_url", ""))
    return None
