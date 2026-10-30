"""The desktop asset manifest: which profiles, anchors and models the current release uses.

Published on the Gateway as `public/desktop/manifest.json`; every file listed is stored once,
content-addressed, at `public/desktop/files/<sha256>`, so files never change under a URL and
only the manifest has to be re-read. The app downloads what it does not have yet, verifies
each hash, and switches to the new set as a whole.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = 1
MANIFEST_KEY = "public/desktop/manifest.json"
FILES_PREFIX = "public/desktop/files/"
ASSET_DIRS = ("profiles", "anchors", "models")


@dataclass(frozen=True)
class ManifestFile:
    path: str  # relative, forward slashes, e.g. "models/digit_crnn.onnx"
    sha256: str
    size: int


@dataclass(frozen=True)
class Manifest:
    version: str
    min_app_version: str
    files: tuple[ManifestFile, ...]
    created: str = ""

    def to_json(self) -> str:
        data = {
            "schema": SCHEMA,
            "version": self.version,
            "min_app_version": self.min_app_version,
            "created": self.created,
            "files": [{"path": f.path, "sha256": f.sha256, "size": f.size} for f in self.files],
        }
        return json.dumps(data, indent=2) + "\n"


class ManifestError(ValueError):
    pass


def parse_manifest(data: Any) -> Manifest:
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ManifestError("not a ScanLake desktop manifest (schema 1)")
    files = []
    for item in data.get("files", []):
        path = str(item["path"])
        if path.startswith(("/", "\\")) or ".." in Path(path).parts or ":" in path:
            raise ManifestError(f"unsafe path in manifest: {path!r}")
        sha = str(item["sha256"]).lower()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ManifestError(f"bad sha256 for {path}")
        files.append(ManifestFile(path, sha, int(item["size"])))
    return Manifest(
        version=str(data["version"]),
        min_app_version=str(data.get("min_app_version", "0")),
        files=tuple(files),
        created=str(data.get("created", "")),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(source: Path, version: str, min_app_version: str = "0") -> Manifest:
    """The manifest of an asset folder laid out like `assets/` (profiles/, anchors/, models/)."""
    files = []
    for folder in ASSET_DIRS:
        for path in sorted((source / folder).rglob("*")):
            if path.is_file():
                rel = path.relative_to(source).as_posix()
                files.append(ManifestFile(rel, sha256_file(path), path.stat().st_size))
    if not any(f.path == "profiles/manifest.json" for f in files):
        raise ManifestError(f"{source} has no profiles/manifest.json")
    return Manifest(
        version, min_app_version, tuple(files), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


def upload_commands(manifest: Manifest, source: Path, bucket: str = "scanlake-data") -> list[str]:
    """wrangler commands that publish the files and then the manifest (run them by hand)."""
    seen: set[str] = set()
    commands = []
    for f in manifest.files:
        if f.sha256 in seen:
            continue
        seen.add(f.sha256)
        commands.append(
            f'npx wrangler r2 object put "{bucket}/{FILES_PREFIX}{f.sha256}" '
            f'--file "{source / f.path}" --remote'
        )
    commands.append(
        f'npx wrangler r2 object put "{bucket}/{MANIFEST_KEY}" --file manifest.json '
        "--content-type application/json --remote"
    )
    return commands


def version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.replace("-", ".").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)
