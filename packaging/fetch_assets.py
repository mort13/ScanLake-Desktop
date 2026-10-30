"""Download the assets pinned in assets/assets.lock.json and verify their sha256.

The repository never stores model weights, anchors or profiles; this script fetches the
exact versions the lock pins (the same files the web Capture app serves) into `assets/`.

    python packaging/fetch_assets.py            # fetch missing/changed files, verify all
    python packaging/fetch_assets.py --update   # re-download every URL and re-pin the lock

Standard library only, so it runs before `uv sync`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = ROOT / "assets" / "assets.lock.json"
USER_AGENT = "ScanLakeDesktop-fetch-assets/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Downloads `url` to `dest` atomically (temp file in the same folder, then replace)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                tmp.write(chunk)
    Path(tmp.name).replace(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify the pinned assets.")
    parser.add_argument("--update", action="store_true", help="re-download and re-pin every asset")
    parser.add_argument("--dest", type=Path, default=ROOT / "assets", help="target folder")
    args = parser.parse_args(argv)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []

    for asset in lock["assets"]:
        target = args.dest / asset["path"]
        pinned = asset.get("sha256", "")
        if args.update or not target.exists() or sha256_file(target) != pinned:
            print(f"fetch  {asset['path']}  <- {asset['url']}")
            download(asset["url"], target)

        actual = sha256_file(target)
        if args.update:
            asset["sha256"] = actual
            asset["size"] = target.stat().st_size
        elif actual != pinned:
            failures.append(f"{asset['path']}: sha256 {actual} != pinned {pinned}")
        else:
            print(f"ok     {asset['path']}")

    if args.update:
        LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        print(f"re-pinned {len(lock['assets'])} assets in {LOCK_PATH.relative_to(ROOT)}")

    if failures:
        print("\nThe server content no longer matches the lock:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        print("Run with --update if the change is intended.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
