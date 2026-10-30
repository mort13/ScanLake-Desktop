"""Get the current upload format: live from the Gateway, else the last cached copy, else bundled.

The live file is parsed non-strictly (unknown file kinds are skipped), so a format that grew a
kind this app does not know yet still loads; the cache only ever holds a copy that parsed.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from scanlake_core.format import FormatError, ParquetFormat, parse_format

Origin = Literal["live", "cache", "bundled"]


@dataclass(frozen=True)
class LoadedFormat:
    format: ParquetFormat
    origin: Origin
    loaded_at: float
    error: str = ""  # why the live copy was not used


def fetch_format(url: str, timeout: float = 5.0) -> dict[str, Any]:
    """The raw format JSON from `url` (an http(s) URL or a local file path)."""
    if not url.startswith(("http://", "https://")):
        return json.loads(Path(url).read_text(encoding="utf-8"))
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def load_format(
    url: str, cache_path: Path, bundled_path: Path, timeout: float = 5.0
) -> LoadedFormat:
    error = ""
    try:
        raw = fetch_format(url, timeout)
        fmt = parse_format(raw, strict=False)
        _write_atomic(cache_path, json.dumps(raw, indent=2))
        return LoadedFormat(fmt, "live", time.time())
    except (httpx.HTTPError, OSError, ValueError, FormatError) as err:
        error = f"{type(err).__name__}: {err}"

    for origin, path in (("cache", cache_path), ("bundled", bundled_path)):
        try:
            fmt = parse_format(json.loads(path.read_text(encoding="utf-8")), strict=False)
            return LoadedFormat(fmt, origin, time.time(), error)  # type: ignore[arg-type]
        except (OSError, ValueError, FormatError):
            continue
    raise FormatError(f"no usable upload format (live: {error})")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
