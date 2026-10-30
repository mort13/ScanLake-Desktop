"""User settings, stored as JSON in the config folder.

Endpoints are not stored: they default to production and can be overridden per run with
SCANLAKE_STORE_URL / SCANLAKE_GATEWAY_URL / SCANLAKE_AUTH_URL / SCANLAKE_FORMAT_URL.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from scanlake_desktop import paths

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class Region(BaseModel):
    """A rectangle on a monitor, in that monitor's pixels."""

    x: int
    y: int
    w: int
    h: int


class Hotkeys(BaseModel):
    stage: str = "f9"
    save: str = "f10"
    new_cluster: str = "f11"


class Settings(BaseModel):
    schema_version: int = SCHEMA_VERSION
    # Identity (the userId is the anonymous ScanLake id; Discord login can replace it)
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_name: str = ""
    org: str = ""
    # Capture
    ship: str = "mole"
    monitor: int = 1  # mss numbering: 1 = primary monitor
    region_mode: Literal["auto", "manual"] = "auto"
    manual_region: Region | None = None
    fps: int = Field(default=10, ge=1, le=30)
    # Recognition
    backend: str = "auto"  # auto | ort-cpu | ort-dml | torch-cpu | torch-cuda
    auto_backend: str | None = None  # what `auto` resolved to (benchmarked once)
    dml_device_id: int = 0
    vote_window_s: float = Field(default=1.5, ge=0.3, le=10)
    hotkeys: Hotkeys = Field(default_factory=Hotkeys)
    beep: bool = True
    # Location the next session starts with
    system: str = "Stanton"
    gravity_well: str = ""
    region: str = ""
    # Upload
    auto_upload: bool = True
    batch_size: int = Field(default=50, ge=1, le=500)
    export_dir: str = ""
    # Modes
    dev_mode: bool = False
    collect_crops: bool = False  # dev: save ROI crops of saved scans for training


def load_settings(path: Path | None = None) -> Settings:
    path = path or paths.settings_file()
    if not path.exists():
        settings = Settings()
        save_settings(settings, path)
        return settings
    try:
        return Settings.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as err:
        backup = path.with_suffix(".invalid.json")
        path.replace(backup)
        log.warning("settings were unreadable (%s); kept a copy at %s", err, backup)
        settings = Settings()
        save_settings(settings, path)
        return settings


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or paths.settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(settings.model_dump_json(indent=2))
    os.replace(tmp, path)


def peek_backend(path: Path | None = None) -> str:
    """The configured backend, read without pydantic or Qt (for the import order at start-up)."""
    try:
        raw = json.loads((path or paths.settings_file()).read_text(encoding="utf-8"))
        return str(raw.get("backend", "auto"))
    except (OSError, ValueError):
        return "auto"


@dataclass(frozen=True)
class Endpoints:
    store: str
    gateway: str
    auth: str
    format: str

    @classmethod
    def from_env(cls) -> Endpoints:
        gateway = os.environ.get("SCANLAKE_GATEWAY_URL", "https://gateway.scanlake.rocks")
        return cls(
            store=os.environ.get("SCANLAKE_STORE_URL", "https://scanlake-store.mo-55d.workers.dev"),
            gateway=gateway.rstrip("/"),
            auth=os.environ.get("SCANLAKE_AUTH_URL", "https://auth.scanlake.rocks").rstrip("/"),
            format=os.environ.get("SCANLAKE_FORMAT_URL", gateway.rstrip("/") + "/api/format"),
        )
