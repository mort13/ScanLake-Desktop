r"""Where the app keeps things.

    %APPDATA%\ScanLake\Desktop\settings.json   settings (roam with the Windows profile)
    %LOCALAPPDATA%\ScanLake\Desktop\           database, format cache, logs, snapshots

A `portable.flag` file next to the executable keeps everything in a `data` folder beside it.
"""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path

import platformdirs

APP_AUTHOR = "ScanLake"
APP_NAME = "Desktop"


def _install_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


@cache
def portable_root() -> Path | None:
    if getattr(sys, "frozen", False) and (_install_dir() / "portable.flag").exists():
        return _install_dir() / "data"
    return None


@cache
def config_dir() -> Path:
    override = os.environ.get("SCANLAKE_CONFIG_DIR")
    root = portable_root()
    if override:
        path = Path(override)
    else:
        path = root or Path(platformdirs.user_config_dir(APP_NAME, APP_AUTHOR, roaming=True))
    path.mkdir(parents=True, exist_ok=True)
    return path


@cache
def data_dir() -> Path:
    override = os.environ.get("SCANLAKE_DATA_DIR")
    root = portable_root()
    if override:
        path = Path(override)
    elif root:
        path = root
    else:
        path = Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_file() -> Path:
    return config_dir() / "settings.json"


def database_file() -> Path:
    return data_dir() / "scanlake.db"


def format_cache_file() -> Path:
    return data_dir() / "format.json"


def logs_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(exist_ok=True)
    return path


def user_assets_dir() -> Path:
    """Dev-mode edits: profiles/ and anchors/ here override the bundled files."""
    path = data_dir() / "assets"
    path.mkdir(exist_ok=True)
    return path


def training_dir() -> Path:
    """Crops collected while capturing, for labelling and training (dev mode)."""
    path = data_dir() / "training"
    path.mkdir(exist_ok=True)
    return path


def updates_dir() -> Path:
    """Downloaded asset sets (profiles, anchors, models); `active` is used from the next start."""
    path = data_dir() / "updates"
    path.mkdir(exist_ok=True)
    return path


def snapshots_dir() -> Path:
    path = data_dir() / "snapshots"
    path.mkdir(exist_ok=True)
    return path


@cache
def bundled_assets() -> Path:
    """Profiles, anchors, models and the format shipped with this build.

    Frozen builds carry them in `assets/` next to the code; from source they are the repo's
    `assets/` folder (filled by packaging/fetch_assets.py). SCANLAKE_ASSETS overrides both.
    """
    override = os.environ.get("SCANLAKE_ASSETS")
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", _install_dir())) / "assets"
    for parent in Path(__file__).resolve().parents:
        if (parent / "assets" / "assets.lock.json").exists():
            return parent / "assets"
    raise FileNotFoundError("assets folder not found; run packaging/fetch_assets.py")
