"""Load the runtime assets: ship profiles, anchor templates, models, the upload format."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from scanlake_core.preprocess import Gray
from scanlake_core.profile.load import ProfileEntry, load_hud_profile, load_manifest
from scanlake_core.profile.model import HudProfile
from scanlake_core.vision.anchors import load_template

MODEL_NAMES = ("digit_crnn", "word_model2k")


def _sha256(path: Path) -> str:
    return _sha256_cached(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=16)
def _sha256_cached(path: str, _mtime: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Ship:
    key: str
    label: str
    hud: HudProfile
    templates: dict[str, Gray]


@dataclass(frozen=True)
class AssetSet:
    """The bundled assets, with optional per-user overrides (dev mode edits) on top.

    A profile or anchor file in `overrides/profiles` or `overrides/anchors` replaces the bundled
    file of the same name; everything else comes from `root`.
    """

    root: Path
    overrides: Path | None = None

    def profile_path(self, file: str) -> Path:
        return self._pick("profiles", file)

    def anchor_path(self, file: str) -> Path:
        return self._pick("anchors", file)

    def _pick(self, kind: str, file: str) -> Path:
        if self.overrides is not None and (self.overrides / kind / file).exists():
            return self.overrides / kind / file
        return self.root / kind / file

    @property
    def models_dir(self) -> Path:
        """User-installed models (dev mode) when both are complete there, else the bundled ones."""
        if self.overrides is not None:
            user = self.overrides / "models"
            needed = [f"{n}{s}" for n in MODEL_NAMES for s in (".onnx", ".onnx.data", ".json")]
            if all((user / f).exists() for f in needed):
                return user
        return self.root / "models"

    def model_stamp(self) -> str:
        """Changes when the model files change (so capture reloads its engine)."""
        d = self.models_dir
        return f"{d}:" + ",".join(
            str(int((d / f"{n}.onnx.data").stat().st_mtime_ns)) for n in MODEL_NAMES
        )

    @property
    def bundled_format(self) -> Path:
        return self.root / "format" / "format.json"

    def manifest(self) -> list[ProfileEntry]:
        return load_manifest(self.root / "profiles" / "manifest.json")

    def ships(self) -> dict[str, Ship]:
        ships: dict[str, Ship] = {}
        for entry in self.manifest():
            hud = load_hud_profile(self.profile_path(entry.file))
            capture = hud.main
            templates = {
                a.name: load_template(self.anchor_path(a.template_path))
                for a in (*capture.anchors, *capture.sub_anchors)
            }
            ships[entry.key] = Ship(entry.key, entry.label, hud, templates)
        return ships

    def word_classes(self) -> tuple[str, ...]:
        meta = json.loads((self.models_dir / "word_model2k.json").read_text(encoding="utf-8"))
        return tuple(meta["wordClasses"])

    def model_versions(self) -> str:
        """Which model files read a capture (stored with it): name@sha256-prefix of the weights."""
        if self.models_dir != self.root / "models":
            return ",".join(
                f"{n}@{_sha256(self.models_dir / f'{n}.onnx.data')[:12]}" for n in MODEL_NAMES
            )
        try:
            lock = json.loads((self.root / "assets.lock.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        hashes = {a["path"]: a.get("sha256", "") for a in lock.get("assets", [])}
        return ",".join(f"{n}@{hashes.get(f'models/{n}.onnx.data', '')[:12]}" for n in MODEL_NAMES)
