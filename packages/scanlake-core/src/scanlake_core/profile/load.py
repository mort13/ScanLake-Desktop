"""Parse HUD profile JSON (and the profile manifest) into `scanlake_core.profile.model`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scanlake_core.geometry import Rect
from scanlake_core.profile.model import (
    RECOGNITION_MODES,
    AnchorPoint,
    CaptureProfile,
    FilterSettings,
    HudProfile,
    RoiDef,
    SchemaGroup,
    SchemaLeaf,
    SchemaNode,
)


class ProfileError(ValueError):
    """The profile JSON is unusable."""


@dataclass(frozen=True)
class ProfileEntry:
    """One ship in `profiles/manifest.json`."""

    key: str
    label: str
    file: str


def load_manifest(path: Path) -> list[ProfileEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ProfileEntry(key=e["key"], label=e["label"], file=e["file"]) for e in data]


def load_hud_profile(path: Path) -> HudProfile:
    return parse_hud_profile(json.loads(path.read_text(encoding="utf-8")))


def parse_hud_profile(data: dict[str, Any]) -> HudProfile:
    captures_raw = data.get("profiles")
    if not isinstance(captures_raw, dict) or not captures_raw:
        raise ProfileError("profile needs a non-empty 'profiles' object")
    captures = {name: _capture(name, raw) for name, raw in captures_raw.items()}
    schema = tuple(_node(n) for n in data.get("output_schema", []) if n.get("key"))
    return HudProfile(
        name=str(data.get("name", "")), captures=captures, output_schema=schema, source=data
    )


def _rect(raw: dict[str, Any]) -> Rect:
    # Profiles use w/h; older files and the web types also allow width/height
    w = raw.get("w", raw.get("width", 0))
    h = raw.get("h", raw.get("height", 0))
    return Rect(round(raw.get("x", 0)), round(raw.get("y", 0)), round(w), round(h))


def _anchor(raw: dict[str, Any]) -> AnchorPoint:
    region = raw.get("search_region")
    return AnchorPoint(
        name=raw["name"],
        template_path=raw.get("template_path", ""),
        match_threshold=float(raw.get("match_threshold", 0.7)),
        ref_x=float(raw.get("ref_x", 0.0)),
        ref_y=float(raw.get("ref_y", 0.0)),
        search_region=_rect(region) if isinstance(region, dict) else None,
    )


def _filters(raw: dict[str, Any]) -> FilterSettings:
    return FilterSettings(
        brightness=int(raw.get("brightness", 0)),
        contrast=int(raw.get("contrast", 0)),
        threshold=int(raw.get("threshold", 127)),
        threshold_enabled=bool(raw.get("threshold_enabled", False)),
        grayscale=bool(raw.get("grayscale", True)),
        invert=bool(raw.get("invert", False)),
        channel=str(raw.get("channel", "none")),
    )


def _roi(raw: dict[str, Any]) -> RoiDef:
    mode = raw.get("recognition_mode", "")
    if mode not in RECOGNITION_MODES:
        raise ProfileError(
            f"ROI {raw.get('name')!r}: recognition_mode {mode!r} is not supported "
            f"(expected one of {', '.join(RECOGNITION_MODES)})"
        )
    return RoiDef(
        name=raw["name"],
        ref_x=float(raw.get("ref_x", 0.0)),
        ref_y=float(raw.get("ref_y", 0.0)),
        width=int(raw.get("width", 0)),
        height=int(raw.get("height", 0)),
        recognition_mode=mode,
        filters=_filters(raw.get("filters", {})),
        format_pattern=str(raw.get("format_pattern", "")),
        sub_anchor=str(raw.get("sub_anchor", "") or ""),
        enabled=bool(raw.get("enabled", True)),
    )


def _capture(name: str, raw: dict[str, Any]) -> CaptureProfile:
    anchors = tuple(_anchor(a) for a in raw.get("anchors", []))
    if len(anchors) < 2:
        raise ProfileError(f"capture profile {name!r} needs at least 2 anchors")
    return CaptureProfile(
        name=name,
        search_region=_rect(raw.get("search_region", {})),
        anchors=anchors,
        rois=tuple(_roi(r) for r in raw.get("rois", []) if r.get("enabled", True)),
        sub_anchors=tuple(_anchor(a) for a in raw.get("sub_anchors", [])),
        monitor_index=int(raw.get("monitor_index", 0)),
    )


def _node(raw: dict[str, Any]) -> SchemaNode:
    if raw.get("roi"):
        column = raw.get("column")
        return SchemaLeaf(
            key=raw["key"],
            roi=raw["roi"],
            profile=raw.get("profile", ""),
            column=column if isinstance(column, str) and column else None,
        )
    kind = "array" if raw.get("type") == "array" else "object"
    children = tuple(_node(c) for c in raw.get("children", []) if c.get("key"))
    return SchemaGroup(key=raw["key"], type=kind, children=children)
