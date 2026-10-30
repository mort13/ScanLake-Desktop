"""Editing a HUD profile's JSON: ROIs, anchors, search region and column bindings.

Works on the raw JSON dict, so fields this app does not know (the web app's) survive a save. A
ROI's data goes where its schema leaf points: a scan-level leaf fills a `scans` column, a leaf
inside the composition array fills that row's `compositions` column. `column` on a leaf binds
it to a format column explicitly.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from scanlake_core.format import Coverage, ParquetFormat, coverage
from scanlake_core.profile.load import parse_hud_profile
from scanlake_core.profile.model import HudProfile
from scanlake_core.schema_map import field_slots, slot_layout

Json = dict[str, Any]

ROI_DEFAULTS: Json = {
    "x_offset": 0,
    "y_offset": 0,
    "filters": {
        "brightness": 0,
        "contrast": 0,
        "threshold": 127,
        "threshold_enabled": False,
        "grayscale": True,
        "invert": False,
        "channel": "none",
    },
    "seg_mode": "projection",
    "char_width": 0,
    "char_count": 0,
    "allowed_chars": "",
    "format_pattern": "",
    "dot_width": 0,
    "enabled": True,
    "template_dir": "",
    "csv_index": 0,
}


class EditableProfile:
    def __init__(self, key: str, file: str, data: Json) -> None:
        self.key = key
        self.file = file
        self.data = copy.deepcopy(data)
        self.dirty = False

    @classmethod
    def load(cls, key: str, path: Path) -> EditableProfile:
        return cls(key, path.name, json.loads(path.read_text(encoding="utf-8")))

    # -- structure ------------------------------------------------------------

    @property
    def capture_name(self) -> str:
        return next(iter(self.data["profiles"]))

    @property
    def capture(self) -> Json:
        return self.data["profiles"][self.capture_name]

    def hud(self) -> HudProfile:
        return parse_hud_profile(self.data)

    def coverage(self, fmt: ParquetFormat) -> Coverage:
        return coverage(fmt, slot_layout(self.hud()))

    # -- ROIs -----------------------------------------------------------------

    def rois(self) -> list[Json]:
        return self.capture.setdefault("rois", [])

    def roi(self, name: str) -> Json:
        found = next((r for r in self.rois() if r["name"] == name), None)
        if found is None:
            raise KeyError(name)
        return found

    def binding(self, name: str) -> tuple[str, str, int | None] | None:
        """(file kind, column, composition row) the ROI fills, or None when no leaf uses it."""
        slot = next((s for s in field_slots(self.hud()) if s.roi == name), None)
        return (slot.kind, slot.column, slot.row) if slot else None

    def update_roi(self, name: str, /, **fields: Any) -> None:
        """Updates ROI fields; `name=` renames it (and its schema leaves)."""
        roi = self.roi(name)
        new_name = fields.pop("name", name)
        roi.update(fields)
        if new_name != name:
            if any(r["name"] == new_name for r in self.rois()):
                raise ValueError(f"a ROI named {new_name!r} exists already")
            roi["name"] = new_name
            for leaf in self._leaves():
                if leaf.get("roi") == name:
                    leaf["roi"] = new_name
        self.dirty = True

    def add_roi(
        self,
        name: str,
        mode: str,
        rect: tuple[float, float, int, int],
        column: str,
        row: int | None = None,
    ) -> None:
        """A new ROI at `rect` (reference frame), filling `column` (of row `row` if given)."""
        if any(r["name"] == name for r in self.rois()):
            raise ValueError(f"a ROI named {name!r} exists already")
        x, y, w, h = rect
        template = copy.deepcopy(self.rois()[0]) if self.rois() else copy.deepcopy(ROI_DEFAULTS)
        template.pop("sub_anchor", None)
        template.update(
            {
                "name": name,
                "recognition_mode": mode,
                "ref_x": float(x),
                "ref_y": float(y),
                "width": int(w),
                "height": int(h),
            }
        )
        self.rois().append(template)
        leaf = {"profile": self.capture_name, "roi": name, "key": column, "column": column}
        self._group_for(row)["children"].append(leaf)
        self.dirty = True

    def delete_roi(self, name: str) -> None:
        self.capture["rois"] = [r for r in self.rois() if r["name"] != name]
        self._drop_leaves(self.data.get("output_schema", []), name)
        self.dirty = True

    def set_column(self, name: str, column: str) -> None:
        for leaf in self._leaves():
            if leaf.get("roi") == name:
                leaf["column"] = column
        self.dirty = True

    # -- anchors and region ---------------------------------------------------

    def anchors(self) -> list[Json]:
        return self.capture.setdefault("anchors", [])

    def update_anchor(self, name: str, /, **fields: Any) -> None:
        anchor = next(a for a in self.anchors() if a["name"] == name)
        anchor.update(fields)
        self.dirty = True

    def search_region(self) -> tuple[int, int, int, int]:
        r = self.capture.get("search_region", {})
        return (
            int(r.get("x", 0)),
            int(r.get("y", 0)),
            int(r.get("w", r.get("width", 0))),
            int(r.get("h", r.get("height", 0))),
        )

    def set_search_region(self, x: int, y: int, w: int, h: int) -> None:
        self.capture["search_region"] = {"x": x, "y": y, "w": w, "h": h}
        self.dirty = True

    # -- saving ---------------------------------------------------------------

    def to_json(self) -> str:
        self.hud()  # refuse to write something the app could not load
        return json.dumps(self.data, indent=2, ensure_ascii=False) + "\n"

    def save(self, user_assets: Path) -> Path:
        path = user_assets / "profiles" / self.file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        self.dirty = False
        return path

    # -- schema helpers -------------------------------------------------------

    def _leaves(self) -> Iterator[Json]:
        def walk(nodes: list[Json]) -> Iterator[Json]:
            for node in nodes:
                if node.get("roi"):
                    yield node
                yield from walk(node.get("children") or [])

        yield from walk(self.data.get("output_schema", []))

    def _drop_leaves(self, nodes: list[Json], roi: str) -> None:
        nodes[:] = [n for n in nodes if n.get("roi") != roi]
        for node in nodes:
            if node.get("children"):
                self._drop_leaves(node["children"], roi)

    def _group_for(self, row: int | None) -> Json:
        """The schema group a new leaf goes into: the scan object, or composition row `row`."""
        schema: list[Json] = self.data.setdefault("output_schema", [])
        top = next(
            (n for n in schema if n.get("type", "object") == "object" and not n.get("roi")), None
        )
        if top is None:
            top = {"key": "scan", "type": "object", "children": []}
            schema.append(top)
        if row is None:
            return top
        array = next((n for n in top["children"] if n.get("type") == "array"), None)
        if array is None:
            array = {"key": "composition", "type": "array", "children": []}
            top["children"].append(array)
        while len(array["children"]) <= row:
            n = len(array["children"]) + 1
            array["children"].append({"key": f"material{n}", "type": "object", "children": []})
        return array["children"][row]
