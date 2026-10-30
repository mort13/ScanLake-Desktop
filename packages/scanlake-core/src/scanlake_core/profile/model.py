"""HUD profiles: where the anchors and ROIs of one ship's scan HUD are, and how ROIs map to data.

The JSON format is the one ScanLake Capture serves (`public/profiles/*.json`, originally from
Rock Capture CNN). It is shared with the web app, so changes to it stay additive: fields this
app does not use are kept in `extra` and written back unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from scanlake_core.geometry import Rect

DIGIT_CRNN = "digit_crnn"
WORD_CNN = "word_cnn"
RECOGNITION_MODES = (DIGIT_CRNN, WORD_CNN)

Json = dict[str, Any]


@dataclass(frozen=True)
class FilterSettings:
    brightness: int = 0
    contrast: int = 0
    threshold: int = 127
    threshold_enabled: bool = False
    grayscale: bool = True
    invert: bool = False
    channel: str = "none"  # none | red | green | blue

    @property
    def is_identity_gray(self) -> bool:
        """Only a grayscale conversion: the common case, cheap to special-case."""
        return (
            self.brightness == 0
            and self.contrast == 0
            and not self.threshold_enabled
            and not self.invert
            and self.channel == "none"
        )


@dataclass(frozen=True)
class AnchorPoint:
    name: str
    template_path: str
    match_threshold: float = 0.7
    ref_x: float = 0.0
    ref_y: float = 0.0
    search_region: Rect | None = None  # reference frame; sub-anchors only


@dataclass(frozen=True)
class RoiDef:
    name: str
    ref_x: float
    ref_y: float
    width: int
    height: int
    recognition_mode: str
    filters: FilterSettings = FilterSettings()
    format_pattern: str = ""
    sub_anchor: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class CaptureProfile:
    """One capture area of a HUD (today every ship has exactly one: `scan_results`)."""

    name: str
    search_region: Rect  # reference frame (2560x1440)
    anchors: tuple[AnchorPoint, ...]
    rois: tuple[RoiDef, ...]
    sub_anchors: tuple[AnchorPoint, ...] = ()
    monitor_index: int = 0

    def roi(self, name: str) -> RoiDef | None:
        return next((r for r in self.rois if r.name == name), None)

    def sub_anchor(self, name: str) -> AnchorPoint | None:
        return next((a for a in self.sub_anchors if a.name == name), None)


@dataclass(frozen=True)
class SchemaLeaf:
    """An output value read from one ROI. `column` names the format column it fills."""

    key: str
    roi: str
    profile: str
    column: str | None = None


@dataclass(frozen=True)
class SchemaGroup:
    key: str
    type: Literal["object", "array"]
    children: tuple[SchemaNode, ...]


SchemaNode = SchemaLeaf | SchemaGroup


@dataclass(frozen=True)
class HudProfile:
    name: str
    captures: dict[str, CaptureProfile]
    output_schema: tuple[SchemaNode, ...]
    source: Json = field(default_factory=dict, compare=False, repr=False)

    @property
    def main(self) -> CaptureProfile:
        """The capture area the pipeline runs (the first one)."""
        return next(iter(self.captures.values()))
