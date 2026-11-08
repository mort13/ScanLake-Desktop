"""Where the ship was when a scan was taken, from the game's debug text (r_DisplayInfo).

The game draws plain text in the top-right corner of the screen, e.g.

    CamDir: -155  45 -179 FOV: 60 Focal: 3.00 FStop: 64.0
    Zone: MISC_Prospector_915731471510 Pos: -0.01m 9.72m 0.41m
    Zone: ab_mine_pyro_regionb_sml_001 Pos: -18.8743km -3354.15m 12.6396km
    Zone: SolarSystem_906804811333 Pos: -14427194.5960km 7825291.9814km -14901.0881km
    Zone: Root Pos: -14427194.5960km 7825291.9814km -14901.0881km

One `Zone:` line per reference frame, innermost first; how many there are depends on where the
ship is. `parse_position` turns the recognised lines into a `Position` with every coordinate in
metres. Only CamDir and Zone lines are kept: the rest of the debug text (session, server, ...)
never leaves the reader.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

UNKNOWN_GLYPH = "?"  # what the text reader writes where it recognises no glyph

_NUM = r"[-+]?\d+(?:\.\d+)?"
_UNIT = r"(km|m)"
_CAM_DIR = re.compile(rf"(?:^|\s)CamDir:\s*({_NUM})\s+({_NUM})\s+({_NUM})(?=\s|$)")
# A Zone line even when its numbers are unreadable ("CamPos Planet Zone: ..." has no Pos:)
_ZONE_LINE = re.compile(r"(?:^|\s)Zone:\s+\S.*\sP[o?][s?]:")
_ZONE = re.compile(
    rf"(?:^|\s)Zone:\s+(\S(?:.*?\S)?)\s+Pos:\s*"
    rf"({_NUM}){_UNIT}\s+({_NUM}){_UNIT}\s+({_NUM}){_UNIT}(?=\s|$)"
)
_METRES = {"m": 1.0, "km": 1000.0}

# Zones that span the whole system: the zone just inside them is the "local" one (asteroid
# cluster, planet, station) that places a scan on a map.
_SYSTEM_ZONES = ("SolarSystem", "Root")


@dataclass(frozen=True)
class ZonePosition:
    index: int  # 0-based position of the Zone line in the debug text, top first
    zone: str
    x: float  # metres
    y: float
    z: float
    confidence: float = 1.0  # lowest glyph match score of the line, 0..1

    def readable(self) -> str:
        """x / y / z in km, or in m when the position is within 10 km of the zone's origin."""
        values = (self.x, self.y, self.z)
        if all(abs(v) < 10_000 for v in values):
            return " / ".join(f"{v:.1f}" for v in values) + " m"
        return " / ".join(f"{v / 1000:.2f}" for v in values) + " km"


@dataclass(frozen=True)
class Position:
    zones: tuple[ZonePosition, ...]
    cam_dir: tuple[float, float, float] | None = None

    @property
    def confidence(self) -> float:
        return min((z.confidence for z in self.zones), default=0.0)

    @property
    def local_zone(self) -> ZonePosition:
        """The zone just inside the solar system (the asteroid cluster, planet or station);
        the innermost zone when there is none."""
        local = self.zones[0]
        for zone in self.zones:
            if zone.zone.startswith(_SYSTEM_ZONES):
                break
            local = zone
        return local

    def summary(self) -> str:
        local = self.local_zone
        return f"{local.zone}: {local.readable()}"

    def details(self) -> str:
        lines = []
        if self.cam_dir is not None:
            lines.append("CamDir: " + " ".join(f"{v:g}" for v in self.cam_dir))
        lines += [f"{z.zone}: {z.readable()}" for z in self.zones]
        return "\n".join(lines)


def parse_position(lines: Iterable[str | tuple[str, float]]) -> Position | None:
    """The CamDir and Zone values in recognised debug-text lines; None without a Zone line.

    `lines` are texts or (text, score) pairs, top first. A Zone line whose numbers cannot be
    read is dropped but still counts for the index of the lines below it, so an index always
    names the same line of the game's text. An unrecognised glyph in a zone name is kept.
    Zone names are engine identifiers without spaces, so a gap in one is an underscore the
    reader saw too faintly (a 1 px line under the text) and becomes "_" again.
    """
    cam_dir: tuple[float, float, float] | None = None
    zones: list[ZonePosition] = []
    zone_lines = 0
    for line in lines:
        text, score = (line, 1.0) if isinstance(line, str) else line
        if cam_dir is None and (found := _CAM_DIR.search(text)):
            values = tuple(float(v) for v in found.groups())
            if all(math.isfinite(v) for v in values):
                cam_dir = (values[0], values[1], values[2])
            continue
        if not _ZONE_LINE.search(text):
            continue
        index = zone_lines
        zone_lines += 1
        found = _ZONE.search(text)
        if found is None:
            continue
        name, *parts = found.groups()
        name = re.sub(r"\s", "_", name)
        # 4 places: the game shows km with 4 decimals (0.1 mm); rounding drops float noise
        coords = [round(float(parts[i]) * _METRES[parts[i + 1]], 4) for i in (0, 2, 4)]
        if not all(math.isfinite(v) for v in coords):
            continue
        zones.append(ZonePosition(index, name, *coords, confidence=round(float(score), 4)))
    if not zones:
        return None
    return Position(zones=tuple(zones), cam_dir=cam_dir)
