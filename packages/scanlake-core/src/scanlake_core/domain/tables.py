"""Known deposits, materials, systems, gravity wells and regions.

`data/tables.json` is generated from ScanLake Capture's `src/data/*.ts` so both apps offer the
same values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(frozen=True)
class Tables:
    deposits: tuple[str, ...]
    materials: tuple[str, ...]
    systems: tuple[str, ...]
    gravity_wells: dict[str, tuple[str, ...]]
    regions_by_gravity_well: dict[str, tuple[str, ...]]

    def gravity_wells_of(self, system: str) -> tuple[str, ...]:
        return self.gravity_wells.get(system, ())

    def regions_of(self, gravity_well: str) -> tuple[str, ...]:
        return self.regions_by_gravity_well.get(gravity_well, ())


@cache
def tables() -> Tables:
    raw = json.loads(
        resources.files("scanlake_core.domain").joinpath("data/tables.json").read_text("utf-8")
    )
    return Tables(
        deposits=tuple(raw["deposits"]),
        materials=tuple(raw["materials"]),
        systems=tuple(raw["systems"]),
        gravity_wells={k: tuple(v) for k, v in raw["gravity_wells"].items()},
        regions_by_gravity_well={k: tuple(v) for k, v in raw["regions_by_gravity_well"].items()},
    )
