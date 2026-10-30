"""Where each ROI's value goes: from a HUD profile's `output_schema` to format columns.

Leaves under an `array` group become rows of the `compositions` file (one row per array
element); every other leaf fills a column of the `scans` file. A leaf's `column` attribute
names the column; leaves without one use their `key`, with two legacy keys of the published
profiles translated (`deposit_name` -> `deposit`, composition `name` -> `type`).
"""

from __future__ import annotations

from dataclasses import dataclass

from scanlake_core.profile.model import HudProfile, SchemaLeaf, SchemaNode

SCANS = "scans"
COMPOSITIONS = "compositions"

LEGACY_KEY_COLUMNS: dict[str, dict[str, str]] = {
    SCANS: {"deposit_name": "deposit"},
    COMPOSITIONS: {"name": "type"},
}


@dataclass(frozen=True)
class FieldSlot:
    path: str  # dotted schema path, e.g. "scan.composition.material1.amount"
    roi: str
    kind: str  # SCANS | COMPOSITIONS
    column: str
    row: int | None = None  # composition row (0-based array position), None for scans


def field_slots(profile: HudProfile) -> list[FieldSlot]:
    slots: list[FieldSlot] = []

    def leaf(node: SchemaLeaf, path: str, row: int | None) -> None:
        kind = SCANS if row is None else COMPOSITIONS
        column = node.column or LEGACY_KEY_COLUMNS[kind].get(node.key, node.key)
        slots.append(FieldSlot(path=path, roi=node.roi, kind=kind, column=column, row=row))

    def walk(nodes: tuple[SchemaNode, ...], prefix: str, row: int | None) -> None:
        for node in nodes:
            path = f"{prefix}.{node.key}" if prefix else node.key
            if isinstance(node, SchemaLeaf):
                leaf(node, path, row)
            elif node.type == "array" and row is None:
                for index, child in enumerate(node.children):
                    child_path = f"{path}.{child.key}"
                    if isinstance(child, SchemaLeaf):
                        leaf(child, child_path, index)
                    else:
                        walk(child.children, child_path, index)
            else:
                walk(node.children, path, row)

    walk(profile.output_schema, "", None)
    return slots


@dataclass(frozen=True)
class SlotLayout:
    """The slots of one profile, grouped the way the review form shows them."""

    scan: tuple[FieldSlot, ...]
    rows: tuple[tuple[FieldSlot, ...], ...]  # rows[i] = the slots of composition row i

    @property
    def scan_columns(self) -> tuple[str, ...]:
        return tuple(s.column for s in self.scan)

    @property
    def row_columns(self) -> tuple[str, ...]:
        """Composition columns in first-seen order across all rows."""
        seen: dict[str, None] = {}
        for row in self.rows:
            for slot in row:
                seen.setdefault(slot.column, None)
        return tuple(seen)


def slot_layout(profile: HudProfile) -> SlotLayout:
    slots = field_slots(profile)
    n_rows = 1 + max((s.row for s in slots if s.row is not None), default=-1)
    return SlotLayout(
        scan=tuple(s for s in slots if s.row is None),
        rows=tuple(tuple(s for s in slots if s.row == i) for i in range(n_rows)),
    )
