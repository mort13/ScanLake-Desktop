"""Checks a capture before it is saved (port of ScanLake Capture's `ScanForm.tsx` validate).

Rules about specific columns (amount sum, mass vs volume, ...) only run when the profile
fills those columns, so a format with other data still validates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from scanlake_core.domain.capture import Capture, row_complete, row_filled
from scanlake_core.fields import field_spec
from scanlake_core.schema_map import COMPOSITIONS, SCANS

AMOUNT_TARGET = 100.0
AMOUNT_TOLERANCE_PER_ROW = 0.009
MASS_PER_VOLUME = 100
MAX_QUALITY = 1000


@dataclass(frozen=True)
class Issue:
    message: str
    kind: str = SCANS
    column: str | None = None
    row: int | None = None  # index into Capture.rows


def validate(
    capture: Capture,
    scan_columns: tuple[str, ...],
    row_columns: tuple[str, ...],
    *,
    required_scan_columns: Iterable[str] = (),
    cluster_deposit: str | None = None,
) -> list[Issue]:
    """Every problem that blocks saving `capture`; empty when it may be saved."""
    issues: list[Issue] = []

    for column in scan_columns:
        value = capture.scan.get(column)
        if value and not value.empty and field_spec(column).normalize(value.text) is None:
            issues.append(Issue(f"{_label(column)} {value.text!r} is not valid", SCANS, column))
    for index, row in enumerate(capture.rows):
        for column, value in row.items():
            if not value.empty and field_spec(column).normalize(value.text) is None:
                issues.append(
                    Issue(
                        f"{_label(column)} {value.text!r} is not valid", COMPOSITIONS, column, index
                    )
                )

    complete: list[int] = []
    for index, row in enumerate(capture.rows):
        if not row_filled(row):
            continue
        if row_complete(row, row_columns):
            complete.append(index)
        else:
            missing = [c for c in row_columns if c not in row or row[c].empty]
            issues.append(
                Issue(
                    f"Material row {index + 1} is incomplete (needs {', '.join(missing)})",
                    COMPOSITIONS,
                    missing[0] if missing else None,
                    index,
                )
            )

    if "amount" in row_columns and complete:
        total = sum(_number(capture.rows[i]["amount"].text) for i in complete)
        tolerance = AMOUNT_TOLERANCE_PER_ROW * len(complete)
        if abs(total - AMOUNT_TARGET) > tolerance:
            issues.append(
                Issue(
                    f"Material amounts sum to {total:.2f}%, expected 100% (±{tolerance:.3f}%)",
                    COMPOSITIONS,
                    "amount",
                )
            )

    if "quality" in row_columns:
        for i in complete:
            quality = _number(capture.rows[i]["quality"].text)
            if quality > MAX_QUALITY:
                issues.append(
                    Issue(
                        f"Quality cannot exceed {MAX_QUALITY} (got {quality:g})",
                        COMPOSITIONS,
                        "quality",
                        i,
                    )
                )

    mass, volume = _number(capture.value("mass")), _number(capture.value("volume"))
    if mass and volume and mass < MASS_PER_VOLUME * volume:
        issues.append(
            Issue(
                f"Mass ({mass:g}) must be at least {MASS_PER_VOLUME}× volume ({volume:g})"
                f" = {MASS_PER_VOLUME * volume:.1f}",
                SCANS,
                "mass",
            )
        )
    if "." in capture.value("mass") and volume >= 1:
        issues.append(
            Issue(
                f"Mass ({mass:g}) has decimals, which the HUD only shows for deposits under"
                " 1 SCU (volume in mSCU)",
                SCANS,
                "mass",
            )
        )

    deposit = capture.value("deposit")
    if cluster_deposit and deposit and deposit != cluster_deposit:
        issues.append(
            Issue(f'Cluster deposit must be "{cluster_deposit}", got "{deposit}"', SCANS, "deposit")
        )

    for column in required_scan_columns:
        if capture.value(column) == "":
            issues.append(Issue(f"{_label(column)} is required", SCANS, column))

    if row_columns and not complete:
        issues.append(Issue("At least one complete material row is required", COMPOSITIONS))
    return issues


def is_duplicate(
    capture: Capture, existing: Iterable[Capture], row_columns: tuple[str, ...]
) -> bool:
    """True when the session already holds a scan with identical values (web duplicate check)."""

    def signature(c: Capture) -> tuple[object, ...]:
        rows = sorted(
            tuple(row[col].text for col in row_columns) for row in c.complete_rows(row_columns)
        )
        scan = tuple(sorted((k, v.text) for k, v in c.scan.items()))
        return (c.cluster_id, c.context.get("region", ""), scan, tuple(rows))

    mine = signature(capture)
    return any(
        other.capture_id != capture.capture_id and signature(other) == mine for other in existing
    )


def _number(text: str) -> float:
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return 0.0


def _label(column: str) -> str:
    return column.replace("_", " ").capitalize()
