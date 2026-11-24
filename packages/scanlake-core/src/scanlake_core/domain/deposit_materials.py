"""Which materials each deposit type can contain, and how that steers the word model.

Both directions: the deposit narrows its material slots (`pick_material`), and the material
slots settle an unsure deposit read (`pick_deposit`).

`data/deposit_materials.json` is the same file as ScanLake Capture's
`src/data/deposit_materials.json` (whose src/ocr/DepositPrior.ts does the same in the web app);
keep both identical. Deposits it does not list are not restricted. Lookups ignore case.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cache
from importlib import resources
from typing import Any

import numpy as np

from scanlake_core.decode import Floats

# How much a material counts that the deposit is not known to contain, relative to one it is.
# At 0.01 the model has to be 100x surer of an outside material than of every listed one before
# it wins: ordinary misreads (an iron deposit's "Iron" read as "Tin") snap back to a listed
# material, while a material the table lacks can still be read when the model is certain.
OUTSIDE_DEPOSIT_WEIGHT = 0.01

# A deposit name the deposit slot gives less probability than this is never chosen because its
# materials fit: the materials only settle between names the deposit read itself finds plausible.
PLAUSIBLE_DEPOSIT = 0.02


@cache
def _raw() -> dict[str, Any]:
    return json.loads(
        resources.files("scanlake_core.domain")
        .joinpath("data/deposit_materials.json")
        .read_text("utf-8")
    )


@cache
def deposit_table() -> dict[str, tuple[str, ...]]:
    """Deposit -> the materials it can contain, as written in the file."""
    return {deposit: tuple(materials) for deposit, materials in _raw()["deposits"].items()}


@cache
def one_entry_deposits() -> tuple[str, ...]:
    """Deposits that show a single composition row, their own material, as written in the file."""
    return tuple(_raw().get("one_entry", ()))


@cache
def _by_deposit() -> dict[str, frozenset[str]]:
    return {d.lower(): frozenset(m.lower() for m in ms) for d, ms in deposit_table().items()}


@cache
def _one_entry() -> frozenset[str]:
    return frozenset(d.lower() for d in one_entry_deposits())


def allowed_materials(deposit: str) -> frozenset[str] | None:
    """The materials `deposit` can contain (lower case), or None when it is not restricted."""
    return _by_deposit().get(deposit.strip().lower())


def row_materials(deposit: str, row: int) -> frozenset[str] | None:
    """The materials composition row `row` (0 = top) of `deposit` can show: as
    `allowed_materials`, except that below the single row of a one_entry deposit nothing can
    (an empty set: blank only)."""
    key = deposit.strip().lower()
    if row > 0 and key in _one_entry():
        return frozenset()
    return _by_deposit().get(key)


def pick_material(
    probs: Floats, class_materials: Sequence[str], allowed: frozenset[str]
) -> tuple[int, float]:
    """The word class to read for a material slot of a deposit that can hold `allowed`.

    `class_materials[i]` is word class i as a material name ("" for the blank class, which is
    always allowed). Classes outside `allowed` are weighted by OUTSIDE_DEPOSIT_WEIGHT; returns the
    best class index and its renormalised probability.
    """
    p, weights, _ = _weighted(probs, class_materials, allowed)
    scores = p * weights
    best = int(scores.argmax())
    total = float(scores.sum())
    return best, float(scores[best]) / total if total > 0 else 0.0


def slot_fit(probs: Floats, class_materials: Sequence[str], allowed: frozenset[str]) -> float:
    """How well one material slot fits a deposit that can hold `allowed`, against a deposit with
    no restriction (1 = no evidence either way, as for a blank slot).

    The word model's probabilities assume every material equally likely; this is how much more
    likely the slot's reading becomes when only the deposit's materials are (OUTSIDE_DEPOSIT_WEIGHT
    for the rest). A slot read as a listed material counts for the deposit, the more so the
    shorter its list; one read as a material it cannot hold counts against it. With nothing
    allowed (below a one_entry deposit's row) any material counts against it.
    """
    p, weights, blank = _weighted(probs, class_materials, allowed)
    material = ~blank
    if not allowed:
        return float(p[blank].sum() + OUTSIDE_DEPOSIT_WEIGHT * p[material].sum())
    norm = float(weights[material].sum())
    if norm == 0:
        return 1.0
    return float(p[blank].sum() + p[material] @ weights[material] * material.sum() / norm)


def pick_deposit(
    deposit_probs: Floats,
    class_deposits: Sequence[str],
    slots: Mapping[int, Floats],
    class_materials: Sequence[str],
) -> tuple[int, float]:
    """The deposit class to read, from the deposit slot AND the material slots together.

    `class_deposits[i]` is word class i as a deposit name. The plausible names
    (>= PLAUSIBLE_DEPOSIT) share out their probability anew, each weighted by how well every
    material slot fits it (names the table does not list fit any material). The materials never
    move probability to or from the other names, so a deposit read as the only plausible name
    keeps its probability whatever the materials say. Returns the best class index and its new
    probability.

    `slots` maps composition row (0 = top) to that row's word probabilities.
    """
    q = np.asarray(deposit_probs, dtype=np.float64)[: len(class_deposits)]
    plausible = np.flatnonzero(q >= PLAUSIBLE_DEPOSIT)
    scores = q[plausible].copy()
    for k, i in enumerate(plausible):
        deposit = class_deposits[int(i)]
        for row, slot in slots.items():
            allowed = row_materials(deposit, row)
            if allowed is not None:
                scores[k] *= slot_fit(slot, class_materials, allowed)
    total = float(scores.sum())
    reweighed = q.copy()
    if total > 0:
        reweighed[plausible] = q[plausible].sum() * scores / total
    best = int(reweighed.argmax())
    return best, float(reweighed[best])


def _weighted(
    probs: Floats, class_materials: Sequence[str], allowed: frozenset[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class probabilities, their weights for `allowed`, and which classes are blank."""
    n = min(len(probs), len(class_materials))
    blank = np.array([m == "" for m in class_materials[:n]], dtype=bool)
    weights = np.array(
        [
            1.0 if m == "" or m.lower() in allowed else OUTSIDE_DEPOSIT_WEIGHT
            for m in class_materials[:n]
        ]
    )
    return np.asarray(probs, dtype=np.float64)[:n], weights, blank
