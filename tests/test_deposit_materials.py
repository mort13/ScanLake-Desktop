"""Which materials a deposit can hold; the recognizer reads deposit and materials with it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scanlake_core.domain.deposit_materials import (
    allowed_materials,
    deposit_table,
    one_entry_deposits,
    pick_deposit,
    pick_material,
    row_materials,
    slot_fit,
)
from scanlake_core.domain.tables import tables
from scanlake_core.geometry import Rect
from scanlake_core.profile.model import WORD_CNN, HudProfile
from scanlake_core.recognition import FrameRecognizer, RoiReading

CLASSES = (
    "bexalite", "borase", "empty", "gold", "hadanite", "hephaestanite", "inert_materials", "iron",
    "laranite", "silicon", "taranite", "tin", "tungsten",
)  # fmt: skip
MATERIALS = tuple(
    "" if c == "empty" else "Inert Materials" if c == "inert_materials" else c.title()
    for c in CLASSES
)
DEPOSITS = tuple(c.title() for c in CLASSES)  # "Empty", "Inert_Materials": not in the table
CAPTURE_COPY = (
    Path(__file__).resolve().parents[2]
    / "ScanLake Capture"
    / "src"
    / "data"
    / "deposit_materials.json"
)


def _probs(**given: float) -> np.ndarray:
    """Class probabilities over CLASSES; classes not given get a tiny share."""
    return np.array([given.get(c, 1e-4) for c in CLASSES])


def test_table_names_known_deposits_and_materials() -> None:
    t = tables()
    for deposit, materials in deposit_table().items():
        assert deposit in t.deposits
        assert set(materials) <= set(t.materials), deposit
    for deposit in one_entry_deposits():
        assert deposit_table()[deposit] == (deposit,)  # a single row of its own material


@pytest.mark.skipif(
    not CAPTURE_COPY.exists(), reason="ScanLake Capture checkout not next to this one"
)
def test_table_matches_capture() -> None:
    ours = json.loads(
        (Path(__file__).resolve().parents[1] / "packages/scanlake-core/src/scanlake_core/domain"
         / "data/deposit_materials.json").read_text("utf-8")
    )  # fmt: skip
    assert ours == json.loads(CAPTURE_COPY.read_text("utf-8"))


def test_allowed_materials_ignores_case_and_leaves_unknown_deposits_free() -> None:
    assert allowed_materials("iron") == {"iron", "inert materials"}
    assert allowed_materials(" BEXALITE ") == {"bexalite", "borase", "gold", "inert materials"}
    assert allowed_materials("None") is None
    assert allowed_materials("") is None


def test_row_materials_allow_nothing_below_a_one_entry_deposit() -> None:
    assert row_materials("hadanite", 0) == {"hadanite"}
    assert row_materials("Hadanite", 1) == frozenset()
    assert row_materials("Iron", 3) == allowed_materials("Iron")
    assert row_materials("None", 1) is None


def test_pick_material() -> None:
    iron = allowed_materials("Iron")
    assert iron is not None
    best, p = pick_material(_probs(tin=0.7, iron=0.25), MATERIALS, iron)
    assert MATERIALS[best] == "Iron" and p > 0.95  # misread snaps back
    best, _ = pick_material(_probs(empty=0.9, tin=0.08), MATERIALS, iron)
    assert MATERIALS[best] == ""  # blank stays blank
    best, _ = pick_material(_probs(tin=0.999), MATERIALS, iron)
    assert MATERIALS[best] == "Tin"  # near-certain outside evidence still wins
    bexalite = allowed_materials("Bexalite")
    assert bexalite is not None
    best, p = pick_material(_probs(gold=0.55, bexalite=0.4), MATERIALS, bexalite)
    assert MATERIALS[best] == "Gold" and p == pytest.approx(
        0.55 / 0.95, abs=1e-3
    )  # no false certainty


def test_slot_fit() -> None:
    taranite = allowed_materials("Taranite")
    assert taranite is not None
    assert slot_fit(_probs(taranite=0.95), MATERIALS, taranite) > 3  # listed: counts for it
    assert slot_fit(_probs(tin=0.95), MATERIALS, taranite) < 0.1  # cannot hold: counts against
    assert slot_fit(_probs(empty=0.99), MATERIALS, taranite) == pytest.approx(1, abs=0.05)
    # below a one_entry deposit's row: any material counts against it, blank is neutral
    assert slot_fit(_probs(silicon=0.95), MATERIALS, frozenset()) < 0.02
    assert slot_fit(_probs(empty=0.99), MATERIALS, frozenset()) == pytest.approx(1, abs=0.05)


def _pick_deposit(deposit: dict[str, float], *slots: dict[str, float]) -> tuple[str, float]:
    best, p = pick_deposit(
        _probs(**deposit),
        DEPOSITS,
        {row: _probs(**slot) for row, slot in enumerate(slots)},
        MATERIALS,
    )
    return DEPOSITS[best], p


def test_pick_deposit() -> None:
    # Tungsten deposits cannot hold Taranite; Taranite deposits hold nothing else
    name, p = _pick_deposit(
        {"tungsten": 0.55, "taranite": 0.4},
        {"taranite": 0.95}, {"taranite": 0.95}, {"inert_materials": 0.9},
    )  # fmt: skip
    assert name == "Taranite" and p > 0.9  # materials settle an unsure read
    name, p = _pick_deposit({"iron": 0.6, "tin": 0.35}, {"iron": 0.95}, {"inert_materials": 0.9})
    assert name == "Iron" and p > 0.9  # materials confirm an unsure read
    name, p = _pick_deposit({"iron": 0.99}, {"tin": 0.95}, {"tin": 0.95})
    assert (name, p) == ("Iron", pytest.approx(0.99))  # a confident read stays, whatever
    # Bexalite and Gold deposits can hold the same materials: the read decides, unchanged
    name, p = _pick_deposit({"bexalite": 0.6, "gold": 0.38}, {"bexalite": 0.95}, {"gold": 0.9})
    assert (name, p) == ("Bexalite", pytest.approx(0.6, abs=1e-3))
    # Hadanite deposits show one row, Hadanite; Hephaestanite ones hold Silicon
    unsure = {"hadanite": 0.5, "hephaestanite": 0.45}
    name, _ = _pick_deposit(unsure, {"hephaestanite": 0.9}, {"silicon": 0.9})
    assert name == "Hephaestanite"
    name, p = _pick_deposit(unsure, {"hadanite": 0.9}, {"empty": 0.99})
    assert name == "Hadanite" and p > 0.9


class _WordsOnly:
    word_classes = CLASSES


def _reading(roi: str, text: str, confidence: float, probs: np.ndarray | None = None) -> RoiReading:
    return RoiReading(roi, WORD_CNN, Rect(0, 0, 1, 1), text, confidence, raw_text=text, probs=probs)


@pytest.mark.parametrize(
    ("deposit", "expected"),
    [
        # A confident deposit narrows the materials
        ({"iron": 0.98}, ("Iron", "Iron", "Inert Materials")),
        # The materials settle an unsure deposit, which then narrows them
        ({"tungsten": 0.55, "taranite": 0.4}, ("Taranite", "Taranite", "Inert Materials")),
        # Unsure, and the materials fit both names: no restriction
        ({"bexalite": 0.6, "gold": 0.38}, ("Bexalite", "Tin", "Inert Materials")),
        # Not in the table (and, as before, no deposit name either)
        ({"inert_materials": 0.98}, ("inert_materials", "Tin", "Inert Materials")),
    ],
)
def test_recognizer_reads_deposit_and_materials_together(
    profiles: dict[str, HudProfile], deposit: dict[str, float], expected: tuple[str, str, str]
) -> None:
    recognizer = FrameRecognizer(profiles["mole"], {}, _WordsOnly())  # type: ignore[arg-type]
    d = _probs(**deposit)
    readings = {
        "deposit": _reading("deposit", DEPOSITS[int(d.argmax())], float(d.max()), d),
        "material1": _reading("material1", "Tin", 0.6, _probs(tin=0.6, iron=0.2, taranite=0.15)),
        "material2": _reading("material2", "Inert Materials", 0.9, _probs(inert_materials=0.9)),
        "material3": _reading("material3", "", 1.0),  # blank canvas: no model run, no probs
    }
    recognizer._apply_deposit(readings)
    texts = tuple(readings[n].text for n in ("deposit", "material1", "material2"))
    assert texts == expected
    assert readings["material1"].raw_text == "Tin"  # the model's own pick stays visible
    assert readings["material3"].text == ""


def test_recognizer_reads_one_row_for_a_one_entry_deposit(profiles: dict[str, HudProfile]) -> None:
    recognizer = FrameRecognizer(profiles["mole"], {}, _WordsOnly())  # type: ignore[arg-type]
    d = _probs(hadanite=0.98)
    readings = {
        "deposit": _reading("deposit", "Hadanite", 0.98, d),
        "material1": _reading(
            "material1", "Hadanite", 0.6, _probs(hadanite=0.6, hephaestanite=0.3)
        ),
        # a bright background behind the empty second row gives the model something to misread
        "material2": _reading("material2", "Silicon", 0.8, _probs(silicon=0.8, empty=0.15)),
    }
    recognizer._apply_deposit(readings)
    assert tuple(readings[n].text for n in readings) == ("Hadanite", "Hadanite", "")
