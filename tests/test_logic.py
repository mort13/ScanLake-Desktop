"""Decoding, field specs, voting and validation."""

from __future__ import annotations

import numpy as np
import pytest

from scanlake_core.decode import ctc_greedy, softmax
from scanlake_core.domain.capture import Capture, FieldValue, utc_timestamp
from scanlake_core.domain.fuzzy import fuzzy_match
from scanlake_core.domain.validation import is_duplicate, validate
from scanlake_core.fields import field_spec
from scanlake_core.voting import Observation, RoiVoter, Stability

CLASSES = "0123456789.%"


def _log_probs(seq: list[int], p: float = 0.9, c: int = 13) -> np.ndarray:
    """[T, 1, C] log-probs whose arg-max sequence is `seq`."""
    out = np.full((len(seq), 1, c), (1 - p) / (c - 1), dtype=np.float32)
    for t, idx in enumerate(seq):
        out[t, 0, idx] = p
    return np.log(out)


def idx(ch: str) -> int:
    return CLASSES.index(ch) + 1


def test_ctc_greedy_collapses_repeats_and_blanks() -> None:
    seq = [0, idx("4"), idx("4"), 0, idx("2"), idx("."), 0, idx("0"), 0, idx("0"), idx("%"), 0]
    [result] = ctc_greedy(_log_probs(seq), CLASSES)
    assert result.text == "42.00%"
    assert result.confidence == pytest.approx(0.9, abs=1e-5)
    assert len(result.char_confidences) == 6


def test_ctc_batch_and_all_blank() -> None:
    a = _log_probs([idx("7"), 0, 0])
    b = _log_probs([0, 0, 0])
    results = ctc_greedy(np.concatenate([a, b], axis=1), CLASSES)
    assert [r.text for r in results] == ["7", ""]
    assert results[1].confidence == 0.0


def test_softmax_rows_sum_to_one() -> None:
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(p.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("column", "text", "expected"),
    [
        ("mass", "11826", "11826"),
        ("mass", "1182.6", None),
        ("resistance", "12%", "12"),
        ("resistance", "120", None),
        ("instability", "4.58", "4.58"),
        ("volume", "42,1", "42.1"),
        ("amount", "42.09%", "42.09"),
        ("amount", "7.5", "7.50"),
        ("amount", "142.00", None),
        ("quality", "1001", None),
        ("quality", "512", "512"),
        ("type", "inert_materials", "Inert Materials"),
        ("type", "empty", ""),
        ("type", "aluminium", "Aluminium"),
        ("deposit", "aluminium", "Aluminum"),
        ("deposit", "quantanium", "Quantanium"),
        ("mass", "", ""),
    ],
)
def test_field_specs(column: str, text: str, expected: str | None) -> None:
    assert field_spec(column).normalize(text) == expected


def test_unknown_columns_get_generic_specs() -> None:
    assert field_spec("signal", "INT32").normalize("1234") == "1234"
    assert field_spec("signal", "INT32").normalize("12.5") is None
    assert field_spec("ratio", "DOUBLE").normalize("0.5") == "0.5"


def test_fuzzy_match_keeps_unknown_words() -> None:
    assert fuzzy_match("Laranit", ["Laranite", "Iron"]) == "Laranite"
    assert fuzzy_match("xyz", ["Laranite", "Iron"]) == "xyz"


def _feed(voter: RoiVoter, texts: list[str | None], conf: float = 0.95, t0: float = 0.0) -> None:
    for i, text in enumerate(texts):
        voter.add(Observation(t0 + i * 0.1, text, conf))


def test_vote_becomes_stable() -> None:
    voter = RoiVoter()
    _feed(voter, ["11826", "11828", "11826"])
    assert voter.last.text == "11826" and voter.last.stability is Stability.UNSTABLE
    _feed(voter, ["11826", "11826"], t0=0.3)
    assert voter.last.stable
    assert voter.last.candidates[0][0] == "11826"


def test_misreads_lower_confidence_but_not_the_winner() -> None:
    voter = RoiVoter()
    _feed(voter, ["512", None, "512", None, "512", "512"])
    assert voter.last.text == "512"
    assert 0.5 < voter.last.confidence < 0.95


def test_new_rock_switches_immediately() -> None:
    voter = RoiVoter()
    _feed(voter, ["11826"] * 8)
    assert voter.last.stable
    _feed(voter, ["9035"] * 3, t0=0.8)
    assert voter.last.text == "9035"
    _feed(voter, ["9035"], t0=1.1)
    assert voter.last.stable


def test_old_observations_expire() -> None:
    voter = RoiVoter(window_s=1.0)
    _feed(voter, ["1"] * 5)
    assert voter.vote(10.0).stability is Stability.EMPTY


def _capture(rows: list[tuple[str, str, str]], **scan: str) -> Capture:
    values = {"deposit": "Quartz", "mass": "11826", "resistance": "0", "instability": "4.58",
              "volume": "42.10"} | scan  # fmt: skip
    return Capture(
        capture_id="c",
        session_id="s",
        cluster_id="k",
        timestamp=utc_timestamp(),
        scan={k: FieldValue(text=v) for k, v in values.items()},
        rows=[
            {
                "type": FieldValue(text=t),
                "amount": FieldValue(text=a),
                "quality": FieldValue(text=q),
            }
            for t, a, q in rows
        ],
    )


SCAN = ("deposit", "mass", "resistance", "instability", "volume")
ROW = ("type", "amount", "quality")


def _messages(capture: Capture, **kw: object) -> list[str]:
    return [i.message for i in validate(capture, SCAN, ROW, required_scan_columns=SCAN, **kw)]  # type: ignore[arg-type]


def test_valid_capture() -> None:
    capture = _capture(
        [("Quartz", "58.40", "512"), ("Inert Materials", "41.60", "0"), ("", "", "")]
    )
    assert _messages(capture) == []


def test_amount_sum_and_incomplete_rows() -> None:
    msgs = _messages(_capture([("Quartz", "58.40", "512"), ("Iron", "40.00", "")]))
    assert any("incomplete" in m for m in msgs)
    assert any("sum to 58.40%" in m for m in msgs)


def test_mass_volume_and_cluster_deposit() -> None:
    capture = _capture([("Quartz", "100.00", "1")], mass="100", volume="2.00")
    msgs = _messages(capture, cluster_deposit="Iron")
    assert any("at least 100× volume" in m for m in msgs)
    assert any('Cluster deposit must be "Iron"' in m for m in msgs)


def test_required_and_invalid_values() -> None:
    capture = _capture([("Quartz", "100.00", "1")], mass="", resistance="abc")
    issues = validate(capture, SCAN, ROW, required_scan_columns=SCAN)
    assert {i.column for i in issues} >= {"mass", "resistance"}


def test_duplicate_detection_ignores_row_order() -> None:
    a = _capture([("Quartz", "58.40", "512"), ("Iron", "41.60", "3")])
    b = _capture([("Iron", "41.60", "3"), ("Quartz", "58.40", "512")])
    b.capture_id = "other"
    assert is_duplicate(a, [b], ROW)
    b.scan["mass"] = FieldValue(text="1")
    assert not is_duplicate(a, [b], ROW)
