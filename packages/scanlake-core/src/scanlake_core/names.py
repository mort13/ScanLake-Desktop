"""Deposit/material names read letter by letter: a CTC read matched against the known names.

The letter model (a DigitCRNN with the alphabet `NAME_CHARS`) reads what the HUD shows, e.g.
"COPPER (ORE)", "RAW ICE", or the visible part of a long name that the HUD clips and scrolls
("ANTAINIUM (RAW"). `NameScorer` turns its [T, N, C] log-probabilities into one score per name
class (domain/data/names.json), in the shape the word classifier used to return, so recognition
and the deposit/material narrowing work on it unchanged:

- each class is scored by its best form (spelling with optional prefix and suffix): the
  log-probability of the best CTC path that spells it (Viterbi; summing over alignments instead
  lets the many paths of a one-letter "clipped" match outweigh an empty row). A form may be
  clipped on either side (the scrolling window) at CLIP_PENALTY per hidden letter, so the full
  text wins whenever it is all visible; a clipped match still holds at least one letter
- "empty" is the all-blank path
- "unknown" is the model's own best path, less UNKNOWN_PENALTY: a read that fits no name gives
  it the most probability, which is then spread evenly over all classes, so an unreadable or
  unknown name comes out with low confidence instead of a confident wrong one
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np

from scanlake_core.decode import Floats

NAME_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ ()"
EMPTY = "empty"  # the class of an empty composition row (as the word classifier named it)

# Log-probability cost per letter hidden by the HUD's scrolling window. Small: only long names
# are clipped, and they then still show most of their letters.
CLIP_PENALTY = 0.25
# How much likelier than every known name the model's own reading must be to count as unknown
# (natural log): one misread letter keeps the name, garbage spreads over all classes.
UNKNOWN_PENALTY = 4.0

_NEG = np.float32(-1e30)


@dataclass(frozen=True)
class Lexicon:
    classes: tuple[str, ...]  # class names (tables.json materials) and EMPTY, last
    forms: tuple[str, ...]  # prefix + spelling + suffix, e.g. "RAW COPPER (ORE)"
    cores: tuple[tuple[int, int], ...]  # per form: where its spelling starts and ends
    form_class: tuple[int, ...]  # index into classes, per form


def build_lexicon(names: dict[str, list[str]], suffixes: list[str], prefixes: list[str]) -> Lexicon:
    """One form per spelling, prefix and suffix ("RAW COPPER (ORE)", "RAW COPPER (RAW)").
    `NameScorer` treats the prefix and the suffix as optional, at no cost."""
    classes = (*names, EMPTY)
    forms: list[str] = []
    cores: list[tuple[int, int]] = []
    owner: list[int] = []
    for k, spellings in enumerate(names.values()):
        for spelling in spellings:
            for prefix in prefixes or [""]:
                for suffix in suffixes or [""]:
                    forms.append(prefix + spelling + suffix)
                    cores.append((len(prefix), len(prefix) + len(spelling)))
                    owner.append(k)
    return Lexicon(classes, tuple(forms), tuple(cores), tuple(owner))


@cache
def lexicon() -> Lexicon:
    raw = json.loads(
        resources.files("scanlake_core.domain").joinpath("data/names.json").read_text("utf-8")
    )
    return build_lexicon(raw["names"], raw.get("suffixes", []), raw.get("prefixes", []))


class NameScorer:
    """[T, N, C] CTC log-probabilities -> [N, K] log-probabilities over the lexicon's classes."""

    def __init__(
        self,
        lex: Lexicon,
        alphabet: str = NAME_CHARS,
        blank: int = 0,
        clip_penalty: float = CLIP_PENALTY,
        unknown_penalty: float = UNKNOWN_PENALTY,
    ) -> None:
        if blank != 0:
            raise ValueError("the letter model's blank must be class 0")
        index = {c: i + 1 for i, c in enumerate(alphabet)}  # class i > 0 is alphabet[i - 1]
        missing = {c for f in lex.forms for c in f if c not in index}
        if missing:
            raise ValueError(f"names use letters the model cannot read: {''.join(sorted(missing))}")
        self.lexicon = lex
        self.unknown_penalty = unknown_penalty
        never = len(alphabet) + 1  # an extra input column that is always _NEG
        lengths = np.array([len(f) for f in lex.forms])[:, None]
        starts = np.array([a for a, _ in lex.cores])[:, None]
        ends = np.array([b for _, b in lex.cores])[:, None]
        states = 2 * int(lengths.max()) + 1
        # extended label per form: blank, l1, blank, l2, ..., blank; padding never fires
        ext = np.full((len(lex.forms), states), never, dtype=np.int64)
        for f, text in enumerate(lex.forms):
            ext[f, : 2 * len(text) + 1] = 0
            ext[f, 1 : 2 * len(text) : 2] = [index[c] for c in text]
        self._ext = ext
        # a letter state may also be reached from two states back when the letters differ
        self._skip = np.where((ext[:, 2:] != 0) & (ext[:, 2:] != ext[:, :-2]), 0.0, _NEG)
        # Clipping. A match may start at letter j and end after letter j' (the HUD's scrolling
        # window), at clip_penalty per hidden letter of the spelling; a whole prefix or suffix
        # may be missing for free ("COPPER" for "RAW COPPER (ORE)"). Blanks before a clipped
        # start run in one extra "lead" state per input, blanks after a clipped end in one
        # "trail" state per form, so a match always holds a letter: an empty row never matches
        # a clipped name.
        s = np.arange(states)[None, :]
        j = (s - 1) // 2  # the letter of letter state 2j+1
        letter = (s % 2 == 1) & (j < lengths)
        hidden_before = np.where(j < starts, j, j - starts)
        self._enter = np.where(letter, -clip_penalty * hidden_before, _NEG)
        self._enter_late = np.where(j >= 1, self._enter, _NEG)  # from the lead state
        last = lengths - 1
        hidden_after = np.where(j < ends - 1, ends - 1 - j, np.where(j == ends - 1, 0, last - j))
        self._leave = np.where(letter & (j < last), -clip_penalty * hidden_after, _NEG)
        self._final = np.where((s == 2 * lengths) | (s == 2 * lengths - 1), 0.0, _NEG)
        for name in ("_skip", "_enter", "_enter_late", "_leave", "_final"):
            setattr(self, name, getattr(self, name).astype(np.float32))
        self._owner = np.asarray(lex.form_class)
        self._empty = lex.classes.index(EMPTY)

    def form_scores(self, log_probs: Floats) -> Floats:
        """[T, N, C] -> [N, F]: log-probability of the best path spelling each form (clipped as
        fits best, less its clipping penalty)."""
        lp = log_probs.astype(np.float32)
        lp = np.concatenate([lp, np.full((*lp.shape[:2], 1), _NEG, np.float32)], axis=2)
        blank = lp[:, :, 0]  # [T, N]
        emit = lp[:, :, self._ext]  # [T, N, F, S]
        alpha = np.full(emit.shape[1:], _NEG, dtype=np.float32)
        alpha[:, :, 0] = emit[0, :, :, 0]  # leading blank of an unclipped start
        alpha = np.maximum(alpha, self._enter[None] + emit[0])
        lead = blank[0]  # [N]
        trail = np.full(alpha.shape[:2], _NEG, dtype=np.float32)  # [N, F]
        skip, enter, leave = self._skip[None], self._enter_late[None], self._leave[None]
        for t in range(1, lp.shape[0]):
            into = alpha.copy()
            np.maximum(into[:, :, 1:], alpha[:, :, :-1], out=into[:, :, 1:])
            np.maximum(into[:, :, 2:], alpha[:, :, :-2] + skip, out=into[:, :, 2:])
            np.maximum(into, lead[:, None, None] + enter, out=into)
            trail = np.maximum(trail, (alpha + leave).max(axis=2)) + blank[t][:, None]
            lead = lead + blank[t]
            alpha = np.maximum(into + emit[t], _NEG)  # unreachable states stay at _NEG
        unclipped = (alpha + self._final[None]).max(axis=2)
        clipped = (alpha + leave).max(axis=2)  # ending on a letter
        return np.maximum(np.maximum(unclipped, clipped), trail)

    def scores(self, log_probs: Floats) -> Floats:
        """[T, N, C] -> [N, K] log-probabilities over the classes (rows sum to 1 in exp)."""
        lp = log_probs.astype(np.float64)
        forms = self.form_scores(log_probs).astype(np.float64)  # [N, F]
        n, k = forms.shape[0], len(self.lexicon.classes)
        classes = np.full((n, k), _NEG, dtype=np.float64)
        np.maximum.at(classes.T, self._owner, forms.T)
        classes[:, self._empty] = lp[:, :, 0].sum(axis=0)  # all blank
        unknown = lp.max(axis=2).sum(axis=0) - self.unknown_penalty  # the model's own best path
        both = np.concatenate([classes, unknown[:, None]], axis=1)
        both -= both.max(axis=1, keepdims=True)
        p = np.exp(both)
        p /= p.sum(axis=1, keepdims=True)
        spread = p[:, :k] + p[:, k:] / k
        return np.log(spread).astype(np.float32)
