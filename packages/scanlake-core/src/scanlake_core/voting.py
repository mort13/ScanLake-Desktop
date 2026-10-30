"""Multi-frame voting: one stable value per ROI out of many noisy frames.

Each ROI keeps the observations of the last `window_s` seconds, weighted by recency
(half-life window_s / 2) and by their own confidence.

The winner is the text with the highest summed weight x confidence. Its confidence is that
support divided by the total weight of all observations, so frames that read nothing or
something else pull it down. Digits and words vote alike, on the normalized text (a word's
confidence is its class probability).

A value is STABLE once it won at least `min_votes` observations with agreement >= 0.7 and the
last two observations agree with it. When a new rock is scanned, three confident observations
that disagree with a stable winner replace the window, so the reading switches immediately.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

CONFIDENT = 0.75


class Stability(StrEnum):
    EMPTY = "empty"  # nothing readable yet
    UNSTABLE = "unstable"
    STABLE = "stable"


@dataclass(frozen=True)
class Observation:
    t: float
    text: str | None  # None = unreadable
    confidence: float


@dataclass(frozen=True)
class Vote:
    text: str
    confidence: float
    agreement: float
    count: int  # observations that read the winner
    stability: Stability
    candidates: tuple[tuple[str, float], ...] = ()  # (text, share of support), best first

    @property
    def stable(self) -> bool:
        return self.stability is Stability.STABLE


EMPTY_VOTE = Vote("", 0.0, 0.0, 0, Stability.EMPTY)


@dataclass
class RoiVoter:
    window_s: float = 1.5
    max_observations: int = 30
    min_votes: int = 3
    min_agreement: float = 0.7
    _obs: deque[Observation] = field(default_factory=deque, init=False)
    _last: Vote = field(default=EMPTY_VOTE, init=False)
    _stable_text: str | None = field(default=None, init=False)  # last value that was STABLE

    def reset(self) -> None:
        self._obs.clear()
        self._last = EMPTY_VOTE
        self._stable_text = None

    def add(self, obs: Observation) -> Vote:
        self._obs.append(obs)
        while len(self._obs) > self.max_observations:
            self._obs.popleft()
        if self._stable_text is not None and self._switched(self._stable_text):
            recent = list(self._obs)[-3:]
            self._obs.clear()
            self._obs.extend(recent)
            self._stable_text = None
        self._last = self.vote(obs.t)
        if self._last.stable:
            self._stable_text = self._last.text
        return self._last

    @property
    def last(self) -> Vote:
        return self._last

    def _switched(self, winner: str) -> bool:
        recent = list(self._obs)[-3:]
        if len(recent) < 3:
            return False
        texts = {o.text for o in recent}
        return (
            len(texts) == 1
            and None not in texts
            and winner not in texts
            and all(o.confidence >= CONFIDENT for o in recent)
        )

    def vote(self, now: float) -> Vote:
        live = [o for o in self._obs if now - o.t <= self.window_s]
        if not live:
            return EMPTY_VOTE
        half_life = self.window_s / 2
        weights = [0.5 ** ((now - o.t) / half_life) for o in live]
        total = sum(weights)

        support: dict[str, float] = {}
        counts: dict[str, int] = {}
        for o, w in zip(live, weights, strict=True):
            if o.text is None:
                continue
            support[o.text] = support.get(o.text, 0.0) + w * o.confidence
            counts[o.text] = counts.get(o.text, 0) + 1
        if not support:
            return EMPTY_VOTE
        ranked = sorted(support.items(), key=lambda kv: kv[1], reverse=True)
        winner, best = ranked[0]
        all_support = sum(support.values())
        agreement = best / all_support if all_support else 0.0
        tail = [o.text for o in live[-2:]]
        stable = (
            counts[winner] >= self.min_votes
            and agreement >= self.min_agreement
            and all(t == winner for t in tail)
        )
        return Vote(
            text=winner,
            confidence=best / total,
            agreement=agreement,
            count=counts[winner],
            stability=Stability.STABLE if stable else Stability.UNSTABLE,
            candidates=tuple((t, s / all_support) for t, s in ranked[:3]),
        )
