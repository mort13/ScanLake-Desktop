"""Snap a recognised word to the closest known name (port of ScanLake Capture's ResultMapper)."""

from __future__ import annotations

from collections.abc import Sequence


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def fuzzy_match(text: str, candidates: Sequence[str], max_ratio: float = 0.4) -> str:
    """The candidate closest to `text`, or `text` itself when nothing is within `max_ratio`.

    Case-insensitive; underscores count as spaces (model labels use `inert_materials`).
    """
    if not text:
        return ""
    wanted = text.replace("_", " ").strip().lower()
    for candidate in candidates:
        if candidate.lower() == wanted:
            return candidate

    best, best_dist = "", None
    for candidate in candidates:
        dist = levenshtein(wanted, candidate.lower())
        if best_dist is None or dist < best_dist:
            best, best_dist = candidate, dist
    if best_dist is not None and best_dist <= max(len(wanted), len(best)) * max_ratio:
        return best
    return text
