"""Model outputs -> text: greedy CTC for the digit CRNN, softmax top-k for the word CNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float32]


@dataclass(frozen=True)
class CtcResult:
    text: str
    confidence: float  # mean max-probability over non-blank timesteps (Rock Capture CNN's)
    char_confidences: tuple[float, ...]  # per emitted character: max prob over its timesteps


def ctc_greedy(log_probs: Floats, classes: str, blank: int = 0) -> list[CtcResult]:
    """Decodes a [T, N, C] log-probability batch (the DigitCRNN output layout).

    Class index i (i != blank) is `classes[i - 1]`; the blank is index 0.
    An all-blank sequence gives text "" with confidence 0.
    """
    probs = np.exp(log_probs.astype(np.float64))
    best = probs.argmax(axis=2)  # [T, N]
    best_p = probs.max(axis=2)  # [T, N]
    results: list[CtcResult] = []
    for n in range(best.shape[1]):
        seq, p = best[:, n], best_p[:, n]
        chars: list[str] = []
        char_conf: list[float] = []
        prev = -1
        for t, idx in enumerate(seq):
            idx = int(idx)
            if idx != prev and idx != blank:
                chars.append(classes[idx - 1] if 0 < idx <= len(classes) else "")
                char_conf.append(float(p[t]))
            elif idx == prev and idx != blank and char_conf:
                char_conf[-1] = max(char_conf[-1], float(p[t]))
            prev = idx
        non_blank = seq != blank
        conf = float(p[non_blank].mean()) if non_blank.any() else 0.0
        results.append(CtcResult("".join(chars), conf if chars else 0.0, tuple(char_conf)))
    return results


def softmax(logits: Floats) -> Floats:
    """Row-wise softmax of [N, K] logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
