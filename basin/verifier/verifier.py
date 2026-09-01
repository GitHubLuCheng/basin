"""
Lightweight verifier implementations.

A verifier assigns a score in [0, 1] to a reasoning state, estimating how
likely it is to be correct.  The score enters the metadynamics candidate
selection objective:

    score(z) = alpha * verifier_score(z) + beta * confidence(z) - lambda * bias(z)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List

from basin.state.reasoning_state import ReasoningState


class BaseVerifier(ABC):
    """Abstract verifier interface."""

    @abstractmethod
    def score(self, state: ReasoningState) -> float:
        """Return a verification score in [0, 1]. Higher = more likely correct."""

    def score_batch(self, states: List[ReasoningState]) -> List[float]:
        """Score a list of states.  Default: call score() in a loop."""
        return [self.score(s) for s in states]


# ---------------------------------------------------------------------------
# Heuristic verifier (no API calls)
# ---------------------------------------------------------------------------

class HeuristicVerifier(BaseVerifier):
    """
    Fast verifier based on text heuristics.

    Combines:
    * Self-eval score from the state (already extracted).
    * Contradiction penalty.
    * Answer format check (do numbers appear plausibly?).
    * Length and structure reward.
    """

    def score(self, state: ReasoningState) -> float:
        score = state.self_eval_score
        # Penalise contradictions.
        score -= 0.3 * state.contradiction_score
        # Penalise "unknown" or empty answers.
        ans = state.provisional_answer.lower().strip()
        if ans in ("unknown", "", "none", "n/a"):
            score -= 0.2
        # Reward structured reasoning.
        if re.search(r"step \d|therefore|thus|so the answer", state.trace_text, re.I):
            score += 0.05
        return float(max(0.0, min(1.0, score)))


def _normalise_answer(answer: str) -> str:
    """Normalise answer for comparison: lowercase, strip whitespace/punctuation."""
    a = answer.lower().strip().rstrip(".")
    # Remove common filler phrases.
    a = re.sub(r"the answer is|therefore|thus|=|equals?", "", a)
    # Remove currency symbols.
    a = re.sub(r"[$€£¥]", "", a)
    # Collapse whitespace and remove commas in numbers.
    a = re.sub(r",", "", a)
    a = re.sub(r"\s+", " ", a).strip()
    # If the string contains a number, extract just the numeric part for comparison.
    m = re.search(r"-?\d+\.?\d*", a)
    if m:
        a = m.group()
    return a
