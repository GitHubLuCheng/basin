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
from typing import List, Optional

from basin.llm.base import BaseLLMClient
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


# ---------------------------------------------------------------------------
# LLM-based verifier
# ---------------------------------------------------------------------------

_VERIFIER_PROMPT = """\
You are a careful mathematics and reasoning verifier.

Question: {question}

Proposed reasoning and answer:
\"\"\"
{trace}
\"\"\"

Is this reasoning correct?  Score it from 0.0 (clearly wrong) to 1.0 (certainly correct).
Return ONLY a single float between 0.0 and 1.0, nothing else.
"""


class LLMVerifier(BaseVerifier):
    """
    Second-pass verifier that calls an LLM to judge the reasoning.

    Parameters
    ----------
    llm_client:
        The LLM backend to use.
    temperature:
        Low temperature for more deterministic judgements.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.1,
    ) -> None:
        self._llm = llm_client
        self._temperature = temperature

    def score(self, state: ReasoningState) -> float:
        prompt = _VERIFIER_PROMPT.format(
            question=state.question,
            trace=state.trace_text[:1500],
        )
        responses = self._llm.generate(
            prompt, temperature=self._temperature, max_tokens=16, n=1
        )
        raw = responses[0].text.strip()
        try:
            val = float(re.search(r"\d+\.?\d*", raw).group())  # type: ignore
            return float(max(0.0, min(1.0, val)))
        except (AttributeError, ValueError):
            return 0.5  # neutral on parse failure


# ---------------------------------------------------------------------------
# Consistency verifier (answer agreement across multiple states)
# ---------------------------------------------------------------------------

class ConsistencyVerifier(BaseVerifier):
    """
    Scores a state by how often its answer agrees with the plurality answer
    in a collection of comparison states.

    Parameters
    ----------
    comparison_states:
        Pool of states to compare against.  Typically the current memory.
    """

    def __init__(self, comparison_states: Optional[List[ReasoningState]] = None) -> None:
        self.comparison_states: List[ReasoningState] = comparison_states or []

    def update(self, states: List[ReasoningState]) -> None:
        """Replace the comparison pool."""
        self.comparison_states = states

    def score(self, state: ReasoningState) -> float:
        if not self.comparison_states:
            return 0.5
        target = _normalise_answer(state.provisional_answer)
        matches = sum(
            1 for s in self.comparison_states
            if _normalise_answer(s.provisional_answer) == target
        )
        return matches / len(self.comparison_states)


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
