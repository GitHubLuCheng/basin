"""
Final-answer confidence estimation via basin aggregation.

For each candidate answer a, we aggregate support from all basins that
ended in a:

    P(a)  ∝  sum_{B_j → a}  exp(-F(B_j)) * verifier(B_j)

where F(B_j) ≈ -log(visit_count_j) is the approximate pseudo-free-energy
and verifier(B_j) is the mean verifier score in basin j.

This mirrors the reweighting scheme in well-tempered metadynamics and gives
a calibrated probability estimate even when individual samples are noisy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from basin.memory.basin_memory import BasinMemory, BasinStats
from basin.verifier.verifier import _normalise_answer


@dataclass
class AnswerResult:
    """
    Final aggregated result after exploring all basins.

    Attributes
    ----------
    predicted_answer:
        The most-supported answer.
    confidence:
        Estimated probability for the predicted answer in [0, 1].
    answer_distribution:
        Mapping from normalised answer string to probability.
    n_basins_explored:
        Number of distinct basins found.
    converged:
        True if multiple basins agree on the same answer (a convergence
        signal that is analogous to free energy convergence in MD).
    visit_history:
        Sequence of basin IDs visited at each step.
    """

    predicted_answer: str
    confidence: float
    answer_distribution: Dict[str, float] = field(default_factory=dict)
    n_basins_explored: int = 0
    converged: bool = False
    visit_history: List[int] = field(default_factory=list)
    # Number of distinct basins that produced each normalised answer.
    # Populated by ConfidenceEstimator and CalibratedConfidenceEstimator.
    n_basins_per_answer: Dict[str, int] = field(default_factory=dict)
    # Total trajectories selected (= len(visit_history) when populated).
    n_trajectories: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        top = sorted(self.answer_distribution.items(), key=lambda x: -x[1])[:3]
        dist_str = ", ".join(f"{a!r}:{p:.2f}" for a, p in top)
        return (
            f"AnswerResult(answer={self.predicted_answer!r}, "
            f"conf={self.confidence:.2f}, basins={self.n_basins_explored}, "
            f"converged={self.converged}, dist=[{dist_str}])"
        )


class ConfidenceEstimator:
    """
    Aggregates basin memory into a final answer and confidence estimate.

    Parameters
    ----------
    temperature:
        Softmax temperature for the basin-weight softmax.
        Lower → sharper, more concentrated probability on the best basin.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def aggregate(self, memory: BasinMemory) -> AnswerResult:
        """
        Derive the final answer from all states in *memory*.

        Parameters
        ----------
        memory:
            A :class:`~basin.memory.basin_memory.BasinMemory` after
            the metadynamics exploration loop has completed.
        """
        if not memory.states:
            return AnswerResult(
                predicted_answer="unknown",
                confidence=0.0,
                n_basins_explored=0,
            )

        basin_stats = memory.get_basin_stats()

        # Compute a weight for each basin:
        #   w(B) = exp(-F(B) / T) * verifier(B)
        #        = visit_count^(1/T) * verifier(B)
        # (using F(B) = -log(visit_count), so exp(-F/T) = count^(1/T))
        answer_weights: Dict[str, float] = defaultdict(float)
        for bs in basin_stats:
            # Basin weight = verifier score (quality) * log(1+visits) (exploration coverage).
            # Verifier score dominates so the best-reasoned basin wins;
            # log-visit multiplier gives a small bonus for consensus basins.
            f_b = bs.pseudo_free_energy  # = -log(visit_count), <= 0
            visit_mult = math.exp(-f_b / self.temperature)  # = visit_count^(1/T)
            # Normalise visit multiplier to [1, 2] range to prevent domination.
            visit_bonus = 1.0 + math.log1p(bs.visit_count) / 5.0
            basin_weight = bs.mean_verifier_score * visit_bonus
            normed_answer = _normalise_answer(bs.dominant_answer)
            answer_weights[normed_answer] += basin_weight

        # Normalise to probabilities.
        total = sum(answer_weights.values())
        if total < 1e-12:
            # No useful signal; fall back to the most-visited basin's answer.
            best_bs = max(basin_stats, key=lambda b: b.visit_count)
            return AnswerResult(
                predicted_answer=best_bs.dominant_answer,
                confidence=0.0,
                n_basins_explored=len(basin_stats),
            )

        probs = {a: w / total for a, w in answer_weights.items()}
        predicted = max(probs, key=probs.get)  # type: ignore
        confidence = probs[predicted]

        # Convergence: multiple basins with different IDs agree on the answer.
        answer_basin_counts: Dict[str, int] = defaultdict(int)
        for bs in basin_stats:
            answer_basin_counts[_normalise_answer(bs.dominant_answer)] += 1
        converged = answer_basin_counts.get(predicted, 0) > 1

        visit_hist = memory.visit_history()
        return AnswerResult(
            predicted_answer=predicted,
            confidence=confidence,
            answer_distribution=dict(probs),
            n_basins_explored=len(basin_stats),
            converged=converged,
            visit_history=visit_hist,
            n_basins_per_answer=dict(answer_basin_counts),
            n_trajectories=len(visit_hist),
        )
