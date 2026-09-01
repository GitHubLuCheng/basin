"""
Calibrated confidence estimator.

The default ``ConfidenceEstimator`` can saturate to confidence = 1.0 whenever
all basins happen to agree on the same answer, even if that agreement is driven
by a small number of trajectories.  This module provides a drop-in replacement
that uses a Laplace-smoothed trajectory vote fraction, which:

  - Cannot reach 1.0 unless the prior is overwhelmed by evidence.
  - Naturally reflects sample size (few trajectories → confidence closer to 0.5).
  - Is well-calibrated by construction for binomial outcomes.

Confidence formula
------------------
Let N be the total number of selected trajectories (= n_rounds) and n_a the
number of trajectories whose dominant basin has normalised answer *a*.

    raw_vote_fraction(a) = n_a / N

    laplace_conf(a)      = (n_a + 1) / (N + 2)          [Laplace / add-1 smoothing]

    verifier_mod(a)      = mean verifier score of all basins supporting a

    conf(a)  = (1 - verifier_weight) * laplace_conf(a)
               + verifier_weight     * verifier_mod(a)

The predicted answer is still the plurality answer (highest raw vote fraction).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict

from metadyn.confidence.estimator import AnswerResult
from metadyn.memory.basin_memory import BasinMemory
from metadyn.verifier.verifier import _normalise_answer


class CalibratedConfidenceEstimator:
    """
    Laplace-smoothed trajectory-vote confidence estimator.

    Unlike the softmax-based ``ConfidenceEstimator``, this class:

    * Uses trajectory visit counts (not free-energy-weighted basin scores)
      as the primary confidence signal.
    * Applies Laplace (add-1) smoothing so confidence stays in (0, 1)
      regardless of sample size.
    * Blends with mean verifier score to penalise low-quality winning basins.

    Parameters
    ----------
    verifier_weight:
        Weight given to mean verifier score in the final blend.
        0.0 → pure vote fraction; 1.0 → pure verifier score.
    """

    def __init__(self, verifier_weight: float = 0.3) -> None:
        self.verifier_weight = verifier_weight

    def aggregate(self, memory: BasinMemory) -> AnswerResult:
        """
        Derive the final answer from *memory* using calibrated confidence.

        Parameters
        ----------
        memory:
            A :class:`~metadyn.memory.basin_memory.BasinMemory` after the
            exploration loop has completed.
        """
        if not memory.states:
            return AnswerResult(
                predicted_answer="unknown",
                confidence=0.0,
                n_basins_explored=0,
            )

        basin_stats = memory.get_basin_stats()
        visit_hist = memory.visit_history()
        n_total = sum(bs.visit_count for bs in basin_stats)  # = len(visit_hist)

        # ------------------------------------------------------------------
        # Accumulate per-answer trajectory counts and verifier scores.
        # ------------------------------------------------------------------
        answer_visits: Dict[str, int] = defaultdict(int)
        answer_verifier_sum: Dict[str, float] = defaultdict(float)
        answer_basin_counts: Dict[str, int] = defaultdict(int)

        for bs in basin_stats:
            normed = _normalise_answer(bs.dominant_answer)
            answer_visits[normed] += bs.visit_count
            answer_verifier_sum[normed] += bs.mean_verifier_score * bs.visit_count
            answer_basin_counts[normed] += 1

        # ------------------------------------------------------------------
        # Select predicted answer by plurality vote.
        # ------------------------------------------------------------------
        predicted = max(answer_visits, key=answer_visits.__getitem__)
        n_agree = answer_visits[predicted]

        # ------------------------------------------------------------------
        # Laplace-smoothed vote fraction.
        # ------------------------------------------------------------------
        laplace_conf = (n_agree + 1) / (n_total + 2)

        # ------------------------------------------------------------------
        # Mean verifier score for the winning answer (visit-weighted).
        # ------------------------------------------------------------------
        verifier_conf = (
            answer_verifier_sum[predicted] / n_agree
            if n_agree > 0
            else 0.5
        )

        # Blend the two signals.
        w = self.verifier_weight
        confidence = (1.0 - w) * laplace_conf + w * verifier_conf

        # ------------------------------------------------------------------
        # Build answer probability distribution (still Laplace-normalised).
        # ------------------------------------------------------------------
        probs: Dict[str, float] = {}
        for ans, visits in answer_visits.items():
            probs[ans] = (visits + 1) / (n_total + len(answer_visits))

        # Convergence: multiple basins agree on the winning answer.
        converged = answer_basin_counts.get(predicted, 0) > 1

        return AnswerResult(
            predicted_answer=predicted,
            confidence=confidence,
            answer_distribution=probs,
            n_basins_explored=len(basin_stats),
            converged=converged,
            visit_history=visit_hist,
            n_basins_per_answer=dict(answer_basin_counts),
            n_trajectories=n_total,
        )
