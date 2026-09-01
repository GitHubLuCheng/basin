"""
Evaluation metrics for the metadynamics reasoning prototype.

Metrics:
  * Accuracy — fraction of problems answered correctly.
  * Expected Calibration Error (ECE) — |confidence - accuracy| per bin.
  * Diversity — average number of unique basins explored per problem.
  * Escape rate — fraction of problems where the initial wrong basin was
    eventually abandoned in favour of the correct one.
  * Total model calls / tokens (compute cost).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from metadyn.confidence.estimator import AnswerResult
from metadyn.datasets.loader import Problem
from metadyn.verifier.verifier import _normalise_answer


@dataclass
class EvalResult:
    """Aggregated evaluation results over a dataset."""

    n_problems: int
    accuracy: float
    ece: float
    mean_basins: float
    escape_rate: float
    mean_confidence: float
    # Raw per-problem data for detailed analysis.
    per_problem: List[dict] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EvalResult("
            f"acc={self.accuracy:.3f}, "
            f"ECE={self.ece:.3f}, "
            f"basins={self.mean_basins:.1f}, "
            f"escape={self.escape_rate:.3f})"
        )


def accuracy(
    results: List[AnswerResult],
    problems: List[Problem],
) -> float:
    """
    Fraction of problems where the predicted answer matches the reference.

    Comparison is done after :func:`~metadyn.verifier.verifier._normalise_answer`.
    """
    if not results:
        return 0.0
    correct = sum(
        _normalise_answer(r.predicted_answer) == p.normalised_answer()
        for r, p in zip(results, problems)
    )
    return correct / len(results)


def expected_calibration_error(
    results: List[AnswerResult],
    problems: List[Problem],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) using equal-width confidence bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    n_bins:
        Number of confidence bins in [0, 1].
    """
    if not results:
        return 0.0

    bins: List[List[Tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for r, p in zip(results, problems):
        conf = r.confidence
        correct = _normalise_answer(r.predicted_answer) == p.normalised_answer()
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    ece = 0.0
    n = len(results)
    for b in bins:
        if not b:
            continue
        confs, corrects = zip(*b)
        acc_b = sum(corrects) / len(b)
        conf_b = sum(confs) / len(b)
        ece += (len(b) / n) * abs(acc_b - conf_b)
    return ece


def diversity_score(results: List[AnswerResult]) -> float:
    """Average number of unique basins explored per problem."""
    if not results:
        return 0.0
    return float(np.mean([r.n_basins_explored for r in results]))


def escape_rate(
    results: List[AnswerResult],
    problems: List[Problem],
) -> float:
    """
    Fraction of problems where the final answer is correct even though
    the *first* state selected was incorrect.

    This measures the algorithm's ability to escape initially attractive
    wrong basins — a key property distinguishing metadynamics from greedy.
    """
    if not results:
        return 0.0
    n_escaped = 0
    n_initially_wrong = 0
    for r, p in zip(results, problems):
        if not r.visit_history:
            continue
        # First-selected answer: look at per-problem state if available.
        # We use the visit_history to infer but don't have direct first-answer
        # stored in AnswerResult.  Mark as "escaped" if n_basins > 1 and correct.
        if r.n_basins_explored > 1:
            n_initially_wrong += 1
            if _normalise_answer(r.predicted_answer) == p.normalised_answer():
                n_escaped += 1
    if n_initially_wrong == 0:
        return 0.0
    return n_escaped / n_initially_wrong


def compute_all(
    results: List[AnswerResult],
    problems: List[Problem],
) -> EvalResult:
    """Compute all metrics and return an :class:`EvalResult`."""
    per_problem = []
    for r, p in zip(results, problems):
        per_problem.append(
            {
                "id": p.problem_id,
                "question": p.question[:60] + "...",
                "reference": p.answer,
                "predicted": r.predicted_answer,
                "correct": _normalise_answer(r.predicted_answer) == p.normalised_answer(),
                "confidence": r.confidence,
                "n_basins": r.n_basins_explored,
                "converged": r.converged,
            }
        )
    return EvalResult(
        n_problems=len(results),
        accuracy=accuracy(results, problems),
        ece=expected_calibration_error(results, problems),
        mean_basins=diversity_score(results),
        escape_rate=escape_rate(results, problems),
        mean_confidence=float(np.mean([r.confidence for r in results])) if results else 0.0,
        per_problem=per_problem,
    )
