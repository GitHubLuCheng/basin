"""
Extended evaluation metrics for stress-testing the MetaDynamics prototype.

Goes beyond basic accuracy / ECE to diagnose:
  - Basin diversity (effective basin count Neff)
  - Confidence calibration quality (saturation rate, reliability diagram data)
  - Escape behaviour (was the initial wrong basin actually abandoned?)
  - Answer convergence patterns (competing vs consistent basins)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from basin.confidence.estimator import AnswerResult
from basin.datasets.loader import Problem
from basin.verifier.verifier import _normalise_answer


# ---------------------------------------------------------------------------
# Scalar basin-diversity statistics
# ---------------------------------------------------------------------------


def effective_basin_count(visit_history: List[int]) -> float:
    """
    Effective number of basins (Hill number N1 / inverse Simpson index).

        Neff = 1 / sum_j p_j^2

    where p_j = fraction of steps spent in basin j.

    Interpretation:
    * Neff = 1   → all steps in one basin (no exploration).
    * Neff = k   → k basins explored equally.
    * Neff close to n_basins_explored → balanced exploration.
    """
    if not visit_history:
        return 0.0
    counts = Counter(visit_history)
    n = len(visit_history)
    probs = [c / n for c in counts.values()]
    return 1.0 / sum(p * p for p in probs)


def basin_occupancy_histogram(visit_history: List[int]) -> Dict[int, int]:
    """Return a mapping basin_id → visit count."""
    return dict(Counter(visit_history))


# ---------------------------------------------------------------------------
# Confidence saturation diagnostics
# ---------------------------------------------------------------------------


def fraction_saturated_confidence(
    results: List[AnswerResult],
    threshold: float = 0.99,
) -> float:
    """
    Fraction of problems where confidence >= *threshold*.

    A high value (e.g. > 0.8) indicates that the confidence estimator is
    saturating and the scores carry little information about uncertainty.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.confidence >= threshold) / len(results)


def confidence_histogram(
    results: List[AnswerResult],
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (bin_centres, counts) for a histogram of confidence values.

    Parameters
    ----------
    n_bins:
        Number of equal-width bins in [0, 1].
    """
    confs = np.array([r.confidence for r in results])
    counts, edges = np.histogram(confs, bins=n_bins, range=(0.0, 1.0))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts


def reliability_diagram_data(
    results: List[AnswerResult],
    problems: List[Problem],
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for a reliability (calibration) diagram.

    Returns
    -------
    bin_conf : np.ndarray shape (n_bins,)
        Mean confidence in each bin.
    bin_acc  : np.ndarray shape (n_bins,)
        Fraction correct in each bin.
    bin_frac : np.ndarray shape (n_bins,)
        Fraction of all examples falling in each bin.
    """
    bins: List[List[Tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for r, p in zip(results, problems):
        correct = _normalise_answer(r.predicted_answer) == p.normalised_answer()
        idx = min(int(r.confidence * n_bins), n_bins - 1)
        bins[idx].append((r.confidence, correct))

    n = len(results)
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_frac = np.zeros(n_bins)
    for i, b in enumerate(bins):
        if b:
            confs, corrects = zip(*b)
            bin_conf[i] = np.mean(confs)
            bin_acc[i] = np.mean(corrects)
            bin_frac[i] = len(b) / n
    return bin_conf, bin_acc, bin_frac


# ---------------------------------------------------------------------------
# Answer convergence statistics
# ---------------------------------------------------------------------------


def mean_consistent_basins(
    results: List[AnswerResult],
    problems: List[Problem],
) -> float:
    """
    Mean number of distinct basins agreeing on the *predicted* answer.

    If this is consistently 1 while confidence is 1.0, the high confidence
    is driven by the softmax normalisation rather than genuine agreement.
    Requires ``AnswerResult.n_basins_per_answer`` to be populated.
    """
    if not results:
        return 0.0
    counts = []
    for r, p in zip(results, problems):
        if r.n_basins_per_answer:
            predicted_norm = _normalise_answer(r.predicted_answer)
            counts.append(r.n_basins_per_answer.get(predicted_norm, 1))
        else:
            # Fallback: infer from converged flag.
            counts.append(2 if r.converged else 1)
    return float(np.mean(counts))


def mean_competing_basins(results: List[AnswerResult]) -> float:
    """
    Mean number of *distinct answer groups* per problem.

    High values indicate unresolved disagreement; value = 1 means all
    explored basins gave the same answer.
    """
    if not results:
        return 0.0
    counts = []
    for r in results:
        if r.n_basins_per_answer:
            counts.append(len(r.n_basins_per_answer))
        else:
            counts.append(len(r.answer_distribution) if r.answer_distribution else 1)
    return float(np.mean(counts))


def basin_convergence_rate(results: List[AnswerResult]) -> float:
    """
    Fraction of problems where multiple distinct basins converge on the same answer.

    When high AND accuracy is high, the convergence signal is meaningful.
    When confidence saturates without genuine multi-basin agreement, this will
    remain low even if confidence = 1.0.
    """
    if not results:
        return 0.0
    return float(np.mean([r.converged for r in results]))


# ---------------------------------------------------------------------------
# Escape rate (cleaner definition)
# ---------------------------------------------------------------------------


def strict_escape_rate(
    results: List[AnswerResult],
    problems: List[Problem],
    first_answers: Optional[List[str]] = None,
) -> float:
    """
    Fraction of problems where:
      * The first selected trajectory was *wrong*, AND
      * The final predicted answer is *correct*.

    This is a stronger signal than the original ``escape_rate`` in
    ``basin.metrics.metrics``, which only checks if n_basins > 1.

    Parameters
    ----------
    first_answers:
        If provided, the answer produced at round 0 for each problem.
        If None, the method falls back to the heuristic: treat the
        visit_history[0] basin as wrong if the final answer differs.
    """
    if not results:
        return 0.0
    n_initially_wrong = 0
    n_escaped = 0
    for i, (r, p) in enumerate(zip(results, problems)):
        ref = p.normalised_answer()
        final_correct = _normalise_answer(r.predicted_answer) == ref

        if first_answers is not None:
            first_wrong = _normalise_answer(first_answers[i]) != ref
        else:
            # Heuristic: assume first trajectory is wrong if there are >1 basins
            # and the first basin visited differs from the final answer's basin.
            first_wrong = r.n_basins_explored > 1 and not final_correct

            # Better heuristic: if n_basins > 1, assume we started wrong.
            if r.n_basins_explored > 1:
                first_wrong = True  # conservative — counts exploration as "started wrong"

        if first_wrong:
            n_initially_wrong += 1
            if final_correct:
                n_escaped += 1

    return n_escaped / n_initially_wrong if n_initially_wrong > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-problem diagnostic dict
# ---------------------------------------------------------------------------


def per_problem_extended(
    result: AnswerResult,
    problem: Problem,
    first_answer: Optional[str] = None,
) -> dict:
    """
    Full diagnostic dictionary for one problem.

    Parameters
    ----------
    result:
        The :class:`~basin.confidence.estimator.AnswerResult` produced by
        any method (MetaDyn or baseline).
    problem:
        The ground-truth problem.
    first_answer:
        Optional: the provisional answer at step 0 (for escape-rate tracking).
    """
    predicted_norm = _normalise_answer(result.predicted_answer)
    ref_norm = problem.normalised_answer()
    correct = predicted_norm == ref_norm

    neff = effective_basin_count(result.visit_history)
    occ = basin_occupancy_histogram(result.visit_history)

    if result.n_basins_per_answer:
        n_answer_groups = len(result.n_basins_per_answer)
        consistent = result.n_basins_per_answer.get(predicted_norm, 1)
        competing = n_answer_groups - (1 if consistent > 0 else 0)
    else:
        n_answer_groups = len(result.answer_distribution) if result.answer_distribution else 1
        consistent = 1
        competing = n_answer_groups - 1

    first_correct: Optional[bool] = None
    if first_answer is not None:
        first_correct = _normalise_answer(first_answer) == ref_norm
    elif result.visit_history:
        # We don't have the first answer, so leave as None.
        pass

    return {
        "id": problem.problem_id,
        "question": problem.question[:80] + ("..." if len(problem.question) > 80 else ""),
        "reference": ref_norm,
        "predicted": predicted_norm,
        "correct": correct,
        "confidence": round(result.confidence, 4),
        "n_basins": result.n_basins_explored,
        "neff": round(neff, 3),
        "n_trajectories": result.n_trajectories,
        "n_answer_groups": n_answer_groups,
        "consistent_basins": consistent,
        "competing_basins": competing,
        "converged": result.converged,
        "basin_occupancy": occ,
        "answer_distribution": {k: round(v, 4) for k, v in result.answer_distribution.items()},
        "first_answer": _normalise_answer(first_answer) if first_answer else None,
        "first_correct": first_correct,
    }


# ---------------------------------------------------------------------------
# Aggregate extended evaluation
# ---------------------------------------------------------------------------


@dataclass
class ExtendedEvalResult:
    """
    Full evaluation result including all extended diagnostics.

    Fields mirror :class:`~basin.metrics.metrics.EvalResult` but add
    basin-diversity, confidence-saturation, and convergence statistics.
    """

    method: str
    n_problems: int
    accuracy: float
    ece: float
    mean_basins: float
    mean_neff: float
    escape_rate: float
    mean_confidence: float
    frac_saturated_conf: float          # fraction with confidence >= 0.99
    basin_convergence_rate: float       # fraction where multiple basins agree
    mean_consistent_basins: float       # mean basins backing predicted answer
    mean_competing_basins: float        # mean distinct answer groups
    n_model_calls: int                  # total API / LLM calls
    per_problem: List[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        """Return a flat dict suitable for a results table."""
        return {
            "method": self.method,
            "acc": round(self.accuracy, 3),
            "ece": round(self.ece, 3),
            "mean_basins": round(self.mean_basins, 2),
            "mean_neff": round(self.mean_neff, 2),
            "escape_rate": round(self.escape_rate, 3),
            "mean_conf": round(self.mean_confidence, 3),
            "frac_sat": round(self.frac_saturated_conf, 3),
            "conv_rate": round(self.basin_convergence_rate, 3),
            "consist_basins": round(self.mean_consistent_basins, 2),
            "compete_basins": round(self.mean_competing_basins, 2),
            "n_calls": self.n_model_calls,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ExtendedEvalResult({self.method!r}: "
            f"acc={self.accuracy:.3f}, ece={self.ece:.3f}, "
            f"basins={self.mean_basins:.1f}, neff={self.mean_neff:.2f}, "
            f"escape={self.escape_rate:.3f}, conf={self.mean_confidence:.3f}, "
            f"sat={self.frac_saturated_conf:.3f})"
        )


def compute_extended_eval(
    results: List[AnswerResult],
    problems: List[Problem],
    method: str = "unknown",
    n_model_calls: int = 0,
    first_answers: Optional[List[str]] = None,
) -> ExtendedEvalResult:
    """
    Compute all extended metrics for a list of results.

    Parameters
    ----------
    results:
        List of :class:`~basin.confidence.estimator.AnswerResult` objects.
    problems:
        Corresponding ground-truth problems.
    method:
        Human-readable method name (for reports).
    n_model_calls:
        Total number of LLM API calls made.
    first_answers:
        Optional first-step answers for accurate escape-rate computation.
    """
    from basin.metrics.metrics import (
        accuracy as _acc,
        expected_calibration_error as _ece,
    )

    per_problem_diags = [
        per_problem_extended(
            r, p,
            first_answer=first_answers[i] if first_answers else None,
        )
        for i, (r, p) in enumerate(zip(results, problems))
    ]

    neff_vals = [effective_basin_count(r.visit_history) for r in results]

    return ExtendedEvalResult(
        method=method,
        n_problems=len(results),
        accuracy=_acc(results, problems),
        ece=_ece(results, problems),
        mean_basins=float(np.mean([r.n_basins_explored for r in results])) if results else 0.0,
        mean_neff=float(np.mean(neff_vals)) if neff_vals else 0.0,
        escape_rate=strict_escape_rate(results, problems, first_answers),
        mean_confidence=float(np.mean([r.confidence for r in results])) if results else 0.0,
        frac_saturated_conf=fraction_saturated_confidence(results),
        basin_convergence_rate=basin_convergence_rate(results),
        mean_consistent_basins=mean_consistent_basins(results, problems),
        mean_competing_basins=mean_competing_basins(results),
        n_model_calls=n_model_calls,
        per_problem=per_problem_diags,
    )
