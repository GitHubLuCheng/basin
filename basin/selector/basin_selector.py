"""
basin/selector/basin_selector.py
=====================================
Final-basin selector for MetaDynamics reasoning.

After the metadynamics exploration loop, each discovered basin B is scored:

    S(B) = alpha * log(1 + visits_B)
           + beta  * verifier_score(B)
           + gamma * internal_consistency(B)

where:
  visits_B             — number of selected states assigned to basin B
  verifier_score(B)    — mean HeuristicVerifier score of states in B  (∈ [0,1])
  internal_consistency(B) — mean pairwise NLI E-C score within B,
                           mapped to [0,1] via (score+1)/2
                           (= 1.0 for singleton basins, = 0.5 when NLI unavail)

The basin with the highest S(B) is selected; its dominant answer is predicted.

Four preset selectors are exposed via PRESET_SELECTORS:
  "current"        alpha=1, beta=0, gamma=0  (mass-only ≈ majority vote)
  "mass_verifier"  alpha=1, beta=1, gamma=0
  "full"           alpha=1, beta=1, gamma=1  (the new composite scorer)
  "verifier_only"  alpha=0, beta=1, gamma=0
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from basin.state.reasoning_state import ReasoningState
from basin.verifier.verifier import HeuristicVerifier


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelectorResult:
    """Per-problem result for one basin selector."""

    name: str
    predicted: str          # predicted answer letter (lowercase) or "?"
    correct: bool
    confidence: float       # softmax-normalised score of selected basin ∈ [0,1]
    selected_basin_id: int  # basin id chosen by this selector
    most_visited_basin_id: int  # basin with highest visit count
    differs_from_most_visited: bool  # True when selected ≠ most-visited basin

    # Set to True when ≥1 basin contains the correct answer AND this selector
    # also selects a correct-answer basin.  Set to False when ≥1 correct basin
    # exists but this selector picks a wrong one.  None when pass@k is False.
    correct_when_correct_exists: Optional[bool]


# ---------------------------------------------------------------------------
# BasinSelector
# ---------------------------------------------------------------------------

_VERIFIER = HeuristicVerifier()


class BasinSelector:
    """
    Scores basins and selects the best one.

    Parameters
    ----------
    alpha:
        Weight of log-visit term (exploration mass).
    beta:
        Weight of mean verifier score term (reasoning quality).
    gamma:
        Weight of internal consistency term (NLI coherence within basin).
    name:
        Human-readable label for this selector variant.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        name: str = "custom",
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.name = name

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def select(
        self,
        history: List[ReasoningState],
        state_answers: List[str],
        reference: str,
        nli_pairwise: Optional[np.ndarray] = None,
    ) -> SelectorResult:
        """
        Apply this selector to a completed MetaDyn run.

        Parameters
        ----------
        history:
            Ordered list of selected ReasoningState objects (from
            ``MetadynamicsController.selected_history()``).  Each state must
            have a valid ``.basin_id`` attribute.
        state_answers:
            Answer extracted for each state in *history* (same length).
            Lowercase letter or "?".
        reference:
            Ground-truth answer letter (lowercase or uppercase).
        nli_pairwise:
            Square (N×N) NLI E-C score matrix for the *history* states
            (from ``NLIBasinClusterer.last_pairwise``).  May be None.

        Returns
        -------
        SelectorResult
        """
        if not history:
            return SelectorResult(
                name=self.name,
                predicted="?",
                correct=False,
                confidence=0.0,
                selected_basin_id=-1,
                most_visited_basin_id=-1,
                differs_from_most_visited=False,
                correct_when_correct_exists=None,
            )

        ref_norm = reference.lower().strip()

        # Group states / answers by basin.
        basin_states: Dict[int, List[int]] = defaultdict(list)   # bid -> indices
        for i, state in enumerate(history):
            basin_states[state.basin_id].append(i)

        # Most-visited basin.
        most_visited_bid = max(basin_states, key=lambda b: len(basin_states[b]))

        # Compute per-basin score and dominant answer.
        basin_score: Dict[int, float] = {}
        basin_answer: Dict[int, str] = {}

        for bid, idxs in basin_states.items():
            visits = len(idxs)
            states_b = [history[i] for i in idxs]
            answers_b = [state_answers[i] for i in idxs if i < len(state_answers)]

            # ---- verifier score ----
            v_score = float(np.mean([_VERIFIER.score(s) for s in states_b]))

            # ---- internal consistency ----
            consistency = self._basin_consistency(bid, idxs, nli_pairwise)

            # ---- composite score ----
            score = (
                self.alpha * math.log(1 + visits)
                + self.beta * v_score
                + self.gamma * consistency
            )
            basin_score[bid] = score

            # Dominant answer for this basin.
            valid_ans = [a for a in answers_b if a not in ("?", "")]
            if valid_ans:
                basin_answer[bid] = Counter(valid_ans).most_common(1)[0][0]
            elif answers_b:
                basin_answer[bid] = answers_b[0]
            else:
                basin_answer[bid] = "?"

        # Select best basin.
        best_bid = max(basin_score, key=basin_score.__getitem__)
        predicted = basin_answer.get(best_bid, "?")

        # Confidence via softmax over basin scores.
        scores_arr = np.array(list(basin_score.values()), dtype=float)
        scores_arr -= scores_arr.max()          # numerical stability
        exp_scores = np.exp(scores_arr)
        exp_sum = exp_scores.sum()
        bids_list = list(basin_score.keys())
        best_pos = bids_list.index(best_bid)
        confidence = float(exp_scores[best_pos] / exp_sum) if exp_sum > 1e-12 else 0.0

        correct = (predicted.lower() == ref_norm)
        differs = (best_bid != most_visited_bid)

        # correct_when_correct_exists:
        pass_k = any(a.lower() == ref_norm for a in state_answers if a not in ("?", ""))
        if pass_k:
            correct_when_correct_exists: Optional[bool] = correct
        else:
            correct_when_correct_exists = None

        return SelectorResult(
            name=self.name,
            predicted=predicted,
            correct=correct,
            confidence=confidence,
            selected_basin_id=best_bid,
            most_visited_basin_id=most_visited_bid,
            differs_from_most_visited=differs,
            correct_when_correct_exists=correct_when_correct_exists,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _basin_consistency(
        self,
        bid: int,
        idxs: List[int],
        nli_pairwise: Optional[np.ndarray],
    ) -> float:
        """
        Internal consistency of basin *bid* ∈ [0, 1].

        For singleton basins or when NLI data is unavailable, returns a
        neutral value (1.0 for singletons, 0.5 otherwise).

        NLI E-C scores lie in [-1, 1]; they are linearly mapped to [0, 1]
        via (score + 1) / 2 before averaging.
        """
        if len(idxs) == 1:
            return 1.0      # trivially consistent

        if nli_pairwise is None:
            return 0.5

        n_mat = nli_pairwise.shape[0]
        nli_vals: List[float] = []
        for i in idxs:
            for j in idxs:
                if i == j:
                    continue
                if i >= n_mat or j >= n_mat:
                    continue
                s = float(nli_pairwise[i, j])
                if s > -1.5:            # skip sentinel −2.0
                    nli_vals.append(s)

        if not nli_vals:
            return 0.5

        mean_nli = float(np.mean(nli_vals))
        return (mean_nli + 1.0) / 2.0   # map [-1,1] → [0,1]

    def __repr__(self) -> str:
        return (
            f"BasinSelector(name={self.name!r}, "
            f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})"
        )


# ---------------------------------------------------------------------------
# Preset selector variants
# ---------------------------------------------------------------------------

PRESET_SELECTORS: Dict[str, BasinSelector] = {
    # current: pure mass → equivalent to majority-vote at basin level
    "current": BasinSelector(alpha=1.0, beta=0.0, gamma=0.0, name="current"),
    # mass + verifier quality
    "mass_verifier": BasinSelector(alpha=1.0, beta=1.0, gamma=0.0, name="mass_verifier"),
    # full composite: mass + verifier + NLI internal consistency
    "full": BasinSelector(alpha=1.0, beta=1.0, gamma=1.0, name="full"),
    # verifier only: pick the basin with the best average reasoning quality
    "verifier_only": BasinSelector(alpha=0.0, beta=1.0, gamma=0.0, name="verifier_only"),
}
