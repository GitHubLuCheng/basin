"""
Metadynamics controller — the main exploration loop.

Algorithm per iteration t:
  1. Generate n_candidates reasoning traces.
  2. Extract state z_i for each candidate.
  3. Embed each z_i.
  4. Compute score_i = alpha * verifier(z_i) + beta * confidence(z_i)
                       - lambda_ * bias(z_i)
  5. Select the highest-scoring candidate z*.
  6. Add z* to basin memory.
  7. Repeat for T rounds.

After T rounds, aggregate the final answer via the confidence estimator.

The history-dependent bias discourages the controller from repeatedly
selecting states in already-explored basins, driving exploration toward
novel regions of reasoning space — exactly the role of the Gaussian hills
in classical metadynamics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from basin.confidence.estimator import AnswerResult, ConfidenceEstimator
from basin.embedding.base import BaseEmbedder
from basin.memory.basin_memory import BasinMemory
from basin.state.reasoning_state import ReasoningState
from basin.verifier.verifier import BaseVerifier
from .candidate_gen import CandidateGenerator

logger = logging.getLogger(__name__)


@dataclass
class MetadynamicsConfig:
    """
    Hyper-parameters for the metadynamics reasoning loop.

    Attributes
    ----------
    alpha:
        Weight of the verifier score in the candidate selection objective.
    beta:
        Weight of the confidence (logprob) signal.
    lambda_:
        Strength of the history-dependent revisit penalty.
        lambda_=0 reduces to verifier reranking with no exploration pressure.
    n_rounds:
        Number of metadynamics iterations.
    n_candidates:
        Number of candidates generated per round.
    temperature:
        LLM sampling temperature.
    use_continuation:
        If True, after round 0 pass the previous best trace as context
        to the LLM, encouraging it to explore from that starting point.
    """

    alpha: float = 1.0
    beta: float = 0.5
    lambda_: float = 1.5
    n_rounds: int = 8
    n_candidates: int = 4
    temperature: float = 0.8
    use_continuation: bool = False
    quality_aware: bool = False


class MetadynamicsController:
    """
    Orchestrates the metadynamics reasoning exploration loop.

    Parameters
    ----------
    candidate_generator:
        Generates and embeds candidate reasoning states.
    verifier:
        Scores each candidate for correctness likelihood.
    memory:
        Basin memory that stores visited states and computes bias.
    confidence_estimator:
        Aggregates basin memory into a final answer.
    config:
        Hyper-parameters.
    """

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        verifier: BaseVerifier,
        memory: BasinMemory,
        confidence_estimator: ConfidenceEstimator,
        config: Optional[MetadynamicsConfig] = None,
    ) -> None:
        self.gen = candidate_generator
        self.verifier = verifier
        self.memory = memory
        self.estimator = confidence_estimator
        self.cfg = config or MetadynamicsConfig()

        # Bookkeeping for analysis.
        self._score_history: List[List[float]] = []   # per round
        self._selected_history: List[ReasoningState] = []
        self._last_verifier_scores: List[float] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        question: str,
        n_rounds: Optional[int] = None,
        n_candidates: Optional[int] = None,
    ) -> AnswerResult:
        """
        Run the full metadynamics loop and return the final answer.

        Parameters
        ----------
        question:
            The problem to solve.
        n_rounds:
            Override :attr:`MetadynamicsConfig.n_rounds`.
        n_candidates:
            Override :attr:`MetadynamicsConfig.n_candidates`.
        """
        n_rounds = n_rounds or self.cfg.n_rounds
        n_candidates = n_candidates or self.cfg.n_candidates

        # Reset state for a fresh run.
        self.memory.__init__(
            kernel_bandwidth=self.memory.kernel_bandwidth,
            base_weight=self.memory.base_weight,
            count_scaling=self.memory.count_scaling,
            clusterer=self.memory.clusterer,
        )
        self._score_history = []
        self._selected_history = []

        previous_trace: Optional[str] = None

        for round_idx in range(n_rounds):
            logger.debug("Round %d/%d", round_idx + 1, n_rounds)

            candidates = self.gen.generate_candidates(
                question=question,
                n=n_candidates,
                step_index=round_idx,
                previous_trace=previous_trace if self.cfg.use_continuation else None,
            )

            # Re-embed all stored memory states with the current vocabulary
            # so basin assignments stay consistent as the vocabulary grows.
            self.memory.reembed_states(self.gen.embedder)

            scores = self._score_candidates(candidates)
            self._score_history.append(scores)

            best_idx = int(np.argmax(scores))
            best_state = candidates[best_idx]
            best_quality = self._last_verifier_scores[best_idx] if self.cfg.quality_aware else 0.0
            self.memory.add_state(best_state, quality=best_quality)
            self._selected_history.append(best_state)

            if self.cfg.use_continuation:
                previous_trace = best_state.trace_text

            logger.debug(
                "  Selected: answer=%r score=%.3f basin=%d",
                best_state.provisional_answer,
                scores[best_idx],
                best_state.basin_id,
            )

        return self.estimator.aggregate(self.memory)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_candidates(self, candidates: List[ReasoningState]) -> List[float]:
        """
        Compute the metadynamics score for each candidate:

            score(z) = alpha * verifier(z) + beta * confidence(z)
                       - lambda_ * bias(z)

        When ``cfg.quality_aware`` is set, bias(z) is scaled by
        ``(1 - q(z))``, where q(z) is the kernel-weighted average quality
        (verifier score) of nearby stored states -- the continuous analogue
        of Eq. 6 for this kernel-based memory (see
        ``BasinMemory.compute_quality``).
        """
        cfg = self.cfg
        verifier_scores = self.verifier.score_batch(candidates)
        self._last_verifier_scores = verifier_scores

        # Vectorised bias computation.
        embeddings = np.vstack(
            [s.embedding_vector for s in candidates if s.embedding_vector is not None]
        )
        if len(embeddings) == len(candidates):
            biases = self.memory.compute_bias_batch(embeddings)
            qualities = (self.memory.compute_quality_batch(embeddings)
                         if cfg.quality_aware else None)
        else:
            biases = np.array([self.memory.compute_bias(s) for s in candidates])
            qualities = (np.array([self.memory.compute_quality(s) for s in candidates])
                         if cfg.quality_aware else None)

        scores = []
        for i, state in enumerate(candidates):
            v_score = verifier_scores[i]
            c_score = state.confidence_proxy
            b_score = float(biases[i]) if i < len(biases) else 0.0
            if cfg.quality_aware and qualities is not None:
                q = float(qualities[i]) if i < len(qualities) else 0.0
                b_score = b_score * (1.0 - q)
            s = cfg.alpha * v_score + cfg.beta * c_score - cfg.lambda_ * b_score
            scores.append(s)

        return scores

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def score_history(self) -> List[List[float]]:
        """Per-round list of candidate scores (for plotting / ablation)."""
        return self._score_history

    def selected_history(self) -> List[ReasoningState]:
        """Sequence of selected states across all rounds."""
        return self._selected_history

    def bias_evolution(self) -> List[float]:
        """Bias value of the selected state at each round."""
        return [self.memory.compute_bias(s) for s in self._selected_history]
