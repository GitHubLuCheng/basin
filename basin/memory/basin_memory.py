"""
Basin memory — the core data structure for metadynamics.

This is the analogue of the Gaussian hill-sum history in classical
metadynamics.  Instead of adding Gaussian hills in CV space, we
accumulate a kernel-weighted penalty over the embedding space of
reasoning states.

The revisit bias at a query state z is:

    bias(z) = sum_j  w_j * K(z, z_j)

where:
  z_j are all previously selected states
  w_j = weight based on visitation count and failure history
  K   is the Gaussian kernel on cosine similarity
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from basin.embedding.similarity import gaussian_kernel, pairwise_gaussian_kernel
from basin.state.reasoning_state import ReasoningState
from .clustering import BasinClusterer


@dataclass
class BasinStats:
    """Aggregate statistics for a single basin."""

    basin_id: int
    visit_count: int
    dominant_answer: str
    mean_verifier_score: float
    mean_self_eval: float
    state_indices: List[int] = field(default_factory=list)

    @property
    def pseudo_free_energy(self) -> float:
        """
        Approximate free energy: F(B) = -log p_hat(B).

        p_hat(B) is estimated as visit_count / total visits.
        We return the unnormalised log-count for basin-relative comparisons.
        """
        return -math.log(max(self.visit_count, 1))


class BasinMemory:
    """
    Tracks visited reasoning states and computes the metadynamics revisit bias.

    Parameters
    ----------
    kernel_bandwidth:
        Bandwidth sigma of the Gaussian kernel K(z, z').
        Smaller → sharper, more localised penalty (distinguishes basins
        with less overlap); larger → broader suppression.
    base_weight:
        Weight w_0 for a state visited once.
    count_scaling:
        How quickly weights grow with revisit count.
        w_j = base_weight * (1 + count_scaling * log(visit_count_j)).
    clusterer:
        Basin clustering algorithm.  If None, a default threshold
        clusterer is used.
    """

    def __init__(
        self,
        kernel_bandwidth: float = 0.5,
        base_weight: float = 1.0,
        count_scaling: float = 0.5,
        clusterer: Optional[BasinClusterer] = None,
    ) -> None:
        self.kernel_bandwidth = kernel_bandwidth
        self.base_weight = base_weight
        self.count_scaling = count_scaling
        self.clusterer = clusterer or BasinClusterer(algorithm="threshold")

        # All selected states, in order of selection.
        self.states: List[ReasoningState] = []
        # Cached basin assignment array (length = len(states)).
        self._basin_labels: np.ndarray = np.array([], dtype=int)
        # Per-basin visit counts: basin_id -> count
        self._basin_counts: Dict[int, int] = defaultdict(int)
        # Per-state weights (used in bias sum).
        self._weights: List[float] = []
        # Per-state quality proxy (only used when quality-aware scoring is
        # enabled; e.g. QA-BASIN's Eq. 6 kernel-weighted analogue).
        self._quality: List[float] = []

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def add_state(self, state: ReasoningState, quality: float = 0.0) -> None:
        """
        Add a newly selected state to memory and recluster.

        The basin ID of *state* is set in-place after clustering.

        *quality* is an optional per-state quality proxy in [0, 1] (e.g. the
        verifier score that selected it), only consumed by
        ``compute_quality``/``compute_quality_batch`` for quality-aware
        (QA-BASIN) scoring; ignored by the standard bias computation.
        """
        self.states.append(state)
        self._recluster()
        # Update visit count for the state's assigned basin.
        bid = state.basin_id
        self._basin_counts[bid] += 1
        count = self._basin_counts[bid]
        w = self.base_weight * (1.0 + self.count_scaling * math.log(count))
        self._weights.append(w)
        self._quality.append(quality)

    def reembed_states(self, embedder) -> None:
        """
        Re-embed all stored states with *embedder* and recluster.

        Call this after the TF-IDF vocabulary has grown (i.e., after each
        round of candidate generation) to ensure all stored embeddings are
        in the same vector space.  Without this, spurious basin splits occur
        because the same text gets different-dimensional vectors at different
        times as the vocabulary expands.
        """
        if not self.states:
            return
        texts = [s.trace_text for s in self.states]
        new_embeddings = embedder.embed_batch(texts)
        for i, state in enumerate(self.states):
            state.embedding_vector = new_embeddings[i]
        self._recluster()

    def _recluster(self) -> None:
        """Recompute basin assignments for all stored states."""
        embeddings = self._get_embeddings()
        if embeddings is None or embeddings.shape[0] == 0:
            return
        labels = self.clusterer.fit_predict(embeddings)
        self._basin_labels = labels
        for i, state in enumerate(self.states):
            state.basin_id = int(labels[i])

    def _get_embeddings(self) -> Optional[np.ndarray]:
        vecs = [s.embedding_vector for s in self.states if s.embedding_vector is not None]
        if not vecs:
            return None
        # Pad all vectors to the same (maximum) dimension before stacking.
        # This handles the case where TF-IDF vocabulary has grown over time.
        max_dim = max(v.shape[0] for v in vecs)
        padded = [
            np.pad(v, (0, max_dim - v.shape[0])) if v.shape[0] < max_dim else v
            for v in vecs
        ]
        return np.vstack(padded)

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def compute_bias(self, query_state: ReasoningState) -> float:
        """
        Compute the history-dependent revisit penalty for *query_state*.

            bias(z) = sum_j  w_j * K(z, z_j)

        Returns 0.0 if memory is empty or the state has no embedding.
        """
        if not self.states or query_state.embedding_vector is None:
            return 0.0

        q_vec = query_state.embedding_vector
        total = 0.0
        for state, w in zip(self.states, self._weights):
            if state.embedding_vector is None:
                continue
            m_vec = state.embedding_vector
            # Align dimensions if vocabulary has grown.
            if q_vec.shape[0] != m_vec.shape[0]:
                target = max(q_vec.shape[0], m_vec.shape[0])
                q_vec = np.pad(q_vec, (0, target - q_vec.shape[0]))
                m_vec = np.pad(m_vec, (0, target - m_vec.shape[0]))
            k = gaussian_kernel(q_vec, m_vec, self.kernel_bandwidth)
            total += w * k
        return total

    def compute_quality(self, query_state: ReasoningState) -> float:
        """
        Kernel-weighted average quality near *query_state*, the continuous
        analogue of Eq. 6's per-basin q_B for this kernel-based memory (as
        opposed to the discrete visit-count schemes used elsewhere in the
        repo). Returns 0.0 if memory is empty (no bias reduction, same as
        an unvisited basin having q_B=0 elsewhere).

            q(z) = [ sum_j w_j * K(z, z_j) * quality_j ] / [ sum_j w_j * K(z, z_j) ]
        """
        if not self.states or query_state.embedding_vector is None:
            return 0.0

        q_vec = query_state.embedding_vector
        num, den = 0.0, 0.0
        for state, w, quality in zip(self.states, self._weights, self._quality):
            if state.embedding_vector is None:
                continue
            m_vec = state.embedding_vector
            if q_vec.shape[0] != m_vec.shape[0]:
                target = max(q_vec.shape[0], m_vec.shape[0])
                q_vec = np.pad(q_vec, (0, target - q_vec.shape[0]))
                m_vec = np.pad(m_vec, (0, target - m_vec.shape[0]))
            k = gaussian_kernel(q_vec, m_vec, self.kernel_bandwidth)
            num += w * k * quality
            den += w * k
        return num / den if den > 1e-12 else 0.0

    def compute_quality_batch(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Vectorised analogue of ``compute_quality`` (see ``compute_bias_batch``)."""
        memory_vecs = self._get_embeddings()
        if memory_vecs is None or not self._quality:
            return np.zeros(query_embeddings.shape[0])

        q_dim = query_embeddings.shape[1]
        m_dim = memory_vecs.shape[1]
        if q_dim != m_dim:
            target = max(q_dim, m_dim)
            if q_dim < target:
                query_embeddings = np.pad(query_embeddings, ((0, 0), (0, target - q_dim)))
            if m_dim < target:
                memory_vecs = np.pad(memory_vecs, ((0, 0), (0, target - m_dim)))

        weights = np.array(self._weights[: memory_vecs.shape[0]])
        quality = np.array(self._quality[: memory_vecs.shape[0]])
        q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        m_norms = np.linalg.norm(memory_vecs, axis=1, keepdims=True)
        q_n = np.where(q_norms < 1e-9, 1.0, q_norms)
        m_n = np.where(m_norms < 1e-9, 1.0, m_norms)
        cos_sim = (query_embeddings / q_n) @ (memory_vecs / m_n).T
        dist_sq = (1.0 - cos_sim) ** 2
        kernel_matrix = np.exp(-dist_sq / (2.0 * self.kernel_bandwidth ** 2))
        num = kernel_matrix @ (weights * quality)
        den = kernel_matrix @ weights
        return np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), 0.0)

    def compute_bias_batch(self, query_embeddings: np.ndarray) -> np.ndarray:
        """
        Vectorised bias computation for multiple query states at once.

        Parameters
        ----------
        query_embeddings:
            Array of shape ``(M, D)``.

        Returns
        -------
        np.ndarray of shape ``(M,)`` with one bias value per query.
        """
        memory_vecs = self._get_embeddings()
        if memory_vecs is None:
            return np.zeros(query_embeddings.shape[0])

        # Align dimensions: the TF-IDF vocabulary may have grown since
        # memory states were embedded, causing dimension mismatches.
        # Pad the shorter dimension with zeros (safe because new TF-IDF
        # dimensions correspond to unseen vocabulary → zero contribution).
        q_dim = query_embeddings.shape[1]
        m_dim = memory_vecs.shape[1]
        if q_dim != m_dim:
            target = max(q_dim, m_dim)
            if q_dim < target:
                query_embeddings = np.pad(
                    query_embeddings, ((0, 0), (0, target - q_dim))
                )
            if m_dim < target:
                memory_vecs = np.pad(
                    memory_vecs, ((0, 0), (0, target - m_dim))
                )

        weights = np.array(self._weights[: memory_vecs.shape[0]])
        q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        m_norms = np.linalg.norm(memory_vecs, axis=1, keepdims=True)
        q_n = np.where(q_norms < 1e-9, 1.0, q_norms)
        m_n = np.where(m_norms < 1e-9, 1.0, m_norms)
        cos_sim = (query_embeddings / q_n) @ (memory_vecs / m_n).T
        dist_sq = (1.0 - cos_sim) ** 2
        kernel_matrix = np.exp(-dist_sq / (2.0 * self.kernel_bandwidth ** 2))
        return kernel_matrix @ weights

    # ------------------------------------------------------------------
    # Basin-level aggregation
    # ------------------------------------------------------------------

    def get_basin_stats(self) -> List[BasinStats]:
        """Return summary statistics for each discovered basin."""
        if not self.states:
            return []

        basin_groups: Dict[int, List[ReasoningState]] = defaultdict(list)
        for i, state in enumerate(self.states):
            basin_groups[state.basin_id].append(state)

        stats = []
        for bid, group in sorted(basin_groups.items()):
            answers = [s.provisional_answer for s in group]
            dominant = max(set(answers), key=answers.count)
            stats.append(
                BasinStats(
                    basin_id=bid,
                    visit_count=self._basin_counts[bid],
                    dominant_answer=dominant,
                    mean_verifier_score=float(
                        np.mean([s.self_eval_score for s in group])
                    ),
                    mean_self_eval=float(
                        np.mean([s.self_eval_score for s in group])
                    ),
                    state_indices=[
                        i for i, s in enumerate(self.states) if s.basin_id == bid
                    ],
                )
            )
        return stats

    def n_basins(self) -> int:
        """Number of distinct basins discovered so far."""
        if len(self._basin_labels) == 0:
            return 0
        return int(np.max(self._basin_labels)) + 1

    def visit_history(self) -> List[int]:
        """Ordered list of basin IDs visited at each step."""
        return [s.basin_id for s in self.states]

    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BasinMemory(n_states={len(self)}, "
            f"n_basins={self.n_basins()}, "
            f"bandwidth={self.kernel_bandwidth})"
        )
