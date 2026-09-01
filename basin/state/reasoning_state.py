"""
Core data structure representing an observable reasoning state z_t.

z_t = phi(y_{<=t}, c_t, v_t)

where:
  y_{<=t}  — reasoning trace text generated so far
  c_t      — confidence signal (logprobs or None)
  v_t      — verifier signals (self-eval, contradiction, answer support)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Re-export for convenience so callers only need to import from state.
from basin.llm.base import LogprobStats


@dataclass
class ReasoningState:
    """
    Observable state of a reasoning trajectory at step *t*.

    This is the central object that the metadynamics controller tracks,
    stores in basin memory, and uses to compute the revisit bias.

    Attributes
    ----------
    question:
        The original question being answered.
    trace_text:
        The full chain-of-thought text generated so far.
    provisional_answer:
        The current best guess extracted from *trace_text*.
    key_assumptions:
        Short strings summarising the main assumptions the trace makes.
    unresolved_issues:
        Detected gaps, contradictions, or open sub-problems.
    self_eval_score:
        Float in [0, 1] rating the internal quality of the reasoning.
        Higher = more confident / consistent.
    contradiction_score:
        Float in [0, 1] measuring detected contradictions.
        Higher = more contradictions (bad).
    logprob_stats:
        Token log-probability statistics from the LLM backend, or None.
    embedding_vector:
        Dense vector representation used for similarity / bias computation.
        Set by :class:`~basin.embedding.base.BaseEmbedder` after creation.
    step_index:
        Which metadynamics iteration produced this state.
    basin_id:
        Cluster / basin index assigned after clustering. -1 = unassigned.
    """

    question: str
    trace_text: str
    provisional_answer: str
    key_assumptions: List[str] = field(default_factory=list)
    unresolved_issues: List[str] = field(default_factory=list)
    self_eval_score: float = 0.5
    contradiction_score: float = 0.0
    logprob_stats: Optional[LogprobStats] = None
    embedding_vector: Optional[np.ndarray] = field(default=None, repr=False)
    step_index: int = 0
    basin_id: int = -1

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def confidence_proxy(self) -> float:
        """
        Composite confidence in [0, 1].

        Combines self-evaluation and logprob signal when available.
        """
        if self.logprob_stats is not None:
            import math
            lp_conf = 1.0 / (1.0 + math.exp(self.logprob_stats.mean + 1.0))
            return 0.5 * self.self_eval_score + 0.5 * lp_conf
        return self.self_eval_score

    @property
    def quality_score(self) -> float:
        """Heuristic quality = confidence minus contradiction penalty."""
        return self.confidence_proxy - 0.5 * self.contradiction_score

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ReasoningState(answer={self.provisional_answer!r}, "
            f"self_eval={self.self_eval_score:.2f}, "
            f"contradiction={self.contradiction_score:.2f}, "
            f"basin={self.basin_id}, step={self.step_index})"
        )
