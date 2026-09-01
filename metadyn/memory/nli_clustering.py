"""
NLI-based basin clustering for MuSR structured reasoning states.

Unlike SBERT cosine similarity, this clusterer uses a cross-encoder NLI model
to decide whether two reasoning states belong to the same basin.

Basin-membership rule
---------------------
States i and j are in the **same basin** iff:
  1. ``final_answer[i] == final_answer[j]``   (hard gate)
  2. ``score(i, j) > entail_threshold``        (soft NLI gate)

where:
  score(i,j) = 0.5*(E_ij + E_ji) - 0.5*(C_ij + C_ji)
  E_ij = P(entailment | premise=hyp_i, hypothesis=hyp_j)
  C_ij = P(contradiction | ...)

score ∈ [-1, 1].  Positive → similar reasoning.  Negative → conflicting.

The text compared is ``main_hypothesis`` (or ``reasoning_summary`` as fallback).
If structured extraction failed the state gets its own singleton basin.

Model
-----
Default: ``cross-encoder/nli-deberta-v3-small``
  label order: {0: contradiction, 1: entailment, 2: neutral}
  fast on CPU (~8 ms/pair on MacBook M-series)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Label indices for cross-encoder/nli-deberta-v3-small
_IDX_CONTRADICTION = 0
_IDX_ENTAILMENT = 1
_IDX_NEUTRAL = 2


class NLIBasinClusterer:
    """
    Basin clusterer that uses NLI entailment instead of cosine similarity.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for a sequence-classification NLI model.
    entail_threshold:
        Minimum ``score(i,j)`` to merge two same-answer states into one basin.
        Range [-1, 1].  Default 0.5 is conservative.
    max_contradict:
        Hard cap on mean contradiction probability — even if score passes the
        threshold, two states with average contradiction > this are kept separate.
    text_field:
        Which structured field to compare: ``"main_hypothesis"`` or
        ``"reasoning_summary"``.  If the chosen field is empty, falls back to
        the other.
    batch_size:
        Number of NLI pairs to process per forward pass.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entail_threshold: float = 0.45,
        max_contradict: float = 0.3,
        text_field: str = "main_hypothesis",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.entail_threshold = entail_threshold
        self.max_contradict = max_contradict
        self.text_field = text_field
        self.batch_size = batch_size

        self._tok = None
        self._mdl = None

        # Holds the structured states for the next fit_predict_from_states call.
        # Set by NLIBasinMemory._recluster() before calling fit_predict().
        self._pending_states: Optional[list] = None

        # Diagnostic storage — populated per clustering call.
        self.last_pairwise: Optional[np.ndarray] = None   # (N,N) score matrix
        self.last_contradict: Optional[np.ndarray] = None # (N,N) mean contradiction

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._tok is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info("Loading NLI model %s …", self.model_name)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._mdl.eval()
        self._torch = torch
        logger.info("NLI model loaded.")

    # ------------------------------------------------------------------
    # Main clustering entry points
    # ------------------------------------------------------------------

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compatibility shim for BasinMemory.  Ignores embeddings; uses
        ``self._pending_states`` set by NLIBasinMemory before this call.

        Falls back to a single-basin assignment if no states are pending.
        """
        if self._pending_states is None:
            n = embeddings.shape[0]
            return np.arange(n, dtype=int)  # each state its own basin
        states = self._pending_states
        self._pending_states = None
        return self.fit_predict_from_states(states)

    def fit_predict_from_states(self, states: list) -> np.ndarray:
        """
        Assign basin IDs given a list of ReasoningState objects that have
        ``structured_state`` attributes.

        Parameters
        ----------
        states:
            List of ReasoningState.  Each should have ``.structured_state`` dict
            with keys ``final_answer``, ``main_hypothesis``, ``reasoning_summary``,
            ``extraction_failed``.

        Returns
        -------
        np.ndarray of shape (N,) with integer basin IDs.
        """
        n = len(states)
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)

        texts = [self._get_text(s) for s in states]
        answers = [self._get_answer(s) for s in states]
        failed = [self._is_failed(s) for s in states]

        # Compute pairwise NLI scores only for same-answer, non-failed pairs.
        scores = np.full((n, n), -2.0)   # sentinel: not computed
        contrad = np.zeros((n, n))
        np.fill_diagonal(scores, 1.0)    # self-similarity = 1

        pairs_to_run = []
        for i in range(n):
            for j in range(i + 1, n):
                if failed[i] or failed[j]:
                    scores[i, j] = scores[j, i] = -2.0  # force separate
                elif answers[i] != answers[j]:
                    scores[i, j] = scores[j, i] = -2.0  # different answer → separate
                else:
                    pairs_to_run.append((i, j))

        if pairs_to_run:
            self._load()
            s_mat, c_mat = self._nli_batch(texts, pairs_to_run)
            for idx, (i, j) in enumerate(pairs_to_run):
                scores[i, j] = scores[j, i] = s_mat[idx]
                contrad[i, j] = contrad[j, i] = c_mat[idx]

        self.last_pairwise = scores
        self.last_contradict = contrad

        # Greedy clustering: merge if score > threshold AND contradiction low.
        labels = np.full(n, -1, dtype=int)
        next_basin = 0
        for i in range(n):
            if failed[i]:
                labels[i] = next_basin
                next_basin += 1
                continue
            best = -1
            for b in range(next_basin):
                # Find any member of basin b.
                members = np.where(labels[:i] == b)[0]
                if len(members) == 0:
                    continue
                rep = members[0]
                s = scores[i, rep]
                c = contrad[i, rep]
                if s > self.entail_threshold and c < self.max_contradict:
                    best = b
                    break
            if best >= 0:
                labels[i] = best
            else:
                labels[i] = next_basin
                next_basin += 1

        return labels

    # ------------------------------------------------------------------
    # NLI forward pass
    # ------------------------------------------------------------------

    def _nli_batch(
        self,
        texts: List[str],
        pairs: List[tuple],
    ):
        """
        Run NLI for each pair (i,j) in both directions (i→j and j→i).

        Returns
        -------
        scores : list of float   symmetric score(i,j)
        contrad: list of float   mean contradiction
        """
        import torch

        # Build forward + backward batch.
        premises, hypotheses = [], []
        for i, j in pairs:
            premises.append(texts[i])
            hypotheses.append(texts[j])
            premises.append(texts[j])
            hypotheses.append(texts[i])

        all_probs = []
        for start in range(0, len(premises), self.batch_size):
            P = premises[start:start + self.batch_size]
            H = hypotheses[start:start + self.batch_size]
            enc = self._tok(
                P, H,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            with torch.no_grad():
                logits = self._mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())

        scores, contrad = [], []
        for k in range(len(pairs)):
            p_fwd = all_probs[2 * k]      # premise=i, hyp=j
            p_bwd = all_probs[2 * k + 1]  # premise=j, hyp=i
            e_fwd = p_fwd[_IDX_ENTAILMENT]
            e_bwd = p_bwd[_IDX_ENTAILMENT]
            c_fwd = p_fwd[_IDX_CONTRADICTION]
            c_bwd = p_bwd[_IDX_CONTRADICTION]
            score = 0.5 * (e_fwd + e_bwd) - 0.5 * (c_fwd + c_bwd)
            c_mean = 0.5 * (c_fwd + c_bwd)
            scores.append(float(score))
            contrad.append(float(c_mean))

        return scores, contrad

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_text(self, state) -> str:
        struct = getattr(state, "structured_state", {})
        t = struct.get(self.text_field, "").strip()
        if not t:
            # Fallback to other field.
            alt = "reasoning_summary" if self.text_field == "main_hypothesis" else "main_hypothesis"
            t = struct.get(alt, "").strip()
        return t or "unknown"

    def _get_answer(self, state) -> str:
        struct = getattr(state, "structured_state", {})
        return struct.get("final_answer", "?").upper()

    def _is_failed(self, state) -> bool:
        struct = getattr(state, "structured_state", {})
        # Route to singleton if extraction failed OR if answer parsing failed.
        return bool(struct.get("extraction_failed", False) or struct.get("_parse_failed", False))
