"""
Sentence-Transformer embedding backend.

Uses a small pre-trained model (default: all-MiniLM-L6-v2, 384-dim) to
produce semantically meaningful embeddings.  Unlike TF-IDF, embeddings are
fixed-dim, stable across the run, and capture paraphrase similarity rather
than surface lexical overlap — making basin boundaries more meaningful for
narrative reasoning tasks like MuSR.

Install: pip install sentence-transformers
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedding backend using sentence-transformers.

    Parameters
    ----------
    model_name:
        Any model from https://www.sbert.net/docs/pretrained_models.html.
        ``all-MiniLM-L6-v2`` (80 MB) is a good speed/quality trade-off.
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not found. Install with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return vec[0].astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        ).astype(np.float32)
