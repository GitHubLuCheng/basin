"""
TF-IDF embedding backend.

Works entirely with scikit-learn; no external API calls or model downloads.
This is the default for offline/prototype use.

To swap in sentence-transformers, implement a ``SentenceTransformerEmbedder``
following the same :class:`~basin.embedding.base.BaseEmbedder` interface.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseEmbedder


class TFIDFEmbedder(BaseEmbedder):
    """
    Incrementally-fitted TF-IDF embedder.

    The vocabulary grows as new texts are seen via :meth:`embed` or
    :meth:`embed_batch`.  This makes it suitable for the online setting
    of metadynamics where states arrive one at a time.

    Parameters
    ----------
    max_features:
        Maximum vocabulary size.
    ngram_range:
        n-gram range for the TF-IDF vectoriser.
    sublinear_tf:
        If True, use log(1+tf) instead of raw tf.
    """

    def __init__(
        self,
        max_features: int = 1024,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
        )
        self._corpus: List[str] = []
        self._fitted = False
        self._max_features = max_features

    @property
    def dim(self) -> int:
        """Current vocabulary size (grows as more texts are seen)."""
        if self._fitted:
            return len(self._vectorizer.vocabulary_)
        return self._max_features

    def _refit(self) -> None:
        """Refit the vectoriser on the full corpus seen so far."""
        self._vectorizer.fit(self._corpus)
        self._fitted = True

    def _to_unit_vector(self, sparse_row) -> np.ndarray:
        """Convert a sparse TF-IDF row to a dense unit vector."""
        vec = sparse_row.toarray().squeeze().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text, refitting the vocabulary if necessary.

        Side-effect: *text* is added to the internal corpus.
        """
        if text not in self._corpus:
            self._corpus.append(text)
            self._refit()
        sparse = self._vectorizer.transform([text])
        return self._to_unit_vector(sparse[0])

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts, updating the corpus with any new ones.

        Returns
        -------
        np.ndarray of shape ``(len(texts), vocab_size)``.
        """
        new = [t for t in texts if t not in self._corpus]
        if new:
            self._corpus.extend(new)
            self._refit()
        sparse = self._vectorizer.transform(texts)
        rows = []
        for i in range(sparse.shape[0]):
            rows.append(self._to_unit_vector(sparse[i]))
        return np.vstack(rows)

    def reembed_corpus(self) -> np.ndarray:
        """
        Re-embed the entire stored corpus after a vocabulary update.

        Useful when you want to recompute all stored vectors with a
        stabilised vocabulary.
        """
        if not self._fitted:
            raise RuntimeError("No texts have been embedded yet.")
        return self.embed_batch(self._corpus)


# ---------------------------------------------------------------------------
# Optional: sentence-transformers backend (uncomment if installed)
# ---------------------------------------------------------------------------

# class SentenceTransformerEmbedder(BaseEmbedder):
#     """Embedding backend using sentence-transformers (requires GPU/CPU model)."""
#
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
#         from sentence_transformers import SentenceTransformer
#         self._model = SentenceTransformer(model_name)
#
#     @property
#     def dim(self) -> int:
#         return self._model.get_sentence_embedding_dimension()
#
#     def embed(self, text: str) -> np.ndarray:
#         vec = self._model.encode([text], normalize_embeddings=True)
#         return vec[0].astype(np.float32)
#
#     def embed_batch(self, texts: List[str]) -> np.ndarray:
#         return self._model.encode(texts, normalize_embeddings=True).astype(np.float32)
