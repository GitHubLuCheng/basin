"""
Word n-gram (Diverse-Beam-Search-style) embedding backend.

Unlike SentenceTransformerEmbedder (paraphrase/semantic similarity), this
backend represents each trace purely by its surface lexical content —
hashed word 1-3-grams, L2-normalised — so cosine similarity between two
traces approximates lexical (n-gram) overlap rather than meaning overlap.

This is used to build a "surface-diversity" baseline analogous to Diverse
Beam Search (Vijayakumar et al., 2016): candidates that repeat the same
wording as previously selected states are penalised, regardless of whether
they express the same or a different underlying reasoning strategy. It is
a controlled contrast to BASIN's NLI-based semantic basin definition,
answering whether a purely lexical diversity penalty recovers similar
gains without any semantic processing.

No fitting/vocabulary required (hashing trick) — stateless and fast
(CPU-only, no LLM calls), so it's a drop-in replacement for
SentenceTransformerEmbedder wherever BasinMemory/BasinClusterer expect a
BaseEmbedder.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .base import BaseEmbedder


class NgramEmbedder(BaseEmbedder):
    def __init__(self, n_features: int = 4096, ngram_range=(1, 3)) -> None:
        self._n_features = n_features
        self._vectorizer = HashingVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            n_features=n_features,
            norm="l2",
            alternate_sign=False,
        )

    @property
    def dim(self) -> int:
        return self._n_features

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        mat = self._vectorizer.transform(texts)
        return mat.toarray().astype(np.float32)
