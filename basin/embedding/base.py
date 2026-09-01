"""Abstract base for text embedding backends."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseEmbedder(ABC):
    """
    Converts text strings into dense vector representations.

    The vectors are used to measure similarity between reasoning states
    and to compute the metadynamics revisit bias.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a unit-normalised 1-D numpy array for *text*."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Return a 2-D array of shape ``(len(texts), dim)`` with one
        row per text, each row unit-normalised.
        """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
