from .base import BaseEmbedder
from .tfidf_backend import TFIDFEmbedder
from .similarity import cosine_similarity, gaussian_kernel, pairwise_cosine

__all__ = [
    "BaseEmbedder",
    "TFIDFEmbedder",
    "cosine_similarity",
    "gaussian_kernel",
    "pairwise_cosine",
]
