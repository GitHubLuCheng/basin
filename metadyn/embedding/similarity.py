"""
Similarity functions and kernels used by basin memory and clustering.

The Gaussian kernel K(z, z') drives the metadynamics bias:

    bias(z) = sum_j  w_j * K(z, z_j)

where z, z_j are embedding vectors and w_j are per-state weights.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity in [-1, 1] between two 1-D vectors.

    Assumes unit-normalised inputs from the embedder; falls back to
    explicit normalisation otherwise.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def gaussian_kernel(
    a: np.ndarray,
    b: np.ndarray,
    bandwidth: float = 0.5,
) -> float:
    """
    Gaussian (RBF) kernel on the cosine-distance between two vectors:

        K(a, b) = exp( -(1 - cos(a, b))^2 / (2 * bandwidth^2) )

    Returns a value in (0, 1].  At bandwidth=0.5 the kernel drops to
    ~0.14 when cos similarity = 0 (orthogonal), and is 1.0 for identical
    vectors.

    Parameters
    ----------
    bandwidth:
        Controls how quickly similarity decays with distance.
        Smaller → sharper, more localised penalty.
    """
    sim = cosine_similarity(a, b)
    dist_sq = (1.0 - sim) ** 2
    return float(np.exp(-dist_sq / (2.0 * bandwidth ** 2)))


def pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity matrix for an (N, D) array.

    Parameters
    ----------
    matrix:
        Embedding matrix of shape ``(N, D)``.

    Returns
    -------
    np.ndarray of shape ``(N, N)`` with values in [-1, 1].
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    normed = matrix / norms
    return normed @ normed.T


def pairwise_gaussian_kernel(
    matrix: np.ndarray,
    bandwidth: float = 0.5,
) -> np.ndarray:
    """
    Pairwise Gaussian kernel matrix for an (N, D) embedding matrix.

    Parameters
    ----------
    matrix:
        Embedding matrix of shape ``(N, D)``.
    bandwidth:
        Kernel bandwidth (same meaning as in :func:`gaussian_kernel`).

    Returns
    -------
    np.ndarray of shape ``(N, N)`` with values in (0, 1].
    """
    cos_sim = pairwise_cosine(matrix)
    dist_sq = (1.0 - cos_sim) ** 2
    return np.exp(-dist_sq / (2.0 * bandwidth ** 2))
