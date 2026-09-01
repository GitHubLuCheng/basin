"""
Clusters reasoning states into basins using embedding similarity.

Two algorithms are provided:

* **threshold** — simple greedy graph clustering based on a cosine
  similarity threshold.  No external dependencies, fast for small N.
* **dbscan** / **agglomerative** — sklearn algorithms for larger corpora.

The basin ID -1 means "not yet assigned".
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from basin.embedding.similarity import pairwise_cosine


class BasinClusterer:
    """
    Assigns reasoning states to discrete basins.

    Parameters
    ----------
    algorithm:
        One of ``"threshold"``, ``"dbscan"``, ``"agglomerative"``.
    similarity_threshold:
        For ``"threshold"`` algorithm: two states are in the same basin
        if their cosine similarity exceeds this value.
    n_clusters:
        For ``"agglomerative"`` only: target number of basins.
    dbscan_eps:
        For ``"dbscan"``: neighbourhood radius in cosine-distance space.
    """

    def __init__(
        self,
        algorithm: str = "threshold",
        similarity_threshold: float = 0.70,
        n_clusters: Optional[int] = None,
        dbscan_eps: float = 0.30,
        dbscan_min_samples: int = 1,
    ) -> None:
        if algorithm not in ("threshold", "dbscan", "agglomerative"):
            raise ValueError(f"Unknown algorithm {algorithm!r}")
        self.algorithm = algorithm
        self.similarity_threshold = similarity_threshold
        self.n_clusters = n_clusters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign basin IDs to an (N, D) embedding matrix.

        Parameters
        ----------
        embeddings:
            Array of shape ``(N, D)``.

        Returns
        -------
        np.ndarray of shape ``(N,)`` with integer basin IDs (0-indexed).
        Noise points in DBSCAN are labelled -1.
        """
        if embeddings.shape[0] == 0:
            return np.array([], dtype=int)
        if embeddings.shape[0] == 1:
            return np.array([0], dtype=int)

        if self.algorithm == "threshold":
            return self._threshold_cluster(embeddings)
        elif self.algorithm == "dbscan":
            return self._dbscan_cluster(embeddings)
        else:
            return self._agglomerative_cluster(embeddings)

    # ------------------------------------------------------------------

    def _threshold_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Greedy threshold clustering.

        The first state starts basin 0.  Each subsequent state joins the
        basin of its most similar existing centroid if that similarity
        exceeds *similarity_threshold*, otherwise it starts a new basin.
        """
        n = embeddings.shape[0]
        labels = np.full(n, -1, dtype=int)
        centroids: List[np.ndarray] = []

        for i in range(n):
            vec = embeddings[i]
            best_basin = -1
            best_sim = -1.0
            for bid, centroid in enumerate(centroids):
                sim = float(np.dot(vec, centroid))  # unit vectors assumed
                if sim > best_sim:
                    best_sim = sim
                    best_basin = bid

            if best_sim >= self.similarity_threshold:
                labels[i] = best_basin
                # Update centroid as running mean (re-normalise).
                count = np.sum(labels[:i] == best_basin) + 1
                centroids[best_basin] = (
                    centroids[best_basin] * (count - 1) / count + vec / count
                )
                norm = np.linalg.norm(centroids[best_basin])
                if norm > 1e-9:
                    centroids[best_basin] /= norm
            else:
                labels[i] = len(centroids)
                centroids.append(vec.copy())

        return labels

    def _dbscan_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.cluster import DBSCAN

        # Convert cosine similarity to distance.
        cos_sim = pairwise_cosine(embeddings)
        dist_matrix = np.clip(1.0 - cos_sim, 0.0, 2.0)
        db = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="precomputed",
        )
        return db.fit_predict(dist_matrix)

    def _agglomerative_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        n = embeddings.shape[0]
        n_clusters = min(self.n_clusters or max(2, n // 3), n)
        cos_sim = pairwise_cosine(embeddings)
        dist_matrix = np.clip(1.0 - cos_sim, 0.0, 2.0)
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        return agg.fit_predict(dist_matrix)
