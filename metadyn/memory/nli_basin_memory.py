"""
NLI-aware basin memory.

Subclasses BasinMemory to pass ReasoningState objects (with their
``structured_state`` attributes) to NLIBasinClusterer instead of raw
embedding vectors.  The embedding vectors are still used for the
Gaussian kernel bias computation — only the clustering step changes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from metadyn.memory.basin_memory import BasinMemory
from metadyn.memory.nli_clustering import NLIBasinClusterer


class NLIBasinMemory(BasinMemory):
    """
    Drop-in replacement for BasinMemory that clusters via NLI.

    Parameters are the same as BasinMemory; ``clusterer`` must be an
    ``NLIBasinClusterer`` instance (or None, in which case one is created
    with default settings).
    """

    def __init__(
        self,
        kernel_bandwidth: float = 0.5,
        base_weight: float = 1.0,
        count_scaling: float = 0.5,
        clusterer: Optional[NLIBasinClusterer] = None,
    ) -> None:
        nli_clusterer = clusterer or NLIBasinClusterer()
        super().__init__(
            kernel_bandwidth=kernel_bandwidth,
            base_weight=base_weight,
            count_scaling=count_scaling,
            clusterer=nli_clusterer,
        )

    def _recluster(self) -> None:
        """
        Override: set pending states on the NLI clusterer, then call
        fit_predict() which will use them instead of embeddings.
        """
        if not self.states:
            return
        # Pass structured states to the NLI clusterer.
        self.clusterer._pending_states = list(self.states)
        # fit_predict() will consume _pending_states and ignore the array arg.
        embeddings = self._get_embeddings()
        if embeddings is None:
            # No embeddings yet — use a dummy array of the right shape.
            embeddings = np.zeros((len(self.states), 1))
        labels = self.clusterer.fit_predict(embeddings)
        self._basin_labels = labels
        for i, state in enumerate(self.states):
            state.basin_id = int(labels[i])
