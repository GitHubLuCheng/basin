"""
Visualisation utilities for metadynamics reasoning experiments.

All functions return a ``matplotlib.figure.Figure`` object so they can
be shown interactively (``fig.show()``) or saved (``fig.savefig(path)``).

Requirements: matplotlib (listed in requirements.txt).
For 2-D projection: sklearn is also needed (already a dependency).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Lazy import guard
# ---------------------------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation.  "
            "Install it with: pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Basin visitation timeline
# ---------------------------------------------------------------------------

def plot_basin_visitation(
    visit_history: List[int],
    n_basins: Optional[int] = None,
    title: str = "Basin Visitation Over Time",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """
    Bar / strip chart showing which basin was selected at each step.

    Parameters
    ----------
    visit_history:
        Ordered list of basin IDs (from :meth:`BasinMemory.visit_history`).
    n_basins:
        Total number of basins.  If None, inferred from the history.
    title:
        Plot title.
    """
    plt = _require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_steps = len(visit_history)
    n_basins = n_basins or (max(visit_history) + 1 if visit_history else 1)

    colors = list(mcolors.TABLEAU_COLORS.values())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), height_ratios=[3, 1])

    # Top: scatter plot of basin ID vs step.
    for step, bid in enumerate(visit_history):
        ax1.scatter(
            step, bid,
            color=colors[bid % len(colors)],
            s=100, zorder=3,
        )
    ax1.plot(visit_history, color="grey", alpha=0.4, linewidth=1, zorder=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Basin ID")
    ax1.set_yticks(range(n_basins))
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Bottom: cumulative visitation counts per basin.
    counts = np.zeros(n_basins, dtype=int)
    cumulative = np.zeros((n_steps, n_basins), dtype=int)
    for i, bid in enumerate(visit_history):
        if 0 <= bid < n_basins:
            counts[bid] += 1
        cumulative[i] = counts

    for bid in range(n_basins):
        ax2.plot(
            cumulative[:, bid],
            label=f"Basin {bid}",
            color=colors[bid % len(colors)],
        )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative visits")
    ax2.legend(ncol=n_basins, fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2-D projection of state embeddings
# ---------------------------------------------------------------------------

def plot_2d_projection(
    embeddings: np.ndarray,
    basin_ids: List[int],
    answers: Optional[List[str]] = None,
    method: str = "pca",
    title: str = "State Embeddings (2-D Projection)",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """
    Project high-dimensional state embeddings to 2-D and colour by basin.

    Parameters
    ----------
    embeddings:
        Array of shape ``(N, D)`` — one row per state.
    basin_ids:
        Basin ID for each state.
    answers:
        Optional provisional answer for each state (shown as text labels).
    method:
        ``"pca"`` or ``"tsne"``.
    """
    plt = _require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if embeddings.shape[0] < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough states to project.", ha="center")
        return fig

    # Dimensionality reduction.
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=min(30, embeddings.shape[0] - 1))
    else:
        raise ValueError(f"Unknown projection method: {method!r}")

    coords = reducer.fit_transform(embeddings)

    colors = list(mcolors.TABLEAU_COLORS.values())
    unique_basins = sorted(set(basin_ids))

    fig, ax = plt.subplots(figsize=(8, 6))
    for bid in unique_basins:
        mask = [i for i, b in enumerate(basin_ids) if b == bid]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=f"Basin {bid}",
            color=colors[bid % len(colors)] if bid >= 0 else "grey",
            s=80,
            alpha=0.8,
        )

    # Annotate with step index.
    for i in range(len(basin_ids)):
        ax.annotate(
            str(i),
            (coords[i, 0], coords[i, 1]),
            fontsize=7,
            alpha=0.7,
        )

    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Score evolution
# ---------------------------------------------------------------------------

def plot_score_evolution(
    score_history: List[List[float]],
    title: str = "Candidate Scores Per Round",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """
    Box plot of candidate scores at each metadynamics round.

    Parameters
    ----------
    score_history:
        List of score lists, one per round (from
        :meth:`MetadynamicsController.score_history`).
    """
    plt = _require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(score_history, positions=range(1, len(score_history) + 1))
    ax.set_xlabel("Round")
    ax.set_ylabel("Candidate score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Answer distribution
# ---------------------------------------------------------------------------

def plot_answer_distribution(
    answer_distribution: dict,
    title: str = "Final Answer Distribution",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """
    Horizontal bar chart of the final answer probabilities.

    Parameters
    ----------
    answer_distribution:
        Dict mapping answer string to probability (from
        :attr:`AnswerResult.answer_distribution`).
    """
    plt = _require_matplotlib()
    import matplotlib.pyplot as plt

    if not answer_distribution:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data.", ha="center")
        return fig

    answers = list(answer_distribution.keys())
    probs = [answer_distribution[a] for a in answers]

    sorted_pairs = sorted(zip(probs, answers), reverse=True)
    probs_s, answers_s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, max(3, len(answers) * 0.5)))
    bars = ax.barh(answers_s, probs_s, color="steelblue", alpha=0.8)
    ax.set_xlabel("Probability")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    for bar, p in zip(bars, probs_s):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.2f}",
            va="center",
            fontsize=9,
        )
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig
