"""
metadyn/datasets/math_loader.py
================================
Loader for the Hendrycks MATH benchmark (EleutherAI/hendrycks_math).

Samples problems from the test split, optionally filtered by difficulty
level and subject.  Ground-truth answers are extracted from the
``\\boxed{...}`` in the solution field.

Usage
-----
    from metadyn.datasets.math_loader import MATHLoader
    problems = MATHLoader(levels=[4, 5], max_problems=20, seed=42).load()
"""

from __future__ import annotations

import re
import random
from typing import List, Optional

from metadyn.datasets.loader import Problem


MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _extract_boxed(solution: str) -> str:
    """
    Extract the answer from ``\\boxed{...}`` in a MATH solution string.

    Handles nested braces (e.g. ``\\boxed{\\frac{1}{2}}``).
    Returns the raw LaTeX string, or "?" if not found.
    """
    # Find \boxed{ and extract matching braces
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        idx = solution.rfind(r"\boxed {")
    if idx == -1:
        return "?"

    start = solution.index("{", idx)
    depth = 0
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return solution[start + 1 : i].strip()
    return "?"


class MATHLoader:
    """
    Load a balanced random subset of MATH problems.

    Parameters
    ----------
    levels:
        Difficulty levels to include (1–5).  Default: [4, 5] (hard).
    subjects:
        Subjects to include.  Default: all 7.
    max_problems:
        Total number of problems to sample.
    seed:
        Random seed for reproducible sampling.
    split:
        Dataset split: "test" (default) or "train".
    """

    def __init__(
        self,
        levels: Optional[List[int]] = None,
        subjects: Optional[List[str]] = None,
        max_problems: int = 20,
        seed: int = 42,
        split: str = "test",
    ) -> None:
        self.levels = levels if levels is not None else [4, 5]
        self.subjects = subjects if subjects is not None else MATH_CONFIGS
        self.max_problems = max_problems
        self.seed = seed
        self.split = split

    def load(self) -> List[Problem]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The `datasets` package is required. Install with: pip install datasets"
            ) from e

        level_strs = {f"Level {l}" for l in self.levels}
        candidates: List[Problem] = []

        for subject in self.subjects:
            ds = load_dataset(
                "EleutherAI/hendrycks_math", subject,
                split=self.split,
            )
            for idx, item in enumerate(ds):
                if item.get("level") not in level_strs:
                    continue
                answer = _extract_boxed(item["solution"])
                if answer == "?":
                    continue
                prob = Problem(
                    problem_id=f"math_{subject}_{self.split}_{idx}",
                    question=item["problem"],
                    answer=answer,
                )
                candidates.append(prob)

        rng = random.Random(self.seed)
        n = min(self.max_problems, len(candidates))
        selected = rng.sample(candidates, n)
        # Sort by problem_id for stable ordering.
        selected.sort(key=lambda p: p.problem_id)
        return selected
