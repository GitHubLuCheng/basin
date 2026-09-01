"""
basin/datasets/humaneval_loader.py
======================================
Loader for OpenAI HumanEval (openai/openai_humaneval on the HF Hub).

Each problem ships with an executable ``check(candidate)`` function, giving
*oracle* (execution-based) correctness — unlike MATH, where correctness is a
string match against ``\\boxed{...}``.

Usage
-----
    from basin.datasets.humaneval_loader import HumanEvalLoader
    problems = HumanEvalLoader(max_problems=40, seed=42).load()
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CodeProblem:
    """A single HumanEval problem."""

    task_id: str
    prompt: str             # function signature + docstring
    test: str               # source defining check(candidate)
    entry_point: str        # function name to call in check()
    canonical_solution: str = ""


class HumanEvalLoader:
    """
    Load a (sub)sample of HumanEval problems.

    Parameters
    ----------
    max_problems:
        Number of problems to sample (default: all 164).
    seed:
        Random seed for reproducible sampling.
    """

    def __init__(self, max_problems: Optional[int] = None, seed: int = 42) -> None:
        self.max_problems = max_problems
        self.seed = seed

    def load(self) -> List[CodeProblem]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The `datasets` package is required. Install with: pip install datasets"
            ) from e

        ds = load_dataset("openai_humaneval", split="test")
        problems = [
            CodeProblem(
                task_id=item["task_id"],
                prompt=item["prompt"],
                test=item["test"],
                entry_point=item["entry_point"],
                canonical_solution=item.get("canonical_solution", ""),
            )
            for item in ds
        ]

        if self.max_problems is not None and self.max_problems < len(problems):
            rng = random.Random(self.seed)
            problems = rng.sample(problems, self.max_problems)
            problems.sort(key=lambda p: p.task_id)

        return problems
