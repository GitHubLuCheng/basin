"""
basin/datasets/gsm_hard_loader.py
======================================
GSM-Hard dataset loader (reasoning-machines/gsm-hard on HuggingFace).

Same underlying word problems and template structure as GSM8K, but with
the surface numbers replaced by much larger, harder-to-mentally-compute
values -- created specifically because plain GSM8K saturates for strong
models. Ground truth ('target') is the exact numeric answer, computed by
re-executing the original problem's solution code with the harder numbers.

Usage
-----
    from basin.datasets.gsm_hard_loader import GSMHardLoader
    problems = GSMHardLoader(max_problems=20, seed=42).load()
"""

from __future__ import annotations

import random
from typing import List

from basin.datasets.loader import Problem


def _format_target(t) -> str:
    fv = float(t)
    if fv == int(fv) and abs(fv) < 1e15:
        return str(int(fv))
    return str(fv)


class GSMHardLoader:
    def __init__(self, max_problems: int = 20, seed: int = 42) -> None:
        self.max_problems = max_problems
        self.seed = seed

    def load(self) -> List[Problem]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The `datasets` package is required for GSMHardLoader. "
                "Install with: pip install datasets"
            ) from e

        ds = load_dataset("reasoning-machines/gsm-hard", split="train")
        total = len(ds)
        rng = random.Random(self.seed)
        n = min(self.max_problems, total)
        indices = rng.sample(range(total), n)
        indices.sort()

        problems: List[Problem] = []
        for i in indices:
            item = ds[i]
            problems.append(Problem(
                problem_id=f"gsmhard_{i}",
                question=item["input"],
                answer=_format_target(item["target"]),
            ))
        return problems
