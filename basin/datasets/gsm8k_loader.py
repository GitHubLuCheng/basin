"""
basin/datasets/gsm8k_loader.py
===================================
GSM8K dataset loader for MetaDynamics reasoning experiments.

Loads problems from the HuggingFace ``gsm8k`` dataset (config: ``main``).
Ground-truth answers are extracted from the ``#### NUMBER`` suffix in the
``answer`` field and stored as plain numeric strings (e.g. ``"42"``).

Usage
-----
    from basin.datasets.gsm8k_loader import GSM8KLoader
    problems = GSM8KLoader(max_problems=20, seed=42).load()
    for p in problems:
        print(p.problem_id, p.question[:60], "→", p.answer)
"""

from __future__ import annotations

import re
import random
from typing import List, Optional

from basin.datasets.loader import Problem


def _extract_gt_answer(answer_text: str) -> str:
    """
    Extract the numeric answer from a GSM8K ground-truth string.

    GSM8K stores solutions ending with ``#### <number>``.
    Returns the number as a canonical string (int if whole, else float).
    """
    m = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            fv = float(raw)
            if fv == int(fv) and abs(fv) < 1e15:
                return str(int(fv))
            return str(fv)
        except ValueError:
            return raw.strip()
    # Fallback: last number in the text.
    nums = re.findall(r"[\-]?\d[\d,]*\.?\d*", answer_text)
    if nums:
        raw = nums[-1].replace(",", "")
        try:
            fv = float(raw)
            if fv == int(fv) and abs(fv) < 1e15:
                return str(int(fv))
            return str(fv)
        except ValueError:
            pass
    return answer_text.strip()


class GSM8KLoader:
    """
    Load a random subset of GSM8K problems.

    Parameters
    ----------
    split:
        Dataset split: ``"test"`` (default) or ``"train"``.
    max_problems:
        Number of problems to sample.
    seed:
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        split: str = "test",
        max_problems: int = 20,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.max_problems = max_problems
        self.seed = seed

    def load(self) -> List[Problem]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The `datasets` package is required for GSM8KLoader. "
                "Install with: pip install datasets"
            ) from e

        ds = load_dataset("gsm8k", "main", split=self.split)
        total = len(ds)
        rng = random.Random(self.seed)
        n = min(self.max_problems, total)
        indices = rng.sample(range(total), n)
        indices.sort()  # keep stable order for reproducibility

        problems: List[Problem] = []
        for i in indices:
            item = ds[i]
            gt = _extract_gt_answer(item["answer"])
            prob = Problem(
                problem_id=f"gsm8k_{self.split}_{i}",
                question=item["question"],
                answer=gt,
            )
            problems.append(prob)
        return problems
