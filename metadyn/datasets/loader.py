"""
Dataset loading for evaluation.

Supports:
  * Local JSONL files in GSM8K format ({"question": ..., "answer": ...}).
  * The built-in sample problem set bundled with this package.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class Problem:
    """A single evaluation problem."""

    question: str
    answer: str          # Reference answer (string; may be a number)
    problem_id: str = ""

    def normalised_answer(self) -> str:
        """Lowercase, stripped, comma-removed reference answer."""
        a = self.answer.lower().strip().rstrip(".")
        a = a.replace(",", "")
        return a


_PACKAGE_DIR = Path(__file__).parent
_DEFAULT_JSONL = _PACKAGE_DIR / "sample_problems.jsonl"


class DatasetLoader:
    """
    Loads evaluation problems from a JSONL file.

    Each line must be a JSON object with at minimum a ``"question"`` key
    and an ``"answer"`` key.  Optional ``"id"`` key is used as problem_id.

    Parameters
    ----------
    path:
        Path to a JSONL file.  Defaults to the bundled sample set.
    max_problems:
        If set, truncate the dataset to at most this many problems.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        max_problems: Optional[int] = None,
    ) -> None:
        self.path = Path(path) if path else _DEFAULT_JSONL
        self.max_problems = max_problems

    def load(self) -> List[Problem]:
        """Return the full list of problems."""
        return list(self.iter_problems())

    def iter_problems(self) -> Iterator[Problem]:
        """Yield problems one at a time (memory-efficient for large files)."""
        count = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield Problem(
                    question=data["question"],
                    answer=str(data["answer"]),
                    problem_id=str(data.get("id", count)),
                )
                count += 1
                if self.max_problems and count >= self.max_problems:
                    break
