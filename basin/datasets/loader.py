"""
Shared evaluation-problem representation.
"""

from __future__ import annotations

from dataclasses import dataclass


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
