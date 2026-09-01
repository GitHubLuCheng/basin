"""
Baseline reasoning strategies to compare against metadynamics.

All baselines expose the same ``solve(question) -> AnswerResult`` interface
so they can be swapped in the experiment script without code changes.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

from metadyn.confidence.estimator import AnswerResult
from metadyn.controller.candidate_gen import CandidateGenerator
from metadyn.verifier.verifier import BaseVerifier, _normalise_answer


class GreedyBaseline:
    """
    Single sample at temperature ≈ 0.  Represents the cheapest baseline.

    Parameters
    ----------
    generator:
        Shared candidate generator (will request n=1).
    """

    def __init__(self, generator: CandidateGenerator) -> None:
        self.gen = generator

    def solve(self, question: str) -> AnswerResult:
        """Generate one sample and return it as the answer."""
        states = self.gen.generate_candidates(question, n=1, step_index=0)
        state = states[0]
        return AnswerResult(
            predicted_answer=state.provisional_answer,
            confidence=state.confidence_proxy,
            n_basins_explored=1,
            converged=False,
        )


class SelfConsistencyBaseline:
    """
    Majority-vote over *n* samples (Wang et al., 2023).

    Generates n independent samples and returns the plurality answer,
    with confidence proportional to its vote share.

    Parameters
    ----------
    generator:
        Shared candidate generator.
    n_samples:
        Number of samples to draw.
    """

    def __init__(self, generator: CandidateGenerator, n_samples: int = 8) -> None:
        self.gen = generator
        self.n = n_samples

    def solve(self, question: str) -> AnswerResult:
        states = self.gen.generate_candidates(question, n=self.n, step_index=0)
        answers = [_normalise_answer(s.provisional_answer) for s in states]
        counts = Counter(answers)
        total = sum(counts.values())
        predicted, vote_count = counts.most_common(1)[0]
        confidence = vote_count / total

        distribution = {a: c / total for a, c in counts.items()}
        # Approximate number of unique "basins" = unique answers.
        n_basins = len(counts)

        return AnswerResult(
            predicted_answer=predicted,
            confidence=confidence,
            answer_distribution=distribution,
            n_basins_explored=n_basins,
            converged=confidence > 0.5,
        )


class VerifierRerankingBaseline:
    """
    Generate *n* samples, rerank by verifier score, pick the top-1.

    This tests the value of the verifier signal *without* the
    exploration-encouraging bias term.

    Parameters
    ----------
    generator:
        Shared candidate generator.
    verifier:
        Verifier used for reranking.
    n_samples:
        Number of samples to rerank.
    """

    def __init__(
        self,
        generator: CandidateGenerator,
        verifier: BaseVerifier,
        n_samples: int = 8,
    ) -> None:
        self.gen = generator
        self.verifier = verifier
        self.n = n_samples

    def solve(self, question: str) -> AnswerResult:
        states = self.gen.generate_candidates(question, n=self.n, step_index=0)
        scores = self.verifier.score_batch(states)
        best = states[max(range(len(scores)), key=lambda i: scores[i])]
        return AnswerResult(
            predicted_answer=best.provisional_answer,
            confidence=best.confidence_proxy,
            n_basins_explored=len(set(_normalise_answer(s.provisional_answer) for s in states)),
            converged=False,
        )


class TemperatureSweepBaseline:
    """
    Sample once at each temperature in a sweep and use majority vote.

    Parameters
    ----------
    generator:
        Shared candidate generator.
    temperatures:
        List of temperatures to sweep over.
    """

    def __init__(
        self,
        generator: CandidateGenerator,
        temperatures: Optional[List[float]] = None,
    ) -> None:
        self.gen = generator
        self.temperatures = temperatures or [0.0, 0.3, 0.7, 1.0, 1.2]

    def solve(self, question: str) -> AnswerResult:
        all_answers: List[str] = []
        saved_temp = self.gen.temperature

        for temp in self.temperatures:
            self.gen.temperature = temp
            states = self.gen.generate_candidates(question, n=1, step_index=0)
            all_answers.append(_normalise_answer(states[0].provisional_answer))

        self.gen.temperature = saved_temp  # restore

        counts = Counter(all_answers)
        total = sum(counts.values())
        predicted, vote_count = counts.most_common(1)[0]

        return AnswerResult(
            predicted_answer=predicted,
            confidence=vote_count / total,
            answer_distribution={a: c / total for a, c in counts.items()},
            n_basins_explored=len(counts),
            converged=vote_count / total > 0.5,
        )
