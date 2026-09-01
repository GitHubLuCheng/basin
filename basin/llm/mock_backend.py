"""
Mock LLM backend for offline testing.

Generates plausible-looking chain-of-thought reasoning traces from
parameterised templates.  Each question is paired with a small library of
reasoning *archetypes*; the mock samples one archetype per call.

Swap this backend for :class:`OpenAIClient` or :class:`AnthropicClient`
to run with real API access.
"""

import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseLLMClient, LLMResponse, LogprobStats


# ---------------------------------------------------------------------------
# Reasoning trace templates
# Each template produces a different reasoning path and a (possibly wrong)
# answer.  The simulator module provides a richer version of this idea.
# ---------------------------------------------------------------------------

_MATH_TEMPLATES: List[Dict] = [
    {
        "name": "step_by_step_correct",
        "trace": (
            "Let me work through this carefully, step by step.\n"
            "First, I'll identify what the question is asking: {question_summary}\n"
            "Key quantities: {quantities}\n"
            "Step 1: {step1}\n"
            "Step 2: {step2}\n"
            "Checking my work: {step1} then {step2} = {answer}\n"
            "The answer is {answer}."
        ),
        "mean_logprob": -0.4,
        "verifier_hint": 0.85,
    },
    {
        "name": "direct_approach",
        "trace": (
            "This is a straightforward calculation.\n"
            "Using the formula directly: {step1}\n"
            "So the answer is {answer}."
        ),
        "mean_logprob": -0.5,
        "verifier_hint": 0.70,
    },
    {
        "name": "common_error_off_by_one",
        "trace": (
            "I see a {type} problem. Let me apply the standard approach.\n"
            "Setting up: {step1}\n"
            "Hmm, I need to be careful about the boundary: {step2}\n"
            "Actually, I think the count starts at 0, so: {answer}."
        ),
        "mean_logprob": -0.7,
        "verifier_hint": 0.50,
    },
    {
        "name": "unit_confusion",
        "trace": (
            "Converting to the base unit first.\n"
            "The key relationship is {step1}.\n"
            "Wait, this could also be interpreted as {step2}.\n"
            "Taking the first interpretation: {answer}."
        ),
        "mean_logprob": -0.8,
        "verifier_hint": 0.45,
    },
    {
        "name": "partial_solution",
        "trace": (
            "Let me think about what we know.\n"
            "Given: {quantities}\n"
            "I'll compute {step1}, which gives {answer}.\n"
            "There might be more to consider, but {answer} seems right."
        ),
        "mean_logprob": -0.9,
        "verifier_hint": 0.35,
    },
]


def _make_logprob_stats(mean: float, n_tokens: int = 40) -> LogprobStats:
    """Simulate per-token logprob stats around a given mean."""
    rng = np.random.default_rng()
    tokens = rng.normal(loc=mean, scale=0.3, size=n_tokens).tolist()
    return LogprobStats(
        mean=float(np.mean(tokens)),
        std=float(np.std(tokens)),
        min=float(np.min(tokens)),
        token_logprobs=tokens,
    )


class MockLLMClient(BaseLLMClient):
    """
    Deterministic-ish mock that generates templated reasoning traces.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    template_weights:
        Probability weight for each template archetype.  If ``None``,
        weights are drawn uniformly.
    noise_level:
        Amount of random variation added to logprob stats.
    """

    def __init__(
        self,
        seed: Optional[int] = 42,
        template_weights: Optional[List[float]] = None,
        noise_level: float = 0.1,
    ) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._templates = _MATH_TEMPLATES
        self._weights = template_weights or [1.0] * len(self._templates)
        self._noise = noise_level
        self._call_count = 0

    @property
    def supports_logprobs(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "mock-v1"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        n: int = 1,
    ) -> List[LLMResponse]:
        """
        Return *n* templated completions.  Higher temperature slightly
        increases the diversity of template selection.
        """
        responses: List[LLMResponse] = []
        # Extract a rough summary from the prompt for template filling.
        summary = self._summarise_prompt(prompt)

        for _ in range(n):
            self._call_count += 1
            template = self._sample_template(temperature)
            text = self._fill_template(template, summary)
            lp_mean = template["mean_logprob"] + self._np_rng.normal(
                scale=self._noise
            )
            stats = _make_logprob_stats(lp_mean, n_tokens=max(10, len(text) // 4))
            responses.append(
                LLMResponse(
                    text=text,
                    logprob_stats=stats,
                    finish_reason="stop",
                    tokens_used=len(text.split()),
                )
            )
        return responses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_template(self, temperature: float) -> Dict:
        total = sum(self._weights)
        probs = [w / total for w in self._weights]
        # At higher temperature widen the distribution slightly.
        if temperature > 1.0:
            probs = [p ** (1.0 / temperature) for p in probs]
            s = sum(probs)
            probs = [p / s for p in probs]
        return self._rng.choices(self._templates, weights=probs, k=1)[0]

    @staticmethod
    def _summarise_prompt(prompt: str) -> Dict[str, str]:
        """Extract rough fillers for template slots from the raw prompt."""
        lines = [l.strip() for l in prompt.splitlines() if l.strip()]
        question_line = lines[0] if lines else "the question"
        # Try to find numbers in the prompt.
        numbers = re.findall(r"\d+(?:\.\d+)?", prompt)
        quantities = ", ".join(numbers[:4]) if numbers else "x, y"
        return {
            "question_summary": question_line[:80],
            "quantities": quantities,
            "type": "arithmetic",
            "step1": f"compute {quantities}" if quantities else "set up equations",
            "step2": "combine the results",
            "answer": numbers[-1] if numbers else "42",
        }

    @staticmethod
    def _fill_template(template: Dict, fillers: Dict[str, str]) -> str:
        try:
            return template["trace"].format(**fillers)
        except KeyError:
            # If a slot is missing just leave the placeholder.
            return template["trace"]
