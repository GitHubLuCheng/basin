"""Abstract base classes for LLM clients."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LogprobStats:
    """Summary statistics of token log-probabilities for a response."""

    mean: float
    std: float
    min: float
    token_logprobs: List[float] = field(default_factory=list)

    @property
    def perplexity(self) -> float:
        """Perplexity derived from mean log-probability."""
        return math.exp(-self.mean)


@dataclass
class LLMResponse:
    """A single completion returned by an LLM backend."""

    text: str
    logprob_stats: Optional[LogprobStats] = None
    finish_reason: str = "stop"
    tokens_used: int = 0

    @property
    def confidence_proxy(self) -> float:
        """
        A 0-1 confidence proxy derived from logprobs.
        Falls back to 0.5 when logprobs are unavailable.
        """
        if self.logprob_stats is not None:
            # Map mean logprob (-inf, 0] -> (0, 1] via sigmoid-like transform.
            # At mean=-0.5 this gives ~0.8; at mean=-2 it gives ~0.27.
            return 1.0 / (1.0 + math.exp(self.logprob_stats.mean + 1.0))
        return 0.5


class BaseLLMClient(ABC):
    """
    Abstract interface for LLM backends.

    Concrete subclasses implement :meth:`generate` for a specific API
    (OpenAI, Anthropic, local mock, …).  All metadynamics logic works
    against this interface so backends are interchangeable.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        n: int = 1,
    ) -> List[LLMResponse]:
        """
        Generate *n* independent completions for *prompt*.

        Parameters
        ----------
        prompt:
            The full input prompt string.
        temperature:
            Sampling temperature (0 = greedy, higher = more diverse).
        max_tokens:
            Hard cap on output tokens per completion.
        n:
            Number of independent samples to draw.

        Returns
        -------
        List of :class:`LLMResponse`, one per requested sample.
        """

    @property
    @abstractmethod
    def supports_logprobs(self) -> bool:
        """True if this backend can return per-token log-probabilities."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier for the underlying model."""
