"""
Anthropic backend — plug in a real ``anthropic`` client here.

Install: ``pip install anthropic``
Usage::

    from metadyn.llm.anthropic_backend import AnthropicClient
    client = AnthropicClient(model="claude-opus-4-6", api_key="sk-ant-...")

Note: The Anthropic API does not currently expose per-token log-probabilities,
so ``supports_logprobs`` returns False and confidence is estimated from
self-evaluation prompts instead.
"""

from typing import List, Optional

from .base import BaseLLMClient, LLMResponse


class AnthropicClient(BaseLLMClient):
    """
    Thin wrapper around the ``anthropic`` Python SDK.

    Parameters
    ----------
    model:
        Anthropic model identifier, e.g. ``"claude-opus-4-6"``.
    api_key:
        If ``None``, the ``ANTHROPIC_API_KEY`` environment variable is used.
    system_prompt:
        Optional system prompt prepended to every request.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        system_prompt: str = "You are a careful reasoning assistant.",
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "anthropic package not found.  Install it with: pip install anthropic"
            ) from exc

        import anthropic as _anthropic

        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._system = system_prompt

    @property
    def supports_logprobs(self) -> bool:
        # Anthropic API does not expose logprobs at time of writing.
        return False

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        n: int = 1,
    ) -> List[LLMResponse]:
        """
        Call the Anthropic messages API *n* times (each call returns one sample,
        because the API does not support batched sampling in a single request).
        """
        results: List[LLMResponse] = []
        for _ in range(n):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self._system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            results.append(
                LLMResponse(
                    text=text,
                    logprob_stats=None,
                    finish_reason=response.stop_reason or "stop",
                    tokens_used=response.usage.output_tokens,
                )
            )
        return results
