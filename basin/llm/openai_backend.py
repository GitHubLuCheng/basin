"""
OpenAI backend — plug in a real ``openai`` client here.

Install: ``pip install openai``
Usage::

    from basin.llm.openai_backend import OpenAIClient
    client = OpenAIClient(model="gpt-4o", api_key="sk-...")
"""

import re
from typing import List, Optional

from .base import BaseLLMClient, LLMResponse, LogprobStats


class OpenAIClient(BaseLLMClient):
    """
    Thin wrapper around the ``openai`` Python SDK.

    Parameters
    ----------
    model:
        OpenAI model identifier, e.g. ``"gpt-4o"`` or ``"gpt-3.5-turbo"``.
    api_key:
        If ``None``, the ``OPENAI_API_KEY`` environment variable is used.
    logprobs:
        Whether to request per-token log-probabilities (requires a model
        that supports the ``logprobs`` parameter).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        logprobs: bool = True,
        base_url: Optional[str] = None,
        extra_body: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "openai package not found.  Install it with: pip install openai"
            ) from exc

        import openai as _openai

        client_kwargs = dict(api_key=api_key, timeout=timeout)
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = _openai.OpenAI(**client_kwargs)
        self._model = model
        self._logprobs = logprobs
        self._extra_body = extra_body or {}
        self._system_prompt = system_prompt

    @property
    def supports_logprobs(self) -> bool:
        return self._logprobs

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
        """Call the OpenAI chat completions API."""
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        # o-series models: use max_completion_tokens, temperature must be 1 (default),
        # and n is capped at 1 (loop externally for n>1 since batched sampling adds
        # little value when temperature is fixed).
        _is_o_series = re.match(r"^o\d", self._model) is not None
        if _is_o_series and n > 1:
            results: List[LLMResponse] = []
            for _ in range(n):
                results.extend(self.generate(prompt, temperature=temperature,
                                             max_tokens=max_tokens, n=1))
            return results

        token_key = "max_completion_tokens" if _is_o_series else "max_tokens"
        kwargs: dict = dict(model=self._model, messages=messages, n=n)
        kwargs[token_key] = max_tokens
        if not _is_o_series:
            kwargs["temperature"] = temperature
        if self._logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 1
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        response = self._client.chat.completions.create(**kwargs)

        results = []
        for choice in response.choices:
            lp_stats = None
            if self._logprobs and choice.logprobs and choice.logprobs.content:
                token_lps = [t.logprob for t in choice.logprobs.content]
                import numpy as np
                lp_stats = LogprobStats(
                    mean=float(np.mean(token_lps)),
                    std=float(np.std(token_lps)),
                    min=float(np.min(token_lps)),
                    token_logprobs=token_lps,
                )
            # Some thinking models (e.g. Qwen3) put the visible reply in
            # `content` and reasoning tokens in a separate `reasoning` field.
            # If `content` is empty/None, fall back to `reasoning` so the
            # trace text is never blank.
            text = choice.message.content or ""
            if not text:
                extra = getattr(choice.message, "model_extra", None) or {}
                text = extra.get("reasoning", "") or ""
            results.append(
                LLMResponse(
                    text=text,
                    logprob_stats=lp_stats,
                    finish_reason=choice.finish_reason or "stop",
                    tokens_used=response.usage.completion_tokens // n,
                )
            )
        return results
