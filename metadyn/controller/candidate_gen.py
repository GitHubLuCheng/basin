"""
Candidate generation — prompts the LLM and extracts reasoning states.
"""

from __future__ import annotations

from typing import List, Optional

from metadyn.llm.base import BaseLLMClient
from metadyn.state.extractor import StateExtractor
from metadyn.state.reasoning_state import ReasoningState
from metadyn.embedding.base import BaseEmbedder


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ONE_SHOT_PROMPT = """\
Question: {question}

Please reason carefully step by step, then give your final answer on the last line.
Start your answer line with "Answer:".
"""

_CONTINUATION_PROMPT = """\
Question: {question}

Previous reasoning attempt:
\"\"\"
{previous_trace}
\"\"\"

The above reasoning may have issues.  Please try a different approach:
think from a fresh angle, consider alternative interpretations, and
give your revised answer on the last line starting with "Answer:".
"""


class CandidateGenerator:
    """
    Generates candidate reasoning states for a given question.

    Parameters
    ----------
    llm_client:
        The LLM backend to use for generation.
    extractor:
        Converts raw LLM output to a :class:`ReasoningState`.
    embedder:
        Embeds the trace text for similarity computation.
    temperature:
        Default sampling temperature.
    max_tokens:
        Maximum tokens per candidate.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        extractor: StateExtractor,
        embedder: BaseEmbedder,
        temperature: float = 0.8,
        max_tokens: int = 400,
    ) -> None:
        self.llm = llm_client
        self.extractor = extractor
        self.embedder = embedder
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_candidates(
        self,
        question: str,
        n: int,
        step_index: int = 0,
        previous_trace: Optional[str] = None,
    ) -> List[ReasoningState]:
        """
        Generate *n* candidate reasoning states.

        Parameters
        ----------
        question:
            The question to answer.
        n:
            Number of independent candidates to generate.
        step_index:
            Current metadynamics step (stored in the state).
        previous_trace:
            If provided, the prompt asks the LLM to explore a different
            direction from this prior trace.

        Returns
        -------
        List of :class:`ReasoningState` objects with embedding vectors set.
        """
        if previous_trace:
            prompt = _CONTINUATION_PROMPT.format(
                question=question,
                previous_trace=previous_trace[:800],
            )
        else:
            prompt = _ONE_SHOT_PROMPT.format(question=question)

        responses = self.llm.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n,
        )

        states: List[ReasoningState] = []
        texts: List[str] = []
        for resp in responses:
            state = self.extractor.extract(question, resp, step_index=step_index)
            states.append(state)
            texts.append(state.trace_text)

        # Embed all traces in one batch for efficiency.
        if texts:
            embeddings = self.embedder.embed_batch(texts)
            for i, state in enumerate(states):
                state.embedding_vector = embeddings[i]

        return states
