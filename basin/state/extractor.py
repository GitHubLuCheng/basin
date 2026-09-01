"""
Converts raw LLM output into a structured :class:`ReasoningState`.

Two extraction modes are supported:

* **heuristic** — fast regex/keyword extraction, no extra API calls.
* **llm** — a second LLM call asks the model to produce structured JSON;
  plug in any :class:`~basin.llm.base.BaseLLMClient` for this.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from basin.llm.base import BaseLLMClient, LLMResponse, LogprobStats
from .reasoning_state import ReasoningState


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

_ASSUMPTION_KEYWORDS = [
    "assume", "assuming", "given that", "suppose", "let", "we know",
    "using", "applying",
]

_ISSUE_KEYWORDS = [
    "however", "but", "wait", "unclear", "not sure", "might be wrong",
    "need to check", "alternatively", "hmm", "actually",
]


def _extract_answer(text: str) -> str:
    """
    Return the provisional answer from a reasoning trace.

    Priority order:
      1. Explicit "Answer: <X>" line — searched in reverse so we get the
         *final* answer even if intermediate steps contain "answer:".
      2. "The answer is <X>" phrase.
      3. "Therefore / Thus …" conclusion.
      4. Trailing standalone number on the last non-empty line.
      5. Last non-empty line as fallback.
    """
    # 1. Explicit "Answer:" label — last occurrence wins.
    for line in reversed(text.splitlines()):
        m = re.match(r"\s*answer[:\s]+(.+)", line, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".").strip()[:120]

    # 2. "The answer is …" phrase.
    m = re.search(r"the answer is\s+([^\n.]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")[:120]

    # 3. "Therefore / Thus …" conclusion.
    m = re.search(r"(?:therefore|thus)[,\s]+([^\n.]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")[:120]

    # 4. Standalone number on the last non-empty line.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        if re.fullmatch(r"[\$\s]*(\d[\d\s,./]*)", last):
            return last.strip()
        return last[:120]

    return "unknown"


def _extract_assumptions(text: str) -> List[str]:
    """Pull out sentences that contain assumption-signalling keywords."""
    sentences = re.split(r"[.!?\n]", text)
    found: List[str] = []
    for s in sentences:
        s = s.strip()
        if any(kw in s.lower() for kw in _ASSUMPTION_KEYWORDS):
            found.append(s[:100])
        if len(found) >= 4:
            break
    return found


def _extract_issues(text: str) -> List[str]:
    """Pull out sentences that suggest uncertainty or self-correction."""
    sentences = re.split(r"[.!?\n]", text)
    found: List[str] = []
    for s in sentences:
        s = s.strip()
        if any(kw in s.lower() for kw in _ISSUE_KEYWORDS):
            found.append(s[:100])
        if len(found) >= 3:
            break
    return found


def _heuristic_self_eval(text: str) -> float:
    """
    Rough self-evaluation score in [0, 1] derived from text features.

    Heuristics:
    * Longer, more structured reasoning → higher score.
    * Many hedging words → lower score.
    * Explicit "checking" phrases → higher score.
    """
    score = 0.5
    words = text.lower().split()
    n = len(words)

    # Reward length (up to ~150 tokens).
    score += min(n / 300, 0.15)

    # Reward step-by-step structure.
    if re.search(r"step \d", text, re.I):
        score += 0.10
    if re.search(r"check|verify|confirm", text, re.I):
        score += 0.08

    # Penalise hedging.
    hedge_count = sum(
        1 for w in words if w in {"maybe", "perhaps", "possibly", "unclear", "unsure"}
    )
    score -= hedge_count * 0.05

    return float(max(0.0, min(1.0, score)))


def _heuristic_contradiction(text: str) -> float:
    """
    Simple contradiction score in [0, 1].

    Looks for self-correction phrases and conflicting numbers.
    """
    lower = text.lower()
    count = 0
    for phrase in ["wait,", "actually,", "no, that's wrong", "i made an error",
                   "let me reconsider", "that doesn't seem right"]:
        count += lower.count(phrase)

    # Conflicting numbers close together.
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    unique_numbers = set(numbers)
    if len(unique_numbers) > 5 and len(numbers) > len(unique_numbers) * 1.5:
        count += 1

    return float(min(1.0, count * 0.25))


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

_LLM_EXTRACTION_PROMPT = """\
Given the following reasoning trace, extract structured information as JSON.

Trace:
\"\"\"
{trace}
\"\"\"

Return a JSON object with these keys:
- provisional_answer: string (the final answer the trace reaches)
- key_assumptions: list of short strings (max 4)
- unresolved_issues: list of short strings (max 3)
- self_eval_score: float 0-1 (how sound is the reasoning?)
- contradiction_score: float 0-1 (how many contradictions / self-corrections?)

Output only the JSON object, nothing else.
"""


class StateExtractor:
    """
    Converts a raw LLM response into a :class:`ReasoningState`.

    Parameters
    ----------
    mode:
        ``"heuristic"`` (default) or ``"llm"``.
    llm_client:
        Required when ``mode="llm"``; used to call back the LLM for
        structured extraction.  Not used in heuristic mode.
    """

    def __init__(
        self,
        mode: str = "heuristic",
        llm_client: Optional[BaseLLMClient] = None,
    ) -> None:
        if mode not in ("heuristic", "llm"):
            raise ValueError(f"mode must be 'heuristic' or 'llm', got {mode!r}")
        if mode == "llm" and llm_client is None:
            raise ValueError("llm_client must be provided when mode='llm'")
        self.mode = mode
        self._llm = llm_client

    def extract(
        self,
        question: str,
        response: LLMResponse,
        step_index: int = 0,
    ) -> ReasoningState:
        """
        Build a :class:`ReasoningState` from a raw LLM response.

        Parameters
        ----------
        question:
            The original question.
        response:
            The LLM completion to analyse.
        step_index:
            Which metadynamics iteration produced this response.
        """
        if self.mode == "llm":
            return self._llm_extract(question, response, step_index)
        return self._heuristic_extract(question, response, step_index)

    # ------------------------------------------------------------------

    def _heuristic_extract(
        self,
        question: str,
        response: LLMResponse,
        step_index: int,
    ) -> ReasoningState:
        text = response.text
        return ReasoningState(
            question=question,
            trace_text=text,
            provisional_answer=_extract_answer(text),
            key_assumptions=_extract_assumptions(text),
            unresolved_issues=_extract_issues(text),
            self_eval_score=_heuristic_self_eval(text),
            contradiction_score=_heuristic_contradiction(text),
            logprob_stats=response.logprob_stats,
            step_index=step_index,
        )

    def _llm_extract(
        self,
        question: str,
        response: LLMResponse,
        step_index: int,
    ) -> ReasoningState:
        assert self._llm is not None
        prompt = _LLM_EXTRACTION_PROMPT.format(trace=response.text[:2000])
        ext_responses = self._llm.generate(prompt, temperature=0.0, max_tokens=256, n=1)
        raw = ext_responses[0].text.strip()

        try:
            # Strip markdown code fences if present.
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # Graceful fallback to heuristic.
            return self._heuristic_extract(question, response, step_index)

        return ReasoningState(
            question=question,
            trace_text=response.text,
            provisional_answer=str(data.get("provisional_answer", "unknown"))[:120],
            key_assumptions=data.get("key_assumptions", [])[:4],
            unresolved_issues=data.get("unresolved_issues", [])[:3],
            self_eval_score=float(data.get("self_eval_score", 0.5)),
            contradiction_score=float(data.get("contradiction_score", 0.0)),
            logprob_stats=response.logprob_stats,
            step_index=step_index,
        )
