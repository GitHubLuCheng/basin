"""
Structured reasoning-state extractor for MuSR.

Problem: raw SBERT embeddings of full reasoning traces are dominated by the
shared narrative content of the MuSR story, so all traces for the same problem
cluster into one basin regardless of which answer they reach.

Solution: a second-pass LLM call extracts a compact JSON "reasoning fingerprint"
that focuses on *what the model concluded and why*, stripping out the shared story.
The fingerprint is then embedded with SBERT, giving basin boundaries that reflect
reasoning type rather than narrative overlap.

JSON schema returned by :meth:`extract_structured`:
    {
      "final_answer":     "C",          # always overwritten with predicted_answer
      "main_hypothesis":  "Oliver suits medical because he takes first-aid courses annually",
      "key_evidence":     ["Oliver partakes in first-aid courses", "Jack is squeamish at blood"],
      "eliminated_option":"A because Emily would be wasted as trainer given her medical background",
      "reasoning_summary":"Assign by explicit skill match: sports trainer Jack, medical Emily+Oliver",
      "extraction_failed": false
    }

Embed text (passed to SBERT):
    Success: "Answer:C Hypothesis:<...> Evidence:<...> Eliminated:<...> Strategy:<...>"
    Failure: "Answer:C"   (only the answer letter — no garbage hypothesis text)

Symbolic basin key:
    Success: "C | Oliver suits medical..."    (answer letter + first 60 chars of hypothesis)
    Failure: "FAILED|C"                       (separate fallback bin, doesn't pollute semantic space)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from metadyn.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are given a reasoning trace that answers a multiple-choice question.
Extract the following information as a JSON object with EXACTLY these five keys:

  "final_answer"    : the answer letter chosen (A / B / C / D / E)
  "main_hypothesis" : ONE sentence stating the single most important argument for that answer
  "key_evidence"    : a JSON list of 2-3 short strings citing specific facts from the story
  "eliminated_option": ONE sentence naming which option was ruled out and the decisive reason
  "reasoning_summary": ONE sentence describing the overall reasoning strategy used

Return ONLY the JSON object — no markdown fences, no explanation, nothing else.

Reasoning trace:
\"\"\"
{trace}
\"\"\"
"""


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class StructuredMuSRExtractor:
    """
    Two-pass extractor for MuSR reasoning states.

    Pass 1 (already done upstream): heuristic extraction → raw trace text.
    Pass 2 (this class): LLM JSON extraction → compact structured fingerprint.

    The structured fingerprint is used as the embedding target so that basin
    clustering reflects reasoning diversity rather than narrative overlap.

    Parameters
    ----------
    llm_client:
        LLM used for the structured extraction call.  Typically the same
        model as the main reasoner (temperature=0 for determinism).
    max_tokens:
        Token budget for the JSON extraction response.
    """

    def __init__(self, llm_client: BaseLLMClient, max_tokens: int = 600) -> None:
        self.llm = llm_client
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def extract_structured(self, trace_text: str, predicted_answer: str = "") -> Dict:
        """
        Call the LLM to extract a structured reasoning fingerprint from *trace_text*.

        The ``predicted_answer`` (already extracted from the trace by the caller)
        is **always** used as ``final_answer``.  The LLM's own answer inference is
        ignored to prevent label overwriting.

        Falls back gracefully if the LLM returns invalid JSON or omits required keys.
        On failure all text fields are set to empty strings and ``extraction_failed``
        is set to ``True`` — no sentinel text is injected into the basin representation.

        Parameters
        ----------
        trace_text:
            The full chain-of-thought trace to analyse (trimmed to 1500 chars).
        predicted_answer:
            The already-normalised single uppercase letter (A–E) extracted from
            the trace by the caller (e.g. via ``normalise_musr_answer``).  If empty,
            the extractor will try a best-effort heuristic.

        Returns
        -------
        dict with keys: final_answer, main_hypothesis, key_evidence,
        eliminated_option, reasoning_summary, extraction_failed.
        """
        # Normalise predicted_answer to a single uppercase letter.
        pa = str(predicted_answer).strip().upper()
        pa = pa[0] if pa and pa[0] in "ABCDE" else "?"

        prompt = _EXTRACTION_PROMPT.format(trace=trace_text[:1500])
        try:
            responses = self.llm.generate(
                prompt, temperature=0.0, max_tokens=self.max_tokens, n=1
            )
            raw = responses[0].text.strip()
            # Strip optional markdown code fences.
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            # Find first '{' to last '}'.
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
            data = json.loads(raw)
        except Exception:
            return self._make_failed(pa)

        required = {
            "final_answer", "main_hypothesis", "key_evidence",
            "eliminated_option", "reasoning_summary",
        }
        if not required.issubset(data.keys()):
            return self._make_failed(pa)

        # Check for answer mismatch between LLM inference and trace extraction.
        llm_ans = str(data.get("final_answer", "?")).strip().upper()
        llm_ans = llm_ans[0] if llm_ans and llm_ans[0] in "ABCDE?" else "?"
        answer_mismatch = (llm_ans != pa and pa != "?" and llm_ans != "?")
        if answer_mismatch:
            logger.debug(
                "Structured extractor answer mismatch: LLM said %s, trace predicted %s — using %s",
                llm_ans, pa, pa,
            )

        # Normalise & truncate fields.
        evid = data.get("key_evidence", [])
        if not isinstance(evid, list):
            evid = [str(evid)]

        # Post-check: ensure all text fields are non-None strings.
        hyp  = str(data.get("main_hypothesis", "") or "").strip()[:200]
        elim = str(data.get("eliminated_option", "") or "").strip()[:200]
        summ = str(data.get("reasoning_summary", "") or "").strip()[:200]
        evid = [str(e).strip()[:100] for e in evid[:3] if e]

        return {
            "final_answer":      pa,          # always trust predicted_answer, not LLM
            "main_hypothesis":   hyp,
            "key_evidence":      evid,
            "eliminated_option": elim,
            "reasoning_summary": summ,
            "extraction_failed": False,
            "_llm_answer":       llm_ans,     # diagnostic only — LLM's inference before override
            "_answer_mismatch":  answer_mismatch,
        }

    # ------------------------------------------------------------------
    # Text builders
    # ------------------------------------------------------------------

    @staticmethod
    def make_embed_text(struct: Dict) -> str:
        """
        Flatten the structured state into a single string for SBERT embedding.

        If extraction failed, returns only the answer letter so that failed
        traces do not all collapse to the same vector via shared sentinel text.
        """
        ans = struct.get("final_answer", "?")
        if struct.get("extraction_failed", False):
            # Minimal representation — prevents garbage text from dominating embedding.
            return f"Answer:{ans}"

        hyp  = struct.get("main_hypothesis", "")
        evid = " | ".join(struct.get("key_evidence", []))
        elim = struct.get("eliminated_option", "")
        summ = struct.get("reasoning_summary", "")
        return f"Answer:{ans} Hypothesis:{hyp} Evidence:{evid} Eliminated:{elim} Strategy:{summ}"

    @staticmethod
    def make_basin_key(struct: Dict) -> str:
        """
        Return a short symbolic key.

        - Success:  ``"<ANSWER> | <hypothesis[:60]>"``
        - Failure:  ``"FAILED|<ANSWER>"``  (kept separate to avoid polluting semantic clusters)
        """
        ans = struct.get("final_answer", "?").upper()
        if struct.get("extraction_failed", False):
            return f"FAILED|{ans}"
        hyp = struct.get("main_hypothesis", "")[:60]
        return f"{ans} | {hyp}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_failed(predicted_answer: str) -> Dict:
        """Return a clean failure record.  All text fields are empty strings."""
        return {
            "final_answer":      predicted_answer,
            "main_hypothesis":   "",
            "key_evidence":      [],
            "eliminated_option": "",
            "reasoning_summary": "",
            "extraction_failed": True,
        }

    # Keep old _fallback for backward compatibility (not used internally).
    def _fallback(self, trace_text: str) -> Dict:
        from metadyn.datasets.musr_loader import normalise_musr_answer  # lazy import
        ans = normalise_musr_answer(trace_text).upper()
        if not (ans and ans[0] in "ABCDE"):
            ans = "?"
        return self._make_failed(ans)
