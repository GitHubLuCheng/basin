"""
basin/verifier/tool_verifier.py
===================================
Objective, LLM-free tool-based verifier for math problems.

Uses regex pattern matching + SymPy arithmetic checks to score how
likely a reasoning state contains a correct numeric answer.  No ground-
truth comparison is made — scores are based purely on structural signals
from the trace text.

Scoring heuristic:
  • +0.50  a clean numeric answer was extractable from the trace
  • +0.20  the answer is a non-negative finite number (GSM8K are positive)
  • +0.15  the answer is an integer (most GSM8K answers are whole numbers)
  • +0.15  the last verifiable arithmetic step is internally correct (SymPy)
  = 1.00  maximum

Also exposes:
  parse_math_answer(text) → str       — best-effort numeric string or "?"
  compare_math_answers(a, b) → bool   — numeric equality with 1% tolerance
"""

from __future__ import annotations

import math
import re
from typing import List, Optional

from basin.state.reasoning_state import ReasoningState
from basin.verifier.verifier import BaseVerifier

try:
    from sympy import sympify, N as sym_N
    _SYMPY = True
except ImportError:
    _SYMPY = False


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

# Patterns ordered by specificity.  We look for these in the WHOLE trace.
_PATTERNS = [
    r"####\s*([\-\d,\.]+)",                             # GSM8K GT: #### 42
    r"[Aa]nswer\s*[:=]\s*\$?\s*([\-\d,\.]+)",          # Answer: 42 / Answer: $42
    r"[Tt]he\s+answer\s+is\s+\$?\s*([\-\d,\.]+)",      # The answer is 42
    r"[Tt]herefore[,\s]+\$?\s*([\-\d,\.]+)\s*\.?\s*$", # Therefore, 42.
    r"[Ss]o\s+the\s+answer\s+is\s+\$?\s*([\-\d,\.]+)",
    r"=\s*([\-\d,\.]+)\s*\.?\s*$",                      # = 42 at line end
]


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _extract_numeric_value(text: str) -> Optional[float]:
    """Extract the most likely numeric answer from text.  Returns float or None."""
    # Try specific patterns first.
    for pat in _PATTERNS:
        m = re.search(pat, text)
        if m:
            v = _try_float(m.group(1))
            if v is not None:
                return v
    # Fallback: last standalone integer or decimal in the text.
    nums = re.findall(r"(?<!\d)([\-]?\d[\d,]*\.?\d*)(?!\d)", text)
    if nums:
        for raw in reversed(nums):
            v = _try_float(raw)
            if v is not None:
                return v
    return None


def parse_math_answer(text: str) -> str:
    """
    Extract the numeric answer from *text* as a canonical string.

    Returns the integer string if the value is whole (e.g. "42"),
    the decimal string otherwise (e.g. "3.14"), or "?" on failure.
    """
    v = _extract_numeric_value(text)
    if v is None:
        return "?"
    if math.isfinite(v) and v == math.floor(v) and abs(v) < 1e15:
        return str(int(v))
    return str(v)


def compare_math_answers(pred: str, ref: str, rel_tol: float = 0.01) -> bool:
    """
    Return True if *pred* and *ref* represent the same numeric value.

    Uses 1 % relative tolerance to handle floating-point noise and
    minor formatting differences.  Falls back to string equality if
    either value cannot be parsed.
    """
    if pred in ("?", "") or ref in ("?", ""):
        return False
    pv = _try_float(pred)
    rv = _try_float(ref)
    if pv is None or rv is None:
        return pred.strip() == ref.strip()
    if rv == 0.0:
        return abs(pv) < 1e-6
    return abs(pv - rv) / abs(rv) < rel_tol


# ---------------------------------------------------------------------------
# Arithmetic step checker
# ---------------------------------------------------------------------------

def _verify_last_steps(text: str, n_steps: int = 3) -> float:
    """
    Check up to *n_steps* arithmetic equations near the end of *text*.

    Returns fraction of checkable equations that are correct (SymPy).
    Falls back to 0.5 when SymPy is unavailable or no equations found.
    """
    if not _SYMPY:
        return 0.5

    # Match "lhs = rhs" where lhs looks like a numeric expression.
    eq_pat = re.compile(
        r"([\d\s\+\-\*\/\.\(\)]+)\s*=\s*([\-]?\d[\d,]*\.?\d*)"
    )
    equations = eq_pat.findall(text[-600:])
    if not equations:
        return 0.5

    correct = 0
    checked = 0
    for lhs_raw, rhs_raw in equations[-n_steps:]:
        rhs_v = _try_float(rhs_raw)
        if rhs_v is None:
            continue
        lhs_clean = lhs_raw.replace(",", "").strip()
        try:
            lhs_v = float(sym_N(sympify(lhs_clean), 15))
            checked += 1
            if abs(lhs_v - rhs_v) < max(1e-6, abs(rhs_v) * 1e-4):
                correct += 1
        except Exception:
            pass

    return (correct / checked) if checked > 0 else 0.5


# ---------------------------------------------------------------------------
# MathToolVerifier
# ---------------------------------------------------------------------------

class MathToolVerifier(BaseVerifier):
    """
    Objective, LLM-free verifier for math reasoning states.

    All signal comes from the structure and arithmetic in *state.trace_text*.
    No ground-truth comparison is performed.

    Scoring:
      +0.50  clean numeric answer extractable
      +0.20  answer is non-negative and finite
      +0.15  answer is a whole number (int)
      +0.15  last arithmetic step verified by SymPy
    """

    def score(self, state: ReasoningState) -> float:
        text = state.trace_text
        v = _extract_numeric_value(text)

        if v is None:
            # No answer at all — very weak candidate.
            return 0.10

        s = 0.50  # base: we found an answer

        if math.isfinite(v) and v >= 0.0:
            s += 0.20

        if math.isfinite(v) and v == math.floor(v) and abs(v) < 1e12:
            s += 0.15

        step_score = _verify_last_steps(text)
        s += 0.15 * step_score

        return min(1.0, max(0.0, s))

    def extract_answer(self, text: str) -> str:
        """Convenience wrapper — returns canonical answer string."""
        return parse_math_answer(text)
