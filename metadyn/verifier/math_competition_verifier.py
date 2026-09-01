"""
metadyn/verifier/math_competition_verifier.py
==============================================
Tool-based verifier for competition math (MATH benchmark).

Answers in MATH are LaTeX expressions: fractions, square roots, polynomials,
sets, intervals, etc.  This module provides:

  parse_competition_answer(text) → str
      Extract the best-guess answer from an LLM trace.
      Looks for \\boxed{...} first, then "Answer: ..." patterns.

  compare_competition_answers(pred, ref) → bool
      Attempt numeric or symbolic equality.  Falls back to
      normalised string comparison.

  MathCompetitionVerifier
      Scores a ReasoningState on structural signals:
        +0.40  a \\boxed{} answer present
        +0.20  math content: fractions / roots / algebra visible
        +0.20  answer is non-trivial (not just a single bare digit)
        +0.20  last arithmetic step verified by SymPy (if available)
"""

from __future__ import annotations

import re
import math
from typing import Optional

from metadyn.state.reasoning_state import ReasoningState
from metadyn.verifier.verifier import BaseVerifier

try:
    from sympy import simplify, sympify, N as sym_N, sqrt, pi, Rational
    from sympy.parsing.latex import parse_latex as _parse_latex
    _SYMPY = True
except ImportError:
    _SYMPY = False


# ---------------------------------------------------------------------------
# LaTeX normalisation helpers
# ---------------------------------------------------------------------------

def _strip_latex_formatting(s: str) -> str:
    """Remove display/text decorators that don't affect value."""
    s = re.sub(r"\\left|\\right|\\,|\\!", "", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\displaystyle\s*", "", s)
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = re.sub(r"\\tfrac", r"\\frac", s)
    s = s.strip()
    return s


def _try_numeric(s: str) -> Optional[float]:
    """Try to evaluate s as a float.  Returns None on failure."""
    clean = s.replace(",", "").replace(" ", "")
    try:
        return float(clean)
    except ValueError:
        pass
    # Simple fraction a/b
    m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", clean)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return num / den if den != 0 else None
    return None


def _sympy_eval(s: str) -> Optional[float]:
    """Try to evaluate LaTeX string numerically via SymPy."""
    if not _SYMPY:
        return None
    try:
        expr = _parse_latex(s)
        val = complex(sym_N(expr, 15))
        if abs(val.imag) < 1e-9:
            return float(val.real)
    except Exception:
        pass
    # Fallback: sympify on cleaned string
    try:
        clean = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
        clean = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", clean)
        clean = re.sub(r"\\sqrt\s+(\S)", r"sqrt(\1)", clean)
        clean = re.sub(r"[\\{}]", "", clean)
        expr = sympify(clean)
        val = complex(sym_N(expr, 15))
        if abs(val.imag) < 1e-9:
            return float(val.real)
    except Exception:
        pass
    return None


def _normalise_string(s: str) -> str:
    """String-level normalisation for fallback comparison."""
    s = _strip_latex_formatting(s)
    # Remove spaces around operators
    s = re.sub(r"\s+", "", s)
    # Normalise minus sign
    s = s.replace("−", "-")
    # Remove trailing zeros after decimal point
    s = re.sub(r"(\.\d*?)0+$", r"\1", s)
    s = re.sub(r"\.$", "", s)
    return s.lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_boxed_from_trace(text: str) -> Optional[str]:
    """Extract \\boxed{...} answer from an LLM trace (handles nested braces)."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        idx = text.rfind(r"\boxed {")
    if idx == -1:
        return None
    try:
        start = text.index("{", idx)
    except ValueError:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i].strip()
    return None


def parse_competition_answer(text: str) -> str:
    """
    Extract the best-guess answer from an LLM trace.

    Priority:
      1. Last \\boxed{...} in the trace.
      2. "Answer: X" / "The answer is X" pattern.
      3. "=" at end of last non-empty line.
      4. "?" on failure.
    """
    # 1. boxed
    boxed = _extract_boxed_from_trace(text)
    if boxed:
        return boxed

    # 2. Answer: X (strip leading $ or \$)
    for pat in [
        r"[Aa]nswer\s*[:=]\s*\$?\s*([^\n$]+)",
        r"[Tt]he\s+answer\s+is\s+\$?\s*([^\n$]+)",
        r"[Tt]herefore[,\s]+(?:the\s+answer\s+is\s+)?\$?\s*([^\n$]+)",
    ]:
        m = re.search(pat, text)
        if m:
            candidate = m.group(1).strip().rstrip(".,;")
            if candidate:
                return candidate

    # 3. Last line ending with "= VALUE"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        m = re.search(r"=\s*([^\s=]+)\s*\.?\s*$", lines[-1])
        if m:
            return m.group(1).strip()

    return "?"


def compare_competition_answers(pred: str, ref: str, rel_tol: float = 0.01) -> bool:
    """
    Return True if *pred* and *ref* represent the same mathematical value.

    Strategy:
      1. String equality after normalisation.
      2. Numeric equality (both parseable as float, within rel_tol).
      3. SymPy symbolic equality (simplify(pred - ref) == 0).
    """
    if pred in ("?", "") or ref in ("?", ""):
        return False

    # 1. Normalised string
    if _normalise_string(pred) == _normalise_string(ref):
        return True

    # 2. Direct numeric
    pv = _try_numeric(pred)
    rv = _try_numeric(ref)
    if pv is not None and rv is not None:
        if rv == 0.0:
            return abs(pv) < 1e-6
        return abs(pv - rv) / abs(rv) < rel_tol

    # 3. SymPy evaluation
    pv_sym = _sympy_eval(pred)
    rv_sym = _sympy_eval(ref)
    if pv_sym is not None and rv_sym is not None:
        if rv_sym == 0.0:
            return abs(pv_sym) < 1e-6
        return abs(pv_sym - rv_sym) / abs(rv_sym) < rel_tol

    # 4. SymPy symbolic simplify (pred - ref == 0)
    if _SYMPY:
        try:
            pred_clean = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", pred)
            pred_clean = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", pred_clean)
            pred_clean = re.sub(r"[\\{}]", "", pred_clean)
            ref_clean = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", ref)
            ref_clean = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", ref_clean)
            ref_clean = re.sub(r"[\\{}]", "", ref_clean)
            diff = simplify(sympify(pred_clean) - sympify(ref_clean))
            if diff == 0:
                return True
        except Exception:
            pass

    return False


# ---------------------------------------------------------------------------
# MathCompetitionVerifier
# ---------------------------------------------------------------------------

class MathCompetitionVerifier(BaseVerifier):
    """
    Tool-based verifier for competition math reasoning states.

    Scoring (no ground truth used):
      +0.40  \\boxed{...} present in trace
      +0.20  math-rich content (fractions / sqrt / equations visible)
      +0.20  answer appears non-trivial (not bare single digit)
      +0.20  last arithmetic step verified by SymPy
    """

    def score(self, state: ReasoningState) -> float:
        text = state.trace_text
        s = 0.0

        # +0.40 boxed answer present
        if _extract_boxed_from_trace(text):
            s += 0.40

        # +0.20 math-rich content
        math_signals = [
            r"\\frac", r"\\sqrt", r"=\s*[\d\-]",
            r"\d+\s*[\+\-\*\/]\s*\d+", r"therefore|thus|so we",
        ]
        if sum(1 for pat in math_signals if re.search(pat, text, re.I)) >= 2:
            s += 0.20

        # +0.20 non-trivial answer
        answer = parse_competition_answer(text)
        if answer != "?" and len(answer) > 1:
            s += 0.20

        # +0.20 last arithmetic step verified
        s += 0.20 * self._verify_steps(text)

        return min(1.0, max(0.0, s))

    @staticmethod
    def _verify_steps(text: str, n_steps: int = 3) -> float:
        """Return fraction of verifiable arithmetic equations that are correct."""
        if not _SYMPY:
            return 0.5
        eq_pat = re.compile(
            r"([\d\s\+\-\*\/\.\(\)]+)\s*=\s*([\-]?\d[\d,]*\.?\d*)\s*[\.,$\n]"
        )
        equations = eq_pat.findall(text[-800:])
        if not equations:
            return 0.5
        correct = 0
        checked = 0
        for lhs_raw, rhs_raw in equations[-n_steps:]:
            rhs_v = _try_numeric(rhs_raw)
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

    def extract_answer(self, text: str) -> str:
        return parse_competition_answer(text)
