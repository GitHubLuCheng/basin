"""
experiments/run_game24_tot_vs_hybrid.py
=======================================
Tree-of-Thoughts (ToT) vs ToT + ThoughtScape Basin Bias on Game of 24.

Reproduces the exact benchmark subset used in the ToT paper:
  puzzles with indices 900–999 (0-based) from the standard 24.csv.

Methods
-------
  cot              — Chain-of-Thought (single-shot, temperature=0.7)
  self_consistency — CoT × 10 samples, majority vote
  tot_standard     — BFS-ToT, pure exhaustive-solver state value
  tot_hybrid       — BFS-ToT + ThoughtScape basin bias (λ=1.5)

Verification
------------
  Exact Python arithmetic using Fraction — no LLM verifier.
  Checks: uses exactly the given numbers, evaluates to 24.

ToT Setup (faithful to original paper)
---------------------------------------
  breadth  = 5   (thoughts proposed per beam state)
  beam_width = 5 (states kept at each depth level)
  depth    = 3   (4 numbers → 3 → 2 → 1 == 24)
  proposal temperature = 0.7
  state value = exhaustive solver (sure=1.0 / impossible=0.0)

Basin Definition (Game of 24)
------------------------------
  basin_key = (sorted_operation_pattern, sorted_remaining_numbers)

  The operation pattern is the sorted tuple of operation types
  (add/sub/mul/div) applied so far.  Combined with the sorted
  remaining numbers, this groups branches pursuing the same
  global arithmetic strategy.

  Examples:
    steps=["4 * 6 = 24 (remaining: ...)"]         → ops=(mul,)
    steps=["4+8=12", "12*2=24"]                   → ops=(add,mul) — sorted
    remaining=(8, 6) → basin "add | 6,8"

Hybrid Scoring
--------------
  score_hybrid = score_tot − λ · log(1 + basin_visits[key])
  λ = 1.5 (default)

  Basin visits are accumulated across BFS levels for a single problem.
  After each level, every state that enters the beam increments its basin
  visit count so future levels are penalised for re-entering the same basin.

Outputs → outputs/game24_tot_vs_hybrid/
  per_example.json
  summary.md
  case_studies.md
  tables/game24_tot_main.csv
  tables/game24_tot_exploration.csv
  tables/game24_tot_failures.csv
  figures/game24_fig1_accuracy.png
  figures/game24_fig2_runtime.png
  figures/game24_fig3_basins.png
  figures/game24_fig4_basin_occupancy.png

Usage
-----
  python experiments/run_game24_tot_vs_hybrid.py \\
      --api_key <NRP_KEY> \\
      [--base_url https://ellm.nrp-nautilus.io/v1] \\
      [--model gpt-oss] \\
      [--breadth 5] [--beam_width 5] [--depth 3] \\
      [--lambda_ 1.5] [--sc_samples 10]
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import operator
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from basin.llm.openai_backend import OpenAIClient

# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------
# These are set dynamically in main() after parsing --output_dir.
_OUTPUT_DIR: Path = _REPO_ROOT / "outputs" / "game24_tot_vs_hybrid"
_TABLE_DIR:  Path = _OUTPUT_DIR / "tables"
_FIG_DIR:    Path = _OUTPUT_DIR / "figures"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _init_output_dirs(output_dir: str) -> None:
    """Set module-level output path globals and create directories."""
    global _OUTPUT_DIR, _TABLE_DIR, _FIG_DIR
    _OUTPUT_DIR = Path(output_dir)
    _TABLE_DIR  = _OUTPUT_DIR / "tables"
    _FIG_DIR    = _OUTPUT_DIR / "figures"
    for d in (_OUTPUT_DIR, _TABLE_DIR, _FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # Add file handler now that we know the path
    fh = logging.FileHandler(_OUTPUT_DIR / "experiment.log", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                      datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
_GAME24_DATA_DIR  = _REPO_ROOT / "data" / "game24"
_GAME24_CSV_PATH  = _GAME24_DATA_DIR / "24.csv"
_GAME24_URL = (
    "https://raw.githubusercontent.com/princeton-nlp/"
    "tree-of-thought-llm/master/src/tot/data/24/24.csv"
)


def _download_game24() -> Path:
    """Download 24.csv from the ToT paper's public repo."""
    _GAME24_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading 24.csv from ToT paper repo …")
    urllib.request.urlretrieve(_GAME24_URL, _GAME24_CSV_PATH)
    logger.info(f"Saved to {_GAME24_CSV_PATH}")
    return _GAME24_CSV_PATH


def load_game24(start_idx: int = 900, end_idx: int = 999) -> List[Tuple[int, Tuple[int, ...]]]:
    """
    Load Game of 24 puzzles with 0-based indices [start_idx, end_idx].
    Returns list of (original_idx, numbers_tuple).

    The CSV has one puzzle per line.  The first column is the 4 numbers
    (space-separated).  Optional second column is a solved-rate annotation
    that we ignore.
    """
    if not _GAME24_CSV_PATH.exists():
        _download_game24()

    puzzles = []
    with open(_GAME24_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # CSV format: Rank,Puzzles,... — skip header (first column is non-numeric)
            first_cell = row[0].strip()
            if re.match(r"[A-Za-z]", first_cell):
                continue
            # Puzzles column is index 1 (rank is index 0); fallback: scan all cells
            puzzle_cell = row[1] if len(row) > 1 else row[0]
            nums_raw = re.findall(r"\d+", puzzle_cell)
            if len(nums_raw) == 4:
                puzzles.append(tuple(int(x) for x in nums_raw))

    if not puzzles:
        raise ValueError(f"No puzzles found in {_GAME24_CSV_PATH}")

    logger.info(f"Loaded {len(puzzles)} Game of 24 puzzles total")
    selected = [(i, puzzles[i]) for i in range(start_idx, min(end_idx + 1, len(puzzles)))]
    logger.info(f"Using indices {start_idx}–{min(end_idx, len(puzzles)-1)} ({len(selected)} puzzles)")
    return selected


# ---------------------------------------------------------------------------
# Exact arithmetic verifier
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node) -> Fraction:
    """Evaluate an AST node with Fraction arithmetic (exact, no float errors)."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return Fraction(node.value).limit_denominator(10**9)
        raise ValueError(f"Non-numeric constant: {node.value}")
    if isinstance(node, ast.BinOp):
        left  = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported op: {node.op}")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary: {node.op}")
        return op_fn(_safe_eval_node(node.operand))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _extract_numbers_from_ast(tree_body) -> List[int]:
    """Extract all numeric literals from an expression AST."""
    nums = []
    for node in ast.walk(tree_body):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            nums.append(int(round(node.value)))
    return nums


def verify_expression(expr_raw: str, numbers: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Verify a candidate Game of 24 expression.

    Returns a dict with:
      success      : bool
      error_type   : None | 'malformed' | 'illegal_op' | 'wrong_numbers'
                     | 'zero_division' | 'wrong_value'
      value        : float | None
      expr_used    : str (cleaned expression)
      numbers_used : list[int] (extracted from expr)
    """
    # --- Strip trailing "= 24" or similar
    expr_clean = expr_raw.strip()
    # Remove "= 24" at the end
    expr_clean = re.sub(r"\s*=\s*24\s*\.?\s*$", "", expr_clean).strip()
    # Remove outer "answer:" prefix
    expr_clean = re.sub(r"^(?:answer|result|expression)\s*[:=]\s*", "", expr_clean, flags=re.I)
    expr_clean = expr_clean.strip()

    # --- Whitelist check: only allow digits, ops, parens, spaces, dots
    if not re.fullmatch(r"[\d+\-*/().\s]+", expr_clean):
        return {
            "success": False, "error_type": "malformed",
            "value": None, "expr_used": expr_clean, "numbers_used": [],
        }

    # --- Parse
    try:
        tree = ast.parse(expr_clean, mode="eval")
    except SyntaxError:
        return {
            "success": False, "error_type": "malformed",
            "value": None, "expr_used": expr_clean, "numbers_used": [],
        }

    # --- Extract numbers
    nums_used = _extract_numbers_from_ast(tree.body)
    if sorted(nums_used) != sorted(numbers):
        return {
            "success": False, "error_type": "wrong_numbers",
            "value": None, "expr_used": expr_clean, "numbers_used": nums_used,
        }

    # --- Evaluate with Fraction
    try:
        result = _safe_eval_node(tree.body)
    except ZeroDivisionError:
        return {
            "success": False, "error_type": "zero_division",
            "value": None, "expr_used": expr_clean, "numbers_used": nums_used,
        }
    except Exception:
        return {
            "success": False, "error_type": "illegal_op",
            "value": None, "expr_used": expr_clean, "numbers_used": nums_used,
        }

    float_val = float(result)
    if abs(float_val - 24.0) < 1e-6:
        return {
            "success": True, "error_type": None,
            "value": float_val, "expr_used": expr_clean, "numbers_used": nums_used,
        }
    return {
        "success": False, "error_type": "wrong_value",
        "value": float_val, "expr_used": expr_clean, "numbers_used": nums_used,
    }


# ---------------------------------------------------------------------------
# Exhaustive state value (Game of 24 solver)
# ---------------------------------------------------------------------------

def _can_reach(nums: Tuple[Fraction, ...], target: Fraction) -> bool:
    """Exact recursive check if any subset of operations on nums reaches target."""
    if len(nums) == 1:
        return nums[0] == target
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = nums[i], nums[j]
            rest = tuple(nums[k] for k in range(n) if k not in (i, j))
            candidates = [a + b, a - b, a * b]
            if b != 0:
                candidates.append(a / b)
            for val in candidates:
                if _can_reach(rest + (val,), target):
                    return True
    return False


def state_is_solvable(remaining: Tuple[float, ...]) -> bool:
    """Return True iff remaining numbers can reach 24 via +−×÷."""
    frac = tuple(Fraction(r).limit_denominator(10**9) for r in remaining)
    return _can_reach(frac, Fraction(24))


def state_value(remaining: Tuple[float, ...]) -> float:
    """State value: 1.0 if solvable, 0.0 otherwise."""
    return 1.0 if state_is_solvable(remaining) else 0.0


# ---------------------------------------------------------------------------
# Basin key extraction (Game of 24)
# ---------------------------------------------------------------------------
_OP_CHARS = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
_STEP_PAT = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)


def extract_step_ops(trace_lines: List[str]) -> Tuple[str, ...]:
    """Return ordered tuple of operation types from a list of step strings."""
    ops = []
    for line in trace_lines:
        m = _STEP_PAT.search(line)
        if m:
            ops.append(_OP_CHARS.get(m.group(2), "?"))
    return tuple(ops)


def make_basin_key(trace_lines: List[str], remaining: Tuple[float, ...]) -> str:
    """
    Lightweight strategy signature for basin grouping.

    Format: "<ordered_ops> | <sorted_remaining>"
    Example: "mul,sub | 4,6"   or  "mul | 12"
    Ops are kept in execution order so mul->sub and sub->mul are distinct basins.
    """
    ops = extract_step_ops(trace_lines)
    ops_str = ",".join(ops) if ops else "none"
    rem_str = ",".join(
        str(int(r)) if r == int(r) else f"{r:.3g}"
        for r in sorted(remaining)
    )
    return f"{ops_str} | {rem_str}"


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_PROPOSE_SYSTEM = (
    "You are a Game of 24 expert. "
    "At each step you are given the CURRENT remaining numbers. "
    "Choose exactly two of them, apply one arithmetic operation (+, -, *, /), "
    "and list possible next steps. "
    "Format each step as: A op B = C (remaining: X Y Z) "
    "where X Y Z are the numbers left after replacing A and B with C. "
    "List up to 5 diverse candidate steps."
)

_FEW_SHOT_EXAMPLES = """\
Input: 4 4 6 8
Possible next steps:
4 + 8 = 12 (remaining: 4 6 12)
6 - 4 = 2 (remaining: 2 4 8)
4 * 6 = 24 (remaining: 4 8)
4 * 8 = 32 (remaining: 4 6)
6 + 8 = 14 (remaining: 4 4 14)

Input: 2 9 10 12
Possible next steps:
12 * 2 = 24 (remaining: 9 10)
2 + 12 = 14 (remaining: 9 10 14)
10 - 2 = 8 (remaining: 8 9 12)
10 - 9 = 1 (remaining: 1 2 12)
9 + 12 = 21 (remaining: 2 10 21)

Input: 9 10
Possible next steps:
10 - 9 = 1 (remaining: 1)
9 + 10 = 19 (remaining: 19)
9 * 10 = 90 (remaining: 90)
10 / 9 = 1.111 (remaining: 1.111)

Input: 4 6 12
Possible next steps:
4 * 6 = 24 (remaining: 12)
12 - 6 = 6 (remaining: 4 6)
6 - 4 = 2 (remaining: 2 12)
12 / 4 = 3 (remaining: 3 6)
4 + 6 = 10 (remaining: 10 12)
"""

_COT_SYSTEM = (
    "You are a Game of 24 expert. "
    "Given 4 numbers, write a step-by-step solution that reaches 24 using each number exactly once "
    "with +, -, *, /. "
    "End your answer with a final expression on its own line, formatted as:\n"
    "Answer: (expression) = 24"
)


def build_proposal_prompt(remaining: Tuple[float, ...]) -> str:
    """
    Build few-shot prompt for proposing next steps.

    Faithful to the original ToT paper: Input = CURRENT remaining numbers,
    not the original 4 numbers + trace. This prevents the model from
    confusing already-used numbers with available ones.
    """
    rem_str = " ".join(
        str(int(r)) if r == int(r) else f"{r:.4g}" for r in remaining
    )
    return f"{_FEW_SHOT_EXAMPLES}\nInput: {rem_str}\nPossible next steps:\n"


def build_cot_prompt(numbers: Tuple[int, ...]) -> str:
    nums_str = " ".join(str(n) for n in numbers)
    return (
        f"Use the numbers {nums_str} to make 24 using +, -, *, /. "
        f"Each number must be used exactly once. Show your work step by step. "
        f"End with: Answer: (your expression) = 24"
    )


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

_STEP_LINE_PAT = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)

# Remaining portion (optional) — for basin-key logging only
_STEP_LINE_REM_PAT = re.compile(
    r"(-?[\d.]+)\s*([+\-*/])\s*(-?[\d.]+)\s*=\s*(-?[\d.]+)\s*\(remaining:\s*([-\d\s./,]+)\)"
)


def _nums_close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-4


def parse_proposal(response_text: str, current_remaining: Tuple[float, ...]) -> List[Dict]:
    """
    Parse proposed next steps from LLM response.

    Flexible parser: finds all "A OP B = C" patterns, verifies A and B are
    in current_remaining, and computes new_remaining from current_remaining
    (does NOT rely on the model's stated remaining numbers).

    Returns list of dicts: {step_text, a, op, b, result, new_remaining, step_op}
    """
    steps = []
    seen_keys = set()

    for m in _STEP_LINE_PAT.finditer(response_text):
        try:
            a_val = float(m.group(1).rstrip("."))
            b_val = float(m.group(3).rstrip("."))
            result_val = float(m.group(4).rstrip("."))
        except ValueError:
            continue
        op = m.group(2)

        # --- Verify arithmetic
        try:
            if op == "+":   expected = a_val + b_val
            elif op == "-": expected = a_val - b_val
            elif op == "*": expected = a_val * b_val
            elif op == "/":
                if abs(b_val) < 1e-9: continue
                expected = a_val / b_val
            else: continue
        except Exception:
            continue
        if not _nums_close(expected, result_val):
            continue

        # --- Verify operands are in current_remaining
        rem_copy = list(current_remaining)
        valid = True
        for num in [a_val, b_val]:
            found = False
            for idx, r in enumerate(rem_copy):
                if _nums_close(r, num):
                    rem_copy.pop(idx)
                    found = True
                    break
            if not found:
                valid = False
                break
        if not valid:
            continue

        # Compute expected new remaining
        rem_copy.append(result_val)
        new_remaining = tuple(sorted(rem_copy))

        key = (round(a_val, 4), op, round(b_val, 4))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        steps.append({
            "step_text":     m.group(0).strip(),
            "a": a_val, "op": op, "b": b_val,
            "result":        result_val,
            "new_remaining": new_remaining,
            "step_op":       _OP_CHARS.get(op, "?"),
        })

    return steps


def _normalize_expr(text: str) -> str:
    """Normalize LaTeX math notation to plain arithmetic."""
    text = re.sub(r"\$+", "", text)          # strip $ delimiters
    text = text.replace(r"\times", "*").replace(r"\cdot", "*")
    text = text.replace(r"\div", "/").replace(r"\frac", "/")
    text = re.sub(r"\\[a-zA-Z]+", "", text)  # remove remaining latex commands
    return text.strip()


def parse_final_answer(response_text: str) -> Optional[str]:
    """
    Extract final expression from CoT/ToT terminal thought.

    Looks for:
      1. "Answer: ..."
      2. "= 24" at end of a line
      3. Any arithmetic expression on its own line
    Handles LaTeX math notation (\\times, $...$, etc.).
    """
    text = _normalize_expr(response_text)

    # Pattern 1: explicit "Answer:/Expression: expr = 24" or "Answer: expr"
    m = re.search(r"(?:[Aa]nswer|[Ee]xpression)\s*[:\-=]\s*(.+?)(?:\s*=\s*24)?\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).strip().rstrip(".,")

    # Pattern 2: line ending with "= 24"
    m = re.search(r"([\d\s+\-*/().]+)\s*=\s*24\s*\.?\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()

    # Pattern 3: last line with numbers and ops
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in reversed(lines[-5:]):
        if re.search(r"\d", line) and re.search(r"[+\-*/]", line):
            clean = re.sub(r"\s*=\s*24\s*\.?\s*$", "", line).strip()
            if clean:
                return clean

    return None


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepNode:
    """A single BFS node in the Game of 24 ToT search."""
    trace_lines:   List[str]
    remaining:     Tuple[float, ...]
    score_tot:     float          # Exhaustive solver value
    score_hybrid:  float          # With basin bias (same as score_tot for standard)
    basin_key:     str
    depth:         int
    parent_idx:    int = -1


@dataclass
class ProblemResult:
    """Full result for one Game of 24 puzzle."""
    idx:          int
    numbers:      Tuple[int, ...]
    method:       str
    success:      bool
    solution:     Optional[str]       # Verified expression
    candidates:   List[str]           # All terminal candidate expressions tried
    verifier_results: List[dict]      # One per candidate
    # Exploration
    n_nodes:      int = 0
    n_terminal:   int = 0
    n_basins:     int = 0
    neff:         float = 0.0
    basin_visits: Dict[str, int] = field(default_factory=dict)
    escape_rate:  float = 0.0
    revisit_rate: float = 0.0
    answer_diversity: float = 0.0
    # Failure types
    n_malformed:  int = 0
    n_wrong_nums: int = 0
    n_rejected:   int = 0
    n_dead_end:   int = 0
    # Meta
    n_calls:      int = 0
    elapsed:      float = 0.0
    # Any candidate correct (pass@k numerator)
    correct_any: bool = False
    # Adaptive λ gating: True if penalty was disabled after depth-0 diversity check
    lambda_gated: bool = False


# ---------------------------------------------------------------------------
# Chain-of-Thought runner
# ---------------------------------------------------------------------------

def run_cot(
    llm: OpenAIClient,
    numbers: Tuple[int, ...],
    n_samples: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> ProblemResult:
    """Run CoT (n_samples=1) or Self-Consistency (n_samples>1)."""
    method = "self_consistency" if n_samples > 1 else "cot"
    t0 = time.time()
    prompt = build_cot_prompt(numbers)

    responses = llm.generate(
        prompt, temperature=temperature, max_tokens=max_tokens, n=n_samples
    )

    candidates = []
    verifier_results = []
    n_malformed = n_wrong_nums = n_rejected = 0

    for resp in responses:
        expr = parse_final_answer(resp.text)
        if expr is None:
            n_malformed += 1
            candidates.append("")
            verifier_results.append({"success": False, "error_type": "malformed",
                                     "value": None, "expr_used": "", "numbers_used": []})
            continue
        v = verify_expression(expr, numbers)
        candidates.append(expr)
        verifier_results.append(v)
        if v["error_type"] == "malformed":       n_malformed  += 1
        elif v["error_type"] == "wrong_numbers": n_wrong_nums += 1
        elif not v["success"]:                   n_rejected   += 1

    # pass@k: any correct candidate
    correct_any = any(v["success"] for v in verifier_results)

    # Majority vote: pick most-common non-empty expression, then verify it
    if n_samples > 1:
        vote_counts: Dict[str, int] = {}
        for c in candidates:
            if c:
                vote_counts[c] = vote_counts.get(c, 0) + 1
        if vote_counts:
            best_expr = max(vote_counts, key=vote_counts.__getitem__)
            best_v = verify_expression(best_expr, numbers)
            selected_success = best_v["success"]
            solution = best_expr if selected_success else None
        else:
            selected_success = False
            solution = None
    else:
        # Single-sample CoT: success = whether that one answer is correct
        selected_success = verifier_results[0]["success"] if verifier_results else False
        solution = candidates[0] if (verifier_results and verifier_results[0]["success"]) else None

    return ProblemResult(
        idx=0, numbers=numbers, method=method,
        success=selected_success, solution=solution,
        candidates=candidates, verifier_results=verifier_results,
        n_nodes=n_samples, n_terminal=n_samples,
        n_malformed=n_malformed, n_wrong_nums=n_wrong_nums, n_rejected=n_rejected,
        n_calls=1, correct_any=correct_any,
        elapsed=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# SC + ThoughtScape runner
# ---------------------------------------------------------------------------

_OP_NAME = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


def _sc_basin_key(response_text: str) -> str:
    """
    Basin key for SC on Game of 24: the full operator sequence across all steps.

    Extracts all arithmetic steps and returns their operators in order, e.g.
    "*-+" for multiply-then-subtract-then-add.  This captures the solution
    strategy more faithfully than just the first step.
    Falls back to 'unknown' if no step lines are found.
    """
    ops = [m.group(2) for m in _STEP_LINE_PAT.finditer(response_text)]
    if not ops:
        return "unknown"
    return "".join(ops)


def _build_diversity_hint(basin_visits: Dict[str, int], numbers: Tuple[int, ...]) -> str:
    """
    Build a diversity hint based on which operator sequences have already been tried.
    Suggests an operator sequence (strategy) not yet explored.
    """
    if not basin_visits:
        return ""

    used_seqs = set(basin_visits.keys()) - {"unknown"}

    # Rank operators by how often they appear as the first op in visited sequences
    first_op_counts: Dict[str, int] = {}
    for seq, cnt in basin_visits.items():
        if seq and seq != "unknown":
            first_op_counts[seq[0]] = first_op_counts.get(seq[0], 0) + cnt

    # Suggest a first operation that has been tried least
    all_ops = ["+", "-", "*", "/"]
    least_used_op = min(all_ops, key=lambda o: first_op_counts.get(o, 0))

    # Suggest a full sequence not yet tried, starting with the least-used op
    op_names = {"+": "addition", "-": "subtraction", "*": "multiplication", "/": "division"}
    hint = f" (Hint: try a strategy that starts with {op_names[least_used_op]}."

    # Also mention if there are completely unexplored length-3 sequences
    explored_len3 = {s[:3] for s in used_seqs if len(s) >= 3}
    all_len3 = [a+b+c for a in all_ops for b in all_ops for c in all_ops]
    unexplored = [s for s in all_len3 if s not in explored_len3]
    if unexplored:
        s = unexplored[0]
        hint += f" For example, try {op_names[s[0]]}, then {op_names[s[1]]}, then {op_names[s[2]]}."
    hint += ")"
    return hint


def run_sc_hybrid(
    llm: OpenAIClient,
    numbers: Tuple[int, ...],
    n_samples: int = 10,
    lambda_: float = 1.5,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> ProblemResult:
    """
    SC + ThoughtScape: sequential sampling with diversity hints + basin-weighted
    aggregation. Basin key = sorted set of arithmetic operations in the solution.
    """
    t0 = time.time()
    basin_visits: Dict[str, int] = {}

    @dataclass
    class _Sample:
        expr: str
        weight: float
        v: dict

    samples: List[_Sample] = []
    n_malformed = n_wrong_nums = n_rejected = 0
    candidates: List[str] = []
    verifier_results: List[dict] = []

    for _ in range(n_samples):
        hint = _build_diversity_hint(basin_visits, numbers)
        base_prompt = build_cot_prompt(numbers)
        prompt = base_prompt + hint

        try:
            responses = llm.generate(
                prompt, temperature=temperature, max_tokens=max_tokens, n=1
            )
            resp_text = responses[0].text
        except Exception as e:
            logger.warning(f"SC hybrid LLM error: {e}")
            n_malformed += 1
            candidates.append("")
            verifier_results.append({"success": False, "error_type": "malformed",
                                     "value": None, "expr_used": "", "numbers_used": []})
            continue

        bk = _sc_basin_key(resp_text)
        # weight computed BEFORE updating visits (lower weight = over-visited basin)
        w = math.exp(-lambda_ * math.log(1 + basin_visits.get(bk, 0)))
        basin_visits[bk] = basin_visits.get(bk, 0) + 1

        expr = parse_final_answer(resp_text)
        if expr is None:
            n_malformed += 1
            candidates.append("")
            v = {"success": False, "error_type": "malformed",
                 "value": None, "expr_used": "", "numbers_used": []}
        else:
            v = verify_expression(expr, numbers)
            candidates.append(expr)
            if v["error_type"] == "malformed":       n_malformed  += 1
            elif v["error_type"] == "wrong_numbers": n_wrong_nums += 1
            elif not v["success"]:                   n_rejected   += 1

        verifier_results.append(v)
        if expr:
            samples.append(_Sample(expr=expr, weight=w, v=v))

    # Basin-weighted aggregation: C1(a) = sum of weights for samples with answer a
    c1: Dict[str, float] = {}
    for s in samples:
        c1[s.expr] = c1.get(s.expr, 0.0) + s.weight

    # Select answer with highest weighted vote
    correct_any = any(v.get("success", False) for v in verifier_results)
    if c1:
        best_expr = max(c1, key=c1.__getitem__)
        best_v = verify_expression(best_expr, numbers)
        selected_success = best_v["success"]
        solution = best_expr if selected_success else (
            next((c for c, v in zip(candidates, verifier_results) if v.get("success")), None)
        )
    else:
        selected_success = False
        solution = None

    # Exploration stats
    n_basins = len(basin_visits)
    if n_basins > 0:
        counts = np.array(list(basin_visits.values()), dtype=float)
        probs = counts / counts.sum()
        neff = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
    else:
        neff = 0.0
    n_revisits = sum(v - 1 for v in basin_visits.values() if v > 1)
    revisit_rate = n_revisits / max(1, n_samples - 1)
    escape_rate = sum(1 for v in basin_visits.values() if v == 1) / max(1, n_basins)

    return ProblemResult(
        idx=0, numbers=numbers, method="sc_hybrid",
        success=selected_success, solution=solution,
        candidates=candidates, verifier_results=verifier_results,
        n_nodes=n_samples, n_terminal=n_samples,
        n_basins=n_basins, neff=neff,
        basin_visits=basin_visits,
        revisit_rate=revisit_rate, escape_rate=escape_rate,
        n_malformed=n_malformed, n_wrong_nums=n_wrong_nums, n_rejected=n_rejected,
        n_calls=n_samples, correct_any=correct_any,
        elapsed=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# SC + ThoughtScape Fix 2 (two-phase) for Game of 24
# ---------------------------------------------------------------------------

def run_sc_twophase(
    llm: OpenAIClient,
    numbers: Tuple[int, ...],
    n_samples: int = 10,
    lambda_: float = 1.5,
    k_explore: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> "ProblemResult":
    """
    SC + ThoughtScape Fix 2 (two-phase):
      Phase 1 (k_explore samples): basin-biased diverse CoT to map operator strategies.
      Select best basin: strategy of any correct Phase 1 sample; if none correct,
        the strategy with the most Phase 1 samples.
      Phase 2 (n-k samples): focused CoT hinting the best operator strategy.
      Accuracy: any Phase 2 sample correct.
      Pass@k:   any sample from either phase correct.
    """
    t0 = time.time()
    k = min(k_explore, n_samples - 1)

    basin_visits: Dict[str, int] = {}
    candidates_p1: List[str] = []
    verifier_results_p1: List[dict] = []
    basin_keys_p1: List[str] = []
    n_malformed = n_wrong_nums = n_rejected = 0

    # ---- Phase 1: diverse exploration ----
    for _ in range(k):
        hint = _build_diversity_hint(basin_visits, numbers)
        prompt = build_cot_prompt(numbers) + hint
        try:
            responses = llm.generate(prompt, temperature=temperature,
                                     max_tokens=max_tokens, n=1)
            resp_text = responses[0].text
        except Exception as e:
            logger.warning(f"SC-2P p1 error: {e}")
            resp_text = ""

        bk = _sc_basin_key(resp_text)
        basin_visits[bk] = basin_visits.get(bk, 0) + 1
        basin_keys_p1.append(bk)

        expr = parse_final_answer(resp_text)
        if expr is None:
            n_malformed += 1
            candidates_p1.append("")
            verifier_results_p1.append({"success": False, "error_type": "malformed",
                                        "value": None, "expr_used": "", "numbers_used": []})
        else:
            v = verify_expression(expr, numbers)
            candidates_p1.append(expr)
            verifier_results_p1.append(v)
            if v["error_type"] == "malformed":       n_malformed  += 1
            elif v["error_type"] == "wrong_numbers": n_wrong_nums += 1
            elif not v["success"]:                   n_rejected   += 1

    # ---- Select best strategy ----
    # Prefer: strategy of a correct Phase 1 sample.
    # Fallback: strategy with most visits.
    best_strategy: Optional[str] = None
    for bk, expr, v in zip(basin_keys_p1, candidates_p1, verifier_results_p1):
        if v.get("success") and bk != "unknown":
            best_strategy = bk
            break
    if best_strategy is None and basin_visits:
        best_strategy = max(basin_visits, key=basin_visits.__getitem__)

    # Build focus hint for Phase 2
    op_names = {"+": "addition", "-": "subtraction", "*": "multiplication", "/": "division"}
    if best_strategy and best_strategy != "unknown":
        ops_readable = " → ".join(op_names.get(c, c) for c in best_strategy)
        focus_hint = f" (Hint: Use the operator strategy: {ops_readable}.)"
    else:
        focus_hint = ""

    # ---- Phase 2: focused exploitation ----
    candidates_p2: List[str] = []
    verifier_results_p2: List[dict] = []
    for _ in range(n_samples - k):
        prompt = build_cot_prompt(numbers) + focus_hint
        try:
            responses = llm.generate(prompt, temperature=temperature,
                                     max_tokens=max_tokens, n=1)
            resp_text = responses[0].text
        except Exception as e:
            logger.warning(f"SC-2P p2 error: {e}")
            resp_text = ""

        expr = parse_final_answer(resp_text)
        if expr is None:
            n_malformed += 1
            candidates_p2.append("")
            verifier_results_p2.append({"success": False, "error_type": "malformed",
                                        "value": None, "expr_used": "", "numbers_used": []})
        else:
            v = verify_expression(expr, numbers)
            candidates_p2.append(expr)
            verifier_results_p2.append(v)
            if v["error_type"] == "malformed":       n_malformed  += 1
            elif v["error_type"] == "wrong_numbers": n_wrong_nums += 1
            elif not v["success"]:                   n_rejected   += 1

    # ---- Aggregate ----
    all_candidates = candidates_p1 + candidates_p2
    all_verifier   = verifier_results_p1 + verifier_results_p2
    correct_any = any(v.get("success") for v in all_verifier)

    # Accuracy: any correct in Phase 2 (exploitation phase)
    p2_correct = [v for v in verifier_results_p2 if v.get("success")]
    selected_success = len(p2_correct) > 0
    solution = next((c for c, v in zip(candidates_p2, verifier_results_p2)
                     if v.get("success")), None)
    if solution is None:
        solution = next((c for c, v in zip(candidates_p1, verifier_results_p1)
                         if v.get("success")), None)

    # Basin metrics over Phase 1 visits
    n_basins = len(basin_visits)
    if n_basins > 0:
        counts_arr = np.array(list(basin_visits.values()), dtype=float)
        probs = counts_arr / counts_arr.sum()
        neff = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
    else:
        neff = 0.0
    n_revisits = sum(v - 1 for v in basin_visits.values() if v > 1)
    revisit_rate = n_revisits / max(1, k - 1)
    escape_rate = sum(1 for v in basin_visits.values() if v == 1) / max(1, n_basins)

    return ProblemResult(
        idx=0, numbers=numbers, method="sc_twophase",
        success=selected_success, solution=solution,
        candidates=all_candidates, verifier_results=all_verifier,
        n_nodes=n_samples, n_terminal=n_samples,
        n_basins=n_basins, neff=neff,
        basin_visits=basin_visits,
        revisit_rate=revisit_rate, escape_rate=escape_rate,
        n_malformed=n_malformed, n_wrong_nums=n_wrong_nums, n_rejected=n_rejected,
        n_calls=n_samples, correct_any=correct_any,
        elapsed=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# ToT BFS runner
# ---------------------------------------------------------------------------

def _propose_next_steps(
    llm: OpenAIClient,
    remaining: Tuple[float, ...],
    breadth: int,
    temperature: float,
    max_tokens: int = 512,
    max_retries: int = 5,
    retry_base_delay: float = 10.0,
) -> List[Dict]:
    """
    Call LLM once to propose up to `breadth` next steps.
    Retries with exponential backoff on transient errors (e.g. "no healthy upstream").

    Faithful to the ToT paper: Input = current remaining numbers.
    The system prompt instructs the model on the format.
    """
    prompt = build_proposal_prompt(remaining)
    full_prompt = f"{_PROPOSE_SYSTEM}\n\n{prompt}"

    for attempt in range(max_retries):
        try:
            responses = llm.generate(
                full_prompt, temperature=temperature, max_tokens=max_tokens, n=1
            )
            parsed = parse_proposal(responses[0].text, remaining)
            if parsed:
                return parsed[:breadth]
            # Empty parse — may be a bad response, retry
            logger.warning(f"Proposal returned no steps (attempt {attempt+1}/{max_retries}), retrying...")
        except Exception as e:
            delay = retry_base_delay * (2 ** attempt)
            logger.warning(f"Proposal LLM error (attempt {attempt+1}/{max_retries}): {e} — retrying in {delay:.0f}s")
            time.sleep(delay)
            continue

        if attempt < max_retries - 1:
            time.sleep(retry_base_delay)

    logger.error(f"Proposal failed after {max_retries} attempts for remaining={remaining}")
    return []


def _generate_terminal_candidates(
    remaining: Tuple[float, ...],
) -> List[str]:
    """
    At final depth (2 remaining numbers), enumerate all 6 arithmetic operations
    and return those that yield exactly 24.  No LLM call needed — exhaustive
    and 100% reliable.
    """
    if len(remaining) == 1:
        return ["(reconstructed from steps)"] if abs(remaining[0] - 24) < 1e-6 else []

    if len(remaining) != 2:
        return []

    a, b = remaining
    candidates = []
    for x, y, label in [(a, b, (a, b)), (b, a, (b, a))]:
        # addition (commutative — only add once)
        if x <= y:
            if abs(x + y - 24) < 1e-6:
                candidates.append(f"{x:g} + {y:g} = 24")
        # subtraction
        if abs(x - y - 24) < 1e-6:
            candidates.append(f"{x:g} - {y:g} = 24")
        # multiplication (commutative — only add once)
        if x <= y:
            if abs(x * y - 24) < 1e-6:
                candidates.append(f"{x:g} * {y:g} = 24")
        # division
        if abs(y) > 1e-9:
            if abs(x / y - 24) < 1e-6:
                candidates.append(f"{x:g} / {y:g} = 24")

    return candidates


def run_tot_bfs(
    llm: OpenAIClient,
    numbers: Tuple[int, ...],
    depth: int = 3,
    breadth: int = 5,
    beam_width: int = 5,
    temperature: float = 0.7,
    lambda_: Optional[float] = None,
    max_tokens: int = 512,
    quality_aware: bool = True,
) -> ProblemResult:
    """
    BFS Tree-of-Thoughts for Game of 24.

    If lambda_ is None: standard ToT (pure exhaustive-solver score).
    If lambda_ is float: hybrid ToT with ThoughtScape basin bias.
    """
    method = "tot_hybrid" if lambda_ is not None else "tot_standard"
    t0 = time.time()

    initial_rem = tuple(float(n) for n in numbers)
    beam: List[StepNode] = [StepNode(
        trace_lines=[], remaining=initial_rem,
        score_tot=1.0, score_hybrid=1.0,
        basin_key=make_basin_key([], initial_rem),
        depth=0,
    )]

    basin_visits: Dict[str, int] = defaultdict(int)
    basin_quality: Dict[str, float] = defaultdict(float)  # running mean score_tot per basin
    all_nodes: List[StepNode] = list(beam)
    n_calls = 0
    effective_lambda = lambda_   # may be zeroed after depth-0 if already diverse
    lambda_gated_off = False

    # BFS main loop
    for d in range(depth):
        is_final = (d == depth - 1)
        candidates: List[StepNode] = []

        for node in beam:
            rem = node.remaining
            if len(rem) < 2:
                continue

            if is_final:
                # At final depth: remaining has 2 numbers → enumerate all ops (no LLM)
                terminal_exprs = _generate_terminal_candidates(rem)
                for expr in terminal_exprs:
                    # These become leaf nodes with "remaining = (result,)"
                    # We'll handle verification separately below
                    leaf = StepNode(
                        trace_lines=node.trace_lines + [expr],
                        remaining=(24.0,) if expr else (0.0,),
                        score_tot=1.0,
                        score_hybrid=1.0,
                        basin_key=make_basin_key(node.trace_lines + [expr], (24.0,)),
                        depth=d + 1,
                        parent_idx=len(all_nodes),
                    )
                    candidates.append(leaf)
                    all_nodes.append(leaf)
            else:
                # Propose intermediate steps from current remaining numbers
                steps = _propose_next_steps(llm, rem, breadth, temperature, max_tokens=max_tokens)
                n_calls += 1
                for step in steps:
                    new_rem = step["new_remaining"]
                    new_trace = node.trace_lines + [step["step_text"]]
                    # Basin key = operator sequence of the NEW step only (current depth strategy)
                    bk = make_basin_key([step["step_text"]], new_rem)
                    score_tot = state_value(new_rem)

                    if effective_lambda is not None:
                        n_b = basin_visits[bk]
                        if quality_aware:
                            q_b = basin_quality[bk]
                            score_hybrid = score_tot - effective_lambda * math.log(1 + n_b) * (1 - q_b)
                            basin_quality[bk] = (q_b * n_b + score_tot) / (n_b + 1)
                        else:
                            score_hybrid = score_tot - effective_lambda * math.log(1 + n_b)
                        basin_visits[bk] += 1
                    else:
                        score_hybrid = score_tot

                    child = StepNode(
                        trace_lines=new_trace,
                        remaining=new_rem,
                        score_tot=score_tot,
                        score_hybrid=score_hybrid,
                        basin_key=bk,
                        depth=d + 1,
                        parent_idx=len(all_nodes),
                    )
                    candidates.append(child)
                    all_nodes.append(child)

        if not candidates:
            break

        if is_final:
            beam = candidates  # Keep all terminals for verification
        else:
            # Sort by hybrid score (ties broken by score_tot)
            candidates.sort(key=lambda n: (n.score_hybrid, n.score_tot), reverse=True)
            beam = candidates[:beam_width]

        # Adaptive λ gating: after depth 0, measure diversity of basin visits.
        # If the model is already near-uniformly distributed across basins
        # (Neff / n_basins > 0.85), the penalty would only hurt — disable it.
        if d == 0 and lambda_ is not None and basin_visits:
            total = sum(basin_visits.values())
            probs = [v / total for v in basin_visits.values()]
            neff_d0 = math.exp(-sum(p * math.log(p + 1e-12) for p in probs))
            diversity_ratio = neff_d0 / len(basin_visits)
            if diversity_ratio > 0.85:
                effective_lambda = None
                lambda_gated_off = True
                logger.debug(f"  Adaptive λ: gated off (neff_d0={neff_d0:.2f}, ratio={diversity_ratio:.2f})")

    # --- Verification phase
    candidates_exprs: List[str] = []
    verifier_results: List[dict] = []
    n_malformed = n_wrong_nums = n_rejected = n_dead_end = 0

    def verify_trace(trace_lines: List[str], orig_numbers: Tuple[int, ...]) -> bool:
        """
        Verify a step trace is a valid solution.

        Each step is "A OP B = C".  parse_proposal already checked operands
        are in current_remaining at generation time, so we just replay the
        chain and confirm the final remaining value is 24.
        """
        remaining = [float(n) for n in orig_numbers]
        for line in trace_lines:
            m = _STEP_LINE_PAT.search(line)
            if not m:
                return False
            try:
                a_val = float(m.group(1).rstrip("."))
                b_val = float(m.group(3).rstrip("."))
                result_val = float(m.group(4).rstrip("."))
            except ValueError:
                return False
            op = m.group(2)
            # Verify arithmetic
            try:
                if op == "+":   exp = a_val + b_val
                elif op == "-": exp = a_val - b_val
                elif op == "*": exp = a_val * b_val
                elif op == "/":
                    if abs(b_val) < 1e-9: return False
                    exp = a_val / b_val
                else: return False
            except Exception:
                return False
            if not _nums_close(exp, result_val):
                return False
            # Remove operands from remaining
            rem_copy = list(remaining)
            for num in [a_val, b_val]:
                found = False
                for i, r in enumerate(rem_copy):
                    if _nums_close(r, num):
                        rem_copy.pop(i)
                        found = True
                        break
                if not found:
                    return False
            rem_copy.append(result_val)
            remaining = rem_copy
        return len(remaining) == 1 and abs(remaining[0] - 24.0) < 1e-6

    # Collect terminal nodes (depth == depth) and any node whose last step gives 24
    terminal_nodes = [n for n in all_nodes if n.depth == depth]

    seen_traces: set = set()
    for node in all_nodes:
        if not node.trace_lines:
            continue
        trace_key = tuple(node.trace_lines)
        if trace_key in seen_traces:
            continue
        seen_traces.add(trace_key)

        # Check if this trace leads to 24
        if verify_trace(node.trace_lines, numbers):
            solution_str = " → ".join(node.trace_lines)
            candidates_exprs.append(solution_str)
            verifier_results.append({
                "success": True, "error_type": None,
                "value": 24.0, "expr_used": solution_str, "numbers_used": list(numbers),
            })
        else:
            # Record as attempted (for failure stats)
            last_line = node.trace_lines[-1]
            m = _STEP_LINE_PAT.search(last_line)
            if m:
                try:
                    result_val = float(m.group(4).rstrip("."))
                    if abs(result_val - 24.0) >= 1e-6:
                        # Last step aimed at 24 but trace is invalid
                        candidates_exprs.append(last_line)
                        verifier_results.append({
                            "success": False, "error_type": "wrong_numbers",
                            "value": result_val, "expr_used": last_line, "numbers_used": [],
                        })
                        n_wrong_nums += 1
                except ValueError:
                    pass

    if not candidates_exprs:
        n_dead_end = 1

    success = any(v.get("success", False) for v in verifier_results)
    solution = next(
        (c for c, v in zip(candidates_exprs, verifier_results) if v.get("success")),
        None
    )

    # --- Exploration metrics
    visited_basins = Counter(n.basin_key for n in all_nodes[1:])  # exclude root
    n_basins = len(visited_basins)

    # Effective number of basins (Neff = exp(entropy of basin visit distribution))
    if n_basins > 0:
        total_visits = sum(visited_basins.values())
        probs = np.array([v / total_visits for v in visited_basins.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        neff = math.exp(entropy)
    else:
        neff = 0.0

    # Escape rate: fraction of BFS levels where selected beam changes basin
    non_root = [n for n in all_nodes if n.depth > 0]
    escape_count = 0
    if len(non_root) >= 2:
        basins_by_depth = defaultdict(set)
        for n in non_root:
            basins_by_depth[n.depth].add(n.basin_key)
        depths = sorted(basins_by_depth.keys())
        for i in range(1, len(depths)):
            prev = basins_by_depth[depths[i-1]]
            curr = basins_by_depth[depths[i]]
            if curr - prev:  # new basin entered
                escape_count += 1
        escape_rate = escape_count / max(1, len(depths) - 1)
    else:
        escape_rate = 0.0

    # Revisit rate: fraction of nodes sharing basin with an earlier node
    revisit = sum(1 for n in all_nodes[1:] if n.basin_key in
                  {nn.basin_key for nn in all_nodes[:all_nodes.index(n)]})
    revisit_rate = revisit / max(1, len(all_nodes) - 1)

    # Answer diversity
    results_24 = [v.get("value", 0.0) or 0.0 for v in verifier_results]
    answer_diversity = len(set(round(v, 1) for v in results_24)) / max(1, len(results_24))

    return ProblemResult(
        idx=0, numbers=numbers, method=method,
        success=success, solution=solution, correct_any=success,
        lambda_gated=lambda_gated_off,
        candidates=candidates_exprs,
        verifier_results=verifier_results,
        n_nodes=len(all_nodes),
        n_terminal=len(terminal_nodes),
        n_basins=n_basins,
        neff=round(neff, 3),
        basin_visits=dict(basin_visits),
        escape_rate=round(escape_rate, 3),
        revisit_rate=round(revisit_rate, 3),
        answer_diversity=round(answer_diversity, 3),
        n_malformed=n_malformed,
        n_wrong_nums=n_wrong_nums,
        n_rejected=n_rejected,
        n_dead_end=n_dead_end,
        n_calls=n_calls,
        elapsed=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Experiment main loop
# ---------------------------------------------------------------------------

METHODS = ["self_consistency", "sc_hybrid", "sc_twophase", "tot_standard", "tot_hybrid"]
LABELS  = {
    "self_consistency":    "Self-Consistency (×10)",
    "sc_hybrid":           "SC + ThoughtScape (λ=1.5)",
    "sc_twophase":         "SC + ThoughtScape 2-phase (λ=1.5)",
    "tot_standard":        "ToT (standard)",
    "tot_hybrid":          "ToT + ThoughtScape (λ=1.5)",
    "tot_hybrid_adaptive": "ToT + ThoughtScape Adaptive (λ=1.5)",
}


def run_one_problem(
    idx: int,
    numbers: Tuple[int, ...],
    llm: OpenAIClient,
    args,
) -> List[ProblemResult]:
    """Run all methods on one Game of 24 puzzle. Returns list of ProblemResult."""
    results = []

    for method in METHODS:
        t0 = time.time()
        try:
            if method == "self_consistency":
                r = run_cot(llm, numbers, n_samples=args.sc_samples,
                            max_tokens=args.max_tokens)
            elif method == "sc_hybrid":
                r = run_sc_hybrid(
                    llm, numbers,
                    n_samples=args.sc_samples,
                    lambda_=args.lambda_,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            elif method == "sc_twophase":
                r = run_sc_twophase(
                    llm, numbers,
                    n_samples=args.sc_samples,
                    lambda_=args.lambda_,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            elif method == "tot_standard":
                r = run_tot_bfs(
                    llm, numbers,
                    depth=args.depth, breadth=args.breadth,
                    beam_width=args.beam_width,
                    temperature=args.temperature,
                    lambda_=None,
                    max_tokens=args.max_tokens,
                )
            elif method == "tot_hybrid":
                r = run_tot_bfs(
                    llm, numbers,
                    depth=args.depth, breadth=args.breadth,
                    beam_width=args.beam_width,
                    temperature=args.temperature,
                    lambda_=args.lambda_,
                    max_tokens=args.max_tokens,
                    quality_aware=args.quality_aware,
                )
            else:
                continue
            r.idx = idx
            r.method = method
        except Exception as e:
            logger.warning(f"  [{method}] FAILED: {e}", exc_info=True)
            r = ProblemResult(
                idx=idx, numbers=numbers, method=method,
                success=False, solution=None,
                candidates=[], verifier_results=[],
                n_dead_end=1, elapsed=time.time() - t0,
            )

        elapsed = time.time() - t0
        logger.info(
            f"  {method:<22s} success={r.success!s:<5} "
            f"basins={r.n_basins} neff={r.neff:.2f} "
            f"nodes={r.n_nodes} calls={r.n_calls} ({elapsed:.1f}s)"
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def method_stats(results: List[ProblemResult]) -> Dict[str, Any]:
    n = len(results)
    succ = sum(r.success for r in results)
    ci_lo, ci_hi = wilson_ci(succ, n)

    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    # Pass@k: fraction of problems where at least one candidate is correct
    pass_at_k = sum(1 for r in results if r.correct_any) / max(1, n)

    # Failure breakdown
    n_mal  = sum(r.n_malformed  for r in results)
    n_wn   = sum(r.n_wrong_nums for r in results)
    n_rej  = sum(r.n_rejected   for r in results)
    n_de   = sum(r.n_dead_end   for r in results)
    total_cands = max(1, sum(len(r.candidates) for r in results))

    return {
        "n": n,
        "accuracy": succ / n,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "pass_at_k": pass_at_k,
        "mean_nodes": safe_mean([r.n_nodes for r in results]),
        "mean_terminal": safe_mean([r.n_terminal for r in results]),
        "mean_calls": safe_mean([r.n_calls for r in results]),
        "mean_elapsed": safe_mean([r.elapsed for r in results]),
        "mean_basins": safe_mean([r.n_basins for r in results]),
        "mean_neff": safe_mean([r.neff for r in results]),
        "frac_ge2_basins": sum(1 for r in results if r.n_basins >= 2) / n,
        "escape_rate": safe_mean([r.escape_rate for r in results]),
        "revisit_rate": safe_mean([r.revisit_rate for r in results]),
        "malformed_pct": 100 * n_mal / total_cands,
        "wrong_num_pct": 100 * n_wn / total_cands,
        "rejected_pct":  100 * n_rej / total_cands,
        "dead_end_pct":  100 * n_de / n,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_tables(all_results: Dict[str, List[ProblemResult]], stats: Dict[str, Dict]) -> None:
    # Table 1: Main comparison
    with open(_TABLE_DIR / "game24_tot_main.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Accuracy", "CI_lo", "CI_hi", "Pass@k",
                    "Runtime_s", "Calls", "Nodes", "Basins", "Neff"])
        for m in METHODS:
            if m not in stats: continue
            s = stats[m]
            w.writerow([
                LABELS[m],
                f"{s['accuracy']:.3f}", f"{s['ci_lo']:.3f}", f"{s['ci_hi']:.3f}",
                f"{s['pass_at_k']:.3f}",
                f"{s['mean_elapsed']:.1f}", f"{s['mean_calls']:.1f}",
                f"{s['mean_nodes']:.1f}",
                f"{s['mean_basins']:.2f}", f"{s['mean_neff']:.2f}",
            ])

    # Table 2: Exploration diagnostics
    with open(_TABLE_DIR / "game24_tot_exploration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Basins", "Neff", "Revisit%", "Escape%", "Frac>=2B"])
        for m in METHODS:
            if m not in stats: continue
            s = stats[m]
            w.writerow([
                LABELS[m],
                f"{s['mean_basins']:.2f}", f"{s['mean_neff']:.2f}",
                f"{100*s['revisit_rate']:.1f}", f"{100*s['escape_rate']:.1f}",
                f"{s['frac_ge2_basins']:.3f}",
            ])

    # Table 3: Failure analysis
    with open(_TABLE_DIR / "game24_tot_failures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Malformed%", "WrongNum%", "Rejected%", "DeadEnd%"])
        for m in METHODS:
            if m not in stats: continue
            s = stats[m]
            w.writerow([
                LABELS[m],
                f"{s['malformed_pct']:.1f}", f"{s['wrong_num_pct']:.1f}",
                f"{s['rejected_pct']:.1f}", f"{s['dead_end_pct']:.1f}",
            ])

    logger.info("Tables written.")


def write_summary(all_results: Dict[str, List[ProblemResult]], stats: Dict[str, Dict],
                  model: str = "gpt-4o-mini") -> None:
    lines = [
        "# Game of 24: SC vs SC+ThoughtScape vs ToT vs ToT+ThoughtScape",
        "",
        f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
        f"**Puzzles:** indices 900–999 (n=100)",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]

    lines += [
        f"| Model | {model} |",
        "| Depth | 3 |",
        "| Breadth | 5 |",
        "| Beam width | 5 |",
        "| λ (hybrid) | 1.5 |",
        "| SC samples | 10 |",
        "",
        "## Main Results",
        "",
        "| Method | Acc | 95% CI | Pass@k | Runtime/prob | Calls | Nodes | Basins | Neff |",
        "|--------|-----|--------|--------|-------------|-------|-------|--------|------|",
    ]
    for m in METHODS:
        if m not in stats: continue
        s = stats[m]
        lines.append(
            f"| {LABELS[m]} | {s['accuracy']:.3f} | "
            f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}] | {s['pass_at_k']:.3f} | "
            f"{s['mean_elapsed']:.1f}s | {s['mean_calls']:.1f} | {s['mean_nodes']:.1f} | "
            f"{s['mean_basins']:.2f} | {s['mean_neff']:.2f} |"
        )

    lines += [
        "",
        "## Exploration Diagnostics",
        "",
        "| Method | Basins | Neff | Revisit% | Escape% | Frac≥2B |",
        "|--------|--------|------|---------|--------|---------|",
    ]
    for m in METHODS:
        if m not in stats: continue
        s = stats[m]
        lines.append(
            f"| {LABELS[m]} | {s['mean_basins']:.2f} | {s['mean_neff']:.2f} | "
            f"{100*s['revisit_rate']:.1f}% | {100*s['escape_rate']:.1f}% | "
            f"{s['frac_ge2_basins']:.3f} |"
        )

    lines += [
        "",
        "## Failure Analysis",
        "",
        "| Method | Malformed% | WrongNum% | Rejected% | DeadEnd% |",
        "|--------|-----------|----------|---------|---------|",
    ]
    for m in METHODS:
        if m not in stats: continue
        s = stats[m]
        lines.append(
            f"| {LABELS[m]} | {s['malformed_pct']:.1f}% | {s['wrong_num_pct']:.1f}% | "
            f"{s['rejected_pct']:.1f}% | {s['dead_end_pct']:.1f}% |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "1. **Does SC+ThoughtScape improve over SC?**  "
        + ("Yes" if stats.get("sc_hybrid", {}).get("accuracy", 0) >
           stats.get("self_consistency", {}).get("accuracy", 0) else "No")
        + f" (SC+TS {stats.get('sc_hybrid',{}).get('accuracy',0):.3f} vs "
        + f"SC {stats.get('self_consistency',{}).get('accuracy',0):.3f}).",
        "",
        "2. **Does ToT+ThoughtScape improve over ToT?**  "
        + ("Yes" if stats.get("tot_hybrid", {}).get("accuracy", 0) >
           stats.get("tot_standard", {}).get("accuracy", 0) else "No")
        + f" (ToT+TS {stats.get('tot_hybrid',{}).get('accuracy',0):.3f} vs "
        + f"ToT {stats.get('tot_standard',{}).get('accuracy',0):.3f}).",
        "",
        "3. **Does basin bias increase diversity?**  "
        "Basin bias penalises repeated operation strategies, pushing search "
        "toward unexplored arithmetic approaches (e.g., divide-first vs add-first).",
        "",
        "4. **When it helps:**  "
        "Puzzles requiring a non-obvious operation sequence (e.g., divide-first "
        "when the model keeps trying addition-heavy strategies).",
        "",
        "5. **When it hurts:**  "
        "Puzzles where the single correct strategy maps to one basin and "
        "the bias discourages returning to it.",
    ]

    with open(_OUTPUT_DIR / "game24_tot_report.md", "w") as f:
        f.write("\n".join(lines))
    logger.info("Summary report written.")


def write_case_studies(all_results: Dict[str, List[ProblemResult]]) -> None:
    """Write 3 illustrative case studies."""
    tot_std = all_results.get("tot_standard", [])
    tot_hyb = all_results.get("tot_hybrid", [])

    lines = ["# Game of 24 — Case Studies", ""]

    def format_case(title: str, r_std: ProblemResult, r_hyb: ProblemResult) -> List[str]:
        nums_str = " ".join(str(n) for n in r_std.numbers)
        out = [
            f"## {title}",
            f"**Input:** `{nums_str}`",
            "",
            f"### ToT Standard",
            f"- Success: {r_std.success}  |  Nodes: {r_std.n_nodes}  |  Basins: {r_std.n_basins}  |  Neff: {r_std.neff:.2f}",
            f"- Solution: `{r_std.solution or 'None'}`",
            f"- Basins visited:",
        ]
        for bk, cnt in sorted(r_std.basin_visits.items(), key=lambda x: -x[1])[:5]:
            out.append(f"  - `{bk}` (×{cnt})")
        out += [
            "",
            f"### ToT + ThoughtScape (λ=1.5)",
            f"- Success: {r_hyb.success}  |  Nodes: {r_hyb.n_nodes}  |  Basins: {r_hyb.n_basins}  |  Neff: {r_hyb.neff:.2f}",
            f"- Solution: `{r_hyb.solution or 'None'}`",
            f"- Basins visited:",
        ]
        for bk, cnt in sorted(r_hyb.basin_visits.items(), key=lambda x: -x[1])[:5]:
            out.append(f"  - `{bk}` (×{cnt})")
        out.append("")
        return out

    # Case 1: ToT fails, hybrid succeeds
    for std_r, hyb_r in zip(tot_std, tot_hyb):
        if not std_r.success and hyb_r.success:
            lines += format_case("Case 1: ToT Fails, Hybrid Succeeds", std_r, hyb_r)
            break
    else:
        lines += ["## Case 1: (no case found where ToT fails and hybrid succeeds)", ""]

    # Case 2: hybrid fails, ToT succeeds (basin bias hurt)
    for std_r, hyb_r in zip(tot_std, tot_hyb):
        if std_r.success and not hyb_r.success:
            lines += format_case("Case 2: Basin Bias Hurts (ToT Succeeds, Hybrid Fails)", std_r, hyb_r)
            break
    else:
        lines += ["## Case 2: (no case found where hybrid fails but ToT succeeds)", ""]

    # Case 3: both succeed but hybrid uses fewer nodes
    for std_r, hyb_r in zip(tot_std, tot_hyb):
        if std_r.success and hyb_r.success and hyb_r.n_nodes < std_r.n_nodes:
            lines += format_case("Case 3: Both Succeed, Hybrid More Efficient", std_r, hyb_r)
            break
    else:
        lines += ["## Case 3: (no efficiency-advantage case found)", ""]

    with open(_OUTPUT_DIR / "case_studies.md", "w") as f:
        f.write("\n".join(lines))
    logger.info("Case studies written.")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(all_results: Dict[str, List[ProblemResult]], stats: Dict[str, Dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping figures")
        return

    method_order = [m for m in METHODS if m in stats]
    labels       = [LABELS[m] for m in method_order]
    colors       = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"][:len(method_order)]

    # Fig 1: Accuracy comparison with CI bars
    fig, ax = plt.subplots(figsize=(8, 5))
    accs    = [stats[m]["accuracy"]    for m in method_order]
    ci_los  = [stats[m]["ci_lo"]       for m in method_order]
    ci_his  = [stats[m]["ci_hi"]       for m in method_order]
    yerr_lo = [a - lo for a, lo in zip(accs, ci_los)]
    yerr_hi = [hi - a for a, hi in zip(accs, ci_his)]
    bars = ax.bar(labels, accs, color=colors, alpha=0.85,
                  yerr=[yerr_lo, yerr_hi], capsize=6, error_kw=dict(elinewidth=1.5))
    ax.set_ylabel("Accuracy (Success Rate)")
    ax.set_title("Game of 24 — Method Comparison (n=100, puzzles 900–999)")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "game24_fig1_accuracy.png", dpi=150)
    plt.close()

    # Fig 2: Runtime and calls comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    runtimes = [stats[m]["mean_elapsed"] for m in method_order]
    n_calls  = [stats[m]["mean_calls"]   for m in method_order]

    axes[0].bar(labels, runtimes, color=colors, alpha=0.85)
    axes[0].set_title("Mean Runtime per Problem (s)")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(labels, n_calls, color=colors, alpha=0.85)
    axes[1].set_title("Mean LLM Calls per Problem")
    axes[1].set_ylabel("API Calls")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(_FIG_DIR / "game24_fig2_runtime.png", dpi=150)
    plt.close()

    # Fig 3: Basin diversity (Neff, escape rate)
    tot_methods = [m for m in ["tot_standard", "tot_hybrid"] if m in stats]
    if len(tot_methods) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        neff_vals   = [stats[m]["mean_neff"]       for m in tot_methods]
        escape_vals = [100*stats[m]["escape_rate"]  for m in tot_methods]
        revisit_vals= [100*stats[m]["revisit_rate"] for m in tot_methods]
        tot_labels  = [LABELS[m] for m in tot_methods]
        tot_colors  = [colors[METHODS.index(m)] for m in tot_methods]

        for ax, vals, title, ylabel in [
            (axes[0], neff_vals,    "Mean Neff (Effective Basins)", "Neff"),
            (axes[1], escape_vals,  "Basin Escape Rate (%)",        "% problems"),
            (axes[2], revisit_vals, "Basin Revisit Rate (%)",       "% nodes"),
        ]:
            ax.bar(tot_labels, vals, color=tot_colors, alpha=0.85)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=15)
            for j, (bar_val, lbl) in enumerate(zip(vals, tot_labels)):
                ax.text(j, bar_val + 0.01 * max(vals, default=1),
                        f"{bar_val:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(_FIG_DIR / "game24_fig3_basins.png", dpi=150)
        plt.close()

    # Fig 4: Basin occupancy — example per-problem basin counts (histogram)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m in zip(axes, ["tot_standard", "tot_hybrid"]):
        if m not in all_results: continue
        basin_counts = [r.n_basins for r in all_results[m]]
        ax.hist(basin_counts, bins=range(0, max(basin_counts)+2), color=colors[METHODS.index(m)],
                alpha=0.8, edgecolor="white")
        ax.set_title(f"{LABELS[m]}\nBasins per Problem")
        ax.set_xlabel("Number of Basins")
        ax.set_ylabel("Problems")
    plt.suptitle("Basin Occupancy Distribution")
    plt.tight_layout()
    plt.savefig(_FIG_DIR / "game24_fig4_basin_occupancy.png", dpi=150)
    plt.close()

    logger.info("Figures saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Game of 24: ToT vs ToT + ThoughtScape")
    p.add_argument("--api_key",   required=True, help="LLM API key")
    p.add_argument("--base_url",  default="https://ellm.nrp-nautilus.io/v1",
                   help="LLM base URL")
    p.add_argument("--model",     default="gpt-oss")
    p.add_argument("--no_thinking", action="store_true",
                   help="Disable thinking mode (for Qwen3 and other reasoning models)")
    p.add_argument("--breadth",   type=int,   default=5)
    p.add_argument("--beam_width",type=int,   default=5)
    p.add_argument("--depth",     type=int,   default=3)
    p.add_argument("--lambda_",   type=float, default=1.5)
    p.add_argument("--quality_aware", action="store_true",
                   help="Use quality-aware basin penalty: bias=λ·log(1+n_B)·(1−q_B)")
    p.add_argument("--sc_samples",type=int,   default=10)
    p.add_argument("--max_tokens",type=int,   default=512)
    p.add_argument("--temperature",type=float,default=0.7)
    p.add_argument("--start_idx", type=int,   default=900)
    p.add_argument("--end_idx",   type=int,   default=999)
    p.add_argument("--skip_cot",  action="store_true",
                   help="Skip CoT and SC methods (run only ToT variants)")
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: outputs/game24_tot_vs_hybrid)")
    p.add_argument("--methods", default=None,
                   help="Comma-separated subset of methods to run, e.g. 'self_consistency,sc_hybrid'")
    return p.parse_args()


def main():
    args = parse_args()

    out = args.output_dir or str(_REPO_ROOT / "outputs" / "game24_tot_vs_hybrid")
    _init_output_dirs(out)

    extra_body = {"thinking": {"type": "disabled"}} if args.no_thinking else {}
    llm = OpenAIClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        logprobs=False,
        extra_body=extra_body if extra_body else None,
    )

    global METHODS
    if args.skip_cot:
        METHODS = [m for m in METHODS if m not in ("self_consistency", "sc_hybrid")]
    if args.methods:
        selected = [m.strip() for m in args.methods.split(",")]
        METHODS = [m for m in METHODS if m in selected]

    logger.info("=" * 60)
    logger.info("  Game of 24: ToT vs ToT + ThoughtScape Basin Bias")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k:<20s} {v}")
    logger.info("=" * 60)

    # Load dataset
    puzzles = load_game24(args.start_idx, args.end_idx)

    # Check for checkpoint
    ckpt_path = _OUTPUT_DIR / "checkpoint.json"
    per_example_path = _OUTPUT_DIR / "per_example.json"
    completed_keys: set = set()
    all_problem_results: List[dict] = []

    if ckpt_path.exists():
        with open(ckpt_path) as f:
            all_problem_results = json.load(f)
        for r in all_problem_results:
            completed_keys.add((r["idx"], r["method"]))
        logger.info(f"Resumed from checkpoint: {len(all_problem_results)} records, "
                    f"{len(set(r['idx'] for r in all_problem_results))} problems")

    # Main loop
    for i, (orig_idx, numbers) in enumerate(puzzles):
        # Skip if all methods done
        remaining_methods = [m for m in METHODS if (orig_idx, m) not in completed_keys]
        if not remaining_methods:
            continue

        logger.info(f"[{i+1}/{len(puzzles)}] idx={orig_idx}  numbers={numbers}")

        results = run_one_problem(orig_idx, numbers, llm, args)
        for r in results:
            if r.method not in remaining_methods:
                continue
            all_problem_results.append({
                "idx": r.idx,
                "numbers": list(r.numbers),
                "method": r.method,
                "success": r.success,
                "solution": r.solution,
                "candidates": r.candidates[:5],  # truncate for storage
                "n_nodes": r.n_nodes,
                "n_terminal": r.n_terminal,
                "n_basins": r.n_basins,
                "neff": r.neff,
                "escape_rate": r.escape_rate,
                "revisit_rate": r.revisit_rate,
                "n_malformed": r.n_malformed,
                "n_wrong_nums": r.n_wrong_nums,
                "n_rejected": r.n_rejected,
                "n_dead_end": r.n_dead_end,
                "n_calls": r.n_calls,
                "elapsed": round(r.elapsed, 2),
                "basin_visits": r.basin_visits,
                "correct_any": r.correct_any,
                "lambda_gated": getattr(r, "lambda_gated", False),
            })
            completed_keys.add((r.idx, r.method))

        # Checkpoint
        if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(puzzles):
            with open(ckpt_path, "w") as f:
                json.dump(all_problem_results, f, indent=2)
            logger.info(f"  Checkpoint saved ({i+1}/{len(puzzles)})")

    # Save full per-example
    with open(per_example_path, "w") as f:
        json.dump(all_problem_results, f, indent=2)
    logger.info(f"Saved {len(all_problem_results)} total records to per_example.json")

    # Build result objects for analysis
    all_results: Dict[str, List[ProblemResult]] = {m: [] for m in METHODS}
    for rec in all_problem_results:
        m = rec["method"]
        if m not in all_results:
            continue
        r = ProblemResult(
            idx=rec["idx"],
            numbers=tuple(rec["numbers"]),
            method=m,
            success=rec["success"],
            solution=rec.get("solution"),
            candidates=rec.get("candidates", []),
            verifier_results=[],
            n_nodes=rec.get("n_nodes", 0),
            n_terminal=rec.get("n_terminal", 0),
            n_basins=rec.get("n_basins", 0),
            neff=rec.get("neff", 0.0),
            escape_rate=rec.get("escape_rate", 0.0),
            revisit_rate=rec.get("revisit_rate", 0.0),
            n_malformed=rec.get("n_malformed", 0),
            n_wrong_nums=rec.get("n_wrong_nums", 0),
            n_rejected=rec.get("n_rejected", 0),
            n_dead_end=rec.get("n_dead_end", 0),
            n_calls=rec.get("n_calls", 0),
            elapsed=rec.get("elapsed", 0.0),
            basin_visits=rec.get("basin_visits", {}),
            correct_any=rec.get("correct_any", rec.get("success", False)),
        )
        all_results[m].append(r)

    # Compute stats
    stats = {}
    for m in METHODS:
        if all_results[m]:
            stats[m] = method_stats(all_results[m])
            s = stats[m]
            logger.info(
                f"{LABELS[m]:<35s}  acc={s['accuracy']:.3f} "
                f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]  "
                f"pass@k={s['pass_at_k']:.3f}  "
                f"basins={s['mean_basins']:.2f}  neff={s['mean_neff']:.2f}"
            )

    # Write outputs
    write_tables(all_results, stats)
    write_summary(all_results, stats, model=args.model)
    write_case_studies(all_results)
    make_figures(all_results, stats)

    logger.info("=" * 60)
    logger.info("  DONE — outputs in outputs/game24_tot_vs_hybrid/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
