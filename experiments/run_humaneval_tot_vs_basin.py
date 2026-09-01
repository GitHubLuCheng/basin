"""
experiments/run_humaneval_tot_vs_basin.py
============================================
BASIN on a code-generation domain (HumanEval), addressing reviewer 9rgM's
scope concern ("...not enough to fully establish generality across
broader reasoning domains such as code generation, theorem proving,
planning, math word problems, or tool-use agents").

Basin definition: exact, symbolic, AST-normalized structural key
(basin.verifier.ast_basin.structural_basin_key) — the code-generation
analogue of Game of 24's exact operator-sequence basin key. Deliberately
NOT execution-outcome-based (that would make correct/incorrect basin
membership circular) and NOT semantic/NLI-based (code has a free, exact
structural invariant, so there's no need for an approximate one).

Search procedure: iterative refinement, matching the same controlled-
comparison methodology used throughout the paper (same prompts, same
model, same per-round budget; ToT and ToT+BASIN differ ONLY in the
candidate-selection formula, Eq. 3):

  score_hybrid(s) = f(s) - lambda * log(1 + visits[key(s)])

f(s) is a cheap deterministic verifier (parses without error AND defines
the required entry point) — same "exact, free verifier" philosophy as
Game of 24's symbolic check, not a learned or LLM-judged score.

Each round generates n_candidates full-function completions (round 0 from
the bare problem statement; later rounds conditioned on the previously
selected attempt, asking for a different approach). After n_rounds, the
finally-selected candidate is executed against the HELD-OUT hidden tests
(basin.verifier.code_executor.run_check) for the accuracy metric --
hidden tests are never used to guide search, only to score the final
answer, mirroring how Game of 24's exact-24 check and MuSR's gold-answer
check are used only at the end.

Usage
-----
  python experiments/run_humaneval_tot_vs_basin.py \\
      --api_key <NRP_KEY> --n_examples 50 --max_workers 8
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basin.datasets.humaneval_loader import CodeProblem, HumanEvalLoader
from basin.llm.openai_backend import OpenAIClient
from basin.verifier.ast_basin import structural_basin_key
from basin.verifier.code_executor import extract_code, run_check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def _output_dir(model: str) -> Path:
    d = _REPO_ROOT / "outputs" / f"humaneval_tot_vs_basin_{model}"
    d.mkdir(parents=True, exist_ok=True)
    return d

N_ROUNDS = 6
N_CANDIDATES = 2
MAX_TOKENS = 512
TEMPERATURE = 0.8

SYSTEM_PROMPT = (
    "You are a careful, precise Python programmer. When asked to implement "
    "a function, respond with ONLY the complete function definition "
    "(signature + body) inside a single ```python code fence. Do not "
    "include explanations, tests, or any text outside the fence."
)

ROUND0_TEMPLATE = """\
Complete the following Python function.

{prompt}
"""

CONTINUATION_TEMPLATE = """\
Here is a previous attempt at this function:

```python
{previous_code}
```

This attempt may be incorrect or incomplete. Write a complete, corrected \
implementation. If the previous approach seems wrong, try a genuinely \
different algorithm or control-flow structure rather than a minor edit.

Original problem:
{prompt}
"""


def _neff(visit_history: List[str]) -> float:
    if not visit_history:
        return 1.0
    counts = Counter(visit_history)
    n = len(visit_history)
    H = -sum((c / n) * math.log(c / n) for c in counts.values())
    return math.exp(H)


def _verifier_score(code: str, entry_point: str) -> float:
    """Cheap deterministic verifier: parses AND defines the entry point."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return 0.0
    names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return 1.0 if entry_point in names else 0.3  # parses, but no matching def


@dataclass
class ProblemResult:
    task_id: str
    method: str          # "tot_standard" or "tot_basin"
    lambda_: float
    passed: bool
    final_basin_key: str
    n_basins: int
    neff: float
    n_rounds: int
    elapsed_s: float
    error: Optional[str] = None


def run_one(problem: CodeProblem, llm, lambda_: float, method_name: str,
            quality_aware: bool = False) -> ProblemResult:
    t0 = time.time()
    visits: dict = {}
    quality: dict = {}  # running mean of the free verifier score per basin (Eq. 6)
    basin_history: List[str] = []
    previous_code: Optional[str] = None
    selected_code: Optional[str] = None

    try:
        for round_idx in range(N_ROUNDS):
            if previous_code is None:
                prompt = ROUND0_TEMPLATE.format(prompt=problem.prompt)
            else:
                prompt = CONTINUATION_TEMPLATE.format(
                    previous_code=previous_code, prompt=problem.prompt
                )

            responses = llm.generate(
                prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, n=N_CANDIDATES
            )

            candidates = []
            for resp in responses:
                code = extract_code(resp.text)
                key = structural_basin_key(code)
                f_s = _verifier_score(code, problem.entry_point)
                n_b = visits.get(key, 0)
                if quality_aware:
                    q_b = quality.get(key, 0.0)
                    penalty = lambda_ * math.log(1 + n_b) * (1 - q_b)
                else:
                    penalty = lambda_ * math.log(1 + n_b)
                score = f_s - penalty
                candidates.append((score, code, key, f_s))

            best_score, best_code, best_key, best_f = max(candidates, key=lambda c: c[0])
            if quality_aware:
                n_prev = visits.get(best_key, 0)
                quality[best_key] = (quality.get(best_key, 0.0) * n_prev + best_f) / (n_prev + 1)
            visits[best_key] = visits.get(best_key, 0) + 1
            basin_history.append(best_key)
            previous_code = best_code
            selected_code = best_code

        passed, exec_key, _ = run_check(selected_code, problem.test, problem.entry_point)

        return ProblemResult(
            task_id=problem.task_id,
            method=method_name,
            lambda_=lambda_,
            passed=passed,
            final_basin_key=exec_key,
            n_basins=len(set(basin_history)),
            neff=_neff(basin_history),
            n_rounds=N_ROUNDS,
            elapsed_s=time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("  [%s/%s] FAILED: %s", problem.task_id, method_name, exc, exc_info=True)
        return ProblemResult(
            task_id=problem.task_id, method=method_name, lambda_=lambda_,
            passed=False, final_basin_key="ERROR", n_basins=0, neff=1.0,
            n_rounds=N_ROUNDS, elapsed_s=time.time() - t0, error=str(exc),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss")
    ap.add_argument("--backend", choices=["nrp", "openai", "together"], default="nrp")
    ap.add_argument("--base_url", default=None, help="Override the backend-derived base URL")
    ap.add_argument("--n_examples", type=int, default=50)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 3.0])
    ap.add_argument("--qa_lambda", type=float, default=0.5)
    ap.add_argument("--include_qabasin", action="store_true",
                     help="Also run a quality-aware BASIN (Eq. 6) condition, using the "
                          "existing free deterministic verifier score as the per-basin "
                          "quality proxy.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--api_key", default=None, help="NRP or OpenAI API key (matches --backend)")
    ap.add_argument("--no_thinking", action="store_true",
                     help="Disable extended thinking mode (e.g. for qwen3) so the token "
                          "budget goes to the actual completion, not a reasoning preamble")
    args = ap.parse_args()

    if args.backend == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        base_url = None
    elif args.backend == "together":
        api_key = args.api_key or os.environ.get("TOGETHER_API_KEY")
        base_url = "https://api.together.xyz/v1"
    else:
        api_key = args.api_key or os.environ.get("NRP_API_KEY")
        base_url = "https://ellm.nrp-nautilus.io/v1"
    if args.base_url:
        base_url = args.base_url
    if not api_key:
        raise RuntimeError(f"Missing API key for backend={args.backend}.")

    extra_body = {"thinking": {"type": "disabled"}} if args.no_thinking else None
    llm = OpenAIClient(
        model=args.model, api_key=api_key,
        base_url=base_url, extra_body=extra_body,
        logprobs=False, system_prompt=SYSTEM_PROMPT,
    )

    problems = HumanEvalLoader(max_problems=args.n_examples, seed=args.seed).load()
    logger.info("Loaded %d HumanEval problems", len(problems))

    method_name = {0.0: "tot_standard", 3.0: "tot_basin"}
    conditions = [(lam, method_name.get(lam, f"lambda_{lam}"), False) for lam in args.lambdas]
    if args.include_qabasin:
        conditions.append((args.qa_lambda, "tot_qabasin", True))

    jobs = []
    for prob in problems:
        for lam, name, qa in conditions:
            jobs.append((prob, lam, name, qa))

    logger.info(
        "Running %d problem x condition jobs (model=%s, n_rounds=%d, n_candidates=%d, "
        "n_workers=%d)", len(jobs), args.model, N_ROUNDS, N_CANDIDATES, args.max_workers,
    )

    t0 = time.time()
    results: List[Optional[ProblemResult]] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                run_one, prob, llm, lam, name, qa
            ): i
            for i, (prob, lam, name, qa) in enumerate(jobs)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            r = results[i]
            logger.info(
                "  [%d/%d] %s (%s)  passed=%-5s  basins=%d  Neff=%.2f  (%.1fs)",
                done, len(jobs), r.task_id, r.method, r.passed, r.n_basins,
                r.neff, r.elapsed_s,
            )

    elapsed = time.time() - t0

    by_method: dict = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    logger.info("=" * 60)
    for m, rs in by_method.items():
        n = len(rs)
        acc = sum(r.passed for r in rs) / n
        mean_basins = sum(r.n_basins for r in rs) / n
        mean_neff = sum(r.neff for r in rs) / n
        n_errors = sum(1 for r in rs if r.error is not None)
        logger.info(
            "%s: n=%d  Acc=%.3f  mean_#basins=%.2f  mean_Neff=%.2f  errors=%d",
            m, n, acc, mean_basins, mean_neff, n_errors,
        )
    logger.info("Total elapsed: %.0fs", elapsed)

    out = {
        "config": {
            "model": args.model,
            "n_examples": args.n_examples,
            "n_rounds": N_ROUNDS,
            "n_candidates": N_CANDIDATES,
            "lambdas": args.lambdas,
            "basin_definition": "AST-normalized structural key (exact match)",
            "seed": args.seed,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "aggregates": {
            m: {
                "n": len(rs),
                "accuracy": sum(r.passed for r in rs) / len(rs),
                "mean_basins": sum(r.n_basins for r in rs) / len(rs),
                "mean_neff": sum(r.neff for r in rs) / len(rs),
                "n_errors": sum(1 for r in rs if r.error is not None),
            }
            for m, rs in by_method.items()
        },
        "per_example": [
            {
                "task_id": r.task_id, "method": r.method, "lambda": r.lambda_,
                "passed": r.passed, "final_basin_key": r.final_basin_key,
                "n_basins": r.n_basins, "neff": r.neff, "error": r.error,
                "elapsed_s": r.elapsed_s,
            }
            for r in results
        ],
    }
    out_path = _output_dir(args.model) / "results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
