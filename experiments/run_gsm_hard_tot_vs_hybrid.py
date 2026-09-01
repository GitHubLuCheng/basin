"""
experiments/run_gsm_hard_tot_vs_hybrid.py
===========================================
ToT vs ToT+BASIN vs ToT+QA-BASIN on GSM-Hard, using a Game-of-24-style FREE,
DETERMINISTIC verifier instead of an LLM judge.

GSM-Hard's solutions are almost always explicit arithmetic statements
("A op B = C"), so we can verify each stated calculation exactly via
computation -- same "free, exact verifier" philosophy as Game of 24's
symbolic check -- rather than asking an LLM whether a partial derivation
"looks" correct. Final answers are single numbers, so scoring is exact
match, not SymPy-with-tolerance comparison.

  f(s)      = fraction of "A op B = C" statements in the step that are
              numerically correct (free, deterministic; 1.0 if the step
              has no parseable equations but is non-empty; unparsed steps
              score 0.5, a genuine "no signal" case, distinct from a
              verified-wrong one)
  basin(s)  = the ordered tuple of correctly-verified intermediate values
              computed so far in the trace -- ties basin identity to
              VERIFIED COMPUTATION, not to arbitrary substrings of the
              generated text.

Standard/BASIN/QA-BASIN scoring formula follows Eq. 3 / Eq. 6, with the
quality proxy = f(s) reused as the free per-basin quality signal, the same
convention as Game of 24's QA-BASIN.

Usage
-----
  python experiments/run_gsm_hard_tot_vs_hybrid.py \\
      --api_key <KEY> --model gpt-4o-mini --n_examples 100 \\
      --lambda_basin 3.0 --qa_lambda 0.5 --include_qabasin
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basin.llm.openai_backend import OpenAIClient
from basin.datasets.gsm_hard_loader import GSMHardLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

N_ROUNDS = 4
N_CANDIDATES = 4
MAX_TOKENS = 400
TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are a careful math tutor solving grade-school word problems. "
    "Show your work as a sequence of explicit calculations in the form "
    "'A op B = C' (e.g. '3 * 4 = 12'), one per line. "
    "End with a final line reading 'Answer: <number>'."
)

ROUND0_TEMPLATE = """\
Solve this problem step by step, showing explicit calculations.

{question}

Write the next 1-2 calculation steps only (do not finish the whole problem yet \
unless it only takes one step)."""

CONTINUATION_TEMPLATE = """\
Problem:
{question}

Work so far:
{history}

Continue with the next 1-2 calculation steps. If you have reached the final \
answer, end with 'Answer: <number>'."""

FINAL_ROUND_TEMPLATE = """\
Problem:
{question}

Work so far:
{history}

Finish the problem now. Show any remaining calculation(s), then end with \
'Answer: <number>'."""

# ---------------------------------------------------------------------------
# Exact arithmetic verifier
# ---------------------------------------------------------------------------

_EQ_PAT = re.compile(
    r"(-?\d[\d,]*\.?\d*)\s*([+\-*/x×])\s*(-?\d[\d,]*\.?\d*)\s*=\s*(-?\d[\d,]*\.?\d*)"
)
_ANSWER_PAT = re.compile(r"[Aa]nswer\s*[:=]\s*\$?\s*(-?\d[\d,]*\.?\d*)")


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def verify_step(step_text: str) -> Tuple[float, List[float]]:
    """
    Exact, deterministic check of every 'A op B = C' statement in *step_text*.

    Returns (score, verified_values):
      score: fraction of parsed equations that are numerically correct.
             0.5 (no signal, not a verified error) if no equation is found
             at all -- distinct from a parsed-and-wrong equation, which
             counts against the score.
      verified_values: the C values of equations that verified correct, in
                        order of appearance -- feeds the basin key.
    """
    matches = _EQ_PAT.findall(step_text)
    if not matches:
        return 0.5, []
    correct = 0
    verified: List[float] = []
    for a_s, op, b_s, c_s in matches:
        a, b, c = _to_float(a_s), _to_float(b_s), _to_float(c_s)
        if a is None or b is None or c is None:
            continue
        op = "*" if op in ("x", "×") else op
        try:
            if op == "+":
                exp = a + b
            elif op == "-":
                exp = a - b
            elif op == "*":
                exp = a * b
            elif op == "/":
                if abs(b) < 1e-9:
                    continue
                exp = a / b
            else:
                continue
        except Exception:
            continue
        if abs(exp - c) < 1e-4:
            correct += 1
            verified.append(round(c, 4))
    return (correct / len(matches) if matches else 0.5), verified


def extract_final_answer(text: str) -> Optional[str]:
    m = _ANSWER_PAT.search(text)
    if m:
        raw = m.group(1).replace(",", "")
        fv = _to_float(raw)
        if fv is not None:
            return str(int(fv)) if fv == int(fv) else str(fv)
    # Fallback: last verified-correct equation's value.
    _, verified = verify_step(text)
    if verified:
        v = verified[-1]
        return str(int(v)) if v == int(v) else str(v)
    return None


def is_correct(predicted: Optional[str], gold: str) -> bool:
    if predicted is None:
        return False
    pv, gv = _to_float(predicted), _to_float(gold)
    if pv is not None and gv is not None:
        return abs(pv - gv) < 1e-4
    return predicted.strip() == gold.strip()


def make_basin_key(verified_so_far: List[float]) -> str:
    """Basin = ordered tuple of exactly-verified intermediate values so far."""
    if not verified_so_far:
        return "root"
    return "|".join(str(v) for v in verified_so_far)


# ---------------------------------------------------------------------------
# LLM step generation
# ---------------------------------------------------------------------------

def generate_steps(llm: OpenAIClient, question: str, history_steps: List[str],
                    n: int, temperature: float, is_final: bool) -> List[str]:
    history = "\n".join(history_steps) if history_steps else "(nothing yet)"
    template = FINAL_ROUND_TEMPLATE if is_final else (
        ROUND0_TEMPLATE if not history_steps else CONTINUATION_TEMPLATE
    )
    prompt = template.format(question=question, history=history)
    try:
        resps = llm.generate(prompt, temperature=temperature, max_tokens=MAX_TOKENS, n=n)
        return [r.text for r in resps] if resps else []
    except Exception as e:
        logger.warning("generate_steps error: %s", e)
        return []


@dataclass
class Node:
    trace_steps: List[str]
    verified_values: List[float]
    score_tot: float
    score_hybrid: float
    basin_key: str
    answer: Optional[str] = None


def run_tot_bfs(llm: OpenAIClient, question: str, gold: str,
                 breadth: int = N_CANDIDATES, beam_width: int = 4, depth: int = N_ROUNDS,
                 temperature: float = TEMPERATURE, lambda_: Optional[float] = None,
                 quality_aware: bool = False) -> dict:
    method = ("tot_qabasin" if quality_aware else "tot_hybrid") if lambda_ is not None else "tot_standard"
    beam = [Node(trace_steps=[], verified_values=[], score_tot=1.0, score_hybrid=1.0, basin_key="root")]
    basin_visits: Dict[str, int] = defaultdict(int)
    basin_quality: Dict[str, float] = defaultdict(float)
    all_nodes: List[Node] = []
    n_wrong = 0

    for d in range(depth):
        is_final = (d == depth - 1)
        candidates: List[Node] = []
        for parent in beam:
            steps = generate_steps(llm, question, parent.trace_steps, n=breadth,
                                    temperature=temperature, is_final=is_final)
            for step in steps:
                new_trace = parent.trace_steps + [step]
                score, verified = verify_step(step)
                if score == 0.0:
                    n_wrong += 1
                new_verified = parent.verified_values + verified
                bk = make_basin_key(new_verified)
                ans = extract_final_answer(step) if is_final else None

                if lambda_ is not None:
                    n_b = basin_visits[bk]
                    if quality_aware:
                        q_b = basin_quality[bk]
                        hybrid_score = score - lambda_ * math.log(1 + n_b) * (1 - q_b)
                        basin_quality[bk] = (q_b * n_b + score) / (n_b + 1)
                    else:
                        hybrid_score = score - lambda_ * math.log(1 + n_b)
                else:
                    hybrid_score = score

                node = Node(trace_steps=new_trace, verified_values=new_verified,
                            score_tot=score, score_hybrid=hybrid_score, basin_key=bk, answer=ans)
                candidates.append(node)

        if not candidates:
            break
        all_nodes.extend(candidates)
        candidates.sort(key=lambda n: (n.score_hybrid, n.score_tot), reverse=True)
        beam = candidates[:beam_width]
        for node in beam:
            basin_visits[node.basin_key] += 1

    answer_nodes = [n for n in all_nodes if n.answer is not None]
    best = max(answer_nodes, key=lambda n: n.score_hybrid) if answer_nodes else \
        (max(all_nodes, key=lambda n: n.score_hybrid) if all_nodes else None)
    predicted = best.answer if best else None

    n_total = sum(basin_visits.values())
    if n_total > 0:
        probs = [v / n_total for v in basin_visits.values()]
        neff = math.exp(-sum(p * math.log(p + 1e-12) for p in probs))
    else:
        neff = 1.0

    return dict(
        method=method, success=is_correct(predicted, gold), predicted=predicted,
        n_basins=len(basin_visits), neff=round(neff, 4), n_wrong=n_wrong,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--base_url", default="https://api.openai.com/v1")
    ap.add_argument("--n_examples", type=int, default=100)
    ap.add_argument("--lambda_basin", type=float, default=3.0)
    ap.add_argument("--qa_lambda", type=float, default=0.5)
    ap.add_argument("--include_qabasin", action="store_true")
    ap.add_argument("--breadth", type=int, default=N_CANDIDATES)
    ap.add_argument("--beam_width", type=int, default=4)
    ap.add_argument("--depth", type=int, default=N_ROUNDS)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_args()
    llm = OpenAIClient(model=args.model, api_key=args.api_key, base_url=args.base_url,
                        logprobs=False, system_prompt=SYSTEM_PROMPT)

    problems = GSMHardLoader(max_problems=args.n_examples, seed=args.seed).load()
    logger.info("Loaded %d GSM-Hard problems. model=%s", len(problems), args.model)

    conditions = [("tot_standard", None, False), ("tot_hybrid", args.lambda_basin, False)]
    if args.include_qabasin:
        conditions.append(("tot_qabasin", args.qa_lambda, True))
    jobs = [(prob, name, lam, qa) for prob in problems for name, lam, qa in conditions]

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(run_tot_bfs, llm, prob.question, prob.answer,
                        args.breadth, args.beam_width, args.depth, args.temperature,
                        lam, qa): (prob, name)
            for prob, name, lam, qa in jobs
        }
        done = 0
        for fut in as_completed(futures):
            prob, name = futures[fut]
            r = fut.result()
            r.update(problem_id=prob.problem_id, gold=prob.answer)
            results.append(r)
            done += 1
            logger.info("[%d/%d] %s %s success=%s neff=%.2f n_basins=%d n_wrong=%d",
                        done, len(jobs), prob.problem_id, name, r["success"],
                        r["neff"], r["n_basins"], r["n_wrong"])

    elapsed = time.time() - t0
    model_tag = args.model.replace("/", "_").replace("-", "_")
    out_dir = _REPO_ROOT / "outputs" / f"gsm_hard_tot_vs_hybrid_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_example.json"
    out_path.write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    for name, _, _ in conditions:
        rows = [r for r in results if r["method"] == name]
        n = len(rows)
        acc = np.mean([r["success"] for r in rows])
        mean_basins = np.mean([r["n_basins"] for r in rows])
        mean_neff = np.mean([r["neff"] for r in rows])
        pct_never_wrong = np.mean([r["n_wrong"] == 0 for r in rows])
        logger.info("%s: n=%d Acc=%.3f mean_#basins=%.2f mean_Neff=%.2f frac_never_wrong=%.2f",
                    name, n, acc, mean_basins, mean_neff, pct_never_wrong)
    logger.info("Total elapsed: %.0fs", elapsed)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
