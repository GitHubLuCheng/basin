"""
experiments/run_game24_mcts_qabasin.py
=========================================
QA-BASIN (Eq. 6) inside MCTS, extending run_game24_mcts_vs_basin.py the same
way run_game24_qabasin_together.py extends the ToT BASIN comparison to
quality-aware BASIN. Adds a quality-modulated penalty term directly to the
UCT child-selection score:

  mcts_standard : score(child) = UCB(child, c)
  mcts_basin    : score(child) = UCB(child, c) - lambda_basin * log(1 + visits[key])
  mcts_qabasin  : score(child) = UCB(child, c)
                    - lambda_qa * log(1 + visits[key]) * (1 - quality[key])

quality[key] is the running mean of each basin's deterministic state_value
(the same "free" quality signal run_game24_qabasin_together.py uses for
ToT's QA-BASIN, since Game of 24 has an exact solver-based value function
rather than needing a separate LLM verifier) -- updated every time a node
in that basin is selected during tree traversal, mirroring how visits[]
accumulates.

All three methods share the same LLM proposals, n_simulations, n_proposals,
max_depth, and c; they differ only in the selection-score formula.

Usage
-----
  python experiments/run_game24_mcts_qabasin.py --api_key <KEY> \\
      --base_url https://api.openai.com/v1 --model gpt-4o-mini --n_puzzles 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basin.llm.openai_backend import OpenAIClient

_g24_path = str(_REPO_ROOT / "experiments/run_game24_tot_vs_hybrid.py")
_g24 = {"__file__": _g24_path}
exec(open(_g24_path).read(), _g24)
load_game24 = _g24["load_game24"]
state_value = _g24["state_value"]
make_basin_key = _g24["make_basin_key"]
build_proposal_prompt = _g24["build_proposal_prompt"]
parse_proposal = _g24["parse_proposal"]
_PROPOSE_SYSTEM = _g24["_PROPOSE_SYSTEM"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    remaining: Tuple[float, ...]
    trace: List[str]
    parent: Optional["MCTSNode"] = field(default=None, repr=False)
    step_text: Optional[str] = None

    Q: float = 0.0
    N: int = 0
    children: List["MCTSNode"] = field(default_factory=list, repr=False)
    expanded: bool = False
    value: float = field(init=False)

    def __post_init__(self):
        self.value = state_value(self.remaining)

    def is_terminal(self) -> bool:
        return len(self.remaining) == 1

    def is_leaf(self) -> bool:
        return not self.expanded or not self.children

    def ucb(self, c: float) -> float:
        if self.N == 0:
            return float("inf")
        parent_N = self.parent.N if self.parent else self.N
        return self.Q / self.N + c * math.sqrt(math.log(max(parent_N, 1)) / self.N)

    def basin_key(self) -> str:
        return make_basin_key(self.trace, self.remaining)

    def select_score(self, c: float, lambda_: float, visits: Dict[str, int],
                      quality: Optional[Dict[str, float]] = None) -> float:
        u = self.ucb(c)
        if u == float("inf") or lambda_ <= 0:
            return u
        bk = self.basin_key()
        n_b = visits.get(bk, 0)
        if quality is not None:
            q_b = quality.get(bk, 0.0)
            return u - lambda_ * math.log(1 + n_b) * (1 - q_b)
        return u - lambda_ * math.log(1 + n_b)

    def best_child(self, c: float, lambda_: float, visits: Dict[str, int],
                    quality: Optional[Dict[str, float]] = None) -> "MCTSNode":
        return max(self.children, key=lambda ch: ch.select_score(c, lambda_, visits, quality))


def propose_steps(llm: OpenAIClient, remaining: Tuple[float, ...], n: int,
                   temperature: float = 0.7, max_retries: int = 3,
                   retry_delay: float = 10.0) -> List[dict]:
    full_prompt = f"{_PROPOSE_SYSTEM}\n\n{build_proposal_prompt(remaining)}"
    for attempt in range(max_retries):
        try:
            responses = llm.generate(full_prompt, temperature=temperature, max_tokens=400, n=1)
            parsed = parse_proposal(responses[0].text, remaining)
            if parsed:
                return parsed[:n]
        except Exception as e:
            logger.warning("LLM error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
    return []


def _update_quality(quality: Dict[str, float], visits: Dict[str, int], node: "MCTSNode") -> None:
    bk = node.basin_key()
    n_prev = visits.get(bk, 0)
    quality[bk] = (quality.get(bk, 0.0) * n_prev + node.value) / (n_prev + 1)


def mcts_solve(numbers: Tuple[int, ...], llm: OpenAIClient, c: float, lambda_: float,
                n_simulations: int, n_proposals: int, max_depth: int,
                temperature: float = 0.7, quality_aware: bool = False) -> Dict:
    root = MCTSNode(remaining=tuple(float(x) for x in numbers), trace=[])
    n_calls = 0
    visits: Dict[str, int] = defaultdict(int)
    quality: Optional[Dict[str, float]] = defaultdict(float) if quality_aware else None
    sim_answers: List[str] = []

    for _ in range(n_simulations):
        node = root
        depth = 0
        while not node.is_leaf() and not node.is_terminal() and depth < max_depth:
            node = node.best_child(c, lambda_, visits, quality)
            if quality is not None:
                _update_quality(quality, visits, node)
            visits[node.basin_key()] += 1
            depth += 1

        if node.is_terminal():
            v = node.value
        elif depth >= max_depth:
            v = node.value
        else:
            if not node.expanded:
                steps = propose_steps(llm, node.remaining, n_proposals, temperature)
                n_calls += 1
                node.expanded = True
                for s in steps:
                    child = MCTSNode(
                        remaining=s["new_remaining"],
                        trace=node.trace + [s["step_text"]],
                        parent=node, step_text=s["step_text"],
                    )
                    node.children.append(child)

            if not node.children:
                v = 0.0
            else:
                unvisited = [ch for ch in node.children if ch.N == 0]
                child = unvisited[0] if unvisited else node.best_child(c, lambda_, visits, quality)
                if quality is not None:
                    _update_quality(quality, visits, child)
                visits[child.basin_key()] += 1
                v = child.value
                node = child

        sim_answers.append(node.basin_key())

        cur = node
        while cur is not None:
            cur.N += 1
            cur.Q += v
            cur = cur.parent

    def _any_success(nd: MCTSNode) -> bool:
        if nd.is_terminal() and abs(nd.remaining[0] - 24) < 1e-6 and nd.N > 0:
            return True
        return any(_any_success(ch) for ch in nd.children)
    success = _any_success(root)

    answer_counts: Dict[str, int] = defaultdict(int)
    for ans in sim_answers:
        answer_counts[ans] += 1
    n_basins = len(answer_counts)
    total = len(sim_answers)
    if total > 0 and n_basins > 0:
        probs = np.array(list(answer_counts.values())) / total
        neff = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
    else:
        neff = 1.0

    return dict(success=success, n_basins=n_basins, neff=neff, n_calls=n_calls)


def parse_args():
    p = argparse.ArgumentParser(description="Game of 24: MCTS standard vs MCTS+BASIN vs MCTS+QA-BASIN")
    p.add_argument("--api_key", required=True)
    p.add_argument("--base_url", default="https://api.openai.com/v1")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--c", type=float, default=1.0, help="Fixed UCT exploration constant (all methods)")
    p.add_argument("--lambda_basin", type=float, default=3.0)
    p.add_argument("--lambda_qa", type=float, default=0.5)
    p.add_argument("--n_simulations", type=int, default=50)
    p.add_argument("--n_proposals", type=int, default=5)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--n_puzzles", type=int, default=100)
    p.add_argument("--start_idx", type=int, default=900)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    model_tag = args.model.replace("/", "_").replace("-", "_").replace(".", "")
    out_dir = _REPO_ROOT / "outputs" / f"game24_mcts_qabasin_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = OpenAIClient(model=args.model, api_key=args.api_key, base_url=args.base_url, logprobs=False)

    puzzles = load_game24(args.start_idx, args.start_idx + args.n_puzzles - 1)
    logger.info("Loaded %d puzzles. c=%.1f (fixed)  lambda_basin=%.1f  lambda_qa=%.1f  n_sims=%d",
                len(puzzles), args.c, args.lambda_basin, args.lambda_qa, args.n_simulations)

    # (method, lambda_, quality_aware)
    methods = [
        ("mcts_standard", 0.0, False),
        ("mcts_basin", args.lambda_basin, False),
        ("mcts_qabasin", args.lambda_qa, True),
    ]
    jobs = [(idx, numbers, method, lam, qa) for idx, numbers in puzzles for method, lam, qa in methods]

    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                mcts_solve, numbers, llm, args.c, lam,
                args.n_simulations, args.n_proposals, args.max_depth, args.temperature, qa,
            ): (idx, numbers, method)
            for idx, numbers, method, lam, qa in jobs
        }
        done = 0
        for fut in as_completed(futures):
            idx, numbers, method = futures[fut]
            out = fut.result()
            results.append(dict(idx=idx, numbers=list(numbers), method=method, **out))
            done += 1
            logger.info("[%d/%d] idx=%d %s success=%s neff=%.2f n_basins=%d calls=%d",
                        done, len(jobs), idx, method, out["success"], out["neff"],
                        out["n_basins"], out["n_calls"])

    out_path = out_dir / "per_example.json"
    out_path.write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    for method, _, _ in methods:
        rows = [r for r in results if r["method"] == method]
        n = len(rows)
        acc = np.mean([r["success"] for r in rows])
        mean_basins = np.mean([r["n_basins"] for r in rows])
        mean_neff = np.mean([r["neff"] for r in rows])
        logger.info("%s: n=%d Acc=%.3f mean_#basins=%.2f mean_Neff=%.2f", method, n, acc, mean_basins, mean_neff)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
