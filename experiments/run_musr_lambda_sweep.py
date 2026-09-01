"""
experiments/run_musr_lambda_sweep.py
=====================================
Hyperparameter sweep over λ (basin-penalty weight) on MuSR.

Matches the run_musr_200 setup exactly:
  - gpt-oss (NRP) for reasoning generation
  - gpt-4o-mini (OpenAI) as structured state extractor (StructuredMuSRExtractor)
  - NLI clustering (cross-encoder/nli-deberta-v3-small, entail≥0.45, contradict<0.3)
  - SBERT all-MiniLM-L6-v2 bias kernel
  - 4 rounds × 2 candidates per problem (budget=8)
  - Answer extracted via strict_musr_answer(state.trace_text) from history

Lambda grid: [0.5, 1.0, 1.5, 2.0, 3.0]

Outputs → outputs/musr_lambda_sweep/
  results.json          per-example × per-lambda results
  results.csv           flat CSV
  summary.md            markdown report with main table
  lambda_sweep.png      3-panel figure: accuracy, runtime, diversity

Usage
-----
  python experiments/run_musr_lambda_sweep.py \\
      --api_key <NRP_KEY> \\
      --extractor_api_key <OPENAI_KEY> \\
      [--n_examples 100]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # Tuple kept for run_sweep return type

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basin.confidence.estimator import AnswerResult, ConfidenceEstimator
from basin.controller.candidate_gen import CandidateGenerator
from basin.controller.metadynamics import MetadynamicsConfig, MetadynamicsController
from basin.datasets.musr_loader import MuSRLoader, normalise_musr_answer
from basin.datasets.loader import Problem
from basin.embedding.sentence_transformer_backend import SentenceTransformerEmbedder
from basin.llm.openai_backend import OpenAIClient
from basin.memory.nli_basin_memory import NLIBasinMemory
from basin.memory.nli_clustering import NLIBasinClusterer
from basin.state.extractor import StateExtractor
from basin.state.reasoning_state import ReasoningState
from basin.state.structured_musr_extractor import StructuredMuSRExtractor
from basin.verifier.verifier import HeuristicVerifier

# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path(os.environ.get("MUSR_SWEEP_OUTPUT_DIR", str(_REPO_ROOT / "outputs" / "musr_lambda_sweep")))
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_OUTPUT_DIR / "experiment.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LAMBDAS    = [0.5, 1.0, 1.5, 2.0, 3.0]
DEFAULT_N          = 100
N_ROUNDS           = 4
N_CANDIDATES       = 2   # budget = 8 per lambda
MAX_TOKENS         = 700
SBERT_MODEL        = "all-MiniLM-L6-v2"
NLI_MODEL          = "cross-encoder/nli-deberta-v3-small"
NLI_ENTAIL_THRESH  = 0.45
NLI_CONTRADICT_MAX = 0.3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neff(visit_history: List[int]) -> float:
    if not visit_history:
        return 1.0
    counts = Counter(visit_history)
    n = len(visit_history)
    H = -sum((c / n) * math.log(c / n) for c in counts.values())
    return math.exp(H)


def _basin_entropy(visit_history: List[int]) -> float:
    if not visit_history:
        return 0.0
    counts = Counter(visit_history)
    n = len(visit_history)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def _escape_rate(visit_history: List[int]) -> float:
    if len(visit_history) < 2:
        return 0.0
    seen: set = set()
    new = 0
    for b in visit_history:
        if b not in seen:
            new += 1
        seen.add(b)
    return max(0, new - 1) / max(1, len(visit_history) - 1)


def _revisit_rate(visit_history: List[int]) -> float:
    if len(visit_history) < 2:
        return 0.0
    seen: set = set()
    revisits = 0
    for b in visit_history:
        if b in seen:
            revisits += 1
        seen.add(b)
    return revisits / len(visit_history)


def is_correct(predicted: str, reference: str) -> bool:
    return normalise_musr_answer(predicted) == normalise_musr_answer(reference)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProblemResult:
    problem_id: str
    task: str
    lambda_: float
    reference: str
    predicted: str
    correct: bool
    confidence: float
    n_basins: int
    neff: float
    basin_entropy: float
    escape_rate: float
    revisit_rate: float
    pass_at_k: bool
    runtime_s: float = 0.0
    visit_history: List[int] = field(default_factory=list)
    all_predictions: List[str] = field(default_factory=list)


@dataclass
class LambdaAggregate:
    lambda_: float
    n: int
    accuracy: float
    pass_at_k: float
    mean_runtime_s: float
    mean_basins: float
    mean_neff: float
    mean_basin_entropy: float
    mean_answer_diversity: float
    mean_escape_rate: float
    mean_revisit_rate: float
    mean_confidence: float


# ---------------------------------------------------------------------------
# StructuredCandidateGenerator (copied from run_musr_200)
# ---------------------------------------------------------------------------

import re as _re

def strict_musr_answer(trace_text: str) -> str:
    """Extract a single A-E letter from trace text, or '?'."""
    text = trace_text.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        m = _re.search(r"ANSWER\s*[:\-–]?\s*\**([A-E])\b", line.upper())
        if m:
            return m.group(1)
    for line in reversed(lines):
        m = _re.search(r"(?:OPTION|CHOICE|LETTER)\s+([A-E])\b", line.upper())
        if m:
            return m.group(1)
    for line in reversed(lines):
        m = _re.search(r"\*{1,2}([A-E])\*{1,2}|\(([A-E])\)", line.upper())
        if m:
            return m.group(1) or m.group(2)
    for line in reversed(lines[-10:]):
        m = _re.match(r"^\s*([A-E])[.\s]*$", line.upper())
        if m:
            return m.group(1)
    m = _re.search(r"\b([A-E])\b", text.upper())
    if m:
        return m.group(1)
    return "?"


class StructuredCandidateGenerator(CandidateGenerator):
    def __init__(self, struct_extractor: StructuredMuSRExtractor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.struct_extractor = struct_extractor

    def generate_candidates(
        self,
        question: str,
        n: int,
        step_index: int = 0,
        previous_trace: Optional[str] = None,
    ) -> List[ReasoningState]:
        states = super().generate_candidates(question, n, step_index, previous_trace)
        struct_texts: List[str] = []
        for state in states:
            predicted_answer = strict_musr_answer(state.trace_text)
            struct = self.struct_extractor.extract_structured(
                state.trace_text, predicted_answer=predicted_answer
            )
            embed_text = StructuredMuSRExtractor.make_embed_text(struct)
            basin_key  = StructuredMuSRExtractor.make_basin_key(struct)
            state.structured_state      = struct
            state.structured_embed_text = embed_text
            state.basin_key             = basin_key
            state.parse_failed          = (predicted_answer == "?")
            struct_texts.append(embed_text)
        structured_embeddings = self.embedder.embed_batch(struct_texts)
        for i, state in enumerate(states):
            state.embedding_vector = structured_embeddings[i]
        return states


# ---------------------------------------------------------------------------
# Build MetaDyn controller (matches run_musr_200 metadyn_nli setup)
# ---------------------------------------------------------------------------

def build_metadyn(
    llm,
    struct_llm,
    embedder: SentenceTransformerEmbedder,
    nli_clusterer: NLIBasinClusterer,
    lambda_: float,
    n_rounds: int = N_ROUNDS,
    n_candidates: int = N_CANDIDATES,
    max_tokens: int = MAX_TOKENS,
    quality_aware: bool = False,
) -> MetadynamicsController:
    struct_extractor = StructuredMuSRExtractor(llm_client=struct_llm, max_tokens=600)
    gen = StructuredCandidateGenerator(
        struct_extractor=struct_extractor,
        llm_client=llm,
        extractor=StateExtractor(mode="heuristic"),
        embedder=embedder,
        temperature=0.8,
        max_tokens=max_tokens,
    )
    memory = NLIBasinMemory(
        kernel_bandwidth=0.5, base_weight=1.0, count_scaling=0.5,
        clusterer=nli_clusterer,
    )
    cfg = MetadynamicsConfig(
        alpha=1.0, beta=0.5, lambda_=lambda_,
        n_rounds=n_rounds, n_candidates=n_candidates, temperature=0.8,
        quality_aware=quality_aware,
    )
    return MetadynamicsController(
        candidate_generator=gen,
        verifier=HeuristicVerifier(),
        memory=memory,
        confidence_estimator=ConfidenceEstimator(temperature=1.0),
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Run one problem × one lambda
# ---------------------------------------------------------------------------

def run_one(
    problem: Problem,
    llm,
    struct_llm,
    embedder: SentenceTransformerEmbedder,
    nli_clusterer: NLIBasinClusterer,
    lambda_: float,
    n_rounds: int,
    n_candidates: int,
    max_tokens: int,
    quality_aware: bool = False,
) -> ProblemResult:
    task = problem.problem_id.split("_")[0] if "_" in problem.problem_id else "unknown"
    _t0 = time.time()
    try:
        ctrl = build_metadyn(
            llm, struct_llm, embedder, nli_clusterer,
            lambda_, n_rounds, n_candidates, max_tokens,
            quality_aware=quality_aware,
        )
        res     = ctrl.solve(problem.question)
        history = ctrl.selected_history()

        # Extract answers via structured state (gpt-4o-mini) then strict fallback
        all_preds = []
        for state in history:
            struct_ans = getattr(state, "structured_state", {}).get("final_answer", "?")
            ans = struct_ans.upper() if struct_ans in "ABCDE" else strict_musr_answer(state.trace_text)
            all_preds.append(ans.lower() if ans != "?" else "?")

        # Predicted = majority vote over history answers
        valid_preds = [p for p in all_preds if p and p != "?"]
        if valid_preds:
            counts = Counter(valid_preds)
            predicted = counts.most_common(1)[0][0]
        elif all_preds:
            predicted = all_preds[-1]  # last state as last resort
        else:
            predicted = "?"

        vh = res.visit_history or []
        correct   = is_correct(predicted, problem.answer)
        pass_at_k = any(is_correct(p, problem.answer) for p in all_preds)

        return ProblemResult(
            problem_id=problem.problem_id,
            task=task,
            lambda_=lambda_,
            reference=problem.answer,
            predicted=predicted,
            correct=correct,
            confidence=float(max(0.0, min(1.0, res.confidence))),
            n_basins=res.n_basins_explored,
            neff=_neff(vh),
            basin_entropy=_basin_entropy(vh),
            escape_rate=_escape_rate(vh),
            revisit_rate=_revisit_rate(vh),
            pass_at_k=pass_at_k,
            runtime_s=time.time() - _t0,
            visit_history=vh,
            all_predictions=all_preds,
        )
    except Exception as exc:
        logger.warning("  [%s] λ=%.1f FAILED: %s", problem.problem_id, lambda_, exc, exc_info=True)
        return ProblemResult(
            problem_id=problem.problem_id, task=task, lambda_=lambda_,
            reference=problem.answer, predicted="?", correct=False,
            confidence=0.0, n_basins=0, neff=1.0, basin_entropy=0.0,
            escape_rate=0.0, revisit_rate=0.0, pass_at_k=False,
            runtime_s=time.time() - _t0,
        )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(lambda_: float, results: List[ProblemResult]) -> LambdaAggregate:
    n = len(results)
    if n == 0:
        return LambdaAggregate(lambda_, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    n_correct = sum(r.correct for r in results)
    return LambdaAggregate(
        lambda_=lambda_,
        n=n,
        accuracy=n_correct / n,
        pass_at_k=sum(r.pass_at_k for r in results) / n,
        mean_runtime_s=sum(r.runtime_s for r in results) / n,
        mean_basins=sum(r.n_basins for r in results) / n,
        mean_neff=sum(r.neff for r in results) / n,
        mean_basin_entropy=sum(r.basin_entropy for r in results) / n,
        mean_answer_diversity=sum(
            len(set(r.all_predictions)) / max(1, len(r.all_predictions))
            for r in results
        ) / n,
        mean_escape_rate=sum(r.escape_rate for r in results) / n,
        mean_revisit_rate=sum(r.revisit_rate for r in results) / n,
        mean_confidence=sum(r.confidence for r in results) / n,
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_sweep(
    llm,
    struct_llm,
    problems: List[Problem],
    lambdas: List[float],
    n_rounds: int,
    n_candidates: int,
    max_tokens: int,
    nli_model: str = NLI_MODEL,
    nli_entail_threshold: float = NLI_ENTAIL_THRESH,
) -> Tuple[List[ProblemResult], Dict[float, LambdaAggregate]]:
    logger.info("Loading SentenceTransformer (%s)…", SBERT_MODEL)
    embedder = SentenceTransformerEmbedder(SBERT_MODEL)

    logger.info("Loading NLI clusterer (%s, entail_threshold=%s)…", nli_model, nli_entail_threshold)
    nli_clusterer = NLIBasinClusterer(
        model_name=nli_model,
        entail_threshold=nli_entail_threshold,
        max_contradict=NLI_CONTRADICT_MAX,
        text_field="main_hypothesis",
    )

    all_results: List[ProblemResult] = []

    for prob_idx, prob in enumerate(problems):
        short_q = prob.question[:70].replace("\n", " ")
        logger.info(
            "[%d/%d] %s  ref=%s  q=%s…",
            prob_idx + 1, len(problems), prob.problem_id, prob.answer, short_q,
        )

        for lam in lambdas:
            result = run_one(
                prob, llm, struct_llm, embedder, nli_clusterer,
                lam, n_rounds, n_candidates, max_tokens,
            )
            logger.info(
                "  λ=%-4.1f  pred=%-3s  correct=%-5s  basins=%d  Neff=%.2f  (%.1fs)",
                lam, result.predicted, result.correct,
                result.n_basins, result.neff, result.runtime_s,
            )
            all_results.append(result)

    per_lambda: Dict[float, List[ProblemResult]] = {lam: [] for lam in lambdas}
    for r in all_results:
        per_lambda[r.lambda_].append(r)

    aggregates = {lam: aggregate(lam, per_lambda[lam]) for lam in lambdas}
    return all_results, aggregates


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_json(
    all_results: List[ProblemResult],
    aggregates: Dict[float, LambdaAggregate],
    cfg: dict,
) -> Path:
    out = {
        "config": cfg,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "aggregates": [
            {
                "lambda": a.lambda_,
                "n": a.n,
                "accuracy": a.accuracy,
                "pass_at_k": a.pass_at_k,
                "mean_runtime_s": a.mean_runtime_s,
                "mean_basins": a.mean_basins,
                "mean_neff": a.mean_neff,
                "mean_basin_entropy": a.mean_basin_entropy,
                "mean_answer_diversity": a.mean_answer_diversity,
                "mean_escape_rate": a.mean_escape_rate,
                "mean_revisit_rate": a.mean_revisit_rate,
                "mean_confidence": a.mean_confidence,
            }
            for a in sorted(aggregates.values(), key=lambda x: x.lambda_)
        ],
        "per_example": [
            {
                "problem_id": r.problem_id,
                "task": r.task,
                "lambda": r.lambda_,
                "reference": r.reference,
                "predicted": r.predicted,
                "correct": r.correct,
                "confidence": r.confidence,
                "n_basins": r.n_basins,
                "neff": r.neff,
                "basin_entropy": r.basin_entropy,
                "escape_rate": r.escape_rate,
                "revisit_rate": r.revisit_rate,
                "pass_at_k": r.pass_at_k,
                "visit_history": r.visit_history,
            }
            for r in all_results
        ],
    }
    path = _OUTPUT_DIR / "results.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return path


def save_csv(all_results: List[ProblemResult]) -> Path:
    path = _OUTPUT_DIR / "results.csv"
    fieldnames = [
        "problem_id", "task", "lambda", "reference", "predicted",
        "correct", "confidence", "n_basins", "neff",
        "basin_entropy", "escape_rate", "revisit_rate", "pass_at_k", "runtime_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "problem_id": r.problem_id,
                "task": r.task,
                "lambda": f"{r.lambda_:.1f}",
                "reference": r.reference,
                "predicted": r.predicted,
                "correct": int(r.correct),
                "confidence": f"{r.confidence:.4f}",
                "n_basins": r.n_basins,
                "neff": f"{r.neff:.4f}",
                "basin_entropy": f"{r.basin_entropy:.4f}",
                "escape_rate": f"{r.escape_rate:.4f}",
                "revisit_rate": f"{r.revisit_rate:.4f}",
                "pass_at_k": int(r.pass_at_k),
                "runtime_s": f"{r.runtime_s:.2f}",
            })
    return path


def save_markdown(
    aggregates: Dict[float, LambdaAggregate],
    cfg: dict,
) -> Path:
    lambdas_sorted = sorted(aggregates.keys())

    lines = [
        "# MuSR λ Sweep — BASIN",
        "",
        f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Dataset:** MuSR (n={cfg['n_examples']}, seed={cfg['seed']})",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | {cfg['model']} |",
        f"| Embedder | sbert:{SBERT_MODEL} |",
        f"| Basin clustering | NLI ({NLI_MODEL}) |",
        f"| Extractor | {cfg['extractor_model']} (OpenAI) |",
        f"| n_rounds | {cfg['n_rounds']} |",
        f"| n_candidates | {cfg['n_candidates']} |",
        f"| Budget/problem/λ | {cfg['n_rounds'] * cfg['n_candidates']} |",
        f"| λ grid | {lambdas_sorted} |",
        "",
        "## Main Results",
        "",
        "| λ | Acc | Pass@k | Runtime/prob | Diversity | Basins | Neff |",
        "|---|-----|--------|-------------|-----------|--------|------|",
    ]

    for lam in lambdas_sorted:
        a = aggregates[lam]
        lines.append(
            f"| {lam:.1f} "
            f"| {a.accuracy:.3f} "
            f"| {a.pass_at_k:.3f} "
            f"| {a.mean_runtime_s:.1f}s "
            f"| {a.mean_answer_diversity:.3f} "
            f"| {a.mean_basins:.2f} "
            f"| {a.mean_neff:.2f} |"
        )

    # Identify best λ by accuracy
    best_lam = max(lambdas_sorted, key=lambda l: aggregates[l].accuracy)
    best_a   = aggregates[best_lam]

    lines += [
        "",
        "## Key Findings",
        "",
        f"- **Best λ:** {best_lam:.1f}  (acc={best_a.accuracy:.3f})",
        "",
        "### Accuracy vs λ",
        "",
    ]
    for lam in lambdas_sorted:
        a = aggregates[lam]
        bar_len = int(a.accuracy * 40)
        lines.append(f"  λ={lam:.1f}  {'█' * bar_len}{'░' * (40 - bar_len)}  {a.accuracy:.3f}")

    lines += ["", "### Runtime vs λ", ""]
    max_rt = max(a.mean_runtime_s for a in aggregates.values()) or 1.0
    for lam in lambdas_sorted:
        a = aggregates[lam]
        bar_len = int((a.mean_runtime_s / max_rt) * 40)
        lines.append(f"  λ={lam:.1f}  {'█' * bar_len}{'░' * (40 - bar_len)}  {a.mean_runtime_s:.1f}s")

    lines += ["", "### Diversity vs λ", ""]
    max_div = max(a.mean_answer_diversity for a in aggregates.values()) or 1.0
    for lam in lambdas_sorted:
        a = aggregates[lam]
        bar_len = int((a.mean_answer_diversity / max_div) * 40)
        lines.append(f"  λ={lam:.1f}  {'█' * bar_len}{'░' * (40 - bar_len)}  {a.mean_answer_diversity:.3f}")

    path = _OUTPUT_DIR / "summary.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def save_plot(aggregates: Dict[float, LambdaAggregate]) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return None

    lambdas_sorted = sorted(aggregates.keys())
    x         = np.array(lambdas_sorted)
    accs      = np.array([aggregates[l].accuracy             for l in lambdas_sorted])
    passk     = np.array([aggregates[l].pass_at_k            for l in lambdas_sorted])
    runtimes  = np.array([aggregates[l].mean_runtime_s       for l in lambdas_sorted])
    diversities = np.array([aggregates[l].mean_answer_diversity for l in lambdas_sorted])

    best_idx = int(np.argmax(accs))
    best_lam = lambdas_sorted[best_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MuSR — MetaDyn λ Sweep", fontsize=14, fontweight="bold")

    # Panel 1: Accuracy
    ax = axes[0]
    ax.plot(x, accs, "o-", color="#2196F3", linewidth=2, label="Accuracy")
    ax.plot(x, passk, "s--", color="#4CAF50", alpha=0.75, label="Pass@k")
    ax.axvline(best_lam, color="red", linestyle=":", alpha=0.55, label=f"Best λ={best_lam:.1f}")
    ax.set_xlabel("λ")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs λ")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Compute efficiency (runtime per problem)
    ax = axes[1]
    ax.plot(x, runtimes, "o-", color="#FF9800", linewidth=2)
    ax.axvline(best_lam, color="red", linestyle=":", alpha=0.55)
    ax.set_xlabel("λ")
    ax.set_ylabel("Runtime / problem (s)")
    ax.set_title("Compute Efficiency vs λ")
    ax.grid(True, alpha=0.3)

    # Panel 3: Answer diversity
    ax = axes[2]
    ax.plot(x, diversities, "o-", color="#9C27B0", linewidth=2)
    ax.axvline(best_lam, color="red", linestyle=":", alpha=0.55)
    ax.set_xlabel("λ")
    ax.set_ylabel("Answer diversity")
    ax.set_title("Diversity vs λ")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = _OUTPUT_DIR / "lambda_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="MuSR λ sweep for MetaDynamics")
    p.add_argument("--model", default="gpt-oss")
    p.add_argument("--extractor_model", default="gpt-4o-mini")
    p.add_argument("--n_examples", type=int, default=DEFAULT_N)
    p.add_argument(
        "--lambdas", nargs="+", type=float,
        default=DEFAULT_LAMBDAS,
        help="List of λ values to sweep (default: 0.5 1.0 1.5 2.0 3.0)",
    )
    p.add_argument("--n_rounds",     type=int, default=N_ROUNDS)
    p.add_argument("--n_candidates", type=int, default=N_CANDIDATES)
    p.add_argument("--max_tokens",   type=int, default=MAX_TOKENS)
    p.add_argument(
        "--tasks", nargs="+",
        choices=["murder_mysteries", "object_placements", "team_allocation"],
        default=None,
        help="MuSR subtasks (default: all three)",
    )
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--api_key",           default=None, help="NRP API key")
    p.add_argument("--extractor_api_key", default=None, help="OpenAI API key for gpt-4o-mini")
    p.add_argument("--nli_model",           default=NLI_MODEL)
    p.add_argument("--nli_entail_threshold", type=float, default=NLI_ENTAIL_THRESH)
    return p.parse_args()


def main():
    args = _parse_args()

    system_prompt = (
        "You are a careful reasoning assistant. "
        "Always end your response with 'Answer: X' on its own line, "
        "where X is a single uppercase letter (A, B, C, ...). "
        "Never skip the Answer line."
    )

    cfg = {
        "model": args.model,
        "extractor_model": args.extractor_model,
        "n_examples": args.n_examples,
        "lambdas": sorted(args.lambdas),
        "n_rounds": args.n_rounds,
        "n_candidates": args.n_candidates,
        "budget_per_lambda": args.n_rounds * args.n_candidates,
        "tasks": args.tasks or ["murder_mysteries", "object_placements", "team_allocation"],
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "embedder": f"sbert:{SBERT_MODEL}",
        "nli_model": args.nli_model,
        "nli_entail_threshold": args.nli_entail_threshold,
        "nli_contradict_max": NLI_CONTRADICT_MAX,
    }

    logger.info("=" * 60)
    logger.info("MuSR λ Sweep  —  BASIN")
    logger.info("=" * 60)
    for k, v in cfg.items():
        logger.info("  %-28s %s", k, v)
    logger.info("  Total LLM calls (est.): ~%d",
                args.n_examples * len(args.lambdas) * args.n_rounds * args.n_candidates)

    # ---- LLMs ----
    nrp_key = args.api_key or os.environ.get("NRP_API_KEY")
    oai_key = args.extractor_api_key or os.environ.get("OPENAI_API_KEY") or nrp_key

    _OPENAI_MODEL_PREFIXES = ("gpt-4", "gpt-3.5", "o1", "o3", "o4")
    model_is_openai = args.model.startswith(_OPENAI_MODEL_PREFIXES)
    llm = OpenAIClient(
        model=args.model,
        api_key=(os.environ.get("OPENAI_API_KEY") or oai_key) if model_is_openai else nrp_key,
        base_url=None if model_is_openai else "https://ellm.nrp-nautilus.io/v1",
        logprobs=False,
        system_prompt=system_prompt,
    )
    struct_llm = OpenAIClient(
        model=args.extractor_model,
        api_key=oai_key,
        base_url=None,   # OpenAI default
        logprobs=False,
        system_prompt=system_prompt,
    )

    # ---- Data ----
    loader   = MuSRLoader(tasks=args.tasks, max_problems=args.n_examples, seed=args.seed)
    problems = loader.load()
    logger.info("Loaded %d MuSR problems", len(problems))
    if not problems:
        raise RuntimeError("No problems loaded — check MuSR dataset installation.")

    # ---- Sweep ----
    t0 = time.time()
    all_results, aggregates = run_sweep(
        llm=llm,
        struct_llm=struct_llm,
        problems=problems,
        lambdas=sorted(args.lambdas),
        n_rounds=args.n_rounds,
        n_candidates=args.n_candidates,
        max_tokens=args.max_tokens,
        nli_model=args.nli_model,
        nli_entail_threshold=args.nli_entail_threshold,
    )
    elapsed = time.time() - t0

    # ---- Summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("LAMBDA SWEEP RESULTS  (%.0f s total)", elapsed)
    logger.info("=" * 60)
    header = f"{'λ':>5} {'Acc':>7} {'Pass@k':>8} {'Runtime':>9} {'Diversity':>10} {'Neff':>6}"
    logger.info(header)
    logger.info("-" * len(header))
    for lam in sorted(aggregates.keys()):
        a = aggregates[lam]
        logger.info(
            "%5.1f  %7.3f  %8.3f  %8.1fs  %10.3f  %6.2f",
            lam, a.accuracy, a.pass_at_k, a.mean_runtime_s,
            a.mean_answer_diversity, a.mean_neff,
        )

    # ---- Save ----
    json_path = save_json(all_results, aggregates, cfg)
    csv_path  = save_csv(all_results)
    md_path   = save_markdown(aggregates, cfg)
    plot_path = save_plot(aggregates)

    logger.info("")
    logger.info("Saved: %s", json_path)
    logger.info("Saved: %s", csv_path)
    logger.info("Saved: %s", md_path)
    if plot_path:
        logger.info("Saved: %s", plot_path)
    logger.info("Log:   %s", _OUTPUT_DIR / 'experiment.log')
    logger.info("Done.")


if __name__ == "__main__":
    main()
