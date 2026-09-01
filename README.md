# BASIN: Structure-Aware Search for Inference-Time LLMs

Code to reproduce the main-text results tables from
**"Escaping Redundant Reasoning: Structure-Aware Search for Inference-Time LLMs"** (preprint).

BASIN modifies candidate selection during tree-structured search (e.g. Tree-of-Thoughts)
by penalizing candidates that fall into reasoning "basins" the search has already
visited, encouraging exploration of genuinely distinct reasoning strategies rather than
repeated variants of the same one:

```
f̃(s) = f(s) − λ · log(1 + visits[basin(s)])
```

where `f(s)` is the model/verifier's own score for candidate `s`, `basin(s)` identifies
which cluster of previously-explored reasoning states `s` belongs to, and `visits[·]`
counts how many times that basin has been selected so far. Increasing `λ` increases the
penalty on revisiting already-explored basins.

This repository includes the code for every table appearing in the main body of the
paper (Sections 1–5, before the Appendix):

| Table | Task | Models | Script |
|---|---|---|---|
| Table 1 | Game of 24 | gpt-4o-mini, Qwen3-27B | `experiments/run_game24_tot_vs_hybrid.py` |
| Table 2/3 | MuSR | gpt-oss-120b, gpt-4o-mini | `experiments/run_musr_lambda_sweep.py` |
| Table (Generalization) | HumanEval | gpt-4o-mini, gpt-oss-120b, Qwen2.5-7B-Instruct, Llama-3.3-70B-Instruct | `experiments/run_humaneval_tot_vs_basin.py` |
| Table (Generalization) | GSM-Hard | gpt-4o-mini, gpt-oss-120b, Qwen2.5-7B-Instruct, Llama-3.3-70B-Instruct | `experiments/run_gsm_hard_tot_vs_hybrid.py` |
| Table (MCTS) | Game of 24 (UCT-based MCTS) | gpt-4o-mini, gpt-oss-120b, Qwen2.5-7B-Instruct, Llama-3.3-70B-Instruct | `experiments/run_game24_mcts_qabasin.py` |

All five scripts implement standard search alongside the identical search with the
BASIN penalty applied (and, where the paper reports it, the quality-aware QA-BASIN
variant, Eq. 6), under matched inference budgets, so the conditions can be compared
directly. Not included: the Graph-of-Thoughts result (its full table lives in the
Appendix, though headline numbers are stated in main-text prose), the two post-hoc
analysis figures (collapse-stratified accuracy, the redundancy-gap routing diagnostic),
and inline-only ablations (the Diverse-Beam-Search baseline, the quality-signal
comparison, the temperature sweep) that don't have a dedicated main-body table.

## Setup

```bash
pip install -r requirements.txt
```

MuSR, HumanEval, and GSM-Hard problems are all loaded automatically via the HuggingFace
`datasets` library on first run. Game of 24 puzzles are downloaded automatically from
the standard `tree-of-thought-llm` benchmark CSV on first run.

You'll need:
- An API key for whichever LLM backend you use as the reasoning model (OpenAI, or any
  OpenAI-compatible endpoint serving open-weight models such as gpt-oss).
- An OpenAI API key for `gpt-4o-mini`, used as the structured-state extractor in the
  MuSR pipeline (a fixed, matched-cost component of both the standard and BASIN
  conditions — see the paper's discussion of extraction overhead).

## Reproducing Table 1 (Game of 24)

Beam size 5, branching factor 5, depth 3, λ=3.0, n=100 (puzzle indices 900–999),
matching the paper's Table 1 setup.

```bash
# gpt-4o-mini (via OpenAI)
python experiments/run_game24_tot_vs_hybrid.py \
    --api_key <OPENAI_KEY> --base_url https://api.openai.com/v1 \
    --model gpt-4o-mini --methods tot_standard,tot_hybrid \
    --breadth 5 --beam_width 5 --depth 3 --lambda_ 3.0 \
    --start_idx 900 --end_idx 999

# Qwen3-27B (or any other OpenAI-compatible endpoint) — point --base_url at your provider
python experiments/run_game24_tot_vs_hybrid.py \
    --api_key <PROVIDER_KEY> --base_url <PROVIDER_BASE_URL> \
    --model <MODEL_NAME> --methods tot_standard,tot_hybrid \
    --breadth 5 --beam_width 5 --depth 3 --lambda_ 3.0 \
    --start_idx 900 --end_idx 999
```

`--methods tot_standard,tot_hybrid` restricts the run to the standard-ToT-vs-BASIN
comparison (the script's default method list also includes a few self-consistency
baselines not needed for Table 1). The run writes per-example results plus aggregate
accuracy/#Basins/N_eff to `outputs/game24_tot_vs_hybrid/`.

`tot_hybrid` applies the plain BASIN penalty (Eq. 3) by default. Pass `--quality_aware`
to switch it to the quality-aware variant instead, QA-BASIN
(`f̃(s) = f(s) − λ·log(1+visits[basin(s)])·(1−quality[basin(s)])`, Eq. 6 in the paper) —
run it as a second, separate invocation (with a different `--output_dir`) if you want
both plain-BASIN and QA-BASIN results side by side.

## Reproducing Table 2/3 (MuSR)

n=300, 9 reasoning rounds, λ=3.0, all three MuSR subtasks (murder mysteries, object
placements, team allocation), matching the paper's Table 2/3 setup.

This script sweeps the BASIN penalty strength λ rather than naming separate
"standard"/"BASIN" methods — λ=0 makes the penalty term vanish exactly, which is
standard ToT, so pass both `0.0` and `3.0` to get the paper's standard-vs-BASIN
comparison in one run:

```bash
# gpt-oss-120b (via any OpenAI-compatible endpoint serving it)
python experiments/run_musr_lambda_sweep.py \
    --api_key <PROVIDER_KEY> --extractor_api_key <OPENAI_KEY> \
    --model gpt-oss \
    --n_examples 300 --n_rounds 9 --lambdas 0.0 3.0

# gpt-4o-mini (via OpenAI, for both reasoning and extraction)
python experiments/run_musr_lambda_sweep.py \
    --api_key <OPENAI_KEY> --extractor_api_key <OPENAI_KEY> \
    --model gpt-4o-mini \
    --n_examples 300 --n_rounds 9 --lambdas 0.0 3.0
```

`--model` accepts any OpenAI-model-name prefix (`gpt-4*`, `gpt-3.5*`, `o1*`, `o3*`,
`o4*`) and routes it to `https://api.openai.com/v1` automatically; any other model name
is routed to the NRP-style endpoint passed via `--api_key`. See `--help` for
`--nli_model` / `--nli_entail_threshold` to reproduce the paper's basin-definition
sensitivity analysis.

Outputs (per-lambda aggregate accuracy/Pass@k/#Basins/N_eff, plus per-example results and
a `summary.md` report) are written to `outputs/musr_lambda_sweep/` (override with the
`MUSR_SWEEP_OUTPUT_DIR` environment variable).

## Reproducing the Generalization table (HumanEval, GSM-Hard)

Both tasks use a free, deterministic verifier (test execution for HumanEval; exact
arithmetic recomputation for GSM-Hard) instead of an LLM judge, and a structural basin
key (AST-normalized parse tree for HumanEval; the tuple of exactly-verified intermediate
values for GSM-Hard) instead of NLI clustering — so neither needs the extractor/NLI
machinery MuSR does.

```bash
# HumanEval — full 164-problem test set
python experiments/run_humaneval_tot_vs_basin.py \
    --api_key <KEY> --backend openai --model gpt-4o-mini \
    --n_examples 164 --lambdas 0.0 3.0 --include_qabasin --qa_lambda 0.5

# GSM-Hard
python experiments/run_gsm_hard_tot_vs_hybrid.py \
    --api_key <KEY> --model gpt-4o-mini \
    --n_examples 100 --lambda_basin 3.0 --include_qabasin --qa_lambda 0.5
```

`--backend {nrp,openai,together}` on the HumanEval script picks the base URL for you
(pass `--base_url` directly to override); the GSM-Hard script takes `--base_url`
directly (defaults to `https://api.openai.com/v1`). Both accept any of the paper's four
model names via `--model`.

**Safety note:** HumanEval evaluation executes model-generated code. This repo's
`basin/verifier/code_executor.py` follows the standard HumanEval sandboxing pattern
(subprocess isolation, wall-clock timeout, disabled filesystem/process syscalls) so that
untrusted completions can't affect your machine — the same protections the original
HumanEval release itself recommends running code generation benchmarks with.

## Reproducing the MCTS table (Game of 24 with UCT-based MCTS)

Same 100 puzzles (idx 900–999) as Table 1, but with UCT-based Monte Carlo Tree Search
instead of ToT — the basin penalty is added directly to UCT's child-selection score,
holding the exploration constant `c` and simulation budget fixed and identical across
`mcts_standard`, `mcts_basin`, and `mcts_qabasin`:

```bash
python experiments/run_game24_mcts_qabasin.py \
    --api_key <KEY> --base_url <PROVIDER_BASE_URL> --model gpt-4o-mini \
    --c 1.0 --lambda_basin 3.0 --lambda_qa 0.5 \
    --n_simulations 50 --n_proposals 5 --max_depth 3 \
    --n_puzzles 100 --start_idx 900
```

This script always runs all three conditions (`mcts_standard`, `mcts_basin`,
`mcts_qabasin`) in one invocation.

## Repository structure

```
basin/
├── controller/metadynamics.py   # Core BASIN/QA-BASIN controller (used by the MuSR script)
├── memory/                      # Basin memory + clustering (embedding- and NLI-based)
├── state/                       # Reasoning-state representation and extraction
├── embedding/                   # TF-IDF / sentence-transformer embedders
├── verifier/
│   ├── verifier.py              # Heuristic verifier (MuSR/GoT-style tasks)
│   ├── ast_basin.py             # AST-normalized structural basin key (HumanEval)
│   └── code_executor.py         # Sandboxed test execution (HumanEval)
├── confidence/                  # Basin-weighted final-answer selection (internal to search; not a reported metric)
├── datasets/
│   ├── loader.py                # Shared Problem dataclass
│   ├── musr_loader.py           # MuSR (HuggingFace `datasets`)
│   ├── humaneval_loader.py      # HumanEval (HuggingFace `datasets`)
│   └── gsm_hard_loader.py       # GSM-Hard (HuggingFace `datasets`)
└── llm/                         # OpenAI-compatible LLM backend client
experiments/
├── run_game24_tot_vs_hybrid.py     # Table 1
├── run_musr_lambda_sweep.py        # Table 2/3
├── run_humaneval_tot_vs_basin.py   # Generalization table (HumanEval)
├── run_gsm_hard_tot_vs_hybrid.py   # Generalization table (GSM-Hard)
└── run_game24_mcts_qabasin.py      # MCTS table
```

Trimmed to only what these five scripts actually import (verified by tracing the import
graph and running every script end-to-end), matching exactly the tasks the paper's
main-body tables report. Not included: BBH Logical Deduction, Graph-of-Thoughts, and the
two post-hoc analysis figures — all appendix-table or figure-only content (see the table
above) — or unused code from earlier prototyping (GSM8K, MATH, creative writing, unused
verifier/embedding/LLM backends), none of which this repo's scripts need.

## Citation

This paper is currently a preprint (not yet accepted at a venue). If you use
this code, please cite:

```bibtex
@article{cheng2026basin,
  title  = {Escaping Redundant Reasoning: Structure-Aware Search for Inference-Time LLMs},
  author = {Cheng, Lu},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT — see `LICENSE`.
