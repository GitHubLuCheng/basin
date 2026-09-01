# BASIN: Basin-Aware Search for Inference-Time LLM Reasoning

Code to reproduce the two main-results tables from
**"Escaping Reasoning Basins: Basin-Aware Search for Inference-Time LLM Reasoning."**

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

This repository includes the code for the two headline tables:

| Table | Task | Models | Script |
|---|---|---|---|
| Table 1 | Game of 24 | gpt-4o-mini, Qwen3-397b | `experiments/run_game24_tot_vs_hybrid.py` |
| Table 2/3 | MuSR | gpt-oss-120b, gpt-4o-mini | `experiments/run_musr_lambda_sweep.py` |

Both scripts implement standard ToT search alongside the identical search with the
BASIN penalty applied, under matched inference budgets, so the two conditions can be
compared directly.

## Setup

```bash
pip install -r requirements.txt
```

MuSR problems are loaded automatically via the HuggingFace `datasets` library on first
run. Game of 24 puzzles are downloaded automatically from the standard
`tree-of-thought-llm` benchmark CSV on first run.

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

# Qwen3-397b (or any other OpenAI-compatible endpoint) — point --base_url at your provider
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

## Repository structure

```
metadyn/
├── controller/metadynamics.py   # Core BASIN/QA-BASIN controller
├── memory/                      # Basin memory + clustering (embedding- and NLI-based)
├── state/                       # Reasoning-state representation and extraction
├── embedding/                   # TF-IDF / sentence-transformer / n-gram embedders
├── verifier/                    # Heuristic and task-specific verifiers
├── confidence/                  # Basin-weighted final-answer aggregation
├── datasets/                    # Dataset loaders (MuSR, Game of 24, GSM8K, MATH)
└── llm/                         # LLM backend clients (OpenAI-compatible, Anthropic)
experiments/
├── run_game24_tot_vs_hybrid.py  # Table 1
└── run_musr_lambda_sweep.py     # Table 2/3
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{basin2026,
  title     = {Escaping Reasoning Basins: Basin-Aware Search for Inference-Time LLM Reasoning},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT — see `LICENSE`.
