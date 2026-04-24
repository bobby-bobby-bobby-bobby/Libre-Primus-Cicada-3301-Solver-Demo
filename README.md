# Liber Primus – Cicada 3301 Cryptanalysis Engine

A **complete, production-grade, auto-bootstrapping** distributed
GPU/NPU-accelerated cryptanalysis engine for the unsolved pages of
*Liber Primus* by Cicada 3301.

> **What this is:** A massively parallel hypothesis generator and filtering
> system.  It does **not** claim to solve the cipher – it builds the search
> infrastructure needed to systematically explore the keyspace at scale.

---

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/bobby-bobby-bobby-bobby/Libre-Primus-Cicada-3301-Solver-Demo
cd Libre-Primus-Cicada-3301-Solver-Demo

# 2. (Recommended) Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Run – dependencies install automatically
python run_all.py
```

That single command will:
1. Install `numpy`, `torch`, `scipy`, and `tqdm`.
2. Clone the LP transcription from GitHub into `data/liber_primus/`.
3. Parse all rune pages into structured `Page` objects.
4. Spawn worker processes (default: 4).
5. Begin the genetic-algorithm search loop immediately.
6. Write top candidates to `output/best_result.json` and `output/top_results.json`.

---

## CLI Options

```
python run_all.py [OPTIONS]

Options:
  --config PATH          Path to config.json         (default: config.json)
  --workers N            Number of worker processes   (default: 4)
  --seconds S            Stop after S wall-clock seconds
  --iterations N         Stop after N iterations      (single-process only)
  --strategy STRATEGY    Search strategy: random | grid | beam | genetic
  --batch-size N         Hypotheses per batch
  --single-process       Run without multiprocessing  (good for debugging)
  --no-install           Skip automatic pip install
```

### Examples

```bash
# Debug run – single process, 60-second budget
python run_all.py --single-process --seconds 60

# 8 workers, beam search, 10-minute budget
python run_all.py --workers 8 --strategy beam --seconds 600

# Grid search, 2 workers, no time limit
python run_all.py --workers 2 --strategy grid

# Genetic search with custom config
python run_all.py --config my_config.json --strategy genetic --seconds 3600
```

---

## Architecture

```
run_all.py                   ← single entry point
├── src/data/
│   ├── rune.py              ← Rune class, Gematria Primus table
│   ├── page.py              ← Page class, 3 parallel tensor encodings
│   └── loader.py            ← auto-clone LP repo, parse rune files
├── src/pipeline/
│   ├── accelerator.py       ← GPU/MPS/CPU device detection
│   ├── kernels.py           ← PyTorch vectorised ops (modular, Vigenère, etc.)
│   ├── transforms.py        ← ModularShift, VigenereShift, AtbashVariant,
│   │                          BlockTransposition, IndexPermutation,
│   │                          LearnedSubstitution (unknown transform slot)
│   └── pipeline.py          ← composable, serialisable pipeline
├── src/scoring/
│   ├── statistical.py       ← IoC, entropy, bigram/monogram scoring
│   ├── structural.py        ← line symmetry, repeated n-grams, LP similarity,
│   │                          cross-line coherence
│   ├── dictionary.py        ← word-boundary, common short-word, CV-pattern
│   └── scorer.py            ← multi-layer composite scorer + cross-page coherence
├── src/search/
│   ├── keyspace.py          ← domain definitions + narrowing constraints
│   ├── strategies.py        ← RandomSearch, GridSearch, BeamSearch,
│   │                          GeneticSearch (with adaptive bias + diversity reset)
│   └── engine.py            ← mining loop (single-process)
├── src/distributed/
│   ├── protocol.py          ← WorkItem, WorkResult, ControlMsg
│   ├── coordinator.py       ← work dispatch, result aggregation, checkpointing
│   └── worker.py            ← per-process GPU pipeline + scoring
└── src/utils/
    ├── checkpoint.py        ← JSON-file checkpoints with latest symlink
    ├── profiling.py         ← wall-clock timers, GPU memory stats
    └── logging_utils.py     ← console + JSON-lines structured logging
```

---

## Configuration (`config.json`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `search` | `strategy` | `"genetic"` | Active search strategy |
| `search` | `batch_size` | `1024` | Hypotheses per batch |
| `search` | `top_k` | `10` | Candidates to track |
| `search` | `beam_width` | `100` | Beam search width |
| `search.genetic` | `population_size` | `200` | GA population |
| `search.genetic` | `mutation_rate` | `0.15` | Per-transform mutation probability |
| `search` | `diversity_reset_interval` | `500` | Iterations between diversity resets |
| `constraints` | `modular_multipliers` | `"prime"` | Restrict multipliers to primes |
| `constraints` | `vigenere_key_lengths` | `[3,5,7,11,13]` | Allowed key lengths |
| `pipeline` | `unknown_transform_slots` | `2` | Learned substitution slots |
| `pipeline` | `encodings` | `["gp","index","alt_prime"]` | Parallel encodings |
| `distributed` | `num_workers` | `4` | Worker process count |
| `gpu` | `device` | `"auto"` | CUDA / MPS / CPU |

---

## Tensor Encodings

Three representations are computed and cached for every page:

| Name | Description | Range |
|------|-------------|-------|
| `index` | Futhorc position | 0–28 |
| `gp` | Gematria Primus prime | 2–109 |
| `alt_prime` | GP modulo 29 | 0–28 |

Pipelines run on all three in parallel so no structural information is
lost through premature flattening.

---

## Scoring System

The composite score is a weighted sum of six independent signals:

| Signal | Weight | Measures |
|--------|--------|---------|
| `ioc` | 25% | Index of Coincidence vs English target (0.065) |
| `ngram` | 25% | English bigram + monogram log-probability |
| `entropy` | 15% | 1 − normalised Shannon entropy |
| `word_boundary` | 20% | Vowel/consonant transitions + common short words |
| `structural` | 10% | Line symmetry, repeated n-grams, LP profile similarity |
| `cross_page` | 5% | Consistency of key across multiple pages |

Weights are configurable in `config.json → scoring.weights`.

---

## Search Strategies

| Strategy | Description |
|----------|-------------|
| `random` | Uniform random sampling with adaptive bias tracking |
| `grid` | Exhaustive sweep over the constrained modular-shift keyspace |
| `beam` | Top-K beam with neighbourhood perturbation |
| `genetic` | Evolution: tournament selection, crossover, mutation, elitism, diversity reset |

All strategies track which transform types consistently improve scores
and bias future sampling accordingly (adaptive search).

---

## Output Files

| Path | Contents |
|------|----------|
| `output/best_result.json` | Best candidate found so far (updated in real time) |
| `output/top_results.json` | Top-K list on exit |
| `checkpoints/top_k_latest.json` | Periodic checkpoint |
| `logs/coordinator.jsonl` | Structured coordinator log |
| `logs/worker_N.jsonl` | Per-worker structured logs |
| `logs/experiments.jsonl` | Per-batch experiment journal |

---

## GPU / NPU Support

| Platform | Status |
|----------|--------|
| NVIDIA CUDA | ✅ Auto-detected via PyTorch |
| Apple Metal (M-series) | ✅ Auto-detected via PyTorch MPS |
| CPU (no GPU) | ✅ Full functionality, lower throughput |

The `Accelerator` class handles device detection and tensor movement
automatically.  Batch sizes are auto-tuned based on available VRAM.

---

## Design Notes

### Why multiple encodings?
Gematria Primus is a *designed encoding*, not just a number.  Flattening
everything to integers discards potential positional / relational structure.
Running GP, raw index, and `GP mod 29` in parallel preserves all three
abstractions without tripling memory (tensors share the page data).

### Unknown transform slots
`LearnedSubstitution` is an evolvable substitution table that the genetic
algorithm can mutate freely.  This allows the system to explore
transformations it doesn't know how to name – a direct response to the
observation that LP may not be purely classical.

### Cross-page consistency
The scorer penalises hypotheses where the same key applied to two
different pages produces statistically inconsistent outputs.  This is a
necessary (not sufficient) condition for a valid key.

### Diversity preservation
The genetic search performs a periodic **diversity reset**: the bottom
half of the population is replaced with fresh random hypotheses every
`diversity_reset_interval` iterations.  This prevents the population from
converging on a local optimum ("falling in love with garbage").

---

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24
- torch ≥ 2.0 (CPU build is sufficient; CUDA/MPS builds unlock GPU acceleration)
- scipy ≥ 1.10
- tqdm ≥ 4.65
- git (for auto-cloning the LP repository)

All packages are installed automatically by `run_all.py`.

---

## Disclaimer

This project is a **search tool**, not a solution.  The most likely
outcome of running it is generating millions of near-English artifacts
and developing a philosophical distrust of pattern recognition.  That is
expected and acceptable.  The goal is to scale the search far enough to
find patterns nobody else has found – or to conclusively rule out large
regions of the keyspace.
