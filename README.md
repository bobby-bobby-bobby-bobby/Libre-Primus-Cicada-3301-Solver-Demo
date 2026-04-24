# Liber Primus Cicada 3301 Solver Demo

Production-oriented, auto-bootstrapping cryptanalysis framework for large-scale search over unsolved Liber Primus pages.

> This repository builds the search engine and distributed infrastructure. It does **not** claim to solve the cipher.

## Features

- Auto-bootstrap data source clone (`run_all.py`)
- Rune + page models and Gematria Primus conversion
- Tensorized numeric representation for accelerated kernels
- Composable transform pipelines:
  - `ModularShift`
  - `VigenereShift`
  - `AtbashVariant`
  - `BlockTransposition`
  - `IndexPermutation`
  - `UnknownTransformSlot` (learnable substitution/evolution slot)
- Hardware auto-detection:
  - CUDA via PyTorch
  - Apple MPS via PyTorch
  - CPU fallback (NumPy)
- Distributed multiprocessing coordinator/worker architecture
- Search strategies:
  - random
  - grid
  - beam (with adaptive evolution step)
  - genetic mutation/crossover
- Search-space narrowing constraints from `config.json`
- Multi-layer scoring:
  - statistical (entropy, IOC, n-gram)
  - structural (symmetry, repetition)
  - cross-page consistency
  - lexical coherence proxy
- Checkpoints + experiment logs

## Quick Start

```bash
python run_all.py
```

This command will:
1. Create `config.json` if missing
2. Clone the configured LP data repository
3. Parse rune pages
4. Build tensors
5. Start distributed workers
6. Run search loop and save candidates to `experiments/best_candidates.json`

## Config

Edit `config.json` to control:
- worker count
- strategy
- batch sizes
- max iterations
- search constraints

## Optional Dependencies

- `torch` (recommended for CUDA/MPS acceleration)
- `numpy`

Install:

```bash
pip install torch numpy
```

## Run with Custom Config

```bash
python run_all.py --config /absolute/path/to/config.json
```

## Output

- `experiments/logs/run.log`
- `experiments/checkpoints/iter_*.json`
- `experiments/best_candidates.json`
