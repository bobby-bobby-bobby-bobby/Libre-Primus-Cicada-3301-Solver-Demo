#!/usr/bin/env python3
"""
run_all.py – Single-command entry point for the Liber Primus cryptanalysis engine.

Usage
-----
    python run_all.py [--config config.json] [--workers N] [--seconds S]
                      [--strategy STRATEGY] [--single-process]

The script:
  1. Installs missing Python dependencies (``requirements.txt``).
  2. Creates all runtime directories (data/, logs/, checkpoints/, output/).
  3. Clones / updates the Liber Primus GitHub repository.
  4. Parses rune pages into structured Page objects.
  5. Launches the coordinator + worker processes.
  6. Runs the search loop until interrupted or the time limit is reached.
  7. Prints the top-K candidates on exit.

No arguments are required for a default run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so src/ imports work.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Bootstrap: install dependencies if not present
# ---------------------------------------------------------------------------


def _install_dependencies(requirements_file: str = "requirements.txt") -> None:
    """Install Python packages listed in *requirements_file* if needed."""
    req_path = Path(requirements_file)
    if not req_path.exists():
        print(f"[bootstrap] requirements file not found: {requirements_file}")
        return

    print(f"[bootstrap] Checking / installing dependencies from {requirements_file} …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)],
        capture_output=False,
    )
    if result.returncode != 0:
        print(
            "[bootstrap] WARNING: pip install returned non-zero exit code. "
            "Some packages may be missing."
        )
    else:
        print("[bootstrap] Dependencies OK.")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    """Load and return the JSON configuration file."""
    p = Path(path)
    if not p.exists():
        print(f"[config] Config file not found: {path}. Using defaults.")
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def merge_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI-argument overrides to the config dict in-place."""
    if args.workers is not None:
        config.setdefault("distributed", {})["num_workers"] = args.workers
    if args.strategy is not None:
        config.setdefault("search", {})["strategy"] = args.strategy
    if args.batch_size is not None:
        config.setdefault("search", {})["batch_size"] = args.batch_size


# ---------------------------------------------------------------------------
# Single-process mode (no multiprocessing, for debugging / low-resource envs)
# ---------------------------------------------------------------------------


def run_single_process(
    config: Dict[str, Any],
    max_seconds: Optional[float],
    max_iterations: Optional[int],
) -> None:
    """Run the search engine in a single process (no worker spawning)."""
    from src.utils.logging_utils import setup_logging
    from src.data.loader import load_pages
    from src.search.engine import SearchEngine
    from src.pipeline.accelerator import Accelerator
    from src.data.rune import GematriaPrimus

    data_cfg = config.get("data", {})
    setup_logging(data_cfg.get("log_dir", "logs"), "main")
    log = logging.getLogger("run_all")

    # Load pages.
    pages = load_pages(
        lp_data_dir=data_cfg.get("lp_data_dir", "data/liber_primus"),
        repo_url=data_cfg.get("lp_repo_url", "https://github.com/rtkd/iddqd"),
    )

    if not pages:
        log.error("No pages loaded. Exiting.")
        sys.exit(1)

    log.info("Loaded %d page(s).", len(pages))
    for p in pages:
        log.info("  %r", p)

    # Build and run the engine.
    engine = SearchEngine(pages=pages, config=config)

    log.info("Starting single-process search …")
    top = engine.run(
        max_iterations=max_iterations,
        max_seconds=max_seconds,
    )

    _print_results(top, config, GematriaPrimus)


def _print_results(top, config, GematriaPrimus) -> None:
    """Pretty-print the top results to stdout."""
    print("\n" + "=" * 60)
    print(" TOP RESULTS")
    print("=" * 60)
    if not top:
        print("  No results found.")
        return
    for rank, h in enumerate(top, 1):
        print(f"\n  Rank {rank}  score={h.score:.4f}")
        print(f"  Pipeline : {h.pipeline}")
        # Show a sample Latin transliteration from the first page.
        try:
            import numpy as np
            from src.data.rune import GematriaPrimus as GP

            # We don't have the page outputs here; just show pipeline config.
            print(f"  Config   : {h.pipeline.to_json()[:120]} …")
        except Exception:
            pass
    print("=" * 60)

    # Save to output/top_results.json
    output_dir = Path(config.get("data", {}).get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "top_results.json"
    import json

    data = [
        {"rank": i + 1, "score": h.score, "pipeline": h.pipeline.to_dict()}
        for i, h in enumerate(top)
    ]
    results_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nResults written to {results_path}")


# ---------------------------------------------------------------------------
# Distributed mode
# ---------------------------------------------------------------------------


def run_distributed(
    config: Dict[str, Any],
    max_seconds: Optional[float],
) -> None:
    """Run the full coordinator + worker distributed search."""
    from src.utils.logging_utils import setup_logging
    from src.data.loader import load_pages
    from src.distributed.coordinator import Coordinator

    data_cfg = config.get("data", {})
    setup_logging(data_cfg.get("log_dir", "logs"), "coordinator")
    log = logging.getLogger("run_all")

    # Load pages.
    pages = load_pages(
        lp_data_dir=data_cfg.get("lp_data_dir", "data/liber_primus"),
        repo_url=data_cfg.get("lp_repo_url", "https://github.com/rtkd/iddqd"),
    )

    if not pages:
        log.error("No pages loaded. Exiting.")
        sys.exit(1)

    log.info("Loaded %d page(s):", len(pages))
    for p in pages:
        log.info("  %r", p)

    coordinator = Coordinator(pages=pages, config=config)
    top = coordinator.run(max_seconds=max_seconds)

    print("\n" + "=" * 60)
    print(" TOP RESULTS (distributed)")
    print("=" * 60)
    if not top:
        print("  No results found.")
    for rank, candidate in enumerate(top, 1):
        print(
            f"\n  Rank {rank}  score={candidate.score:.4f}\n"
            f"  Output   : {candidate.latin_output[:80]}\n"
            f"  Pipeline : {json.dumps(candidate.pipeline_dict)[:120]} …"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Liber Primus Cicada 3301 cryptanalysis engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Default run (distributed, 4 workers, genetic search, no time limit):
  python run_all.py

  # Single-process debug run for 60 seconds:
  python run_all.py --single-process --seconds 60

  # 8 workers, beam search, 10-minute budget:
  python run_all.py --workers 8 --strategy beam --seconds 600

  # Grid search, 2 workers, 100 000 evaluations:
  python run_all.py --workers 2 --strategy grid --iterations 100000
""",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (overrides config.json).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Maximum wall-clock search time in seconds.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Maximum search iterations (single-process mode only).",
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "grid", "beam", "genetic"],
        default=None,
        help="Search strategy (overrides config.json).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="Batch size per iteration (overrides config.json).",
    )
    parser.add_argument(
        "--single-process",
        dest="single_process",
        action="store_true",
        help="Run in a single process (no multiprocessing; useful for debugging).",
    )
    parser.add_argument(
        "--no-install",
        dest="no_install",
        action="store_true",
        help="Skip automatic pip install of requirements.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Step 1: install dependencies.
    if not args.no_install:
        _install_dependencies("requirements.txt")

    # Step 2: load config.
    config = load_config(args.config)
    merge_cli_overrides(config, args)

    # Step 3: create runtime directories.
    data_cfg = config.get("data", {})
    for key in ("lp_data_dir", "output_dir", "log_dir", "checkpoint_dir"):
        d = data_cfg.get(key)
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

    # Step 4: run.
    if args.single_process:
        run_single_process(
            config=config,
            max_seconds=args.seconds,
            max_iterations=args.iterations,
        )
    else:
        run_distributed(
            config=config,
            max_seconds=args.seconds,
        )


if __name__ == "__main__":
    # On Windows / macOS with Python >= 3.8 the default multiprocessing start
    # method must be 'spawn' – the guard below is required.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
