from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lp_solver.config import RuntimeConfig
from lp_solver.coordinator import run_coordinator


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-bootstrapping distributed LP cryptanalysis engine")
    parser.add_argument("--config", default="config.json", help="Path to runtime config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config = RuntimeConfig.from_path(cfg_path)
    setup_logging(Path(config.logs_dir))

    best = run_coordinator(config)
    print(f"Completed search. Top score: {best[0]['score'] if best else 'n/a'}")


if __name__ == "__main__":
    main()
