from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from .config import RuntimeConfig
from .data_loader import ensure_repo_cloned, parse_pages, persist_pages
from .distributed import Coordinator

LOG = logging.getLogger(__name__)


def bootstrap_data(config: RuntimeConfig):
    repo_dir = Path(config.data_repo_dir)
    ensure_repo_cloned(config.data_repo_url, repo_dir)
    pages = parse_pages(repo_dir)
    persist_pages(pages, Path(config.parsed_pages_dir))
    if not pages:
        raise RuntimeError("No rune pages parsed from source repository.")
    return pages


def run_coordinator(config: RuntimeConfig) -> List[dict]:
    pages = bootstrap_data(config)
    page_vals = [r.gp_value for r in pages[0].runes]
    ref_vals = [[r.gp_value for r in p.runes] for p in pages[1:10]]

    coord = Coordinator(config)
    best = coord.run(page_vals=page_vals, ref_vals=ref_vals)

    out = Path(config.experiments_dir) / "best_candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best[:512], indent=2), encoding="utf-8")
    LOG.info("wrote candidates to %s", out)
    return best
