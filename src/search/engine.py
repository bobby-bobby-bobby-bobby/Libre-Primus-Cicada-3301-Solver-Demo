"""
Search engine / mining loop.

Orchestrates the end-to-end search:
  1. Pull a batch of hypotheses from the active strategy.
  2. Apply each pipeline to the page tensors (GPU-accelerated).
  3. Score the outputs.
  4. Feed results back to the strategy (adaptive bias + evolution).
  5. Checkpoint and log top candidates.

The engine supports multi-page scoring (cross-page consistency reward)
and runs in a tight loop until stopped or a result limit is reached.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..data.page import Page
from ..pipeline.accelerator import Accelerator
from ..scoring.scorer import Scorer, ScoreBreakdown
from ..search.strategies import (
    BeamSearch,
    GeneticSearch,
    GridSearch,
    Hypothesis,
    RandomSearch,
    SearchStrategy,
)
from ..search.keyspace import Keyspace

log = logging.getLogger(__name__)


def _build_strategy(
    name: str,
    keyspace: Keyspace,
    accelerator: Accelerator,
    config: Dict[str, Any],
) -> SearchStrategy:
    """Instantiate a :class:`SearchStrategy` by name."""
    mapping = {
        "random": RandomSearch,
        "grid": GridSearch,
        "beam": BeamSearch,
        "genetic": GeneticSearch,
    }
    cls = mapping.get(name.lower())
    if cls is None:
        log.warning("Unknown strategy %r; defaulting to GeneticSearch.", name)
        cls = GeneticSearch
    return cls(keyspace=keyspace, accelerator=accelerator, config=config)


class SearchEngine:
    """
    Mining-loop engine: drives hypothesis generation → evaluation → feedback.

    Parameters
    ----------
    pages : List[Page]
        All loaded LP pages (scored in parallel for cross-page consistency).
    config : Dict[str, Any]
        Full configuration dict (from ``config.json``).
    accelerator : Accelerator, optional
        Shared GPU/CPU device manager.
    result_callback : callable, optional
        Called with (Hypothesis, ScoreBreakdown) whenever a new best is found.
    checkpoint_callback : callable, optional
        Called with List[Hypothesis] at each checkpoint interval.
    """

    def __init__(
        self,
        pages: List[Page],
        config: Dict[str, Any],
        accelerator: Optional[Accelerator] = None,
        result_callback: Optional[Callable] = None,
        checkpoint_callback: Optional[Callable] = None,
    ) -> None:
        self.pages = pages
        self.config = config
        self.acc = accelerator or Accelerator(
            device=config.get("gpu", {}).get("device", "auto"),
            precision=config.get("gpu", {}).get("precision", "float32"),
        )

        search_cfg = config.get("search", {})
        self.batch_size: int = self.acc.suggest_batch_size(
            search_cfg.get("batch_size", 1024)
        )
        self.top_k: int = search_cfg.get("top_k", 10)

        # Build keyspace with narrowing constraints.
        self.keyspace = Keyspace(config.get("constraints", {}))

        # Build strategy.
        strategy_name = search_cfg.get("strategy", "genetic")
        strategy_config = {**search_cfg, **config.get("pipeline", {})}
        self.strategy = _build_strategy(
            strategy_name, self.keyspace, self.acc, strategy_config
        )

        # Scorer.
        self.scorer = Scorer(config.get("scoring", {}).get("weights"))

        # State.
        self.best_score: float = -math.inf
        self.best_hypothesis: Optional[Hypothesis] = None
        self.total_evaluated: int = 0
        self.all_top: List[Hypothesis] = []

        self.result_callback = result_callback
        self.checkpoint_callback = checkpoint_callback

        # Precompute page tensors on device.
        pipeline_cfg = config.get("pipeline", {})
        self._encodings: List[str] = pipeline_cfg.get("encodings", ["index"])
        self._page_tensors: Dict[str, List] = {
            enc: [self.acc.from_numpy(p.get_tensor(enc)) for p in pages]
            for enc in self._encodings
        }
        self._page_line_indices: List[List[np.ndarray]] = [
            p.line_indices for p in pages
        ]

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(
        self,
        max_iterations: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> List[Hypothesis]:
        """
        Run the mining loop.

        Parameters
        ----------
        max_iterations : int, optional
            Stop after this many batches.
        max_seconds : float, optional
            Stop after this many wall-clock seconds.

        Returns
        -------
        List[Hypothesis]
            Top-K hypotheses found, sorted descending by score.
        """
        log.info(
            "SearchEngine starting: strategy=%s  batch=%d  pages=%d",
            type(self.strategy).__name__,
            self.batch_size,
            len(self.pages),
        )

        start_time = time.monotonic()
        iteration = 0

        try:
            while True:
                if max_iterations is not None and iteration >= max_iterations:
                    break
                if max_seconds is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= max_seconds:
                        break

                # --- Generate hypotheses ---
                batch = self.strategy.next_batch(self.batch_size)

                # --- Evaluate ---
                scored = self._evaluate_batch(batch)

                # --- Update strategy ---
                self.strategy.update(scored)

                # --- Track best ---
                for h in scored:
                    if h.score > self.best_score:
                        self.best_score = h.score
                        self.best_hypothesis = h
                        log.info(
                            "[iter=%d] New best: score=%.4f  pipeline=%s",
                            iteration,
                            h.score,
                            h.pipeline,
                        )
                        if self.result_callback:
                            self.result_callback(h, None)

                # --- Maintain top-K list ---
                self.all_top.extend(scored)
                self.all_top.sort(key=lambda x: x.score, reverse=True)
                self.all_top = self.all_top[: self.top_k]

                self.total_evaluated += len(batch)

                if self.checkpoint_callback and iteration % 10 == 0:
                    self.checkpoint_callback(list(self.all_top))

                if iteration % 50 == 0:
                    elapsed = time.monotonic() - start_time
                    rate = self.total_evaluated / max(elapsed, 0.001)
                    log.info(
                        "[iter=%d] evaluated=%d  best=%.4f  rate=%.0f/s",
                        iteration,
                        self.total_evaluated,
                        self.best_score,
                        rate,
                    )

                iteration += 1

        except KeyboardInterrupt:
            log.info("Search interrupted by user.")

        log.info(
            "Search complete: %d hypotheses evaluated, best=%.4f",
            self.total_evaluated,
            self.best_score,
        )
        return list(self.all_top)

    # -----------------------------------------------------------------------
    # Batch evaluation
    # -----------------------------------------------------------------------

    def _evaluate_batch(self, batch: List[Hypothesis]) -> List[Hypothesis]:
        """Score every hypothesis in *batch* using the active pages."""
        for h in batch:
            h.score = self._score_hypothesis(h)
        return batch

    def _score_hypothesis(self, h: Hypothesis) -> float:
        """
        Apply pipeline to all pages and compute composite cross-page score.
        """
        enc = h.pipeline.encoding
        if enc not in self._page_tensors:
            enc = self._encodings[0]

        page_outputs: List[np.ndarray] = []
        for pt in self._page_tensors[enc]:
            out_t = h.pipeline.apply(pt)
            page_outputs.append(self.acc.to_numpy(out_t))

        if not page_outputs:
            return 0.0

        # Score each page individually.
        scores: List[float] = []
        for i, out in enumerate(page_outputs):
            lines = self._page_line_indices[i]
            # Use first other page output as cross-page reference.
            ref = page_outputs[1 - i] if len(page_outputs) > 1 else None
            bd = self.scorer.score(out, lines=lines, cross_page_ref=ref)
            scores.append(bd.composite)

        return float(np.mean(scores))
