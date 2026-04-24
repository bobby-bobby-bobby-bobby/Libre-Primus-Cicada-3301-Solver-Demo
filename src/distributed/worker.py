"""
Worker process for distributed cryptanalysis.

Each worker:
  1. Receives a WorkItem (via multiprocessing queue or socket).
  2. Builds / receives pipeline configurations.
  3. Applies each pipeline to the shared page tensors on its local GPU/CPU.
  4. Scores outputs using the composite scorer.
  5. Returns a WorkResult with the top-K candidates.

Workers are designed to run as independent processes (multiprocessing.Process)
and communicate exclusively through queues, so they scale to multiple machines
by substituting the queue transport for a network socket.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from ..data.page import Page
from ..pipeline.accelerator import Accelerator
from ..pipeline.pipeline import Pipeline
from ..pipeline.transforms import transform_from_dict
from ..scoring.scorer import Scorer
from ..search.keyspace import Keyspace
from ..search.strategies import (
    BeamSearch,
    GeneticSearch,
    GridSearch,
    GeneticSearch,
    Hypothesis,
    RandomSearch,
)
from .protocol import (
    CandidateResult,
    ControlCmd,
    ControlMsg,
    MsgType,
    WorkItem,
    WorkResult,
)

log = logging.getLogger(__name__)


class Worker:
    """
    Cryptanalysis worker.

    Parameters
    ----------
    worker_id : str
        Unique identifier (e.g. ``"worker_0"``).
    pages : List[Page]
        Pre-loaded LP pages to evaluate against.
    config : Dict[str, Any]
        Full configuration dict.
    work_queue : multiprocessing.Queue
        Incoming work items.
    result_queue : multiprocessing.Queue
        Where results are placed.
    stop_event : multiprocessing.Event
        Set by the coordinator to trigger clean shutdown.
    """

    def __init__(
        self,
        worker_id: str,
        pages: List[Page],
        config: Dict[str, Any],
        work_queue: "mp.Queue[Any]",
        result_queue: "mp.Queue[Any]",
        stop_event: "mp.Event",
    ) -> None:
        self.worker_id = worker_id
        self.pages = pages
        self.config = config
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

        gpu_cfg = config.get("gpu", {})
        self.acc = Accelerator(
            device=gpu_cfg.get("device", "auto"),
            precision=gpu_cfg.get("precision", "float32"),
        )

        self.scorer = Scorer(config.get("scoring", {}).get("weights"))
        self.keyspace = Keyspace(config.get("constraints", {}))

        pipeline_cfg = config.get("pipeline", {})
        self._encodings: List[str] = pipeline_cfg.get("encodings", ["index"])
        self._top_k: int = config.get("search", {}).get("top_k", 10)

        # Precompute page tensors (device-local).
        self._page_tensors: Dict[str, List] = {
            enc: [self.acc.from_numpy(p.get_tensor(enc)) for p in pages]
            for enc in self._encodings
        }
        self._page_line_indices = [p.line_indices for p in pages]

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Main worker event loop – blocks until stop_event is set."""
        log.info("Worker %s started on device %s.", self.worker_id, self.acc.device)

        while not self.stop_event.is_set():
            try:
                item = self.work_queue.get(timeout=1.0)
            except Exception:
                continue

            if isinstance(item, ControlMsg):
                if item.command == ControlCmd.SHUTDOWN:
                    log.info("Worker %s received SHUTDOWN.", self.worker_id)
                    break
                continue

            if isinstance(item, WorkItem):
                try:
                    result = self._process(item)
                    self.result_queue.put(result)
                except Exception as exc:
                    log.error(
                        "Worker %s error on item %s: %s",
                        self.worker_id,
                        item.item_id,
                        traceback.format_exc(),
                    )

        log.info("Worker %s exiting.", self.worker_id)

    # -----------------------------------------------------------------------
    # Processing
    # -----------------------------------------------------------------------

    def _process(self, item: WorkItem) -> WorkResult:
        """Evaluate a single WorkItem and return a WorkResult."""
        start = time.monotonic()

        # Build or deserialise pipelines.
        if item.pipeline_dicts:
            pipelines = [
                Pipeline.from_dict(d, accelerator=self.acc)
                for d in item.pipeline_dicts
            ]
        else:
            pipelines = self._generate_pipelines(item)

        scored: List[tuple] = []  # (score, pipeline, output_sample)

        enc = item.page_encoding
        if enc not in self._encodings:
            enc = self._encodings[0]

        for pipeline in pipelines:
            score, breakdown, latin = self._evaluate_pipeline(pipeline, enc)
            scored.append((score, pipeline, breakdown, latin))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self._top_k]

        candidates = [
            CandidateResult(
                pipeline_dict=p.to_dict(),
                score=s,
                score_breakdown=bd.to_dict(),
                latin_output=lat,
            )
            for s, p, bd, lat in top
        ]

        elapsed = time.monotonic() - start
        return WorkResult(
            item_id=item.item_id,
            worker_id=self.worker_id,
            top_candidates=candidates,
            evaluated=len(pipelines),
            elapsed_seconds=elapsed,
        )

    def _generate_pipelines(self, item: WorkItem) -> List[Pipeline]:
        """Generate random pipelines for this work item."""
        import random

        strategy_map = {
            "random": RandomSearch,
            "beam": BeamSearch,
            "genetic": GeneticSearch,
            "grid": GridSearch,
        }
        strategy_cls = strategy_map.get(item.strategy, RandomSearch)
        strategy_cfg = {
            **self.config.get("search", {}),
            **self.config.get("pipeline", {}),
        }
        strategy = strategy_cls(
            keyspace=self.keyspace,
            accelerator=self.acc,
            config=strategy_cfg,
        )
        hypotheses = strategy.next_batch(item.batch_size)
        return [h.pipeline for h in hypotheses]

    def _evaluate_pipeline(
        self, pipeline: Pipeline, enc: str
    ) -> tuple:
        """Apply pipeline to all pages and return (mean_score, breakdown, latin_sample)."""
        from ..data.rune import GematriaPrimus

        page_scores = []
        last_output = None

        page_tensors = self._page_tensors.get(enc, self._page_tensors[self._encodings[0]])

        for i, pt in enumerate(page_tensors):
            out_t = pipeline.apply(pt)
            out_np = self.acc.to_numpy(out_t)
            lines = self._page_line_indices[i]
            ref = self.acc.to_numpy(page_tensors[1 - i]) if len(page_tensors) > 1 else None
            bd = self.scorer.score(out_np, lines=lines, cross_page_ref=ref)
            page_scores.append(bd)
            last_output = out_np

        mean_score = float(np.mean([bd.composite for bd in page_scores]))
        avg_bd = page_scores[0] if page_scores else None

        latin = ""
        if last_output is not None:
            latin = GematriaPrimus.indices_to_latin(last_output[:100])

        return mean_score, avg_bd, latin


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------


def worker_process(
    worker_id: str,
    pages: List[Page],
    config: Dict[str, Any],
    work_queue: "mp.Queue",
    result_queue: "mp.Queue",
    stop_event: "mp.Event",
) -> None:
    """Top-level function passed to ``multiprocessing.Process``."""
    # Set up per-process logging.
    from ..utils.logging_utils import setup_logging

    setup_logging(config.get("data", {}).get("log_dir", "logs"), worker_id)
    worker = Worker(
        worker_id=worker_id,
        pages=pages,
        config=config,
        work_queue=work_queue,
        result_queue=result_queue,
        stop_event=stop_event,
    )
    worker.run()
