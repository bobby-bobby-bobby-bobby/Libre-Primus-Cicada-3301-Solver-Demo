"""
Coordinator process.

Responsibilities
----------------
- Spawn and monitor worker processes.
- Assign work batches to workers via a work queue.
- Collect and rank results from the result queue.
- Redistribute promising pipeline configurations to workers (adaptive search).
- Checkpoint the top-K results periodically.
- Log all experiments in structured JSON lines.
- Provide a clean shutdown on KeyboardInterrupt or stop signal.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data.page import Page
from ..utils.checkpoint import Checkpoint
from ..utils.logging_utils import setup_logging
from .protocol import (
    CandidateResult,
    ControlCmd,
    ControlMsg,
    WorkItem,
    WorkResult,
)
from .worker import worker_process

log = logging.getLogger(__name__)


class Coordinator:
    """
    Orchestrates the distributed cryptanalysis search.

    Parameters
    ----------
    pages : List[Page]
        Pre-loaded LP pages.
    config : Dict[str, Any]
        Full configuration dict.
    """

    def __init__(self, pages: List[Page], config: Dict[str, Any]) -> None:
        self.pages = pages
        self.config = config

        dist_cfg = config.get("distributed", {})
        data_cfg = config.get("data", {})
        search_cfg = config.get("search", {})

        self.num_workers: int = int(dist_cfg.get("num_workers", 4))
        self.checkpoint_interval: int = int(dist_cfg.get("checkpoint_interval", 300))
        self.result_queue_size: int = int(dist_cfg.get("result_queue_size", 10000))
        self.top_k: int = int(search_cfg.get("top_k", 10))

        self.output_dir = Path(data_cfg.get("output_dir", "output"))
        self.log_dir = Path(data_cfg.get("log_dir", "logs"))
        self.checkpoint_dir = Path(data_cfg.get("checkpoint_dir", "checkpoints"))

        for d in (self.output_dir, self.log_dir, self.checkpoint_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.checkpoint = Checkpoint(str(self.checkpoint_dir))
        self._experiment_log = (self.log_dir / "experiments.jsonl").open(
            "a", encoding="utf-8"
        )

        self._work_queue: mp.Queue = mp.Queue(maxsize=self.num_workers * 4)
        self._result_queue: mp.Queue = mp.Queue(maxsize=self.result_queue_size)
        self._stop_event: mp.Event = mp.Event()

        self._workers: List[mp.Process] = []
        self._generation: int = 0
        self._total_evaluated: int = 0
        self._best_score: float = -math.inf
        self._best_result: Optional[CandidateResult] = None
        self._top_results: List[CandidateResult] = []

        # Adaptive bias: track which pipeline configs produce high scores.
        self._promising_pipelines: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Spawn all worker processes."""
        log.info("Coordinator: spawning %d worker(s).", self.num_workers)
        for i in range(self.num_workers):
            p = mp.Process(
                target=worker_process,
                args=(
                    f"worker_{i}",
                    self.pages,
                    self.config,
                    self._work_queue,
                    self._result_queue,
                    self._stop_event,
                ),
                name=f"worker_{i}",
                daemon=True,
            )
            p.start()
            self._workers.append(p)
        log.info("All workers started.")

    def stop(self) -> None:
        """Signal all workers to shut down and wait for them."""
        log.info("Coordinator: sending shutdown to all workers.")
        self._stop_event.set()
        for _ in self._workers:
            try:
                self._work_queue.put_nowait(ControlMsg(ControlCmd.SHUTDOWN))
            except Exception:
                pass
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._experiment_log.close()
        log.info("All workers stopped.")

    # -----------------------------------------------------------------------
    # Main coordination loop
    # -----------------------------------------------------------------------

    def run(
        self,
        max_seconds: Optional[float] = None,
        max_evaluations: Optional[int] = None,
    ) -> List[CandidateResult]:
        """
        Drive the search until a time or evaluation limit is reached.

        Returns
        -------
        List[CandidateResult]
            Top-K candidates found, sorted descending by score.
        """
        self.start()
        start_time = time.monotonic()
        last_checkpoint = start_time

        strategy = self.config.get("search", {}).get("strategy", "genetic")
        batch_size = self.config.get("search", {}).get("batch_size", 512)

        log.info(
            "Coordinator: search started  strategy=%s  batch=%d",
            strategy,
            batch_size,
        )

        try:
            while not self._stop_event.is_set():
                elapsed = time.monotonic() - start_time

                # Stop conditions.
                if max_seconds is not None and elapsed >= max_seconds:
                    log.info("Time limit reached (%.0fs).", elapsed)
                    break
                if (
                    max_evaluations is not None
                    and self._total_evaluated >= max_evaluations
                ):
                    log.info("Evaluation limit reached (%d).", self._total_evaluated)
                    break

                # Fill the work queue.
                self._fill_work_queue(strategy, batch_size)

                # Drain results.
                self._drain_results()

                # Periodic checkpoint.
                if time.monotonic() - last_checkpoint >= self.checkpoint_interval:
                    self._do_checkpoint()
                    last_checkpoint = time.monotonic()

                time.sleep(0.05)

        except KeyboardInterrupt:
            log.info("Coordinator interrupted by user.")
        finally:
            self._do_checkpoint()
            self.stop()

        log.info(
            "Coordinator done: %d evaluated, best=%.4f",
            self._total_evaluated,
            self._best_score,
        )
        return list(self._top_results)

    # -----------------------------------------------------------------------
    # Work dispatch
    # -----------------------------------------------------------------------

    def _fill_work_queue(self, strategy: str, batch_size: int) -> None:
        """Push work items until the queue is full."""
        self._generation += 1
        enc = "index"  # Default; can rotate through encodings.

        # Include promising pipelines from previous results.
        pipeline_dicts: List[Dict] = []
        if self._promising_pipelines:
            pipeline_dicts = self._promising_pipelines[: batch_size // 2]

        item = WorkItem(
            item_id=str(uuid.uuid4())[:8],
            strategy=strategy,
            batch_size=batch_size,
            pipeline_dicts=pipeline_dicts,
            page_encoding=enc,
            generation=self._generation,
        )
        try:
            self._work_queue.put_nowait(item)
        except Exception:
            pass  # Queue full; skip this iteration.

    # -----------------------------------------------------------------------
    # Result processing
    # -----------------------------------------------------------------------

    def _drain_results(self) -> None:
        """Process all available results without blocking."""
        drained = 0
        while True:
            try:
                result = self._result_queue.get_nowait()
                self._process_result(result)
                drained += 1
            except Exception:
                break

    def _process_result(self, result: WorkResult) -> None:
        """Integrate a WorkResult into the global top list."""
        self._total_evaluated += result.evaluated

        for candidate in result.top_candidates:
            # Update global best.
            if candidate.score > self._best_score:
                self._best_score = candidate.score
                self._best_result = candidate
                log.info(
                    "NEW BEST  score=%.4f  pipeline=%s  output_sample=%r",
                    candidate.score,
                    json.dumps(candidate.pipeline_dict)[:120],
                    candidate.latin_output[:60],
                )
                # Save to output immediately.
                self._save_best(candidate)

            # Maintain top-K list.
            self._top_results.append(candidate)
            self._top_results.sort(key=lambda c: c.score, reverse=True)
            self._top_results = self._top_results[: self.top_k]

            # Track promising configs for redistribution.
            threshold = self._best_score * 0.90
            if candidate.score >= threshold:
                self._promising_pipelines.append(candidate.pipeline_dict)
            self._promising_pipelines = self._promising_pipelines[-50:]

        # Log to experiment journal.
        self._log_experiment(result)

    # -----------------------------------------------------------------------
    # Logging & checkpointing
    # -----------------------------------------------------------------------

    def _log_experiment(self, result: WorkResult) -> None:
        """Append a structured log entry for this result."""
        entry = {
            "ts": time.time(),
            "item_id": result.item_id,
            "worker_id": result.worker_id,
            "evaluated": result.evaluated,
            "elapsed": round(result.elapsed_seconds, 3),
            "top_score": max(
                (c.score for c in result.top_candidates), default=None
            ),
        }
        self._experiment_log.write(json.dumps(entry) + "\n")
        self._experiment_log.flush()

    def _do_checkpoint(self) -> None:
        """Persist the current top-K results to disk."""
        data = {
            "timestamp": time.time(),
            "total_evaluated": self._total_evaluated,
            "best_score": self._best_score,
            "top_results": [
                {
                    "score": c.score,
                    "pipeline_dict": c.pipeline_dict,
                    "score_breakdown": c.score_breakdown,
                    "latin_output": c.latin_output,
                }
                for c in self._top_results
            ],
        }
        self.checkpoint.save(data, tag="top_k")
        log.info(
            "Checkpoint saved: %d evaluated, best=%.4f",
            self._total_evaluated,
            self._best_score,
        )

    def _save_best(self, candidate: CandidateResult) -> None:
        """Write the new best result to a dedicated file."""
        out_path = self.output_dir / "best_result.json"
        entry = {
            "timestamp": time.time(),
            "score": candidate.score,
            "score_breakdown": candidate.score_breakdown,
            "latin_output": candidate.latin_output,
            "pipeline": candidate.pipeline_dict,
        }
        out_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
