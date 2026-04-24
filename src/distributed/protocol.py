"""
Distributed system message types and work-item protocol.

Work items flow: Coordinator → Worker → Coordinator.

WorkItem    – a batch of pipeline configs + page data reference
WorkResult  – scored results from a worker
ControlMsg  – control signals (shutdown, checkpoint, etc.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MsgType(str, Enum):
    WORK_ITEM = "work_item"
    WORK_RESULT = "work_result"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"


class ControlCmd(str, Enum):
    SHUTDOWN = "shutdown"
    CHECKPOINT = "checkpoint"
    STATUS = "status"


@dataclass
class WorkItem:
    """
    A unit of work dispatched from coordinator to a worker.

    Parameters
    ----------
    item_id : str
        Unique identifier for this batch.
    strategy : str
        Search strategy name (``'random'``, ``'genetic'``, etc.).
    batch_size : int
        Number of hypotheses to evaluate.
    pipeline_dicts : List[Dict]
        Pre-serialised pipeline configurations.  If empty the worker
        generates its own batch using *strategy*.
    page_encoding : str
        Which page encoding to use (``'index'``, ``'gp'``, ``'alt_prime'``).
    generation : int
        Search generation.
    """

    item_id: str
    strategy: str
    batch_size: int
    pipeline_dicts: List[Dict[str, Any]]
    page_encoding: str = "index"
    generation: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": MsgType.WORK_ITEM,
            "item_id": self.item_id,
            "strategy": self.strategy,
            "batch_size": self.batch_size,
            "pipeline_dicts": self.pipeline_dicts,
            "page_encoding": self.page_encoding,
            "generation": self.generation,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkItem":
        return cls(
            item_id=d["item_id"],
            strategy=d["strategy"],
            batch_size=d["batch_size"],
            pipeline_dicts=d["pipeline_dicts"],
            page_encoding=d.get("page_encoding", "index"),
            generation=d.get("generation", 0),
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class CandidateResult:
    """A single scored hypothesis result."""

    pipeline_dict: Dict[str, Any]
    score: float
    score_breakdown: Dict[str, float]
    latin_output: str  # Latin transliteration of the output (first 100 chars)


@dataclass
class WorkResult:
    """
    Results returned by a worker after processing a WorkItem.

    Parameters
    ----------
    item_id : str
        Matches the corresponding WorkItem.
    worker_id : str
        Identifier of the worker that produced this result.
    top_candidates : List[CandidateResult]
        Top-K scored hypotheses.
    evaluated : int
        Total hypotheses evaluated in this batch.
    elapsed_seconds : float
        Wall-clock time the worker spent on this batch.
    """

    item_id: str
    worker_id: str
    top_candidates: List[CandidateResult]
    evaluated: int
    elapsed_seconds: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": MsgType.WORK_RESULT,
            "item_id": self.item_id,
            "worker_id": self.worker_id,
            "top_candidates": [
                {
                    "pipeline_dict": c.pipeline_dict,
                    "score": c.score,
                    "score_breakdown": c.score_breakdown,
                    "latin_output": c.latin_output,
                }
                for c in self.top_candidates
            ],
            "evaluated": self.evaluated,
            "elapsed_seconds": self.elapsed_seconds,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkResult":
        candidates = [
            CandidateResult(
                pipeline_dict=c["pipeline_dict"],
                score=c["score"],
                score_breakdown=c.get("score_breakdown", {}),
                latin_output=c.get("latin_output", ""),
            )
            for c in d.get("top_candidates", [])
        ]
        return cls(
            item_id=d["item_id"],
            worker_id=d["worker_id"],
            top_candidates=candidates,
            evaluated=d.get("evaluated", 0),
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class ControlMsg:
    """Control signal sent from coordinator to workers."""

    command: ControlCmd
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": MsgType.CONTROL,
            "command": self.command,
            "payload": self.payload,
        }
