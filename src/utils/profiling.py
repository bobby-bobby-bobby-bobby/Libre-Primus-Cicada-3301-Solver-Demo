"""
Performance profiling utilities.

Provides lightweight wall-clock and GPU memory profiling for the
pipeline, scoring, and search layers.

Usage
-----
    with Timer("pipeline_apply") as t:
        result = pipeline.apply(x)
    print(t)          # "pipeline_apply: 12.3 ms"

    profiler = Profiler()
    with profiler.timer("score"):
        score = scorer.score_value(out)
    profiler.report()
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class Timer:
    """
    Context-manager wall-clock timer.

    Parameters
    ----------
    name : str
        Label for this measurement.
    """

    def __init__(self, name: str = "timer") -> None:
        self.name = name
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed_ms:.2f} ms"


class Profiler:
    """
    Accumulates timing measurements across multiple calls and reports stats.
    """

    def __init__(self) -> None:
        self._samples: Dict[str, List[float]] = defaultdict(list)

    @contextlib.contextmanager
    def timer(self, name: str):
        """Context-manager that records elapsed time for *name*."""
        t = Timer(name)
        with t:
            yield t
        self._samples[name].append(t.elapsed_ms)

    def record(self, name: str, elapsed_ms: float) -> None:
        """Manually record a measurement."""
        self._samples[name].append(elapsed_ms)

    def stats(self, name: str) -> Dict[str, float]:
        """Return min/max/mean/total for *name*."""
        samples = self._samples.get(name, [])
        if not samples:
            return {}
        import statistics
        return {
            "count": len(samples),
            "mean_ms": statistics.mean(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "total_ms": sum(samples),
        }

    def report(self) -> str:
        """Return a human-readable summary of all recorded timers."""
        if not self._samples:
            return "Profiler: no measurements."
        lines = ["── Profiler report ──────────────────"]
        for name in sorted(self._samples):
            s = self.stats(name)
            lines.append(
                f"  {name:<30}  n={s['count']:>6}  "
                f"mean={s['mean_ms']:>8.2f} ms  "
                f"total={s['total_ms']/1000:>6.2f} s"
            )
        lines.append("─────────────────────────────────────")
        return "\n".join(lines)

    def log_report(self) -> None:
        """Write the report to the logger at INFO level."""
        log.info("\n%s", self.report())


def gpu_memory_stats(device=None) -> Dict[str, int]:
    """
    Return current GPU memory usage in bytes.

    Returns an empty dict if CUDA is not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            dev = device or torch.device("cuda")
            alloc = torch.cuda.memory_allocated(dev)
            reserved = torch.cuda.memory_reserved(dev)
            return {
                "allocated_bytes": alloc,
                "reserved_bytes": reserved,
                "allocated_mb": alloc // (1024 * 1024),
                "reserved_mb": reserved // (1024 * 1024),
            }
    except ImportError:
        pass
    return {}
