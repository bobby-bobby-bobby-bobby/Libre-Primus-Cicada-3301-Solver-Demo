"""
Structured logging setup.

Configures both a human-readable console handler and a machine-readable
JSON-lines file handler for the given process (coordinator or worker).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional


class _JsonLineFormatter(logging.Formatter):
    """Format log records as JSON lines for structured log analysis."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "pid": os.getpid(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def setup_logging(
    log_dir: str,
    process_name: str = "main",
    level: int = logging.INFO,
) -> None:
    """
    Configure root logger with console + file handlers.

    Parameters
    ----------
    log_dir : str
        Directory where log files are written.
    process_name : str
        Label used in the log file name (e.g. ``"coordinator"``).
    level : int
        Logging level (default INFO).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return  # Already configured.

    # Console handler – human-readable.
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console)

    # File handler – JSON lines.
    log_path = Path(log_dir) / f"{process_name}.jsonl"
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(_JsonLineFormatter())
    root.addHandler(file_handler)
