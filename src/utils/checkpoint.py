"""
Checkpoint save / load utilities.

Checkpoints are written as timestamped JSON files in a dedicated directory.
The latest checkpoint for each *tag* is also written as ``<tag>_latest.json``
for easy recovery without scanning directory contents.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class Checkpoint:
    """
    Simple file-based checkpoint manager.

    Parameters
    ----------
    directory : str
        Directory where checkpoint files are stored.
    """

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def save(self, data: Dict[str, Any], tag: str = "checkpoint") -> Path:
        """
        Persist *data* to a timestamped JSON file.

        Also updates the ``<tag>_latest.json`` symlink / copy.

        Parameters
        ----------
        data : dict
            JSON-serialisable data to save.
        tag : str
            Logical name for this checkpoint (e.g. ``"top_k"``).

        Returns
        -------
        Path
            The written checkpoint file path.
        """
        ts = int(time.time())
        filename = f"{tag}_{ts}.json"
        path = self.directory / filename
        latest_path = self.directory / f"{tag}_latest.json"

        payload = {"__tag": tag, "__timestamp": ts, **data}
        text = json.dumps(payload, indent=2, default=str)

        path.write_text(text, encoding="utf-8")
        latest_path.write_text(text, encoding="utf-8")

        log.debug("Checkpoint saved: %s", path)
        return path

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------

    def load_latest(self, tag: str = "checkpoint") -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint for *tag*.

        Returns ``None`` if no checkpoint exists.
        """
        latest_path = self.directory / f"{tag}_latest.json"
        if latest_path.exists():
            try:
                return json.loads(latest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Could not load checkpoint %s: %s", latest_path, exc)

        # Fall back to scanning for the latest timestamped file.
        candidates = sorted(self.directory.glob(f"{tag}_*.json"))
        for path in reversed(candidates):
            if path.name.endswith("_latest.json"):
                continue
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
        return None

    def load(self, path: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint file by path."""
        p = Path(path)
        if not p.exists():
            log.warning("Checkpoint file not found: %s", path)
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.error("Failed to load checkpoint %s: %s", path, exc)
            return None

    def list_checkpoints(self, tag: Optional[str] = None) -> list:
        """Return all checkpoint file paths, optionally filtered by tag."""
        pattern = f"{tag}_*.json" if tag else "*.json"
        return sorted(
            p for p in self.directory.glob(pattern)
            if not p.name.endswith("_latest.json")
        )
