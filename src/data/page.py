"""
Page class: a single Liber Primus page with three parallel tensor encodings.

Three representations are always computed and stored:
  - ``raw_indices``      – futhorc position (0-28); -1 for separators
  - ``gp_tensor``        – Gematria Primus prime values; 0 for separators
  - ``alt_prime_tensor`` – index % 29 (compact ring encoding); -1 for seps

The 2-D view (lines × max_len) is also available for structural analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .rune import GematriaPrimus, RUNE_BY_UNICODE


def _rune_chars() -> frozenset:
    return frozenset(RUNE_BY_UNICODE.keys())


_RUNE_SET = _rune_chars()


@dataclass
class Page:
    """
    Represents a single Liber Primus page.

    Maintains three parallel tensor encodings of the rune content so that
    pipeline transforms can operate on whichever abstraction is appropriate.

    Parameters
    ----------
    page_id : str
        Unique identifier, e.g. ``"page_01"``.
    source_file : str
        Path (or label) of the originating file.
    raw_runes : str
        The original concatenated rune Unicode text (all lines joined with
        ``'\\n'``).
    lines : List[str]
        Individual rune-containing lines from the source.
    metadata : Dict[str, Any]
        Free-form provenance / annotation data.
    """

    page_id: str
    source_file: str
    raw_runes: str
    lines: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Derived tensors – populated by __post_init__
    raw_indices: np.ndarray = field(init=False)
    gp_tensor: np.ndarray = field(init=False)
    alt_prime_tensor: np.ndarray = field(init=False)
    separator_mask: np.ndarray = field(init=False)

    # Per-line tensors
    line_indices: List[np.ndarray] = field(init=False, default_factory=list)
    line_gp: List[np.ndarray] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Derive all tensor representations from *raw_runes*."""
        gp = GematriaPrimus

        # --- flat index encoding -------------------------------------------
        self.raw_indices = gp.string_to_indices(self.raw_runes)

        # --- GP encoding (0 at separator positions) ------------------------
        safe_indices = np.where(self.raw_indices >= 0, self.raw_indices, 0)
        self.gp_tensor = np.where(
            self.raw_indices >= 0,
            gp.indices_to_gp(safe_indices),
            0,
        ).astype(np.int64)

        # --- alt-prime encoding (-1 at separator positions) ----------------
        self.alt_prime_tensor = np.where(
            self.raw_indices >= 0,
            self.raw_indices % 29,
            -1,
        ).astype(np.int32)

        # --- separator mask ------------------------------------------------
        self.separator_mask = self.raw_indices < 0

        # --- per-line tensors ----------------------------------------------
        self.line_indices = []
        self.line_gp = []
        for line in self.lines:
            stripped = line.strip()
            if stripped and any(ch in _RUNE_SET for ch in stripped):
                li = gp.string_to_indices(stripped)
                self.line_indices.append(li)
                safe = np.where(li >= 0, li, 0)
                self.line_gp.append(
                    np.where(li >= 0, gp.indices_to_gp(safe), 0).astype(
                        np.int64
                    )
                )

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def rune_only_indices(self) -> np.ndarray:
        """Index array with all separator positions removed."""
        return self.raw_indices[self.raw_indices >= 0]

    @property
    def rune_only_gp(self) -> np.ndarray:
        """GP array with all separator positions removed."""
        return self.gp_tensor[self.gp_tensor > 0]

    @property
    def num_runes(self) -> int:
        """Total number of rune characters (separators excluded)."""
        return int(np.sum(self.raw_indices >= 0))

    @property
    def num_lines(self) -> int:
        """Number of rune-containing lines."""
        return len(self.line_indices)

    # -----------------------------------------------------------------------
    # Tensor access
    # -----------------------------------------------------------------------

    def get_tensor(self, encoding: str = "index") -> np.ndarray:
        """
        Return the flat 1-D tensor for the requested encoding.

        Parameters
        ----------
        encoding : str
            One of ``'index'``, ``'gp'``, or ``'alt_prime'``.
        """
        if encoding == "index":
            return self.raw_indices
        if encoding == "gp":
            return self.gp_tensor
        if encoding == "alt_prime":
            return self.alt_prime_tensor
        raise ValueError(
            f"Unknown encoding {encoding!r}. "
            "Choose from 'index', 'gp', 'alt_prime'."
        )

    def get_2d_tensor(self, encoding: str = "index") -> np.ndarray:
        """
        Return a 2-D tensor of shape ``(num_lines, max_line_length)``.

        Shorter lines are right-padded with ``-1``.
        """
        if not self.line_indices:
            return self.get_tensor(encoding).reshape(1, -1)

        if encoding == "index":
            line_tensors = self.line_indices
        elif encoding == "gp":
            line_tensors = self.line_gp
        else:  # alt_prime
            line_tensors = [t % 29 for t in self.line_indices]

        max_len = max(len(t) for t in line_tensors)
        result = np.full((len(line_tensors), max_len), -1, dtype=np.int64)
        for i, t in enumerate(line_tensors):
            result[i, : len(t)] = t
        return result

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Lightweight dictionary representation (no large arrays)."""
        return {
            "page_id": self.page_id,
            "source_file": self.source_file,
            "num_runes": self.num_runes,
            "num_lines": self.num_lines,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"Page(id={self.page_id!r}, runes={self.num_runes}, "
            f"lines={self.num_lines})"
        )
