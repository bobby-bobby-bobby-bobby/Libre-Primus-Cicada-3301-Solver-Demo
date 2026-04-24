"""
Multi-layer combined scorer.

Aggregates all sub-scorers into a single weighted composite score and
also provides cross-page coherence scoring (addresses the "pages are not
isolated" feedback).

Score components
----------------
ioc          – Index of Coincidence vs English target
ngram        – English bigram log-probability
entropy      – 1 - normalised Shannon entropy
word_boundary – vowel/consonant transition density
structural   – line symmetry + LP structure similarity
cross_page   – consistency of key across multiple pages (when available)

Weights are read from ``config.json``; defaults are hardcoded as fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .statistical import (
    bigram_score,
    entropy_score,
    ioc_score,
    monogram_chi2_score,
)
from .structural import (
    cross_line_coherence,
    lp_structure_similarity,
    page_symmetry_score,
    repeated_ngram_density,
)
from .dictionary import (
    common_short_word_score,
    letter_pattern_score,
    word_boundary_score,
)

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "ioc": 0.25,
    "ngram": 0.25,
    "entropy": 0.15,
    "word_boundary": 0.20,
    "structural": 0.10,
    "cross_page": 0.05,
}


@dataclass
class ScoreBreakdown:
    """Detailed per-component score for a single hypothesis."""

    ioc: float = 0.0
    ngram: float = 0.0
    entropy: float = 0.0
    word_boundary: float = 0.0
    structural: float = 0.0
    cross_page: float = 0.0
    composite: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "ioc": self.ioc,
            "ngram": self.ngram,
            "entropy": self.entropy,
            "word_boundary": self.word_boundary,
            "structural": self.structural,
            "cross_page": self.cross_page,
            "composite": self.composite,
        }


class Scorer:
    """
    Composite scoring engine.

    Parameters
    ----------
    weights : Dict[str, float], optional
        Per-component weights.  Must sum to approximately 1.0.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        # Normalise weights to sum to 1.
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    # -----------------------------------------------------------------------
    # Single-page scoring
    # -----------------------------------------------------------------------

    def score(
        self,
        indices: np.ndarray,
        lines: Optional[List[np.ndarray]] = None,
        cross_page_ref: Optional[np.ndarray] = None,
    ) -> ScoreBreakdown:
        """
        Score a single hypothesis output.

        Parameters
        ----------
        indices : ndarray (N,)
            Futhorc index array after pipeline transform.
        lines : List[ndarray], optional
            Per-line index arrays for structural scoring.
        cross_page_ref : ndarray, optional
            Reference output from another page under the same key.
            Used for cross-page coherence scoring.

        Returns
        -------
        ScoreBreakdown
        """
        bd = ScoreBreakdown()

        # --- Statistical ---
        bd.ioc = ioc_score(indices)
        bd.ngram = 0.5 * bigram_score(indices) + 0.5 * monogram_chi2_score(indices)
        bd.entropy = entropy_score(indices)

        # --- Word boundary ---
        wb = word_boundary_score(indices)
        cw = common_short_word_score(indices)
        lp_dict = letter_pattern_score(indices)
        bd.word_boundary = 0.4 * wb + 0.4 * cw + 0.2 * lp_dict

        # --- Structural ---
        sym = page_symmetry_score(lines) if lines else 0.0
        rep = repeated_ngram_density(indices, n=3)
        lp_sim = lp_structure_similarity(indices)
        coherence = cross_line_coherence(lines) if lines else 0.0
        bd.structural = 0.25 * sym + 0.25 * rep + 0.25 * lp_sim + 0.25 * coherence

        # --- Cross-page ---
        if cross_page_ref is not None:
            bd.cross_page = self._cross_page_score(indices, cross_page_ref)
        else:
            bd.cross_page = 0.0

        # --- Composite ---
        w = self.weights
        bd.composite = (
            w.get("ioc", 0) * bd.ioc
            + w.get("ngram", 0) * bd.ngram
            + w.get("entropy", 0) * bd.entropy
            + w.get("word_boundary", 0) * bd.word_boundary
            + w.get("structural", 0) * bd.structural
            + w.get("cross_page", 0) * bd.cross_page
        )
        return bd

    def score_value(
        self,
        indices: np.ndarray,
        lines: Optional[List[np.ndarray]] = None,
        cross_page_ref: Optional[np.ndarray] = None,
    ) -> float:
        """Convenience method: return only the composite score."""
        return self.score(indices, lines, cross_page_ref).composite

    # -----------------------------------------------------------------------
    # Cross-page coherence
    # -----------------------------------------------------------------------

    @staticmethod
    def _cross_page_score(a: np.ndarray, b: np.ndarray) -> float:
        """
        Measure symbol-distribution similarity between two page outputs.

        The assumption is that a correct key applied to two LP pages should
        produce outputs with similar symbol statistics.

        Uses cosine similarity of normalised symbol histograms.
        """
        a_rune = a[a >= 0]
        b_rune = b[b >= 0]
        if len(a_rune) == 0 or len(b_rune) == 0:
            return 0.0

        ha = np.bincount(a_rune % 29, minlength=29).astype(np.float64)
        hb = np.bincount(b_rune % 29, minlength=29).astype(np.float64)

        na, nb = np.linalg.norm(ha), np.linalg.norm(hb)
        if na == 0 or nb == 0:
            return 0.0

        cosine = float(np.dot(ha / na, hb / nb))
        return (cosine + 1.0) / 2.0

    # -----------------------------------------------------------------------
    # Batch scoring
    # -----------------------------------------------------------------------

    def score_batch(
        self, batch: List[np.ndarray], lines_batch: Optional[List[List[np.ndarray]]] = None
    ) -> List[ScoreBreakdown]:
        """Score a list of hypothesis outputs and return breakdowns."""
        results = []
        for i, idx in enumerate(batch):
            lines = lines_batch[i] if lines_batch else None
            results.append(self.score(idx, lines=lines))
        return results

    def top_k(
        self,
        batch: List[np.ndarray],
        k: int,
        lines_batch: Optional[List[List[np.ndarray]]] = None,
    ) -> List[tuple]:
        """
        Return the top-K (score, hypothesis_index) pairs from a batch.

        Returns
        -------
        List of (composite_score, original_index) sorted descending.
        """
        breakdowns = self.score_batch(batch, lines_batch)
        scored = sorted(
            enumerate(breakdowns), key=lambda x: x[1].composite, reverse=True
        )
        return [(bd.composite, idx) for idx, bd in scored[:k]]
