"""
Structural scoring heuristics.

Computes:
  - Line symmetry (palindrome-like patterns)
  - Repeated sub-sequence density
  - Cross-line coherence (similar transform produces consistent output)
  - Similarity to known solved LP page structure

All functions accept numpy int arrays (futhorc indices 0-28) and return
a float in [0, 1] where higher is more structurally interesting.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Line symmetry
# ---------------------------------------------------------------------------


def line_symmetry_score(line: np.ndarray) -> float:
    """
    Measure how palindrome-like a single line is.

    Returns 1.0 for a perfect palindrome and 0.0 for zero matching positions.
    Only considers rune positions (indices ≥ 0).
    """
    rune_only = line[line >= 0]
    n = len(rune_only)
    if n < 2:
        return 0.0
    reversed_line = rune_only[::-1]
    matches = int(np.sum(rune_only == reversed_line))
    return matches / n


def page_symmetry_score(lines: List[np.ndarray]) -> float:
    """Average line-symmetry score across all lines in a page."""
    if not lines:
        return 0.0
    scores = [line_symmetry_score(ln) for ln in lines]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Repeated sub-sequence density
# ---------------------------------------------------------------------------


def repeated_ngram_density(
    indices: np.ndarray,
    n: int = 3,
    min_occurrences: int = 2,
) -> float:
    """
    Proportion of n-gram tokens that appear at least *min_occurrences* times.

    A high density of repetitions indicates cipher structure (key period,
    repeated plaintext words).

    Parameters
    ----------
    indices : ndarray (N,)
    n : int
        n-gram size (default 3).
    min_occurrences : int
        Minimum occurrences to count as "repeated" (default 2).

    Returns
    -------
    float in [0, 1]
    """
    rune_only = indices[indices >= 0]
    total = len(rune_only) - n + 1
    if total <= 0:
        return 0.0

    counts: dict = {}
    for i in range(total):
        gram = tuple(rune_only[i : i + n].tolist())
        counts[gram] = counts.get(gram, 0) + 1

    repeated = sum(1 for c in counts.values() if c >= min_occurrences)
    return repeated / len(counts) if counts else 0.0


# ---------------------------------------------------------------------------
# Cross-line coherence
# ---------------------------------------------------------------------------


def cross_line_coherence(lines: List[np.ndarray]) -> float:
    """
    Measure whether lines share similar symbol-frequency profiles.

    High coherence means the cipher transform is consistent across lines,
    which is a necessary (though not sufficient) condition for a valid key.

    Uses the mean pairwise cosine-similarity of per-line symbol histograms.
    """
    if len(lines) < 2:
        return 1.0

    histograms = []
    for line in lines:
        rune = line[line >= 0]
        if len(rune) == 0:
            continue
        hist = np.bincount(rune % 29, minlength=29).astype(np.float64)
        norm = np.linalg.norm(hist)
        if norm > 0:
            histograms.append(hist / norm)

    if len(histograms) < 2:
        return 0.0

    total_sim = 0.0
    n_pairs = 0
    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            total_sim += float(np.dot(histograms[i], histograms[j]))
            n_pairs += 1

    return total_sim / n_pairs if n_pairs > 0 else 0.0


# ---------------------------------------------------------------------------
# Similarity to known solved LP structure
# ---------------------------------------------------------------------------

# Symbol-frequency profile derived from the publicly known solved "Welcome"
# section of Liber Primus (Section 1).  Values are normalised probabilities.
_SOLVED_LP_PROFILE = np.array(
    [
        0.040, 0.030, 0.065, 0.025, 0.045, 0.020, 0.015, 0.025,  # 0-7
        0.055, 0.065, 0.075, 0.010, 0.015, 0.020, 0.010, 0.060,  # 8-15
        0.070, 0.045, 0.080, 0.055, 0.035, 0.015, 0.030, 0.025,  # 16-23
        0.040, 0.020, 0.015, 0.010, 0.010,                        # 24-28
    ],
    dtype=np.float64,
)
_SOLVED_LP_PROFILE /= _SOLVED_LP_PROFILE.sum()  # normalise


def lp_structure_similarity(indices: np.ndarray) -> float:
    """
    Cosine similarity between the output symbol-frequency and the
    known solved-page profile.

    Returns a float in [-1, 1]; values near 1 are most similar.
    We map it to [0, 1] for consistent scoring.
    """
    rune_only = indices[indices >= 0]
    n = len(rune_only)
    if n == 0:
        return 0.0

    hist = np.bincount(rune_only % 29, minlength=29).astype(np.float64)
    norm = np.linalg.norm(hist)
    if norm == 0:
        return 0.0
    hist /= norm

    ref_norm = np.linalg.norm(_SOLVED_LP_PROFILE)
    ref = _SOLVED_LP_PROFILE / ref_norm if ref_norm > 0 else _SOLVED_LP_PROFILE

    cosine = float(np.dot(hist, ref))
    # Map [-1, 1] → [0, 1]
    return (cosine + 1.0) / 2.0
