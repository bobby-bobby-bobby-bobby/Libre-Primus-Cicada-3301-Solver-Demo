"""
Statistical scoring heuristics.

Computes:
  - Index of Coincidence (IoC)
  - Shannon entropy
  - Bigram log-probability against English letter-pair frequencies
  - Monogram chi-squared deviation from English

All functions accept a 1-D numpy int array (futhorc indices 0-28) and
return a float score where *higher* is better (more English-like).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# English monogram frequencies (normalised, a-z order)
# Source: standard corpus analysis
# ---------------------------------------------------------------------------
_ENGLISH_MONO = np.array(
    [
        0.08167, 0.01492, 0.02782, 0.04253, 0.12702,  # a-e
        0.02228, 0.02015, 0.06094, 0.06966, 0.00153,  # f-j
        0.00772, 0.04025, 0.02406, 0.06749, 0.07507,  # k-o
        0.01929, 0.00095, 0.05987, 0.06327, 0.09056,  # p-t
        0.02758, 0.00978, 0.02360, 0.00150, 0.01974,  # u-y
        0.00074,                                       # z
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# English bigram log-probabilities (top frequent pairs).
# Values are natural-log probabilities; unknown pairs → -10.
# ---------------------------------------------------------------------------
_BIGRAM_FREQ: Dict[str, float] = {
    "th": 0.0356, "he": 0.0307, "in": 0.0243, "er": 0.0205, "an": 0.0199,
    "re": 0.0185, "on": 0.0176, "at": 0.0149, "en": 0.0145, "nd": 0.0135,
    "ti": 0.0134, "es": 0.0134, "or": 0.0128, "te": 0.0120, "of": 0.0117,
    "ed": 0.0117, "is": 0.0113, "it": 0.0112, "al": 0.0109, "ar": 0.0107,
    "st": 0.0105, "to": 0.0104, "nt": 0.0104, "ng": 0.0095, "se": 0.0093,
    "ha": 0.0093, "as": 0.0087, "ou": 0.0087, "io": 0.0083, "le": 0.0083,
    "ve": 0.0083, "co": 0.0079, "me": 0.0079, "de": 0.0076, "hi": 0.0076,
    "ri": 0.0073, "ro": 0.0073, "ic": 0.0070, "ne": 0.0069, "ea": 0.0069,
    "ra": 0.0069, "ce": 0.0065, "li": 0.0062, "ch": 0.0060, "ll": 0.0058,
    "be": 0.0058, "ma": 0.0057, "si": 0.0055, "om": 0.0055, "ur": 0.0054,
}
_BIGRAM_LOG: Dict[str, float] = {k: math.log(v) for k, v in _BIGRAM_FREQ.items()}
_BIGRAM_UNKNOWN = math.log(1e-6)

# Latin alphabet labels for the 29 runes (first letter of transliteration, 0-25)
_RUNE_TO_LATIN_LABEL = [
    5, 20, 19, 14, 17, 10, 6, 22,   # F U TH O R K G W
    7, 13, 8,  9,  4, 15, 25, 18,   # H N I J EO P Z S
    19, 1, 4, 12, 11,  8,  3, 14,   # T B E M L ING D OE
    0,  0, 24,  8,  4,              # A AE Y IO EA
]
_LABEL_ARRAY = np.array(_RUNE_TO_LATIN_LABEL, dtype=np.int32)


def _to_labels(indices: np.ndarray) -> np.ndarray:
    """Convert futhorc-index array to 0-25 Latin letter labels."""
    mask = (indices >= 0) & (indices < 29)
    result = np.full_like(indices, -1, dtype=np.int32)
    result[mask] = _LABEL_ARRAY[indices[mask]]
    return result[result >= 0]


# ---------------------------------------------------------------------------
# Index of Coincidence
# ---------------------------------------------------------------------------

def index_of_coincidence(indices: np.ndarray, mod: int = 29) -> float:
    """
    Compute the Index of Coincidence.

    For English text mapped to a 26-letter alphabet, the expected IoC is
    ~0.0655 (and ~0.0385 for uniform random).  We compare against futhorc
    (29-symbol) output here.

    Returns a score in [0, 1]; higher → more regular distribution.
    """
    flat = indices[indices >= 0]
    n = len(flat)
    if n < 2:
        return 0.0
    counts = np.bincount(flat % mod, minlength=mod)
    numerator = float(np.sum(counts * (counts - 1)))
    denominator = float(n * (n - 1))
    return numerator / denominator if denominator > 0 else 0.0


def ioc_score(indices: np.ndarray) -> float:
    """
    Normalised IoC score in [0, 1].

    Maps the raw IoC to a 0-1 scale where 1.0 = perfect English-like
    distribution (IoC ≈ 0.065) and 0.0 = uniform random (IoC ≈ 1/29).
    """
    target_ioc = 0.065
    random_ioc = 1.0 / 29.0
    raw = index_of_coincidence(indices)
    score = (raw - random_ioc) / (target_ioc - random_ioc)
    return float(np.clip(score, 0.0, 2.0))  # allow slight overshoot


# ---------------------------------------------------------------------------
# Shannon Entropy
# ---------------------------------------------------------------------------

def entropy(indices: np.ndarray, mod: int = 29) -> float:
    """
    Compute the normalised Shannon entropy of the symbol distribution.

    Returns a value in [0, 1] where 0 = single symbol (no information)
    and 1 = maximum entropy (uniform).

    For scoring, *lower* entropy indicates more structured (cipher-broken-like)
    output, so callers should negate this when maximising.
    """
    flat = indices[indices >= 0]
    n = len(flat)
    if n == 0:
        return 1.0
    counts = np.bincount(flat % mod, minlength=mod)
    probs = counts[counts > 0] / n
    raw_entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(mod)
    return raw_entropy / max_entropy


def entropy_score(indices: np.ndarray) -> float:
    """
    Normalised entropy-based score in [0, 1].

    Higher = more regular (lower raw entropy), which is what we want.
    """
    return 1.0 - entropy(indices)


# ---------------------------------------------------------------------------
# N-gram scoring
# ---------------------------------------------------------------------------

def bigram_score(indices: np.ndarray) -> float:
    """
    Score based on English bigram log-probability.

    Converts rune indices → Latin letter labels, then sums log-probs of
    consecutive letter pairs.  Returns the mean per-bigram log-prob,
    normalised to [0, 1].
    """
    labels = _to_labels(indices)
    if len(labels) < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(len(labels) - 1):
        a = chr(ord("a") + labels[i])
        b = chr(ord("a") + labels[i + 1])
        total += _BIGRAM_LOG.get(a + b, _BIGRAM_UNKNOWN)
        count += 1

    if count == 0:
        return 0.0

    mean_log_prob = total / count
    # Normalise: range from _BIGRAM_UNKNOWN (-13.8) to log(0.0356) (-3.3)
    low, high = _BIGRAM_UNKNOWN, math.log(0.0356)
    normalised = (mean_log_prob - low) / (high - low)
    return float(np.clip(normalised, 0.0, 1.0))


def monogram_chi2_score(indices: np.ndarray) -> float:
    """
    Chi-squared goodness-of-fit against English monogram frequencies.

    Returns 1/(1 + chi2/n) so that lower chi2 → higher score.
    """
    labels = _to_labels(indices)
    n = len(labels)
    if n == 0:
        return 0.0

    observed = np.bincount(labels, minlength=26).astype(np.float64)
    expected = _ENGLISH_MONO * n
    chi2 = float(np.sum((observed - expected) ** 2 / (expected + 1e-9)))
    return 1.0 / (1.0 + chi2 / max(n, 1))
