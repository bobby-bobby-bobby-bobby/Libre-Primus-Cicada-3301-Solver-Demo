"""
Dictionary / word-boundary proximity scoring.

Avoids AI/semantic scoring while still rewarding outputs that resemble
natural language at the word-boundary level.  Three pure-computational
signals are provided:

  word_boundary_score   – density of plausible vowel/consonant transitions
  common_short_words    – ratio of 2-4 letter sequences matching top English words
  letter_pattern_score  – matches against common English consonant-vowel patterns

All functions accept numpy int arrays (futhorc indices 0-28) and return
a float in [0, 1] where higher is more language-like.
"""

from __future__ import annotations

from typing import Set

import numpy as np

from ..data.rune import RUNE_BY_INDEX

# ---------------------------------------------------------------------------
# Helpers – Latin transliteration
# ---------------------------------------------------------------------------

# Single-character Latin transliterations for the 29 runes (index → char).
# Multi-character transliterations (TH, EO, ING, OE, AE, IO, EA) are
# represented by their first letter.
_LATIN = [
    "f", "u", "t", "o", "r", "k", "g", "w", "h", "n",
    "i", "j", "e", "p", "z", "s", "t", "b", "e", "m",
    "l", "n", "d", "o", "a", "a", "y", "i", "e",
]

_VOWEL_SET: Set[str] = set("aeiou")
_CONSONANT_SET: Set[str] = set("bcdfghjklmnpqrstvwxyz")


def _indices_to_latin_chars(indices: np.ndarray):
    """Return list of single-char Latin strings for rune indices."""
    return [_LATIN[i] if 0 <= i < 29 else None for i in indices]


# ---------------------------------------------------------------------------
# Word-boundary score
# ---------------------------------------------------------------------------


def word_boundary_score(indices: np.ndarray) -> float:
    """
    Estimate word-boundary likelihood.

    Natural language has characteristic vowel/consonant alternation patterns.
    We measure:
      1. The density of V→C and C→V transitions (typical in English).
      2. The proportion of positions where two consecutive vowels or
         two consecutive consonants appear (less common in English).

    The score rewards outputs that look word-like without requiring a
    dictionary.

    Returns a float in [0, 1].
    """
    chars = [c for c in _indices_to_latin_chars(indices) if c is not None]
    n = len(chars)
    if n < 2:
        return 0.0

    vc_transitions = 0
    for i in range(n - 1):
        a_vowel = chars[i] in _VOWEL_SET
        b_vowel = chars[i + 1] in _VOWEL_SET
        if a_vowel != b_vowel:  # V→C or C→V
            vc_transitions += 1

    return vc_transitions / (n - 1)


# ---------------------------------------------------------------------------
# Common short-word density
# ---------------------------------------------------------------------------

_COMMON_WORDS: Set[str] = {
    "a", "i", "an", "as", "at", "be", "by", "do", "go", "he",
    "if", "in", "is", "it", "me", "my", "no", "of", "on", "or",
    "so", "to", "up", "us", "we", "the", "and", "are", "for",
    "has", "his", "not", "one", "she", "was", "you", "but", "can",
    "had", "her", "him", "how", "its", "may", "our", "out", "own",
    "say", "see", "two", "who", "did", "get", "all", "new",
    "that", "with", "this", "from", "they", "will", "have", "been",
    "then", "than", "when", "your", "more", "also", "into", "some",
    "only", "what", "said", "each", "time", "were",
}


def common_short_word_score(indices: np.ndarray) -> float:
    """
    Ratio of 2-4 letter windows that match common English words.

    Scans all substrings of length 2-4 and counts matches against the
    top English short-word list.  Returns matches / total_windows.
    """
    chars = [c for c in _indices_to_latin_chars(indices) if c is not None]
    n = len(chars)
    if n < 2:
        return 0.0

    hits = 0
    total = 0
    for length in (2, 3, 4):
        for start in range(n - length + 1):
            word = "".join(chars[start : start + length])
            total += 1
            if word in _COMMON_WORDS:
                hits += 1

    return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Consonant-vowel pattern score
# ---------------------------------------------------------------------------

def letter_pattern_score(indices: np.ndarray) -> float:
    """
    Score consonant-vowel (CV) pattern regularity.

    English words follow roughly CVCV(C) patterns.  We compute:
      - fraction of positions that fit a CVCVCV... or VCVCVC... template
    Returns 1.0 for perfectly alternating C-V, 0.0 for all-same.
    """
    chars = [c for c in _indices_to_latin_chars(indices) if c is not None]
    n = len(chars)
    if n < 2:
        return 0.0

    cv_flags = [1 if c in _VOWEL_SET else 0 for c in chars]
    # Check how many consecutive pairs alternate
    alternating = sum(
        1 for i in range(n - 1) if cv_flags[i] != cv_flags[i + 1]
    )
    return alternating / (n - 1)
