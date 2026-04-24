from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


def entropy(values: Sequence[int]) -> float:
    cnt = Counter(values)
    n = len(values) or 1
    return -sum((c / n) * math.log2(c / n) for c in cnt.values())


def index_of_coincidence(values: Sequence[int]) -> float:
    cnt = Counter(values)
    n = len(values)
    if n < 2:
        return 0.0
    return sum(c * (c - 1) for c in cnt.values()) / (n * (n - 1))


def ngram_score(values: Sequence[int], n: int = 3) -> float:
    if len(values) < n:
        return 0.0
    grams = [tuple(values[i : i + n]) for i in range(len(values) - n + 1)]
    cnt = Counter(grams)
    top = cnt.most_common(20)
    return sum(c for _, c in top) / len(grams)


def symmetry_score(values: Sequence[int]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    half = n // 2
    left, right = values[:half], values[-half:][::-1]
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return matches / max(half, 1)


def repeated_pattern_score(values: Sequence[int], max_p: int = 12) -> float:
    if not values:
        return 0.0
    best = 0.0
    for p in range(2, min(max_p, len(values)) + 1):
        tpl = tuple(values[:p])
        repeats = 0
        for i in range(0, len(values) - p + 1, p):
            if tuple(values[i : i + p]) == tpl:
                repeats += 1
        best = max(best, repeats * p / len(values))
    return best


def cross_page_consistency(candidate: Sequence[int], references: Iterable[Sequence[int]]) -> float:
    cand = set(candidate)
    refs = [set(r) for r in references if r]
    if not refs:
        return 0.0
    jacc = [len(cand & r) / max(1, len(cand | r)) for r in refs]
    return sum(jacc) / len(jacc)


# Lightweight lexical proxy anchors:
# residue classes modulo 29 corresponding to high-frequency GP symbol buckets
# observed in solved-page transcriptions; used only as a weak coherence prior.
COMMON_WORD_INDICES = {1, 4, 8, 13, 20}


def lexical_coherence(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    hits = sum(1 for v in values if (v % 29) in COMMON_WORD_INDICES)
    return hits / len(values)


@dataclass
class ScoreBreakdown:
    total: float
    parts: Dict[str, float]


class ScoringEngine:
    def score(self, values: Sequence[int], reference_pages: List[Sequence[int]]) -> ScoreBreakdown:
        parts = {
            "entropy": 1.0 / (1.0 + abs(4.2 - entropy(values))),
            "ioc": index_of_coincidence(values),
            "ngram": ngram_score(values),
            "symmetry": symmetry_score(values),
            "pattern": repeated_pattern_score(values),
            "cross_page": cross_page_consistency(values, reference_pages),
            "lexical": lexical_coherence(values),
        }
        weights = {
            "entropy": 1.2,
            "ioc": 0.8,
            "ngram": 1.3,
            "symmetry": 0.7,
            "pattern": 1.0,
            "cross_page": 1.4,
            "lexical": 0.6,
        }
        total = sum(parts[k] * weights[k] for k in parts)
        return ScoreBreakdown(total=total, parts=parts)
