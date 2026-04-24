from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .config import Constraints
from .transforms import (
    AtbashVariant,
    BlockTransposition,
    IndexPermutation,
    ModularShift,
    Pipeline,
    UnknownTransformSlot,
    VigenereShift,
)


@dataclass
class CandidateKey:
    a: int
    b: int
    key: List[int]
    block_pattern: List[int]
    permutation: List[int]
    unknown_table: List[int]


class KeyFactory:
    def __init__(self, constraints: Constraints, mod: int = 109, seed: int = 3301) -> None:
        self.constraints = constraints
        self.mod = mod
        self.rng = random.Random(seed)

    def random_key(self, length: int) -> CandidateKey:
        key_len = self.rng.randint(self.constraints.vigenere_min_len, self.constraints.vigenere_max_len)
        block_size = self.rng.choice(self.constraints.transposition_block_sizes)
        base_pattern = list(range(block_size))
        self.rng.shuffle(base_pattern)
        perm = list(range(length))
        self.rng.shuffle(perm)
        table = list(range(self.mod))
        self.rng.shuffle(table)
        return CandidateKey(
            a=self.rng.choice(self.constraints.modular_primes),
            b=self.rng.choice(self.constraints.offsets),
            key=[self.rng.randrange(self.mod) for _ in range(key_len)],
            block_pattern=base_pattern,
            permutation=perm,
            unknown_table=table,
        )

    def grid(self, length: int, limit: int = 2000) -> Iterable[CandidateKey]:
        count = 0
        block_size = self.constraints.transposition_block_sizes[0]
        patt = list(range(block_size))[::-1]
        perm = list(range(length))
        for a, b, klen in itertools.product(
            self.constraints.modular_primes,
            self.constraints.offsets[: min(64, len(self.constraints.offsets))],
            range(self.constraints.vigenere_min_len, self.constraints.vigenere_max_len + 1),
        ):
            if count >= limit:
                break
            key = [(i * a + b) % self.mod for i in range(klen)]
            table = list(range(self.mod))
            yield CandidateKey(a, b, key, patt, perm, table)
            count += 1

    def mutate(self, ck: CandidateKey, rate: float = 0.1) -> CandidateKey:
        out = CandidateKey(
            a=ck.a,
            b=ck.b,
            key=ck.key[:],
            block_pattern=ck.block_pattern[:],
            permutation=ck.permutation[:],
            unknown_table=ck.unknown_table[:],
        )
        if self.rng.random() < rate:
            out.a = self.rng.choice(self.constraints.modular_primes)
        if self.rng.random() < rate:
            out.b = self.rng.choice(self.constraints.offsets)
        for i in range(len(out.key)):
            if self.rng.random() < rate:
                out.key[i] = (out.key[i] + self.rng.randint(-5, 5)) % self.mod
        for _ in range(max(1, len(out.unknown_table) // 20)):
            if self.rng.random() < rate:
                i, j = self.rng.randrange(len(out.unknown_table)), self.rng.randrange(len(out.unknown_table))
                out.unknown_table[i], out.unknown_table[j] = out.unknown_table[j], out.unknown_table[i]
        return out

    def crossover(self, a: CandidateKey, b: CandidateKey) -> CandidateKey:
        min_len = min(len(a.key), len(b.key))
        if min_len < 2:
            key = (a.key if len(a.key) >= len(b.key) else b.key)[:]
        else:
            split = self.rng.randint(1, min_len - 1)
            key = a.key[:split] + b.key[split:]
        table_split = self.rng.randint(1, len(a.unknown_table) - 1)
        table = a.unknown_table[:table_split] + b.unknown_table[table_split:]
        return CandidateKey(
            a=self.rng.choice([a.a, b.a]),
            b=self.rng.choice([a.b, b.b]),
            key=key,
            block_pattern=self.rng.choice([a.block_pattern, b.block_pattern])[:],
            permutation=self.rng.choice([a.permutation, b.permutation])[:],
            unknown_table=table,
        )


def key_to_pipeline(ck: CandidateKey, mod: int = 109) -> Pipeline:
    return Pipeline(
        transforms=[
            ModularShift(ck.a, ck.b, mod=mod),
            VigenereShift(ck.key, mod=mod),
            AtbashVariant(reverse_range=mod - 1),
            BlockTransposition(pattern=ck.block_pattern),
            IndexPermutation(permutation_vector=ck.permutation),
            UnknownTransformSlot(substitution_table=ck.unknown_table, affine_a=1, affine_b=0, mod=mod),
        ]
    )


class AdaptivePolicy:
    def __init__(self) -> None:
        self.transform_rewards: Dict[str, float] = {
            "modular_shift": 1.0,
            "vigenere_shift": 1.0,
            "atbash_variant": 1.0,
            "block_transposition": 1.0,
            "index_permutation": 1.0,
            "unknown_transform_slot": 1.0,
        }

    def update(self, pipeline_summary: Dict[str, float]) -> None:
        for k, v in pipeline_summary.items():
            if k in self.transform_rewards:
                self.transform_rewards[k] = 0.9 * self.transform_rewards[k] + 0.1 * max(0.01, v)

    def bias(self) -> Dict[str, float]:
        total = sum(self.transform_rewards.values())
        return {k: v / total for k, v in self.transform_rewards.items()}
