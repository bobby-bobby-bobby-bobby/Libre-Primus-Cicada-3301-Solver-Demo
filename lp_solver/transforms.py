from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence

from .tensor_ops import TensorEngine


class Transform:
    name: str = "transform"

    def apply(self, x, eng: TensorEngine):
        raise NotImplementedError

    def serialize(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__, "params": asdict(self)}


@dataclass
class ModularShift(Transform):
    a: int
    b: int
    mod: int
    name: str = "modular_shift"

    def apply(self, x, eng: TensorEngine):
        return eng.mod_affine(x, self.a, self.b, self.mod)


@dataclass
class VigenereShift(Transform):
    key_vector: List[int]
    mod: int = 109
    name: str = "vigenere_shift"

    def apply(self, x, eng: TensorEngine):
        kv = eng.tensor_1d(self.key_vector)
        return eng.vigenere_shift(x, kv, self.mod)


@dataclass
class AtbashVariant(Transform):
    reverse_range: int
    name: str = "atbash_variant"

    def apply(self, x, eng: TensorEngine):
        if eng.hw.backend == "python":
            return [self.reverse_range - v for v in x]
        return self.reverse_range - x


@dataclass
class BlockTransposition(Transform):
    pattern: List[int]
    name: str = "block_transposition"

    def apply(self, x, eng: TensorEngine):
        return eng.transpose_blocks(x, block_size=len(self.pattern), pattern=self.pattern)


@dataclass
class IndexPermutation(Transform):
    permutation_vector: List[int]
    name: str = "index_permutation"

    def apply(self, x, eng: TensorEngine):
        if eng.hw.backend == "torch":
            import torch

            idx_t = torch.tensor(self.permutation_vector, device=eng.hw.device)
            return eng.permute(x, idx_t)
        return eng.permute(x, self.permutation_vector)


@dataclass
class UnknownTransformSlot(Transform):
    """Searchable transform placeholder for learned substitutions/affine tweaks."""

    substitution_table: List[int]
    affine_a: int = 1
    affine_b: int = 0
    mod: int = 109
    name: str = "unknown_transform_slot"

    def apply(self, x, eng: TensorEngine):
        mapped = x
        if eng.hw.backend == "torch":
            import torch

            table = torch.tensor(self.substitution_table, dtype=torch.int64, device=eng.hw.device)
            mapped = table[(x % len(self.substitution_table)).long()]
        elif eng.hw.backend == "numpy":
            import numpy as np

            table = np.array(self.substitution_table, dtype=np.int64)
            mapped = table[x % len(self.substitution_table)]
        else:
            table = self.substitution_table
            mapped = [table[v % len(table)] for v in x]
        return eng.mod_affine(mapped, self.affine_a, self.affine_b, self.mod)


@dataclass
class NumericToLatin(Transform):
    alphabet: str
    mod: int = 29
    name: str = "numeric_to_latin"

    def apply(self, x, eng: TensorEngine):
        # No-op numeric stage; rendering happens outside tensor kernel.
        if eng.hw.backend == "python":
            return [v % self.mod for v in x]
        return x % self.mod


@dataclass
class Pipeline:
    transforms: List[Transform]

    def apply(self, x, eng: TensorEngine):
        out = x
        for t in self.transforms:
            out = t.apply(out, eng)
        return out

    def serialize(self) -> Dict[str, Any]:
        return {"transforms": [t.serialize() for t in self.transforms]}
