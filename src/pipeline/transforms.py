"""
Composable, serialisable cryptanalysis transform classes.

Every transform:
  - accepts a ``torch.Tensor`` (int64, 1-D) and returns a new tensor.
  - is JSON-serialisable via ``to_dict()`` / ``from_dict()``.
  - is GPU-aware: tensors stay on whatever device they arrive on.

Included transforms
-------------------
ModularShift          – affine map modulo N
VigenereShift         – periodic key addition / subtraction
AtbashVariant         – reversal within a range
BlockTransposition    – per-block element permutation
IndexPermutation      – global value-level permutation
NumericToLatinLabels  – map indices → Latin letter class
LearnedSubstitution   – evolvable substitution table ("unknown transform slot")
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from .kernels import (
    atbash_variant,
    block_transposition,
    index_permutation,
    modular_shift,
    numeric_to_latin_labels,
    vigenere_shift,
)


class Transform(ABC):
    """Abstract base class for all pipeline transforms."""

    name: str = "transform"

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform to *x* and return the result."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialise this transform to a JSON-compatible dict."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transform":
        """Deserialise from the dict produced by :meth:`to_dict`."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x)

    def __repr__(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Concrete transforms
# ---------------------------------------------------------------------------


class ModularShift(Transform):
    """
    Affine modular transform: ``y = (a * x + b) % mod``.

    Parameters
    ----------
    a : int
        Multiplicative factor (key component).
    b : int
        Additive offset (key component).
    mod : int
        Modulus; defaults to 29 (futhorc alphabet size).
    """

    name = "ModularShift"

    def __init__(self, a: int, b: int, mod: int = 29) -> None:
        self.a = int(a)
        self.b = int(b)
        self.mod = int(mod)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return modular_shift(x, self.a, self.b, self.mod)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.name, "a": self.a, "b": self.b, "mod": self.mod}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModularShift":
        return cls(d["a"], d["b"], d.get("mod", 29))


class VigenereShift(Transform):
    """
    Vigenère-style periodic key shift: ``y[i] = (x[i] ± key[i % K]) % mod``.

    Parameters
    ----------
    key_vector : List[int]
        Key values (will be reduced modulo *mod* internally).
    mod : int
        Alphabet size (default 29).
    subtract : bool
        If True, subtract the key instead of adding (decryption).
    """

    name = "VigenereShift"

    def __init__(
        self,
        key_vector: List[int],
        mod: int = 29,
        subtract: bool = False,
    ) -> None:
        self.key_vector = [int(k) % mod for k in key_vector]
        self.mod = int(mod)
        self.subtract = bool(subtract)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        key = torch.tensor(self.key_vector, dtype=torch.int64, device=x.device)
        return vigenere_shift(x, key, self.mod, subtract=self.subtract)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "key_vector": self.key_vector,
            "mod": self.mod,
            "subtract": self.subtract,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VigenereShift":
        return cls(d["key_vector"], d.get("mod", 29), d.get("subtract", False))


class AtbashVariant(Transform):
    """
    Reversed-index substitution: ``y[i] = (reverse_range - 1 - x[i]) % reverse_range``.

    Parameters
    ----------
    reverse_range : int
        Alphabet size to reverse within (default 29).
    """

    name = "AtbashVariant"

    def __init__(self, reverse_range: int = 29) -> None:
        self.reverse_range = int(reverse_range)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return atbash_variant(x, self.reverse_range)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.name, "reverse_range": self.reverse_range}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AtbashVariant":
        return cls(d.get("reverse_range", 29))


class BlockTransposition(Transform):
    """
    Per-block permutation: within each block of size ``len(pattern)`` the
    elements are reordered according to *pattern*.

    Parameters
    ----------
    pattern : List[int]
        A permutation of ``range(len(pattern))``.
    """

    name = "BlockTransposition"

    def __init__(self, pattern: List[int]) -> None:
        self.pattern = list(pattern)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        pat = torch.tensor(self.pattern, dtype=torch.int64, device=x.device)
        return block_transposition(x, pat)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.name, "pattern": self.pattern}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BlockTransposition":
        return cls(d["pattern"])


class IndexPermutation(Transform):
    """
    Global value-level permutation table: ``y[i] = perm[x[i] % mod]``.

    Parameters
    ----------
    permutation_vector : List[int]
        A permutation of ``range(mod)``.
    mod : int
        Alphabet size (default 29).
    """

    name = "IndexPermutation"

    def __init__(self, permutation_vector: List[int], mod: int = 29) -> None:
        self.permutation_vector = list(permutation_vector)
        self.mod = int(mod)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        perm = torch.tensor(
            self.permutation_vector, dtype=torch.int64, device=x.device
        )
        return index_permutation(x, perm, self.mod)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "permutation_vector": self.permutation_vector,
            "mod": self.mod,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IndexPermutation":
        return cls(d["permutation_vector"], d.get("mod", 29))


class NumericToLatinLabels(Transform):
    """
    Map futhorc indices → Latin letter class labels (0-25).

    Useful as a final step when scoring against English letter distributions.
    """

    name = "NumericToLatinLabels"

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return numeric_to_latin_labels(x)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.name}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NumericToLatinLabels":
        return cls()


class LearnedSubstitution(Transform):
    """
    Evolvable substitution table – the "unknown transform slot".

    Stores a full substitution table ``table[i] → j`` for i in 0..mod-1.
    The table itself is a parameter that search strategies can mutate or
    evolve without knowing the transform's semantics.

    Parameters
    ----------
    table : List[int]
        Substitution table of length *mod*.  Must be a permutation of
        ``range(mod)`` for reversibility, but the engine doesn't enforce this
        so partial mappings are also accepted.
    mod : int
        Alphabet size (default 29).
    """

    name = "LearnedSubstitution"

    def __init__(self, table: List[int], mod: int = 29) -> None:
        self.table = list(table)
        self.mod = int(mod)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        tbl = torch.tensor(self.table, dtype=torch.int64, device=x.device)
        mask = (x >= 0) & (x < self.mod)
        result = x.to(torch.int64).clone()
        result[mask] = tbl[x[mask].to(torch.int64) % self.mod]
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.name, "table": self.table, "mod": self.mod}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearnedSubstitution":
        return cls(d["table"], d.get("mod", 29))

    @staticmethod
    def random(mod: int = 29) -> "LearnedSubstitution":
        """Generate a random permutation substitution table."""
        import random

        table = list(range(mod))
        random.shuffle(table)
        return LearnedSubstitution(table, mod)


# ---------------------------------------------------------------------------
# Registry for deserialisation
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {
    ModularShift.name: ModularShift,
    VigenereShift.name: VigenereShift,
    AtbashVariant.name: AtbashVariant,
    BlockTransposition.name: BlockTransposition,
    IndexPermutation.name: IndexPermutation,
    NumericToLatinLabels.name: NumericToLatinLabels,
    LearnedSubstitution.name: LearnedSubstitution,
}


def transform_from_dict(d: Dict[str, Any]) -> Transform:
    """
    Deserialise a transform from its dict representation.

    Parameters
    ----------
    d : dict
        Must contain a ``"type"`` key matching one of the registered names.
    """
    t = d.get("type", "")
    cls = _REGISTRY.get(t)
    if cls is None:
        raise ValueError(f"Unknown transform type {t!r}.")
    return cls.from_dict(d)
