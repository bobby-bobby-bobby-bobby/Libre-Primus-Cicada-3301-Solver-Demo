"""
Keyspace definition and search-space narrowing constraints.

Provides helpers to generate valid parameter ranges for each transform type,
respecting the constraints in ``config.json``.

Narrowing rules
---------------
- Modular multipliers are restricted to prime subsets (reduces quadratic
  redundancy in the affine keyspace).
- Offsets are restricted to 0-28 (the valid futhorc index range).
- Vigenère key lengths are restricted to small primes (3, 5, 7, 11, 13)
  which are the most likely period lengths given LP page sizes.
- Transposition block sizes are similarly restricted to prime values ≤ 13.
- All parameters can be further constrained via ``config.json``.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Prime sieve
# ---------------------------------------------------------------------------


def _sieve(limit: int) -> List[int]:
    """Return all primes up to *limit* (inclusive)."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0
    return [i for i, v in enumerate(is_prime) if v]


PRIMES_UP_TO_200 = _sieve(200)
PRIMES_29 = PRIMES_UP_TO_200[:29]  # First 29 primes (GP values)


# ---------------------------------------------------------------------------
# Key descriptors
# ---------------------------------------------------------------------------


class KeySpec:
    """
    Describes a single searchable parameter and its valid domain.

    Parameters
    ----------
    name : str
        Human-readable parameter name.
    domain : list
        Exhaustive list of valid discrete values.
    """

    def __init__(self, name: str, domain: List[Any]) -> None:
        self.name = name
        self.domain = list(domain)

    def random_value(self) -> Any:
        """Sample a uniformly random value from the domain."""
        return random.choice(self.domain)

    def __len__(self) -> int:
        return len(self.domain)

    def __repr__(self) -> str:
        return f"KeySpec({self.name!r}, size={len(self.domain)})"


class Keyspace:
    """
    Defines the full parameter space to search.

    Parameters
    ----------
    constraints : Dict[str, Any]
        The ``"constraints"`` section from ``config.json``.
    """

    def __init__(self, constraints: Optional[Dict[str, Any]] = None) -> None:
        c = constraints or {}

        mod_min = int(c.get("modular_mod_min", 29))
        mod_max = int(c.get("modular_mod_max", 109))
        off_min = int(c.get("offset_min", 0))
        off_max = int(c.get("offset_max", 28))

        mult_type = c.get("modular_multipliers", "prime")
        if mult_type == "prime":
            multipliers = [p for p in PRIMES_UP_TO_200 if 1 <= p <= 29]
        else:
            multipliers = list(range(1, 29))

        vigenere_lengths: List[int] = c.get("vigenere_key_lengths", [3, 5, 7, 11, 13])
        block_sizes: List[int] = c.get(
            "transposition_block_sizes", [3, 5, 7, 11, 13]
        )

        # --- ModularShift space ---
        self.modular_multipliers: KeySpec = KeySpec("a", multipliers)
        self.modular_offsets: KeySpec = KeySpec(
            "b", list(range(off_min, off_max + 1))
        )
        self.modular_mods: KeySpec = KeySpec(
            "mod",
            [p for p in PRIMES_UP_TO_200 if mod_min <= p <= mod_max],
        )

        # --- VigenereShift space ---
        self.vigenere_key_lengths: KeySpec = KeySpec(
            "key_length", [int(k) for k in vigenere_lengths]
        )

        # --- BlockTransposition space ---
        self.block_sizes: KeySpec = KeySpec(
            "block_size", [int(s) for s in block_sizes]
        )

        # --- IndexPermutation: full permutation of 0-28 (too large to enumerate) ---
        # Sampling only.
        self.permutation_mod = 29

    # -----------------------------------------------------------------------
    # Random key generators per transform type
    # -----------------------------------------------------------------------

    def random_modular_shift_params(self) -> Dict[str, int]:
        return {
            "a": self.modular_multipliers.random_value(),
            "b": self.modular_offsets.random_value(),
            "mod": self.modular_mods.random_value(),
        }

    def random_vigenere_key(self) -> List[int]:
        length = self.vigenere_key_lengths.random_value()
        return [random.randint(0, 28) for _ in range(length)]

    def random_block_pattern(self) -> List[int]:
        size = self.block_sizes.random_value()
        pattern = list(range(size))
        random.shuffle(pattern)
        return pattern

    def random_permutation(self, mod: int = 29) -> List[int]:
        perm = list(range(mod))
        random.shuffle(perm)
        return perm

    def random_substitution_table(self, mod: int = 29) -> List[int]:
        """Random (not necessarily bijective) substitution table."""
        return [random.randint(0, mod - 1) for _ in range(mod)]

    # -----------------------------------------------------------------------
    # Size estimation
    # -----------------------------------------------------------------------

    def approx_modular_size(self) -> int:
        return len(self.modular_multipliers) * len(self.modular_offsets) * len(self.modular_mods)

    def approx_vigenere_size(self, alphabet: int = 29) -> int:
        total = 0
        for length in self.vigenere_key_lengths.domain:
            total += alphabet ** length
        return total

    def __repr__(self) -> str:
        return (
            f"Keyspace("
            f"modular_mults={len(self.modular_multipliers)}, "
            f"offsets={len(self.modular_offsets)}, "
            f"vigenere_lengths={self.vigenere_key_lengths.domain}, "
            f"block_sizes={self.block_sizes.domain})"
        )
