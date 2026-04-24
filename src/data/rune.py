"""
Rune data structures and Gematria Primus mapping for Liber Primus cryptanalysis.

The 29 Anglo-Saxon Futhorc runes are each mapped to the nth prime number
(Gematria Primus encoding).  Three parallel encodings are maintained so
pipelines can explore all of them without discarding information:

  index     – futhorc position 0-28
  gp        – Gematria Primus prime value (2..109)
  alt_prime – GP value modulo 29 (compact ring encoding)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# First 29 prime numbers (Gematria Primus value table)
# ---------------------------------------------------------------------------
PRIMES_29: List[int] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109,
]

# ---------------------------------------------------------------------------
# Anglo-Saxon Futhorc rune definitions
# (unicode_char, name, transliteration, index, gp_value)
# ---------------------------------------------------------------------------
_RUNE_DEFINITIONS: List[tuple] = [
    ("\u16A0", "fehu",     "F",   0,  2),
    ("\u16A2", "uruz",     "U",   1,  3),
    ("\u16A6", "thurisaz", "TH",  2,  5),
    ("\u16A9", "os",       "O",   3,  7),
    ("\u16B1", "rad",      "R",   4,  11),
    ("\u16B3", "ken",      "K",   5,  13),
    ("\u16B7", "gyfu",     "G",   6,  17),
    ("\u16B9", "wynn",     "W",   7,  19),
    ("\u16BB", "haegl",    "H",   8,  23),
    ("\u16BE", "nyd",      "N",   9,  29),
    ("\u16C1", "is",       "I",   10, 31),
    ("\u16C4", "ger",      "J",   11, 37),
    ("\u16C7", "eoh",      "EO",  12, 41),
    ("\u16C8", "peor",     "P",   13, 43),
    ("\u16C9", "eolh",     "Z",   14, 47),
    ("\u16CB", "sigel",    "S",   15, 53),
    ("\u16CF", "tir",      "T",   16, 59),
    ("\u16D2", "beorc",    "B",   17, 61),
    ("\u16D6", "eh",       "E",   18, 67),
    ("\u16D7", "man",      "M",   19, 71),
    ("\u16DA", "lagu",     "L",   20, 73),
    ("\u16DC", "ing",      "ING", 21, 79),
    ("\u16DE", "daeg",     "D",   22, 83),
    ("\u16DF", "ethel",    "OE",  23, 89),
    ("\u16AA", "ac",       "A",   24, 97),
    ("\u16AB", "aesc",     "AE",  25, 101),
    ("\u16A3", "yr",       "Y",   26, 103),
    ("\u16E1", "ior",      "IO",  27, 107),
    ("\u16E0", "ear",      "EA",  28, 109),
]


@dataclass(frozen=True)
class Rune:
    """Represents a single Anglo-Saxon Futhorc rune with all its encodings."""

    unicode: str           # Unicode character (e.g. '\u16A0')
    name: str              # Runic name       (e.g. 'fehu')
    transliteration: str   # Latin equivalent (e.g. 'F')
    index: int             # Futhorc position 0-28
    gp_value: int          # Gematria Primus prime

    @property
    def alt_prime_value(self) -> int:
        """GP value modulo 29 (compact ring encoding)."""
        return self.gp_value % 29

    def __repr__(self) -> str:
        return (
            f"Rune({self.unicode!r}, {self.name!r}, "
            f"idx={self.index}, gp={self.gp_value})"
        )


# ---------------------------------------------------------------------------
# Build canonical lists and lookup dictionaries
# ---------------------------------------------------------------------------
RUNES: List[Rune] = [
    Rune(unicode=u, name=n, transliteration=t, index=i, gp_value=g)
    for u, n, t, i, g in _RUNE_DEFINITIONS
]

RUNE_BY_UNICODE: Dict[str, Rune] = {r.unicode: r for r in RUNES}
RUNE_BY_INDEX: Dict[int, Rune] = {r.index: r for r in RUNES}
RUNE_BY_GP: Dict[int, Rune] = {r.gp_value: r for r in RUNES}
RUNE_BY_NAME: Dict[str, Rune] = {r.name: r for r in RUNES}

# Precomputed numpy arrays for fast batch operations
GP_VALUES_ARRAY: np.ndarray = np.array(
    [r.gp_value for r in RUNES], dtype=np.int64
)
INDEX_VALUES_ARRAY: np.ndarray = np.arange(29, dtype=np.int64)


class GematriaPrimus:
    """
    Utility class for Gematria Primus encoding / decoding operations.

    All vectorised operations use numpy for CPU and return int64 arrays.
    PyTorch tensors are handled in the pipeline layer (``src/pipeline/``).
    """

    NUM_RUNES: int = 29
    GP_VALUES: List[int] = PRIMES_29

    # Index → GP look-up table (numpy)
    INDEX_TO_GP: np.ndarray = np.array(PRIMES_29, dtype=np.int64)

    # GP value → index (Python dict for O(1) scalar look-up)
    GP_TO_INDEX: Dict[int, int] = {gp: idx for idx, gp in enumerate(PRIMES_29)}

    # ---------------------------------------------------------------------------
    # Scalar helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def rune_to_index(rune_char: str) -> Optional[int]:
        """Convert a rune Unicode character to its futhorc index (0-28)."""
        r = RUNE_BY_UNICODE.get(rune_char)
        return r.index if r is not None else None

    @staticmethod
    def rune_to_gp(rune_char: str) -> Optional[int]:
        """Convert a rune Unicode character to its Gematria Primus value."""
        r = RUNE_BY_UNICODE.get(rune_char)
        return r.gp_value if r is not None else None

    @staticmethod
    def index_to_gp(idx: int) -> int:
        """Convert a futhorc index to its GP prime value."""
        return PRIMES_29[idx]

    @staticmethod
    def gp_to_index(gp: int) -> Optional[int]:
        """Convert a GP prime value back to its futhorc index."""
        return GematriaPrimus.GP_TO_INDEX.get(gp)

    # ---------------------------------------------------------------------------
    # Vectorised batch converters (numpy)
    # ---------------------------------------------------------------------------

    @staticmethod
    def string_to_indices(text: str) -> np.ndarray:
        """
        Convert a rune Unicode string to an array of futhorc indices.

        Non-rune characters (spaces, separators, punctuation) are encoded
        as ``-1`` so callers can create separator masks.
        """
        return np.array(
            [
                RUNE_BY_UNICODE[ch].index if ch in RUNE_BY_UNICODE else -1
                for ch in text
            ],
            dtype=np.int32,
        )

    @staticmethod
    def string_to_gp_values(text: str) -> np.ndarray:
        """Convert a rune string to a GP-values array (0 for separators)."""
        return np.array(
            [
                RUNE_BY_UNICODE[ch].gp_value if ch in RUNE_BY_UNICODE else 0
                for ch in text
            ],
            dtype=np.int64,
        )

    @staticmethod
    def indices_to_gp(indices: np.ndarray) -> np.ndarray:
        """
        Vectorised conversion from index array to GP values array.

        Separator positions (index < 0) map to 0.
        """
        result = np.zeros(len(indices), dtype=np.int64)
        mask = indices >= 0
        result[mask] = GematriaPrimus.INDEX_TO_GP[indices[mask]]
        return result

    @staticmethod
    def gp_to_indices_array(gp_array: np.ndarray) -> np.ndarray:
        """
        Convert a GP-values array back to a futhorc-index array.

        Values not found in the GP table map to ``-1``.
        """
        result = np.full(len(gp_array), -1, dtype=np.int32)
        for gp, idx in GematriaPrimus.GP_TO_INDEX.items():
            result[gp_array == gp] = idx
        return result

    @staticmethod
    def indices_to_rune_string(indices: np.ndarray) -> str:
        """Convert a futhorc-index array back to a rune Unicode string."""
        return "".join(
            RUNE_BY_INDEX[int(i)].unicode if 0 <= i < 29 else " "
            for i in indices
        )

    @staticmethod
    def indices_to_latin(indices: np.ndarray) -> str:
        """Convert a futhorc-index array to a Latin-transliteration string."""
        return "".join(
            RUNE_BY_INDEX[int(i)].transliteration.lower() if 0 <= i < 29 else " "
            for i in indices
        )
