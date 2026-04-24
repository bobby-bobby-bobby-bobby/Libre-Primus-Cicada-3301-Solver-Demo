"""
Vectorised GPU/CPU kernel operations for cryptanalysis transforms.

All functions operate on :class:`torch.Tensor` inputs and return tensors
on the same device.  They are intentionally stateless so they can be called
inside batched loops without hidden side-effects.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Modular arithmetic
# ---------------------------------------------------------------------------


def modular_shift(
    x: torch.Tensor,
    multiplier: int,
    offset: int,
    mod: int,
) -> torch.Tensor:
    """
    Apply ``y = (multiplier * x + offset) mod mod`` element-wise.

    Parameters
    ----------
    x : Tensor[int64]
        Input index/value tensor.
    multiplier : int
        Multiplicative key component.
    offset : int
        Additive key component (shift).
    mod : int
        Modulus (e.g. 29 for futhorc index space).
    """
    return (multiplier * x + offset) % mod


def modular_shift_batch(
    x: torch.Tensor,
    multipliers: torch.Tensor,
    offsets: torch.Tensor,
    mod: int,
) -> torch.Tensor:
    """
    Batched modular shift: ``y[b, i] = (mult[b] * x[i] + off[b]) % mod``.

    Parameters
    ----------
    x : Tensor[int64] of shape (N,)
    multipliers : Tensor[int64] of shape (B,)
    offsets : Tensor[int64] of shape (B,)

    Returns
    -------
    Tensor of shape (B, N)
    """
    # Broadcast: (B, 1) * (1, N) → (B, N)
    return (multipliers.unsqueeze(1) * x.unsqueeze(0) + offsets.unsqueeze(1)) % mod


# ---------------------------------------------------------------------------
# Vigenère-style shift
# ---------------------------------------------------------------------------


def vigenere_shift(
    x: torch.Tensor,
    key: torch.Tensor,
    mod: int,
    subtract: bool = False,
) -> torch.Tensor:
    """
    Apply a Vigenère-style shift with periodic key extension.

    ``y[i] = (x[i] + key[i % len(key)]) % mod``   (encrypt)
    ``y[i] = (x[i] - key[i % len(key)]) % mod``   (decrypt)

    Parameters
    ----------
    x : Tensor[int64] of shape (N,)
    key : Tensor[int64] of shape (K,)
    mod : int
    subtract : bool
        If True, subtract the key (decryption direction).
    """
    n = x.shape[0]
    k = key.shape[0]
    # Tile the key to length N without a Python loop.
    reps = (n + k - 1) // k
    full_key = key.repeat(reps)[:n]
    if subtract:
        return (x - full_key) % mod
    return (x + full_key) % mod


def vigenere_shift_batch(
    x: torch.Tensor,
    keys: torch.Tensor,
    mod: int,
    subtract: bool = False,
) -> torch.Tensor:
    """
    Batched Vigenère shift.

    Parameters
    ----------
    x    : Tensor (N,)
    keys : Tensor (B, K) – each row is one key vector.

    Returns
    -------
    Tensor (B, N)
    """
    b, k = keys.shape
    n = x.shape[0]
    reps = (n + k - 1) // k
    # Expand keys → (B, reps*K), then slice to (B, N)
    full_keys = keys.repeat(1, reps)[:, :n]  # (B, N)
    x_exp = x.unsqueeze(0).expand(b, -1)     # (B, N)
    if subtract:
        return (x_exp - full_keys) % mod
    return (x_exp + full_keys) % mod


# ---------------------------------------------------------------------------
# Atbash variant
# ---------------------------------------------------------------------------


def atbash_variant(x: torch.Tensor, reverse_range: int) -> torch.Tensor:
    """
    Reversed-index substitution: ``y[i] = (reverse_range - 1 - x[i])``.

    Negative (separator) values are passed through unchanged.
    """
    result = (reverse_range - 1 - x) % reverse_range
    # Restore separator markers (-1)
    result = torch.where(x < 0, x, result)
    return result


# ---------------------------------------------------------------------------
# Block transposition
# ---------------------------------------------------------------------------


def block_transposition(x: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """
    Permute the elements of each block of length ``len(pattern)``.

    Elements that don't fill a complete block are appended unchanged.

    Parameters
    ----------
    x : Tensor (N,)
    pattern : Tensor (P,) – permutation indices into a block
    """
    p = pattern.shape[0]
    n = x.shape[0]
    full_blocks = n // p
    remainder = n % p

    if full_blocks == 0:
        return x.clone()

    blocked = x[: full_blocks * p].view(full_blocks, p)
    # Gather using permutation pattern
    permuted = blocked[:, pattern]  # (full_blocks, p)
    flat = permuted.reshape(-1)

    if remainder > 0:
        flat = torch.cat([flat, x[full_blocks * p :]])
    return flat


# ---------------------------------------------------------------------------
# Index permutation (global)
# ---------------------------------------------------------------------------


def index_permutation(x: torch.Tensor, perm: torch.Tensor, mod: int) -> torch.Tensor:
    """
    Map each value through a global permutation table.

    ``y[i] = perm[x[i] % mod]``   (values outside 0..mod-1 are unchanged)

    Parameters
    ----------
    x    : Tensor (N,)
    perm : Tensor (mod,) – a permutation of 0..mod-1
    mod  : int
    """
    mask = (x >= 0) & (x < mod)
    result = x.to(torch.int64).clone()
    result[mask] = perm[x[mask].to(torch.int64) % mod].to(torch.int64)
    return result


# ---------------------------------------------------------------------------
# Latin / Rune mapping (integer → character class label)
# ---------------------------------------------------------------------------


def numeric_to_latin_labels(x: torch.Tensor) -> torch.Tensor:
    """
    Map futhorc indices (0-28) to simplified Latin letter labels (0-25).

    The mapping uses the first character of each rune's transliteration
    (pre-encoded as a 0-25 integer). Separator values (-1) are unchanged.

    This is useful for scoring outputs against English letter frequencies.
    """
    # Transliteration-first-char → 0-25 mapping (pre-computed constant).
    # Order matches RUNE_DEFINITIONS in rune.py.
    _MAP = torch.tensor(
        [
            5, 20, 19, 14, 17, 10, 6, 22,  # F U TH O R K G W
            7, 13, 8, 9, 4, 15, 25, 18,   # H N I J EO P Z S
            19, 1, 4, 12, 11, 8, 3, 14,   # T B E M L ING D OE
            0, 0, 24, 8, 4,               # A AE Y IO EA
        ],
        dtype=torch.int64,
        device=x.device,
    )
    mask = (x >= 0) & (x < 29)
    result = x.clone()
    result[mask] = _MAP[x[mask]]
    return result
