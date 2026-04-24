from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class HardwareBackend:
    backend: str
    device: str


class TensorEngine:
    def __init__(self) -> None:
        self.hw = self.detect_hardware()

    @staticmethod
    def detect_hardware() -> HardwareBackend:
        if torch is None:
            return HardwareBackend("numpy", "cpu")
        if torch.cuda.is_available():
            return HardwareBackend("torch", "cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return HardwareBackend("torch", "mps")
        return HardwareBackend("torch", "cpu")

    def tensor_1d(self, values: list[int]):
        if self.hw.backend == "torch":
            return torch.tensor(values, dtype=torch.int32, device=self.hw.device)
        return np.array(values, dtype=np.int32)

    def tensor_2d(self, values: list[list[int]]):
        if self.hw.backend == "torch":
            return torch.tensor(values, dtype=torch.int32, device=self.hw.device)
        return np.array(values, dtype=np.int32)

    def mod_affine(self, x, a: int, b: int, mod: int):
        return ((x * a) + b) % mod

    def batch_add(self, x, shifts):
        return x + shifts

    def batch_sub(self, x, shifts):
        return x - shifts

    def vigenere_shift(self, x, key_vec, mod: int):
        if self.hw.backend == "torch":
            n = x.shape[-1]
            tiled = key_vec.repeat((n + key_vec.shape[-1] - 1) // key_vec.shape[-1])[:n]
            return (x + tiled) % mod
        n = x.shape[-1]
        tiled = np.resize(key_vec, n)
        return (x + tiled) % mod

    def permute(self, x, idx):
        return x[..., idx]

    def transpose_blocks(self, x, block_size: int, pattern):
        n = x.shape[-1]
        usable = (n // block_size) * block_size
        head = x[..., :usable]
        tail = x[..., usable:]
        if self.hw.backend == "torch":
            reshaped = head.reshape(*x.shape[:-1], -1, block_size)
            transposed = reshaped[..., pattern]
            return torch.cat([transposed.reshape(*x.shape[:-1], usable), tail], dim=-1)
        reshaped = head.reshape(*x.shape[:-1], -1, block_size)
        transposed = reshaped[..., pattern]
        return np.concatenate([transposed.reshape(*x.shape[:-1], usable), tail], axis=-1)


class Profiler:
    def __init__(self) -> None:
        self.starts: dict[str, float] = {}
        self.metrics: dict[str, float] = {}

    def start(self, key: str) -> None:
        self.starts[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        elapsed = time.perf_counter() - self.starts[key]
        self.metrics[key] = self.metrics.get(key, 0.0) + elapsed
        return elapsed
