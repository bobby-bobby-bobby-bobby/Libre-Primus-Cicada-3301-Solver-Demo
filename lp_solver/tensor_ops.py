from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

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
            if np is not None:
                return HardwareBackend("numpy", "cpu")
            return HardwareBackend("python", "cpu")
        if torch.cuda.is_available():
            return HardwareBackend("torch", "cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return HardwareBackend("torch", "mps")
        if np is not None:
            return HardwareBackend("numpy", "cpu")
        return HardwareBackend("torch", "cpu")

    def tensor_1d(self, values: list[int]):
        if self.hw.backend == "torch":
            return torch.tensor(values, dtype=torch.int32, device=self.hw.device)
        if self.hw.backend == "numpy":
            return np.array(values, dtype=np.int32)
        return list(values)

    def tensor_2d(self, values: list[list[int]]):
        if self.hw.backend == "torch":
            return torch.tensor(values, dtype=torch.int32, device=self.hw.device)
        if self.hw.backend == "numpy":
            return np.array(values, dtype=np.int32)
        return [list(v) for v in values]

    def mod_affine(self, x, a: int, b: int, mod: int):
        if self.hw.backend == "python":
            return [((v * a) + b) % mod for v in x]
        return ((x * a) + b) % mod

    def batch_add(self, x, shifts):
        if self.hw.backend == "python":
            return [a + b for a, b in zip(x, shifts)]
        return x + shifts

    def batch_sub(self, x, shifts):
        if self.hw.backend == "python":
            return [a - b for a, b in zip(x, shifts)]
        return x - shifts

    def vigenere_shift(self, x, key_vec, mod: int):
        if self.hw.backend == "torch":
            n = x.shape[-1]
            tiled = key_vec.repeat((n + key_vec.shape[-1] - 1) // key_vec.shape[-1])[:n]
            return (x + tiled) % mod
        if self.hw.backend == "numpy":
            n = x.shape[-1]
            tiled = np.resize(key_vec, n)
            return (x + tiled) % mod
        n = len(x)
        tiled = [key_vec[i % len(key_vec)] for i in range(n)]
        return [(x[i] + tiled[i]) % mod for i in range(n)]

    def permute(self, x, idx):
        if self.hw.backend == "python":
            return [x[i] for i in idx]
        return x[..., idx]

    def transpose_blocks(self, x, block_size: int, pattern):
        n = x.shape[-1] if hasattr(x, "shape") else len(x)
        usable = (n // block_size) * block_size
        if self.hw.backend == "python":
            head = x[:usable]
            tail = x[usable:]
            out = []
            for i in range(0, usable, block_size):
                block = head[i : i + block_size]
                out.extend([block[j] for j in pattern])
            out.extend(tail)
            return out
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
