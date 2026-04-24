"""
GPU/CPU acceleration manager.

Detects the best available compute device (CUDA → MPS → CPU) and provides
helpers to move tensors to/from device memory with minimal overhead.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

log = logging.getLogger(__name__)


class Accelerator:
    """
    Manages the active compute device and tensor lifecycle.

    Detection order: CUDA → Apple Metal (MPS) → CPU.
    Users can override by passing an explicit *device* string.
    """

    def __init__(self, device: str = "auto", precision: str = "float32") -> None:
        self.device = self._resolve_device(device)
        self.dtype = torch.float32 if precision == "float32" else torch.float16
        self.int_dtype = torch.int64
        log.info("Accelerator: device=%s  precision=%s", self.device, precision)

    # -----------------------------------------------------------------------
    # Device resolution
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            dev = torch.device("cuda")
            log.info("GPU detected: CUDA (%s)", torch.cuda.get_device_name(0))
            return dev

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("GPU detected: Apple Metal (MPS)")
            return torch.device("mps")

        log.info("No GPU found; using CPU.")
        return torch.device("cpu")

    # -----------------------------------------------------------------------
    # Tensor factories / transfer helpers
    # -----------------------------------------------------------------------

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move *tensor* to the active device (non-blocking when possible)."""
        return tensor.to(self.device, non_blocking=True)

    def zeros(self, *shape, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Allocate a zero tensor on the active device."""
        return torch.zeros(*shape, dtype=dtype or self.int_dtype, device=self.device)

    def from_numpy(self, arr) -> torch.Tensor:
        """Convert a numpy array to a device tensor without an extra copy."""
        import numpy as np

        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        t = torch.from_numpy(arr.copy())
        return self.to_device(t)

    def to_numpy(self, tensor: torch.Tensor):
        """Convert a device tensor back to a numpy array (CPU copy)."""
        return tensor.detach().cpu().numpy()

    # -----------------------------------------------------------------------
    # Auto-tuning helpers
    # -----------------------------------------------------------------------

    def suggest_batch_size(self, base: int = 1024) -> int:
        """
        Suggest a good batch size for the current device.

        Scales up aggressively on CUDA (where large batches saturate VRAM),
        keeps it modest on CPU to avoid memory pressure.
        """
        if self.device.type == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                # Rough heuristic: each row in a 1024-element int64 batch is 8 KB.
                suggested = max(base, (free_bytes // (8 * 1024)) // 16)
                # Round down to next power of two for alignment.
                p2 = 1
                while p2 * 2 <= suggested:
                    p2 *= 2
                return min(p2, 1 << 16)  # cap at 65536
            except Exception:
                pass
        if self.device.type == "mps":
            return base * 4
        return base

    @property
    def is_gpu(self) -> bool:
        return self.device.type in ("cuda", "mps")

    def __repr__(self) -> str:
        return f"Accelerator(device={self.device}, dtype={self.dtype})"
