"""
Composable, serialisable Pipeline.

A Pipeline is an ordered list of :class:`~src.pipeline.transforms.Transform`
objects.  Applying the pipeline passes the tensor through each transform in
sequence.  Pipelines can be serialised to / from JSON dicts for checkpointing
and result logging.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .accelerator import Accelerator
from .transforms import Transform, transform_from_dict


class Pipeline:
    """
    Ordered composition of :class:`Transform` objects.

    Parameters
    ----------
    transforms : List[Transform]
        Ordered list of transforms to apply.
    encoding : str
        Which page encoding to use as input: ``'index'``, ``'gp'``, or
        ``'alt_prime'``.
    accelerator : Accelerator, optional
        Device manager.  Created automatically if not provided.
    """

    def __init__(
        self,
        transforms: List[Transform],
        encoding: str = "index",
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        self.transforms = list(transforms)
        self.encoding = encoding
        self.acc = accelerator or Accelerator()

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run *x* through all transforms in order.

        Parameters
        ----------
        x : Tensor (N,) – int64, on any device.

        Returns
        -------
        Tensor (N,) – on the same device as *x*.
        """
        out = x
        for t in self.transforms:
            out = t.apply(out)
        return out

    def apply_numpy(self, arr: np.ndarray) -> np.ndarray:
        """Convenience wrapper: numpy in → numpy out (moves through device)."""
        t = self.acc.from_numpy(arr)
        result = self.apply(t)
        return self.acc.to_numpy(result)

    def apply_batch(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Apply to a batch of sequences of shape ``(B, N)``.

        Each row is processed independently (the transforms are row-wise).
        This is implemented as a simple loop; for truly batched transforms
        use :mod:`src.pipeline.kernels` functions directly.
        """
        return torch.stack([self.apply(xs[i]) for i in range(xs.shape[0])])

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoding": self.encoding,
            "transforms": [t.to_dict() for t in self.transforms],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        accelerator: Optional[Accelerator] = None,
    ) -> "Pipeline":
        transforms = [transform_from_dict(td) for td in d.get("transforms", [])]
        return cls(
            transforms=transforms,
            encoding=d.get("encoding", "index"),
            accelerator=accelerator,
        )

    @classmethod
    def from_json(
        cls, s: str, accelerator: Optional[Accelerator] = None
    ) -> "Pipeline":
        return cls.from_dict(json.loads(s), accelerator=accelerator)

    # -----------------------------------------------------------------------
    # Composition helpers
    # -----------------------------------------------------------------------

    def prepend(self, t: Transform) -> "Pipeline":
        """Return a new Pipeline with *t* added at the front."""
        return Pipeline([t] + self.transforms, self.encoding, self.acc)

    def append(self, t: Transform) -> "Pipeline":
        """Return a new Pipeline with *t* added at the end."""
        return Pipeline(self.transforms + [t], self.encoding, self.acc)

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        steps = " → ".join(t.name for t in self.transforms)
        return f"Pipeline([{steps}], encoding={self.encoding!r})"
