"""
Abstract encoder interface.

Any encoder that can produce the three named vectors (dense, poincare, tangent)
implements this ABC. Switching encoders never touches ingestion or search code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class EncodedVectors:
    """Output of any encoder — maps directly to Qdrant named vectors."""
    dense: np.ndarray        # [B, DENSE_DIM]  float32, L2-normalised
    poincare: np.ndarray     # [B, HYPERBOLIC_DIM]  float32, inside Poincaré ball
    tangent: np.ndarray      # [B, HYPERBOLIC_DIM]  float32, log_map of poincare


class Encoder(ABC):
    """
    Abstract base for all encoders.

    Contract:
    - encode() is deterministic at inference time (no stochastic ops).
    - Returns float32 arrays; shapes match DENSE_DIM / HYPERBOLIC_DIM.
    - poincare vectors are guaranteed inside the ball (||x|| < 1/sqrt(c) - eps).
    - tangent vectors are the log_map_origin of poincare vectors (Euclidean).
    - No Qdrant imports; encoding is pure numpy / torch.
    """

    @property
    @abstractmethod
    def curvature(self) -> float:
        """Curvature used for Poincaré projection."""

    @abstractmethod
    def encode(self, texts: Union[str, list[str]]) -> EncodedVectors:
        """Encode text(s) into named-vector triple."""

    def encode_query(self, text: str) -> EncodedVectors:
        """Encode a single query string. Default: same as encode()."""
        return self.encode(text)
