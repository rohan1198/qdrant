"""
TrainedEncoder — drop-in replacement for PCAEncoder that uses a trained
ProjectionHead instead of PCA + explicit radial projection.

At inference time:
    text → frozen pplx-embed (1024d) → ProjectionHead (128d) → Poincaré ball

The interface is identical to PCAEncoder so all eval scripts work unchanged.
Just swap the encoder object.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch

from hyperbolic.encoder.base import Encoder, EncodedVectors
from hyperbolic.math.poincare import log_map_origin
from hyperbolic.training.projection_head import ProjectionHead

logger = logging.getLogger(__name__)


class TrainedEncoder(Encoder):
    """
    Encoder backed by a trained ProjectionHead.

    Parameters
    ----------
    head_path : str
        Path to a .pt checkpoint saved by ProjectionHead.save().
    base_model : str
        Frozen sentence-transformer model name.
    device : str
        "cuda", "cuda:1", "cpu", etc.
    """

    def __init__(
        self,
        head_path: str,
        base_model: str = "perplexity-ai/pplx-embed-v1-0.6B",
        device: Optional[str] = None,
    ):
        self._base_model_name = base_model
        self._st_model = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._head = ProjectionHead.load(head_path, device=self._device)
        self._head.eval()
        logger.info("Loaded ProjectionHead from %s (device=%s)", head_path, self._device)

    @property
    def curvature(self) -> float:
        return self._head.curvature

    def _load_st(self) -> None:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(
                self._base_model_name,
                trust_remote_code=True,
                device=self._device,
            )

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        self._load_st()
        emb = self._st_model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def encode(
        self,
        texts: Union[str, list[str]],
        tiers: Optional[list[str]] = None,
    ) -> EncodedVectors:
        if isinstance(texts, str):
            texts = [texts]

        # Frozen encoder — no grad needed
        dense_np = self._embed_raw(texts)                        # [B, 1024]
        dense_t  = torch.from_numpy(dense_np).to(self._device)  # [B, 1024]

        with torch.no_grad():
            poincare_t, proj_t = self._head(dense_t, tiers=tiers)

        poincare_np = poincare_t.cpu().numpy().astype(np.float32)  # [B, 128]
        proj_np     = proj_t.cpu().numpy().astype(np.float32)      # [B, 128]

        # Tangent vector = log map of Poincaré position (for tangent pipeline)
        tangent_np = np.stack([
            log_map_origin(v, self.curvature) for v in poincare_np
        ]).astype(np.float32)

        # dense field: use original 1024d frozen embeddings for stage-1 cosine ANN
        # (better recall than the 128d projection for nearest-neighbour search)
        return EncodedVectors(dense=dense_np, poincare=poincare_np, tangent=tangent_np)

    def encode_query(self, text: str, query_tier: str = "content") -> EncodedVectors:
        return self.encode([text], tiers=[query_tier])
