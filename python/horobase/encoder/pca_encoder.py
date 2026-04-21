"""
PCAEncoder — senior engineer's approach (Phase 1-6 reference implementation).

Pipeline:  text → SentenceTransformer (1024d) → PCA (128d) → Poincaré projection
           → log_map_origin → tangent vector

The PCA matrix is fitted once on a representative corpus and cached to disk.
Projection strategies mirror dev/testbench/projection.py exactly so results
are directly comparable with the testbench benchmarks.

Advantage: no training needed, works immediately, calibrated curvature.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Union, Optional

import numpy as np

from hyperbolic.encoder.base import Encoder, EncodedVectors
from hyperbolic.math.poincare import (
    exp_map_origin, log_map_origin, project_to_ball,
    einstein_midpoint, gromov_delta, auto_curvature,
)

logger = logging.getLogger(__name__)

EPS = 1e-5


def _max_ball_norm(curvature: float) -> float:
    """Upper bound on Poincaré vector norm: 1/√c − ε, capped at 0.999."""
    return min(0.999, 1.0 / (curvature ** 0.5) - 1e-4)


STATIC_TIER_SCALING = {
    # generic names
    "root": 0.15, "mid": 0.50, "leaf": 0.90,
    # corpus-specific names — must mirror generator.py tier labels
    "theme": 0.15, "narrative": 0.45, "story": 0.70, "content": 0.90,
}
UNIFORM_SCALING = 0.90


class ProjectionStrategy(str, Enum):
    UNIFORM = "uniform"
    STATIC = "static"
    EINSTEIN = "einstein"
    EINSTEIN_SPREAD = "einstein_spread"
    EXPLICIT_RADIAL = "explicit_radial"   # direction from PCA, radius from tier label
    NATURAL_NORM = "natural_norm"          # direction from PCA, radius from PCA norm percentile rank


class PCAEncoder(Encoder):
    """
    Encoder that replicates the testbench projection pipeline.

    Parameters
    ----------
    base_model:
        SentenceTransformer model name.
    pca_path:
        Path to .npy file storing (128, 1024) PCA matrix. If None, PCA
        is fitted on the first call to fit_pca().
    curvature:
        Poincaré ball curvature.  If None, auto-select via Gromov delta.
    strategy:
        Projection strategy from ProjectionStrategy enum.
    """

    def __init__(
        self,
        base_model: str = "perplexity-ai/pplx-embed-v1-0.6B",
        pca_path: Optional[str] = None,
        curvature: Optional[float] = 1.0,
        strategy: ProjectionStrategy = ProjectionStrategy.EXPLICIT_RADIAL,
        hyperbolic_dim: int = 128,
    ):
        self._base_model_name = base_model
        self._pca_matrix: Optional[np.ndarray] = None  # (hyperbolic_dim, dense_dim)
        self._pca_mean: Optional[np.ndarray] = None
        self._curvature = curvature
        self._strategy = strategy
        self._hyperbolic_dim = hyperbolic_dim
        self._centroid: Optional[np.ndarray] = None  # for einstein strategies
        # Lazy import to avoid hard torch dependency at module load
        self._st_model = None
        # Corpus PCA norm percentiles — updated by fit_pca() and loaded from cache
        self._pca_norm_p5: float = 0.0
        self._pca_norm_p95: float = 1.0

        if pca_path and os.path.exists(pca_path):
            data = np.load(pca_path, allow_pickle=True).item()
            self._pca_matrix = data["components"]
            self._pca_mean = data["mean"]
            self._pca_norm_p5  = float(data.get("pca_norm_p5",  0.0))
            self._pca_norm_p95 = float(data.get("pca_norm_p95", 1.0))
            logger.info("Loaded PCA matrix from %s", pca_path)

    @property
    def curvature(self) -> float:
        return self._curvature or 1.0

    def _load_st(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(
                self._base_model_name,
                trust_remote_code=True,
            )

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        """Return 1024d L2-normalised embeddings."""
        self._load_st()
        emb = self._st_model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def _pca_reduce(self, emb: np.ndarray) -> np.ndarray:
        """Apply fitted PCA: (B, 1024) → (B, 128)."""
        if self._pca_matrix is None:
            raise RuntimeError("PCA not fitted. Call fit_pca() first.")
        if self._pca_mean is not None:
            emb = emb - self._pca_mean
        return (emb @ self._pca_matrix.T).astype(np.float32)

    def fit_pca(
        self,
        texts: list[str],
        save_path: Optional[str] = None,
    ) -> None:
        """Fit PCA on provided texts and optionally save."""
        from sklearn.decomposition import PCA

        self._load_st()
        raw = self._embed_raw(texts)
        n_components = min(self._hyperbolic_dim, raw.shape[0], raw.shape[1])
        pca = PCA(n_components=n_components, whiten=False)
        pca.fit(raw)
        components = pca.components_.astype(np.float32)  # (n_components, dense_dim)
        # Zero-pad to target dim if corpus was smaller than hyperbolic_dim
        if components.shape[0] < self._hyperbolic_dim:
            pad = np.zeros((self._hyperbolic_dim - components.shape[0], components.shape[1]), dtype=np.float32)
            components = np.concatenate([components, pad], axis=0)
        self._pca_matrix = components  # (hyperbolic_dim, dense_dim)
        self._pca_mean = pca.mean_.astype(np.float32)

        # Compute and store norm statistics from the corpus so encode_query
        # can map raw PCA magnitudes to meaningful Poincaré radii.
        pca_vecs = pca.transform(raw).astype(np.float32)
        norms = np.linalg.norm(pca_vecs, axis=-1)
        self._pca_norm_p5  = float(np.percentile(norms, 5))
        self._pca_norm_p95 = float(np.percentile(norms, 95))
        logger.info(
            "PCA norm distribution: p5=%.4f p95=%.4f",
            self._pca_norm_p5, self._pca_norm_p95,
        )

        if save_path:
            np.save(save_path, {
                "components": self._pca_matrix,
                "mean": self._pca_mean,
                "pca_norm_p5":  self._pca_norm_p5,
                "pca_norm_p95": self._pca_norm_p95,
            })
            logger.info("Saved PCA matrix to %s", save_path)

    def _project_uniform(self, pca_vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(pca_vecs, axis=-1, keepdims=True) + EPS
        unit = pca_vecs / norms
        scaled = unit * UNIFORM_SCALING
        return np.stack([exp_map_origin(v, self.curvature) for v in scaled])

    def _project_explicit_radial(
        self,
        pca_vecs: np.ndarray,
        tiers: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Explicit radial tier projection — the correct separation of concerns:

          direction = pca_unit_vec   (captures semantic topic from pplx-embed)
          radius    = STATIC_TIER_SCALING[tier]  (captures hierarchical depth)
          poincare  = direction * radius

        This forces the Poincaré distance formula to actually compute depth:
        two content docs (r=0.90) in similar directions are close; a content
        doc and a theme doc (r=0.15) in the same direction are far apart even
        though they're on the same semantic axis.

        PCA still provides the 128d direction; the tier label provides the radius.
        Neither leaks into the other's job.
        """
        norms = np.linalg.norm(pca_vecs, axis=-1, keepdims=True) + EPS
        unit = (pca_vecs / norms).astype(np.float32)   # semantic direction, unit sphere

        if tiers is None:
            radii = np.full(len(pca_vecs), UNIFORM_SCALING, dtype=np.float32)
        else:
            radii = np.array(
                [STATIC_TIER_SCALING.get(t, UNIFORM_SCALING) for t in tiers],
                dtype=np.float32,
            )

        # Scale unit direction to tier radius — already inside the ball (r < 1)
        poincare = unit * radii[:, None]

        max_norm = _max_ball_norm(self.curvature)
        norms_out = np.linalg.norm(poincare, axis=-1, keepdims=True)
        too_large = norms_out >= max_norm
        poincare = np.where(too_large, poincare * (max_norm / (norms_out + EPS)), poincare)

        return poincare.astype(np.float32)

    def _project_natural_norm(
        self,
        pca_vecs: np.ndarray,
        min_r: float = 0.15,
        max_r: float = 0.90,
    ) -> np.ndarray:
        """
        Data-driven projection: radius comes from the PCA norm percentile rank.

        Hypothesis: general/abstract docs (themes, narratives) have lower PCA norms
        because they describe broad concepts closer to the corpus mean, while
        specific/concrete docs (content) deviate more and have higher norms.

        No tier labels used — if busemann still works here, the model genuinely
        encodes depth. If not, the explicit radial projection was injecting the signal.

        Radius is mapped from PCA norm percentile rank onto [min_r, max_r].
        """
        norms = np.linalg.norm(pca_vecs, axis=-1)          # (B,)
        unit  = pca_vecs / (norms[:, None] + EPS)           # unit direction

        # Rank norms within this batch (percentile within [p5, p95] of corpus)
        p5  = self._pca_norm_p5  if self._pca_norm_p5  > 0 else float(norms.min())
        p95 = self._pca_norm_p95 if self._pca_norm_p95 > 0 else float(norms.max())

        # Clamp to [p5, p95] then linearly map to [min_r, max_r]
        clamped = np.clip(norms, p5, p95)
        if p95 > p5:
            t = (clamped - p5) / (p95 - p5)   # 0 = most general, 1 = most specific
        else:
            t = np.full_like(clamped, 0.5)
        radii = min_r + t * (max_r - min_r)    # (B,)

        poincare = (unit * radii[:, None]).astype(np.float32)

        max_norm = _max_ball_norm(self.curvature)
        out_norms = np.linalg.norm(poincare, axis=-1, keepdims=True)
        too_large = out_norms >= max_norm
        poincare = np.where(too_large, poincare * (max_norm / (out_norms + EPS)), poincare)
        return poincare.astype(np.float32)

    def _project_einstein_spread(
        self,
        pca_vecs: np.ndarray,
        tiers: Optional[list[str]] = None,
        band_width: float = 0.10,
    ) -> np.ndarray:
        norms = np.linalg.norm(pca_vecs, axis=-1, keepdims=True) + EPS
        unit = (pca_vecs / norms).astype(np.float32)

        if tiers is None:
            # No tier info — fall back to uniform
            return self._project_uniform(pca_vecs)

        unique_tiers = sorted(set(tiers), key=lambda t: STATIC_TIER_SCALING.get(t, 0.5))
        tier_radii: dict[str, float] = {}
        for tier in unique_tiers:
            idx = [i for i, t in enumerate(tiers) if t == tier]
            pts = [unit[i].tolist() for i in idx]
            midpoint = einstein_midpoint(pts, self.curvature)
            radius = np.linalg.norm(midpoint)
            tier_radii[tier] = float(radius)

        # Enforce monotonic ordering by STATIC_TIER_SCALING rank
        ordered = sorted(tier_radii.keys(), key=lambda t: STATIC_TIER_SCALING.get(t, 0.5))
        for i in range(1, len(ordered)):
            if tier_radii[ordered[i]] <= tier_radii[ordered[i - 1]]:
                tier_radii[ordered[i]] = tier_radii[ordered[i - 1]] + 0.05

        projected = np.empty_like(unit)
        for tier in unique_tiers:
            idx = [i for i, t in enumerate(tiers) if t == tier]
            t_radius = tier_radii[tier]
            pts_unit = unit[idx]
            # Intra-tier spread
            centroid_dist = np.linalg.norm(pts_unit - pts_unit.mean(0), axis=-1)
            if centroid_dist.max() > EPS:
                norm_dist = (centroid_dist - centroid_dist.min()) / (centroid_dist.max() - centroid_dist.min() + EPS)
            else:
                norm_dist = np.zeros(len(idx))
            scales = t_radius + band_width * (norm_dist - 0.5)
            for j, (orig_i, scale) in enumerate(zip(idx, scales)):
                projected[orig_i] = exp_map_origin(unit[orig_i] * scale, self.curvature)

        return projected.astype(np.float32)

    def encode(
        self,
        texts: Union[str, list[str]],
        tiers: Optional[list[str]] = None,
    ) -> EncodedVectors:
        if isinstance(texts, str):
            texts = [texts]

        dense = self._embed_raw(texts)         # (B, 1024)
        pca = self._pca_reduce(dense)          # (B, 128)

        if self._strategy == ProjectionStrategy.UNIFORM:
            poincare = self._project_uniform(pca)
        elif self._strategy in (ProjectionStrategy.EXPLICIT_RADIAL, ProjectionStrategy.STATIC):
            poincare = self._project_explicit_radial(pca, tiers=tiers)
        elif self._strategy == ProjectionStrategy.NATURAL_NORM:
            poincare = self._project_natural_norm(pca)
        elif self._strategy in (ProjectionStrategy.EINSTEIN, ProjectionStrategy.EINSTEIN_SPREAD):
            poincare = self._project_einstein_spread(pca, tiers=tiers)
        else:
            raise ValueError(f"Unknown projection strategy: {self._strategy!r}")

        tangent = np.stack([log_map_origin(v, self.curvature) for v in poincare]).astype(np.float32)

        return EncodedVectors(dense=dense, poincare=poincare, tangent=tangent)

    def encode_query(self, text: str, query_tier: str = "content") -> "EncodedVectors":
        """Encode a query, projecting it into the same radius band as the tier being retrieved."""
        return self.encode([text], tiers=[query_tier])

    def calibrate_curvature(self, sample_texts: list[str]) -> float:
        """Estimate optimal curvature from Gromov delta of a sample."""
        dense = self._embed_raw(sample_texts)
        pca = self._pca_reduce(dense)
        norms = np.linalg.norm(pca, axis=-1, keepdims=True) + EPS
        unit = (pca / norms * UNIFORM_SCALING).astype(np.float32)
        delta = gromov_delta(unit)
        self._curvature = auto_curvature(delta)
        logger.info("Auto-selected curvature=%.2f (Gromov delta=%.4f)", self._curvature, delta)
        return self._curvature
