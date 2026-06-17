"""
PipelineConfig — single object that controls every tunable knob.

Designed so the same collection can be queried with different configs
without any code changes. Serialisable to YAML/JSON for experiment tracking.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional



@dataclass
class PipelineConfig:
    # --- Encoding ---
    encoder: str = "pca"               # "pca" | "neural"
    curvature: float = 1.0
    projection_strategy: str = "einstein_spread"   # uniform | static | einstein | einstein_spread
    tangent_centroid: str = "origin"   # origin | frechet | einstein

    # --- Search ---
    query_pipeline: str = "alpha"      # alpha | tangent | combined | klein
    prefetch_limit: int = 200
    prune_factor: int = 10             # for tangent pipeline
    stage_sizes: tuple = (200, 50)     # (prefetch, rerank) for alpha/combined
    ef: int = 128

    # --- Fusion ---
    fusion_strategy: str = "busemann"  # busemann | rrf | linear_alpha | depth_band | horo
    fusion_alpha: float = 0.5
    depth_weight: float = 1.0
    horo_steepness: float = 1.0
    horo_intent: str = "auto"          # ancestors | descendants | siblings | auto
    depth_band_width: Optional[float] = None

    # --- Intent routing ---
    intent: str = "auto"               # "auto" | "precision" | "recall" | "alpha" | "text"
                                       # "auto"      — combine Poincaré alpha + text heuristics
                                       # "precision" — always use Poincaré pipeline (high MRR)
                                       # "recall"    — always use cosine pipeline  (high P@10)
                                       # "alpha"     — route using only conformal-factor signal
                                       # "text"      — route using only text-heuristic signal

    # --- Busemann / entailment ---
    K_min: float = 0.10          # minimum norm in corpus (shallowest tier ≈ theme r=0.15)
    cone_weight: float = 2.0     # horoball proximity weight in combined entailment score
    busemann_sigma: Optional[float] = 0.10  # Gaussian radius penalty σ (None = pure Busemann)

    # --- Fallback ---
    poincare_fallback: str = "cosine"  # What to do if Poincare not supported

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Preset configs matching testbench benchmarks (Phases 5-6)
ALPHA_PIPELINE = PipelineConfig(
    query_pipeline="alpha",
    fusion_strategy="busemann",
    curvature=1.0,
    stage_sizes=(200, 50),  # cosine ANN → 200, alpha proxy → 50, exact Poincaré → k
)

TANGENT_PIPELINE = PipelineConfig(
    query_pipeline="tangent",
    fusion_strategy="rrf",
    curvature=1.0,
    tangent_centroid="origin",
    prune_factor=15,         # tangent HNSW fetches k*15 → exact Poincaré → k
)

COMBINED_PIPELINE = PipelineConfig(
    query_pipeline="combined",
    fusion_strategy="busemann",
    curvature=1.0,
    stage_sizes=(120, 60),  # cosine→120 + tangent→60 merged → exact Poincaré → k
)

KLEIN_PIPELINE = PipelineConfig(
    query_pipeline="klein",
    fusion_strategy="rrf",
    curvature=1.0,           # fixed: was 5.0 — index is built at c=1.0
    stage_sizes=(200, 50),  # cosine ANN → 200, Klein proxy → 50, exact Poincaré → k
)

DENSE_COSINE_PIPELINE = PipelineConfig(
    query_pipeline="cosine",
    curvature=1.0,
)

# Cosine + tier filter baseline — isolates whether tier filter or geometry drives gains.
# Real apples-to-apples comparison with hyperbolic pipelines.
DENSE_COSINE_FILTERED_PIPELINE = PipelineConfig(
    query_pipeline="cosine_filtered",
    curvature=1.0,
)

NEURAL_PIPELINE = PipelineConfig(
    encoder="neural",
    query_pipeline="combined",
    fusion_strategy="busemann",
    curvature=1.0,
)

# ---------------------------------------------------------------------------
# Busemann pipeline — no tier filter needed, geometry handles ancestor suppression
# ---------------------------------------------------------------------------
BUSEMANN_PIPELINE = PipelineConfig(
    query_pipeline="busemann",
    curvature=1.0,
    stage_sizes=(2000, 50),   # cosine ANN → 2000, Busemann re-rank → k
)

# ---------------------------------------------------------------------------
# Entailment pipeline — cone selects geometrically valid descendants of query
# ---------------------------------------------------------------------------
ENTAILMENT_PIPELINE = PipelineConfig(
    query_pipeline="entailment",
    curvature=1.0,
    stage_sizes=(200, 80),   # cosine ANN → 200, Busemann filter → 80, entailment → k
    K_min=0.10,              # minimum corpus radius (below theme tier)
    cone_weight=2.0,         # penalty weight for docs outside the cone
)
