"""Per-collection pipeline configuration.

All tunable parameters live here. Values are manually set in Phase 6,
benchmarked on BGC/HWV. Designed to be replaced by SONA-learned values
per collection in future Pythia integration.
"""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the 3-layer hyperbolic search pipeline."""

    # Layer 1: Pre-upload
    curvature: float = 1.0
    projection_strategy: str = "einstein_spread"
    tangent_centroid: str = "origin"  # "origin" | "frechet" | "einstein"

    # Layer 2: Query-time
    query_pipeline: str = "alpha"  # "alpha" | "tangent" | "combined"
    prune_factor: int = 10
    stage_sizes: tuple = (200, 50)

    # Layer 3: Post-processing
    fusion_strategy: str = "busemann"  # "busemann" | "rrf" | "linear_alpha"
    fusion_alpha: float = 0.5
    depth_weight: float = 1.0
    horo_steepness: float = 1.0
    horo_intent: str = "auto"  # "ancestors" | "descendants" | "siblings" | "auto"
    depth_band_width: float | None = None


# Preset configs for benchmarking
PHASE5_BASELINE = PipelineConfig(
    curvature=5.0,
    query_pipeline="klein",
    fusion_strategy="rrf",
    stage_sizes=(200, 50),
)

ALPHA_PIPELINE = PipelineConfig(
    curvature=1.0,
    query_pipeline="alpha",
    fusion_strategy="busemann",
    stage_sizes=(200, 50),
)

TANGENT_PIPELINE = PipelineConfig(
    curvature=1.0,
    query_pipeline="tangent",
    tangent_centroid="origin",
    fusion_strategy="busemann",
    prune_factor=10,
)

COMBINED_PIPELINE = PipelineConfig(
    curvature=1.0,
    query_pipeline="combined",
    tangent_centroid="origin",
    fusion_strategy="busemann",
    prune_factor=10,
    stage_sizes=(100, 50),
)
