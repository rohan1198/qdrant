"""Projection strategies for mapping 128d PCA vectors to the Poincare ball.

Each strategy function takes (vectors, tiers, c) and returns a dict:
  {"vectors": np.ndarray, "metadata": dict}
"""

import numpy as np

from hyperbolic_math import (
    EPS,
    einstein_midpoint,
    exp_map_at_origin,
    project_to_ball,
)

STATIC_TIER_SCALING = {"root": 0.15, "mid": 0.50, "leaf": 0.90}
UNIFORM_SCALING = 0.9
BAND_WIDTH = 0.10
TIER_ORDER = ["root", "mid", "leaf"]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit norm."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return vectors / norms


def _project_batch(vectors: np.ndarray, c: float) -> np.ndarray:
    """Apply exp_map_at_origin + project_to_ball to every row."""
    out = np.empty_like(vectors)
    for i in range(len(vectors)):
        mapped = exp_map_at_origin(vectors[i], c=c)
        out[i] = project_to_ball(mapped, c=c)
    return out


def _norm_stats(vectors: np.ndarray) -> dict:
    """Compute norm statistics for logging."""
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "min": float(norms.min()),
        "max": float(norms.max()),
        "mean": float(norms.mean()),
        "std": float(norms.std()),
    }


def _tier_indices(tiers: list[str]) -> dict[str, list[int]]:
    """Group vector indices by tier."""
    groups: dict[str, list[int]] = {"root": [], "mid": [], "leaf": []}
    for i, t in enumerate(tiers):
        if t in groups:
            groups[t].append(i)
    return groups


def project_uniform(
    vectors: np.ndarray, tiers: list[str], c: float
) -> dict:
    """Uniform scaling: L2-normalize, scale by 0.9, project."""
    normed = _l2_normalize(vectors)
    scaled = normed * UNIFORM_SCALING
    projected = _project_batch(scaled, c)
    return {
        "vectors": projected,
        "metadata": {
            "strategy": "uniform",
            "scaling": UNIFORM_SCALING,
            "norm_stats": _norm_stats(projected),
        },
    }


def project_static(
    vectors: np.ndarray, tiers: list[str], c: float
) -> dict:
    """Static per-tier scaling: L2-normalize, scale by tier, project."""
    normed = _l2_normalize(vectors)
    scaled = np.empty_like(normed)
    groups = _tier_indices(tiers)

    for tier, indices in groups.items():
        if indices:
            scale = STATIC_TIER_SCALING[tier]
            scaled[indices] = normed[indices] * scale

    projected = _project_batch(scaled, c)

    tier_norm_stats = {}
    for tier, indices in groups.items():
        if indices:
            tier_norm_stats[tier] = _norm_stats(projected[indices])

    return {
        "vectors": projected,
        "metadata": {
            "strategy": "static",
            "tier_scaling": STATIC_TIER_SCALING,
            "tier_norm_stats": tier_norm_stats,
            "norm_stats": _norm_stats(projected),
        },
    }


def _calibrate_einstein_radii(
    vectors: np.ndarray, tiers: list[str], c: float
) -> dict[str, float]:
    """Calibration pass: project uniformly, compute Einstein midpoint per tier,
    extract Poincare radius as tier center.

    Returns dict mapping tier name to derived radius.
    Enforces monotonic ordering root < mid < leaf.
    """
    normed = _l2_normalize(vectors)
    scaled = normed * UNIFORM_SCALING
    projected = _project_batch(scaled, c)

    groups = _tier_indices(tiers)
    tier_radii: dict[str, float] = {}

    for tier in TIER_ORDER:
        indices = groups[tier]
        if not indices:
            tier_radii[tier] = STATIC_TIER_SCALING[tier]
            print(f"  Warning: no vectors for tier '{tier}', using static fallback {STATIC_TIER_SCALING[tier]}")
            continue
        tier_points = [projected[i] for i in indices]
        midpoint = einstein_midpoint(tier_points, c)
        radius = float(np.linalg.norm(midpoint))
        tier_radii[tier] = radius
        print(f"  Einstein midpoint radius for '{tier}': {radius:.6f} (n={len(indices)})")

    # Enforce monotonic ordering: root < mid < leaf
    radii_list = [tier_radii[t] for t in TIER_ORDER]
    if not (radii_list[0] < radii_list[1] < radii_list[2]):
        print(f"  Warning: radii not monotonic {radii_list}, sorting and reassigning")
        radii_list.sort()
        for t, r in zip(TIER_ORDER, radii_list):
            tier_radii[t] = r

    return tier_radii


def project_einstein(
    vectors: np.ndarray, tiers: list[str], c: float
) -> dict:
    """Einstein midpoint-derived tier centers: calibrate radii from data, then project."""
    print("  Calibrating tier radii via Einstein midpoint...")
    tier_radii = _calibrate_einstein_radii(vectors, tiers, c)

    normed = _l2_normalize(vectors)
    scaled = np.empty_like(normed)
    groups = _tier_indices(tiers)

    for tier, indices in groups.items():
        if indices:
            scaled[indices] = normed[indices] * tier_radii[tier]

    projected = _project_batch(scaled, c)

    tier_norm_stats = {}
    for tier, indices in groups.items():
        if indices:
            tier_norm_stats[tier] = _norm_stats(projected[indices])

    return {
        "vectors": projected,
        "metadata": {
            "strategy": "einstein",
            "tier_radii": tier_radii,
            "tier_norm_stats": tier_norm_stats,
            "norm_stats": _norm_stats(projected),
        },
    }


def project_einstein_spread(
    vectors: np.ndarray, tiers: list[str], c: float,
    band_width: float = BAND_WIDTH,
) -> dict:
    """Einstein-derived tier centers + intra-tier variance from centroid distance."""
    print("  Calibrating tier radii via Einstein midpoint...")
    tier_radii = _calibrate_einstein_radii(vectors, tiers, c)

    normed = _l2_normalize(vectors)
    scaled = np.empty_like(normed)
    groups = _tier_indices(tiers)

    for tier, indices in groups.items():
        if not indices:
            continue
        tier_center = tier_radii[tier]
        tier_vectors = normed[indices]

        # Euclidean centroid in PCA space (pre-projection)
        centroid = tier_vectors.mean(axis=0)
        distances = np.linalg.norm(tier_vectors - centroid, axis=1)

        # Normalize distances to [0, 1]
        d_min, d_max = distances.min(), distances.max()
        if d_max - d_min > EPS:
            normalized = (distances - d_min) / (d_max - d_min)
        else:
            normalized = np.full_like(distances, 0.5)

        # Scale = tier_center + band_width * (normalized - 0.5)
        per_vector_scale = tier_center + band_width * (normalized - 0.5)

        # Clamp to [EPS, ball_boundary) for safety
        ball_boundary = (1.0 / np.sqrt(c)) - EPS
        per_vector_scale = np.clip(per_vector_scale, EPS, ball_boundary)

        scaled[indices] = tier_vectors * per_vector_scale[:, np.newaxis]

    projected = _project_batch(scaled, c)

    tier_norm_stats = {}
    for tier, indices in groups.items():
        if indices:
            tier_norm_stats[tier] = _norm_stats(projected[indices])

    return {
        "vectors": projected,
        "metadata": {
            "strategy": "einstein_spread",
            "tier_radii": tier_radii,
            "band_width": band_width,
            "tier_norm_stats": tier_norm_stats,
            "norm_stats": _norm_stats(projected),
        },
    }


STRATEGIES = {
    "uniform": project_uniform,
    "static": project_static,
    "einstein": project_einstein,
    "einstein_spread": project_einstein_spread,
}


def auto_curvature(delta: float) -> float:
    """Select Poincare ball curvature from Gromov delta.

    Thresholds are initial estimates — Suite 15 (Curvature Sweep) will
    empirically determine optimal breakpoints on BGC and HWV.

    Lower delta = more tree-like = higher curvature (tighter ball).
    """
    if delta < 0.10:
        return 2.0
    if delta < 0.20:
        return 1.0
    if delta < 0.35:
        return 0.5
    return 0.25
