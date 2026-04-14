"""Tangent space computation for approximate hyperbolic nearest-neighbor search.

Three centroid strategies:
  - origin: zero vector, cheapest (log_map_origin, no Mobius ops)
  - frechet: iterative Frechet mean, most accurate, expensive
  - einstein: closed-form Einstein midpoint, nearly as accurate as frechet

Tangent coordinates are uploaded as a "tangent" named vector with Euclidean
distance. Qdrant's native HNSW indexes them — no fork changes needed.
"""

import numpy as np
from hyperbolic_math import (
    log_map_origin,
    log_map,
    frechet_mean,
    einstein_midpoint,
    project_to_ball,
)


def compute_centroid(
    points: list[np.ndarray],
    strategy: str,
    curvature: float = 1.0,
) -> np.ndarray:
    """Compute the tangent space centroid using the given strategy.

    Args:
        points: list of Poincare ball vectors
        strategy: "origin" | "frechet" | "einstein"
        curvature: Poincare ball curvature

    Returns:
        centroid vector (same dimensionality as input points)
    """
    if strategy == "origin":
        return np.zeros(len(points[0]))
    elif strategy == "frechet":
        return frechet_mean(points, curvature)
    elif strategy == "einstein":
        return einstein_midpoint(points, curvature)
    else:
        raise ValueError(f"Unknown centroid strategy: {strategy}")


def compute_tangent_coords(
    points: list[np.ndarray],
    centroid: np.ndarray,
    curvature: float = 1.0,
) -> np.ndarray:
    """Project all points to tangent space at the given centroid.

    If centroid is the origin (all zeros), uses the fast log_map_origin path.

    Returns:
        np.ndarray of shape (len(points), dim) — tangent coordinates
    """
    is_origin = np.linalg.norm(centroid) < 1e-9

    if is_origin:
        coords = [log_map_origin(p, curvature) for p in points]
    else:
        coords = [log_map(centroid, p, curvature) for p in points]

    return np.array(coords, dtype=np.float32)


def tangent_query(
    query_poincare: np.ndarray,
    centroid: np.ndarray,
    curvature: float = 1.0,
) -> np.ndarray:
    """Project a single query point to tangent space.

    Returns:
        tangent space vector (same dimensionality as query)
    """
    is_origin = np.linalg.norm(centroid) < 1e-9

    if is_origin:
        return log_map_origin(query_poincare, curvature).astype(np.float32)
    else:
        return log_map(centroid, query_poincare, curvature).astype(np.float32)
