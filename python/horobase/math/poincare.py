"""
Poincaré ball math — single source of truth for the entire Python stack.

Mirrors dev/testbench/hyperbolic_math.py exactly so testbench results are
directly reproducible from this library. All ops are pure numpy; no torch.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Sequence

EPS = 1e-5


# ---------------------------------------------------------------------------
# Core ball operations
# ---------------------------------------------------------------------------

def project_to_ball(x: np.ndarray, c: float) -> np.ndarray:
    max_norm = min(0.95, 1.0 / np.sqrt(c) - EPS)
    norm = np.linalg.norm(x)
    return x * (max_norm / (norm + EPS)) if norm >= max_norm else x.copy()


def exp_map_origin(v: np.ndarray, c: float) -> np.ndarray:
    sqrt_c = np.sqrt(c)
    norm_v = np.linalg.norm(v) + EPS
    tanh_val = np.tanh(sqrt_c * norm_v / 2.0)
    return project_to_ball((tanh_val / (sqrt_c * norm_v)) * v, c)


def log_map_origin(x: np.ndarray, c: float) -> np.ndarray:
    sqrt_c = np.sqrt(c)
    norm_x = np.linalg.norm(x) + EPS
    arg = np.clip(sqrt_c * np.linalg.norm(x), 0.0, 1.0 - EPS)
    return (2.0 / sqrt_c) * np.arctanh(arg) / norm_x * x


def mobius_add(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    x2, y2, xy = np.dot(x, x), np.dot(y, y), np.dot(x, y)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    return project_to_ball(num / (denom + EPS), c)


def poincare_dist(x: np.ndarray, y: np.ndarray, c: float) -> float:
    diff = mobius_add(-x, y, c)
    arg = np.clip(np.sqrt(c) * np.linalg.norm(diff), 0.0, 1.0 - EPS)
    return float((2.0 / np.sqrt(c)) * np.arctanh(arg))


def poincare_dist_batch(x: np.ndarray, Y: np.ndarray, c: float) -> np.ndarray:
    """Vectorised: x [d] vs Y [K, d] → [K] distances.
    Computes d(x, y_i) = (2/√c) arctanh(√c ‖(−x) ⊕ y_i‖)
    """
    x2 = np.dot(x, x)
    Y2 = np.sum(Y**2, axis=-1)          # [K]
    xY = Y @ x                           # [K]  (= ⟨y_i, x⟩)
    # Möbius addition (−x) ⊕_c y_i:
    # ⟨−x, y_i⟩ = −xY, ‖−x‖² = x2
    neg_xY = -xY
    num = (1 + 2*c*neg_xY + c*Y2)[:, None]*(-x) + (1 - c*x2)*Y   # [K, d]
    denom = (1 + 2*c*neg_xY + c**2 * x2 * Y2)[:, None]             # [K, 1]
    mobius = num / (denom + EPS)
    norms = np.linalg.norm(mobius, axis=-1)                          # [K]
    args = np.clip(np.sqrt(c) * norms, 0.0, 1.0 - EPS)
    return (2.0 / np.sqrt(c)) * np.arctanh(args)


# ---------------------------------------------------------------------------
# Busemann function retrieval scoring
# ---------------------------------------------------------------------------

def busemann_score_batch(
    query: np.ndarray,
    docs:  np.ndarray,
    c:     float = 1.0,
    radius_sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Score each document by the negative Busemann function relative to the
    query's ideal boundary point ξ = query / ‖query‖.

    In the Poincaré ball the Busemann function at ideal point ξ is:

        b_ξ(d) = log(‖d − ξ‖²) − log(1 − c‖d‖²)

    We return  score(d) = −b_ξ(d) = log(1 − c‖d‖²) − log(‖d − ξ‖²)

    so that *higher score = better match*.

    When ``radius_sigma`` is set, a Gaussian radius-proximity penalty is added:

        score(d) += −(‖d‖ − ‖q‖)² / radius_sigma²

    This suppresses ancestor docs (small ‖d‖) and descendant docs (large ‖d‖)
    relative to the query radius, while leaving same-tier peers untouched.
    It is strictly softer than a binary tier filter and handles continuous
    radius distributions (e.g., from trained encoders) gracefully.

    Choosing radius_sigma
    ---------------------
    • ≈ 0.05 : near-hard tier filter, suppresses even story-tier (|Δr|=0.20)
    • ≈ 0.15 : admits well-aligned story-tier docs, suppresses narratives/themes
    • None   : pure directional Busemann (works when trained radii are natural)

    Parameters
    ----------
    query        : (d,) float32   — query Poincaré vector
    docs         : (K, d) float32 — candidate document Poincaré vectors
    c            : float          — ball curvature (must match index)
    radius_sigma : float | None   — std-dev of Gaussian radius penalty

    Returns
    -------
    scores : (K,) float64  — higher is better, unbounded
    """
    query = np.asarray(query, dtype=np.float64).ravel()
    docs  = np.asarray(docs,  dtype=np.float64)

    norm_q = np.linalg.norm(query)
    xi = query / (norm_q + EPS)                          # ideal boundary point

    doc_norms_sq  = np.sum(docs ** 2, axis=-1)           # [K]
    diff          = docs - xi[None, :]                   # [K, d]
    diff_norms_sq = np.sum(diff ** 2, axis=-1)           # [K]

    log_conformal = np.log(np.maximum(1.0 - c * doc_norms_sq, EPS))  # [K]
    log_proximity = np.log(diff_norms_sq + EPS)                        # [K]

    scores = log_conformal - log_proximity                # [K], direction + depth

    if radius_sigma is not None:
        doc_norms = np.sqrt(doc_norms_sq)
        radius_penalty = (doc_norms - norm_q) ** 2 / (radius_sigma ** 2)
        scores -= radius_penalty

    return scores


def busemann_score_multi_xi(
    anchors: np.ndarray,
    docs:    np.ndarray,
    c:       float = 1.0,
) -> np.ndarray:
    """
    Multi-point Busemann score: score each doc by the BEST (max) Busemann
    score across M ideal boundary points derived from anchor vectors.

        score(d) = max_{k=1..M}  [ log(1 − c‖d‖²) − log(‖d − ξ_k‖²) ]

    where ξ_k = anchors[k] / ‖anchors[k]‖.

    Use case — descendant retrieval from a narrative query whose PCA direction
    is the centroid of 10+ diverse content descendants (≈60° off each one).
    Instead of using the narrative's own ξ (which is the centroid direction and
    misses every descendant cluster), pass the narrative's child STORY docs as
    anchors.  Each story doc's ξ_k points into one sub-cluster, so content
    descendants near any story anchor score well.

    Parameters
    ----------
    anchors : (M, d) — M anchor vectors (e.g. story-tier candidates)
    docs    : (K, d) — candidate document Poincaré vectors
    c       : float  — ball curvature

    Returns
    -------
    scores : (K,) float64 — max Busemann score over all anchors, higher = better
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    docs    = np.asarray(docs,    dtype=np.float64)

    anchor_norms = np.linalg.norm(anchors, axis=-1, keepdims=True) + EPS
    xis = anchors / anchor_norms                      # (M, d) unit vectors

    doc_norms_sq   = np.sum(docs ** 2, axis=-1)       # (K,)
    log_conformal  = np.log(np.maximum(1.0 - c * doc_norms_sq, EPS))  # (K,)

    # Compute log_proximity for every (xi_k, doc_j) pair: (M, K)
    # diff[k, j] = doc[j] - xi[k]
    diff_sq = np.sum((docs[None, :, :] - xis[:, None, :]) ** 2, axis=-1)  # (M, K)
    log_proximity = np.log(diff_sq + EPS)                                    # (M, K)

    # per-anchor Busemann scores: (M, K)
    per_anchor = log_conformal[None, :] - log_proximity   # broadcast log_conformal over M

    return per_anchor.max(axis=0)   # (K,) — best anchor for each doc


# ---------------------------------------------------------------------------
# Log map at an arbitrary base point (batched)
# ---------------------------------------------------------------------------

def log_map_x_batch(x: np.ndarray, Y: np.ndarray, c: float) -> np.ndarray:
    """
    Riemannian logarithm map at base point x applied to each row of Y.

        log_x(y) = (2 / λ_x) · arctanh(√c ‖M‖) / (√c ‖M‖)  ·  M

    where  M = (−x) ⊕_c y  (Möbius addition)
    and    λ_x = 2 / (1 − c‖x‖²).

    Parameters
    ----------
    x : (d,)    — base point
    Y : (K, d)  — target points
    c : float

    Returns
    -------
    tangents : (K, d) — tangent vectors at x pointing toward each y in Y
    """
    x  = np.asarray(x, dtype=np.float64).ravel()
    Y  = np.asarray(Y, dtype=np.float64)

    x2   = np.dot(x, x)
    Y2   = np.sum(Y ** 2, axis=-1)       # [K]
    neg_xY = -(Y @ x)                    # [K]

    # Möbius: (−x) ⊕_c y
    num   = (1 + 2*c*neg_xY + c*Y2)[:, None] * (-x) + (1 - c*x2) * Y  # [K, d]
    denom = (1 + 2*c*neg_xY + c**2 * x2 * Y2)[:, None]                  # [K, 1]
    M     = num / (denom + EPS)                                           # [K, d]

    M_norms = np.linalg.norm(M, axis=-1)               # [K]
    args    = np.clip(np.sqrt(c) * M_norms, 0.0, 1.0 - EPS)
    # scale = (2 / lambda_x) * arctanh(sqrt_c * ‖M‖) / (sqrt_c * ‖M‖)
    lambda_x = 2.0 / max(1.0 - c * x2, EPS)
    scale    = (2.0 / lambda_x) * np.arctanh(args) / (np.sqrt(c) * M_norms + EPS)  # [K]

    return M * scale[:, None]                           # [K, d]


# ---------------------------------------------------------------------------
# Entailment cones (Ganea et al. 2018)
# ---------------------------------------------------------------------------

def entailment_violation_batch(
    query: np.ndarray,
    docs:  np.ndarray,
    c:     float = 1.0,
    K_min: float = 0.10,
) -> np.ndarray:
    """
    Measure how much each document violates the entailment cone of the query.

    Following Ganea et al. (2018), the entailment cone at query q has half-angle

        ψ(q) = arcsin(K_min / ‖q‖)

    where K_min is the minimum allowed norm (≈ the shallowest tier radius).

    A document d is *inside* the cone iff the exterior angle

        φ(q, d) = arccos( ⟨−q, log_q(d)⟩ / (‖q‖ · ‖log_q(d)‖) )

    satisfies φ(q, d) ≤ ψ(q).

    Returns
    -------
    violations : (K,) float — non-negative; 0 = perfectly inside cone.
        To use as a retrieval score, negate: higher negated = better.
    """
    query  = np.asarray(query, dtype=np.float64).ravel()
    docs   = np.asarray(docs,  dtype=np.float64)

    norm_q = np.linalg.norm(query)
    psi    = np.arcsin(np.clip(K_min / (norm_q + EPS), 0.0, 1.0))  # half-angle

    tangents      = log_map_x_batch(query, docs, c)        # [K, d] tangent vecs at q
    tangent_norms = np.linalg.norm(tangents, axis=-1) + EPS # [K]

    # Exterior angle: angle between (−query) and the tangent toward d
    neg_q     = -query                                       # [d]
    neg_q_norm = np.linalg.norm(neg_q) + EPS

    cos_theta = (tangents @ neg_q) / (neg_q_norm * tangent_norms)  # [K]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta     = np.arccos(cos_theta)                         # [K]

    return np.maximum(0.0, theta - psi)                      # [K], lower = better


def horoball_score_batch(
    query: np.ndarray,
    docs:  np.ndarray,
    c:     float = 1.0,
) -> np.ndarray:
    """
    Score documents by horoball proximity to the query.

    Two points lie on the same horoball centered at ideal point ξ when their
    Busemann values are equal:  b_ξ(x) = b_ξ(y).

    Score(q, d) = −|b_ξ(d) − b_ξ(q)|,  ξ = q / ‖q‖

    Properties
    ----------
    • d on exactly the same horoball as q → score = 0 (best possible).
    • d too shallow (ancestor docs) → b_ξ(d) < b_ξ(q) → negative score.
    • d in a different direction → b_ξ(d) ≪ b_ξ(q) → large negative score.

    Unlike the entailment cone, horoball proximity works well when the query
    and target documents are at the same hierarchical depth (same tier).
    The entailment cone is better when the query is a broader concept than
    the target (cross-tier hierarchical retrieval).

    Returns (K,) — higher is better, range (−∞, 0].
    """
    query = np.asarray(query, dtype=np.float64).ravel()
    docs  = np.asarray(docs,  dtype=np.float64)

    xi = query / (np.linalg.norm(query) + EPS)

    # Busemann value of query itself: b_ξ(q)
    q_norm_sq   = np.dot(query, query)
    q_diff      = query - xi
    b_q = np.log(np.dot(q_diff, q_diff) + EPS) - np.log(max(1.0 - c * q_norm_sq, EPS))

    # Busemann values of docs: b_ξ(d)
    doc_norms_sq  = np.sum(docs ** 2, axis=-1)           # [K]
    diff          = docs - xi[None, :]                   # [K, d]
    diff_norms_sq = np.sum(diff ** 2, axis=-1)           # [K]
    b_docs = np.log(diff_norms_sq + EPS) - np.log(np.maximum(1.0 - c * doc_norms_sq, EPS))

    return -np.abs(b_docs - b_q)                          # [K], higher = better


def entailment_score_batch(
    query:  np.ndarray,
    docs:   np.ndarray,
    c:      float = 1.0,
    K_min:  float = 0.10,
    alpha:  float = 1.0,
) -> np.ndarray:
    """
    Combined Busemann + horoball proximity score for retrieval.

    score = busemann(q, d) + alpha · horoball_score(q, d)

    Busemann captures direction + depth jointly; horoball proximity adds a
    strong bonus for documents at exactly the same depth level as the query.
    Together they select docs that are deep, directionally aligned, AND at the
    right tier — no hard tier filter required.

    Returns (K,) — higher is better.
    """
    bus  = busemann_score_batch(query, docs, c)    # [K]  direction + depth (no radius penalty here)
    horo = horoball_score_batch(query, docs, c)    # [K]  depth-level alignment
    return bus + alpha * horo                       # [K]


# ---------------------------------------------------------------------------
# Alpha precomputation (conformal factor — avoids repeated norm computation)
# ---------------------------------------------------------------------------

def alpha_precompute_batch(X: np.ndarray, c: float) -> np.ndarray:
    norms_sq = np.sum(X**2, axis=-1)
    return 1.0 / (1.0 - c * norms_sq + EPS)


def alpha_distance_sq(diff_sq: float, alpha_u: float, alpha_v: float) -> float:
    """Fast proxy: monotonic with Poincaré distance, no acosh."""
    return diff_sq * alpha_u * alpha_v


# ---------------------------------------------------------------------------
# Lorentz model
# ---------------------------------------------------------------------------

def poincare_to_lorentz(p: np.ndarray, c: float) -> np.ndarray:
    norm_sq = np.dot(p, p)
    denom = (1.0 - c * norm_sq)
    if abs(denom) < EPS:
        denom = EPS
    x0 = (1.0 + c * norm_sq) / denom
    sqrt_c = np.sqrt(c)
    spatial = 2.0 * sqrt_c * p / denom
    return np.concatenate([[x0], spatial])


def lorentz_to_poincare(x: np.ndarray, c: float) -> np.ndarray:
    x0, spatial = x[0], x[1:]
    return spatial / (x0 + 1.0 / np.sqrt(c) + EPS)


def lorentz_inner(x: np.ndarray, y: np.ndarray) -> float:
    return float(-x[0]*y[0] + np.dot(x[1:], y[1:]))


def project_hyperboloid(x: np.ndarray, c: float) -> np.ndarray:
    spatial = x[1:]
    x0 = np.sqrt(1.0/c + np.dot(spatial, spatial))
    return np.concatenate([[x0], spatial])


# ---------------------------------------------------------------------------
# Einstein midpoint
# ---------------------------------------------------------------------------

def einstein_midpoint(points: Sequence[Sequence[float]], c: float) -> np.ndarray:
    """Closed-form centroid in hyperbolic space. O(nd), no iteration."""
    pts = [np.asarray(p, dtype=np.float32) for p in points]
    if not pts:
        raise ValueError("empty points")
    dim = len(pts[0])
    weighted_sum = np.zeros(dim + 1, dtype=np.float64)
    gamma_sum = 0.0
    for p in pts:
        lor = poincare_to_lorentz(p, c)
        gamma = lor[0]
        gamma_sum += gamma
        weighted_sum += gamma * lor
    if gamma_sum > EPS:
        weighted_sum /= gamma_sum
    on_hyperboloid = project_hyperboloid(weighted_sum, c)
    poincare = lorentz_to_poincare(on_hyperboloid, c)
    return project_to_ball(poincare.astype(np.float32), c)


# ---------------------------------------------------------------------------
# Busemann depth
# ---------------------------------------------------------------------------

def compute_focal_direction(points: np.ndarray, c: float) -> np.ndarray:
    """Light-like focal vector from centroid of points."""
    mean = points.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < EPS:
        focal = np.zeros(len(mean) + 1)
        focal[1] = 1.0
        return focal
    unit = mean / norm
    lorentz_focal = poincare_to_lorentz(unit * (1.0 - EPS), c)
    return lorentz_focal / (abs(lorentz_inner(lorentz_focal, lorentz_focal)) + EPS)**0.5


def busemann_depth_single(vector: np.ndarray, focal: np.ndarray, c: float) -> float:
    lor = poincare_to_lorentz(vector, c)
    inner = lorentz_inner(lor, focal)
    return float(np.log(max(-inner, EPS)))


def compute_busemann_depths(points: np.ndarray, c: float) -> np.ndarray:
    focal = compute_focal_direction(points, c)
    return np.array([busemann_depth_single(p, focal, c) for p in points], dtype=np.float32)


# ---------------------------------------------------------------------------
# Klein disk (fast distance proxy)
# ---------------------------------------------------------------------------

def poincare_to_klein(p: np.ndarray, c: float) -> np.ndarray:
    return 2.0 * p / (1.0 + c * np.dot(p, p))


def poincare_to_klein_batch(P: np.ndarray, c: float) -> np.ndarray:
    norms_sq = np.sum(P**2, axis=-1, keepdims=True)
    return 2.0 * P / (1.0 + c * norms_sq)


def klein_chord_sq(u_klein: np.ndarray, v_klein: np.ndarray) -> float:
    return float(np.dot(u_klein - v_klein, u_klein - v_klein))


# ---------------------------------------------------------------------------
# Gromov delta and curvature selection
# ---------------------------------------------------------------------------

def gromov_delta(vectors: np.ndarray, num_samples: int = 1000) -> float:
    """Estimate Gromov 4-point delta-hyperbolicity."""
    n = len(vectors)
    if n < 4:
        return 0.0

    def _gp(i: int, j: int, w: int) -> float:
        d_ij = poincare_dist(vectors[i], vectors[j], 1.0)
        d_iw = poincare_dist(vectors[i], vectors[w], 1.0)
        d_jw = poincare_dist(vectors[j], vectors[w], 1.0)
        return 0.5 * (d_iw + d_jw - d_ij)

    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(num_samples, 4))
    max_delta = 0.0
    for a, b, c_i, d in idx:
        s1 = _gp(a, b, c_i) + _gp(d, c_i, b)
        s2 = _gp(a, c_i, b) + _gp(d, b, c_i)
        s3 = _gp(a, d, b)   + _gp(c_i, b, d)
        vals = sorted([s1, s2, s3], reverse=True)
        max_delta = max(max_delta, vals[0] - vals[1])
    return max_delta


def auto_curvature(delta: float) -> float:
    if delta < 0.10:
        return 2.0
    if delta < 0.20:
        return 1.0
    if delta < 0.35:
        return 0.5
    return 0.25


