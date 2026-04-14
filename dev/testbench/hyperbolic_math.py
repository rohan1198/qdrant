"""Shared hyperbolic geometry math for the testbench.

Poincare ball and Lorentz hyperboloid operations used by both
embed.py (projection) and benchmark.py (Busemann scoring).
"""

import numpy as np

EPS = 1e-5


# ---------------------------------------------------------------------------
# Poincare ball operations
# ---------------------------------------------------------------------------


def exp_map_at_origin(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Map a tangent vector at the origin to the Poincare ball."""
    norm_v = np.linalg.norm(v)
    if norm_v < EPS:
        return v.copy()
    sqrt_c = np.sqrt(c)
    coeff = np.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
    return coeff * v


def project_to_ball(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Clip a point to strictly inside the Poincare ball."""
    norm = np.linalg.norm(x)
    if norm < EPS:
        return x
    boundary = (1.0 / np.sqrt(c)) - EPS
    limit = min(0.95, boundary)
    if norm > limit:
        return x * (limit / norm)
    return x


# ---------------------------------------------------------------------------
# Lorentz hyperboloid operations
# ---------------------------------------------------------------------------


def poincare_to_lorentz(p: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Convert Poincare ball point to Lorentz hyperboloid (d -> d+1 dims).

    x_0 = (1 + c||p||^2) / (1 - c||p||^2)
    x_i = 2*sqrt(c)*p_i / (1 - c||p||^2)
    """
    norm_sq = float(np.sum(p ** 2))
    denom = max(1.0 - c * norm_sq, EPS)
    x0 = (1.0 + c * norm_sq) / denom
    spatial = 2.0 * np.sqrt(c) * p / denom
    return np.concatenate([[x0], spatial])


def lorentz_to_poincare(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Convert Lorentz hyperboloid point to Poincare ball (d+1 -> d dims).

    Inverse of poincare_to_lorentz. The forward map scales spatial components
    by 2*sqrt(c), so the inverse must divide by sqrt(c):
        p_i = x_i / (sqrt(c) * (x_0 + 1))
    """
    x0 = x[0]
    denom = max(np.sqrt(c) * (x0 + 1.0), EPS)
    return x[1:] / denom


def lorentz_inner(x: np.ndarray, y: np.ndarray) -> float:
    """Lorentz inner product: <x,y>_L = -x0*y0 + x1*y1 + ... + xn*yn."""
    return float(-x[0] * y[0] + np.dot(x[1:], y[1:]))


def project_hyperboloid(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Project a Lorentz-space vector onto the hyperboloid.

    Sets x0 = sqrt(1/c + ||x_spatial||^2) so that <x,x>_L = -1/c.
    """
    space_norm_sq = float(np.sum(x[1:] ** 2))
    x0 = np.sqrt(max(1.0 / c + space_norm_sq, EPS))
    return np.concatenate([[x0], x[1:]])


# ---------------------------------------------------------------------------
# Busemann scoring
# ---------------------------------------------------------------------------


def busemann_score(x_lorentz: np.ndarray, focal: np.ndarray) -> float:
    """Busemann function: B_xi(x) = log(-<x, xi>_L)."""
    inner = lorentz_inner(x_lorentz, focal)
    return float(np.log(max(-inner, EPS)))


def compute_focal_direction(
    points: list[np.ndarray], c: float = 1.0
) -> np.ndarray:
    """Compute focal direction (light-like vector) from a set of Poincare points."""
    lorentz_points = [poincare_to_lorentz(p, c) for p in points]
    mean = np.mean(lorentz_points, axis=0)
    spatial = mean[1:]
    spatial_norm = max(np.linalg.norm(spatial), EPS)
    spatial_unit = spatial / spatial_norm
    focal = np.zeros(len(mean))
    focal[0] = 1.0
    focal[1:] = -spatial_unit
    return focal


def compute_busemann_depths(
    points: list[dict], c: float = 1.0
) -> list[float]:
    """Return Busemann depth for each point dict (requires 'vector' key)."""
    vectors = [pt["vector"] for pt in points]
    focal = compute_focal_direction(vectors, c=c)
    lorentz_vecs = [poincare_to_lorentz(v, c=c) for v in vectors]
    return [busemann_score(lv, focal) for lv in lorentz_vecs]


def busemann_depth_single(
    vector: np.ndarray, focal: np.ndarray, c: float = 1.0
) -> float:
    """Busemann depth for a single Poincare ball vector given a focal direction."""
    lorentz = poincare_to_lorentz(vector, c)
    return busemann_score(lorentz, focal)


# ---------------------------------------------------------------------------
# Einstein midpoint
# ---------------------------------------------------------------------------


def einstein_midpoint(
    points_poincare: list[np.ndarray], c: float = 1.0
) -> np.ndarray:
    """Closed-form centroid via Lorentz hyperboloid (Einstein midpoint).

    1. Convert each Poincare point to Lorentz coordinates.
    2. Weight by Lorentz gamma (x_0 component).
    3. Average and project back to hyperboloid.
    4. Convert back to Poincare.

    Returns a Poincare ball point.
    """
    if not points_poincare:
        raise ValueError("cannot compute midpoint of empty set")

    dim = len(points_poincare[0])
    weighted_sum = np.zeros(dim + 1, dtype=np.float64)
    gamma_sum = 0.0

    for p in points_poincare:
        lorentz = poincare_to_lorentz(p, c)
        gamma = float(lorentz[0])
        gamma_sum += gamma
        weighted_sum += gamma * lorentz

    if gamma_sum > EPS:
        weighted_sum /= gamma_sum

    on_hyperboloid = project_hyperboloid(weighted_sum, c)
    poincare = lorentz_to_poincare(on_hyperboloid, c)
    return project_to_ball(poincare.astype(np.float32), c)


# ---------------------------------------------------------------------------
# Gromov delta-hyperbolicity
# ---------------------------------------------------------------------------


def gromov_delta(vectors, num_samples=1000):
    """Compute Gromov delta-hyperbolicity via 4-point condition.

    Lower delta = more tree-like = better for hyperbolic geometry.
    Returns (delta, recommendation).

    Source: HyperspaceDB gromov.rs
    """
    n = len(vectors)
    if n < 4:
        return 0.0, "insufficient_data"

    max_delta = 0.0
    rng = np.random.default_rng(42)

    for _ in range(num_samples):
        idx = rng.choice(n, 4, replace=False)
        x, y, u, v = vectors[idx[0]], vectors[idx[1]], vectors[idx[2]], vectors[idx[3]]

        d_xy = np.linalg.norm(x - y)
        d_uv = np.linalg.norm(u - v)
        d_xu = np.linalg.norm(x - u)
        d_yv = np.linalg.norm(y - v)
        d_xv = np.linalg.norm(x - v)
        d_yu = np.linalg.norm(y - u)

        sums = sorted([d_xy + d_uv, d_xu + d_yv, d_xv + d_yu], reverse=True)
        delta = (sums[0] - sums[1]) / 2.0
        max_delta = max(max_delta, delta)

    if max_delta < 0.15:
        rec = "lorentz"
    elif max_delta < 0.30:
        rec = "poincare"
    elif max_delta < 0.50:
        rec = "cosine"
    else:
        rec = "l2"

    return max_delta, rec


# ---------------------------------------------------------------------------
# Alpha precomputation (conformal factor)
# ---------------------------------------------------------------------------


def alpha_precompute(vector, c=1.0):
    """Precompute conformal factor 1/(1 - c*||x||^2) for fast Poincare distance.

    Store this as a payload alongside each vector. Reduces distance
    computation cost by ~30% since the norm doesn't need recomputing.

    Source: HyperspaceDB vector.rs
    """
    sq_norm = np.dot(vector, vector)
    denom = max(1.0 - c * sq_norm, 1e-7)
    return 1.0 / denom


def alpha_precompute_batch(vectors, c=1.0):
    """Batch alpha precomputation for a matrix of vectors."""
    sq_norms = np.sum(vectors ** 2, axis=1)
    denoms = np.maximum(1.0 - c * sq_norms, 1e-7)
    return 1.0 / denoms


# ---------------------------------------------------------------------------
# Fused norms and fast Poincare distance
# ---------------------------------------------------------------------------


def fused_norms(u, v):
    """Compute ||u-v||^2, ||u||^2, ||v||^2 in a single pass.

    Avoids three separate np.dot calls. Most useful in loops;
    for batch operations use vectorized numpy instead.

    Source: RuVector poincare.rs
    """
    diff = u - v
    return np.dot(diff, diff), np.dot(u, u), np.dot(v, v)


def poincare_distance_with_alpha(diff_sq, alpha_u, alpha_v, c=1.0):
    """Poincare distance from precomputed norms and alphas.

    d(u,v) = (1/sqrt(c)) * acosh(1 + 2*c * diff_sq * alpha_u * alpha_v)

    Use with fused_norms() and alpha_precompute() for maximum speed.
    """
    sqrt_c = np.sqrt(c)
    arg = 1.0 + 2.0 * c * diff_sq * alpha_u * alpha_v
    return (1.0 / sqrt_c) * np.arccosh(np.clip(arg, 1.0, None))


# ---------------------------------------------------------------------------
# Klein disk model
# ---------------------------------------------------------------------------

def poincare_to_klein(vector, c=1.0):
    """Convert Poincaré ball point to Klein disk coordinates.

    Klein geodesics are Euclidean straight lines (chords), making
    Klein-space distance a cheap L2 proxy for Poincaré distance.

    k_i = 2 * p_i / (1 + c * ||p||^2)

    Source: HyperspaceDB vector.rs
    """
    sq_norm = np.dot(vector, vector)
    return 2.0 * vector / (1.0 + c * sq_norm)


def poincare_to_klein_batch(vectors, c=1.0):
    """Batch Klein conversion for a matrix of vectors."""
    sq_norms = np.sum(vectors ** 2, axis=1, keepdims=True)
    return 2.0 * vectors / (1.0 + c * sq_norms)


def klein_chord_distance_sq(u_klein, v_klein):
    """Squared Euclidean distance in Klein disk (chord distance).

    This is a fast proxy for Poincaré distance — no acosh needed.
    Preserves ordering for nearby points.
    """
    diff = u_klein - v_klein
    return np.dot(diff, diff)


# ---------------------------------------------------------------------------
# Geometric filters (client-side post-filtering)
# ---------------------------------------------------------------------------

def inball_filter(candidates, center, radius, curvature=1.0):
    """Keep candidates within Poincaré distance `radius` of `center`.

    Args:
        candidates: list of dicts with 'vector' key (Poincaré coordinates)
        center: np.ndarray, center point in Poincaré ball
        radius: float, maximum Poincaré distance from center
        curvature: float, ball curvature

    Returns:
        Filtered list of candidates
    """
    center_sq = np.dot(center, center)
    alpha_c = 1.0 / max(1.0 - curvature * center_sq, 1e-7)
    sqrt_c = np.sqrt(curvature)

    result = []
    for cand in candidates:
        v = cand["vector"]
        diff = center - v
        diff_sq = np.dot(diff, diff)
        v_sq = np.dot(v, v)
        alpha_v = 1.0 / max(1.0 - curvature * v_sq, 1e-7)
        arg = 1.0 + 2.0 * curvature * diff_sq * alpha_c * alpha_v
        dist = np.arccosh(max(arg, 1.0)) / sqrt_c
        if dist <= radius:
            result.append(cand)
    return result


def incone_filter(candidates, axis, aperture, origin=None):
    """Keep candidates within angular aperture of axis direction from origin.

    Measures the angle between the direction from origin to each candidate
    and the axis direction. Keeps candidates within the aperture.

    Args:
        candidates: list of dicts with 'vector' key
        axis: np.ndarray, direction vector (will be normalized)
        aperture: float, half-angle in radians
        origin: np.ndarray or None (defaults to zero vector)

    Returns:
        Filtered list of candidates
    """
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return candidates

    axis_unit = axis / axis_norm

    result = []
    for cand in candidates:
        v = cand["vector"]
        if origin is not None:
            v = v - origin

        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue

        cos_angle = np.dot(v / v_norm, axis_unit)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        if angle <= aperture:
            result.append(cand)
    return result


# ---------------------------------------------------------------------------
# Logarithmic maps and Frechet mean
# ---------------------------------------------------------------------------

def log_map_origin(p: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Logarithmic map at the origin (fast path, no Mobius ops).

    Formula: log_0(p) = (1/sqrt(c)) * arctanh(sqrt(c) * ||p||) * p / ||p||

    At the origin lambda_p = 2 and Mobius addition simplifies to identity,
    so the general log_map formula reduces to (2/(sc*lambda_p)) = (2/(sc*2))
    = 1/sc.  ~3x cheaper than general log_map.
    """
    sc = np.sqrt(c)
    norm = np.linalg.norm(p)
    if norm < EPS:
        return np.zeros_like(p)
    sc_norm = min(sc * norm, 1.0 - EPS)  # clamp for arctanh domain
    coeff = (1.0 / sc) * np.arctanh(sc_norm) / norm
    return p * coeff


def log_map(base: np.ndarray, target: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Logarithmic map at arbitrary base point.

    Formula: log_p(y) = (2/(sqrt(c)*lambda_p)) * arctanh(sqrt(c)*||-p +_c y||)
                        * (-p +_c y) / ||-p +_c y||
    """
    sc = np.sqrt(c)

    # Mobius negation + addition: -p +_c y
    neg_p = -base
    neg_p_sq = np.dot(neg_p, neg_p)
    y_sq = np.dot(target, target)
    neg_p_dot_y = np.dot(neg_p, target)

    num_left = (1.0 + 2.0 * c * neg_p_dot_y + c * y_sq)
    num_right = (1.0 - c * neg_p_sq)
    denom = 1.0 + 2.0 * c * neg_p_dot_y + c**2 * neg_p_sq * y_sq
    denom = max(denom, EPS)

    add = (num_left * neg_p + num_right * target) / denom

    add_norm = np.linalg.norm(add)
    if add_norm < EPS:
        return np.zeros_like(base)

    base_sq = np.dot(base, base)
    lambda_p = 2.0 / max(1.0 - c * base_sq, EPS)

    sc_add_norm = min(sc * add_norm, 1.0 - EPS)
    coeff = (2.0 / (sc * lambda_p)) * np.arctanh(sc_add_norm) / add_norm

    return add * coeff


def frechet_mean(points: list[np.ndarray], c: float = 1.0,
                 max_iter: int = 100, lr: float = 0.1, tol: float = 1e-6) -> np.ndarray:
    """Iterative Frechet mean on the Poincare ball via Riemannian gradient descent."""
    dim = len(points[0])
    n = len(points)

    mean = np.mean(points, axis=0)
    mean = project_to_ball(mean, c)

    for _ in range(max_iter):
        grad = np.zeros(dim)
        for p in points:
            lm = log_map(mean, p, c)
            grad += lm
        grad /= n

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        step = grad * lr
        mean = _exp_map_at(mean, step, c)
        mean = project_to_ball(mean, c)

    return mean


def _exp_map_at(base: np.ndarray, v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Exponential map at arbitrary base point (tangent vector v -> manifold)."""
    sc = np.sqrt(c)
    v_norm = np.linalg.norm(v)
    if v_norm < EPS:
        return base.copy()

    base_sq = np.dot(base, base)
    lambda_p = 2.0 / max(1.0 - c * base_sq, EPS)

    tanh_arg = sc * lambda_p * v_norm / 2.0
    direction = v / v_norm
    second = np.tanh(tanh_arg) * direction / sc

    b_sq = np.dot(base, base)
    s_sq = np.dot(second, second)
    b_dot_s = np.dot(base, second)

    num_left = (1.0 + 2.0 * c * b_dot_s + c * s_sq)
    num_right = (1.0 - c * b_sq)
    denom = 1.0 + 2.0 * c * b_dot_s + c**2 * b_sq * s_sq
    denom = max(denom, EPS)

    return (num_left * base + num_right * second) / denom


def tangent_distance_sq(u: np.ndarray, v: np.ndarray) -> float:
    """Euclidean L2 squared distance between tangent space vectors."""
    diff = u - v
    return float(np.dot(diff, diff))
