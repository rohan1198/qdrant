import math

"""Fusion strategies for dual-space search.

Combines results from cosine and poincare search into a single ranking.
Four strategies available:
  - rrf_fuse: Reciprocal Rank Fusion (rank-based, baseline)
  - linear_alpha_fuse: Score-normalized alpha blending
  - busemann_weighted_fuse: Alpha blend weighted by Busemann depth proximity
  - depth_band_fuse: Hard depth-band filter + alpha blend
"""


def rrf_fuse(
    results_a: list,
    results_b: list,
    k: int = 60,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of two result lists.

    Each result is expected to have an `.id` attribute (qdrant ScoredPoint).

    Args:
        results_a: First ranked list (e.g., cosine results).
        results_b: Second ranked list (e.g., poincare results).
        k: RRF constant (default 60, robust across scales).
        limit: Number of fused results to return.

    Returns:
        List of (point_id, rrf_score) tuples, sorted by descending score.
    """
    scores: dict[int, float] = {}

    for rank, point in enumerate(results_a):
        pid = point.id if hasattr(point, "id") else point
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

    for rank, point in enumerate(results_b):
        pid = point.id if hasattr(point, "id") else point
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:limit]


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] via min-max scaling."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-9:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _extract_scores(results: list) -> tuple[list[int], list[float]]:
    """Extract (ids, scores) from qdrant ScoredPoint results."""
    ids = []
    scores = []
    for r in results:
        pid = r.id if hasattr(r, "id") else r
        score = r.score if hasattr(r, "score") else 0.0
        ids.append(pid)
        scores.append(float(score))
    return ids, scores


def linear_alpha_fuse(
    results_a: list,
    results_b: list,
    alpha: float = 0.5,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Linear alpha fusion with score normalization.

    fused_score = alpha * norm_score_a + (1 - alpha) * norm_score_b

    Both score lists are min-max normalized to [0, 1] before blending.
    Higher alpha = more weight on results_a (cosine).
    """
    ids_a, scores_a = _extract_scores(results_a)
    ids_b, scores_b = _extract_scores(results_b)

    norm_a = _min_max_normalize(scores_a)
    norm_b = _min_max_normalize(scores_b)

    combined: dict[int, float] = {}
    for pid, s in zip(ids_a, norm_a):
        combined[pid] = combined.get(pid, 0.0) + alpha * s
    for pid, s in zip(ids_b, norm_b):
        combined[pid] = combined.get(pid, 0.0) + (1 - alpha) * s

    ranked = sorted(combined.items(), key=lambda x: -x[1])
    return ranked[:limit]


def busemann_weighted_fuse(
    results_a: list,
    results_b: list,
    query_depth: float,
    id_to_depth: dict[int, float],
    alpha: float = 0.5,
    lam: float = 1.0,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Alpha fusion weighted by Busemann depth proximity to query.

    depth_weight = exp(-lambda * |depth(result) - depth(query)|)
    fused_score = depth_weight * (alpha * norm_a + (1 - alpha) * norm_b)

    Points at similar depth to query get boosted.
    """
    import math

    ids_a, scores_a = _extract_scores(results_a)
    ids_b, scores_b = _extract_scores(results_b)

    norm_a = _min_max_normalize(scores_a)
    norm_b = _min_max_normalize(scores_b)

    # Collect alpha-blended base scores
    base_scores: dict[int, float] = {}
    for pid, s in zip(ids_a, norm_a):
        base_scores[pid] = base_scores.get(pid, 0.0) + alpha * s
    for pid, s in zip(ids_b, norm_b):
        base_scores[pid] = base_scores.get(pid, 0.0) + (1 - alpha) * s

    # Apply depth weighting
    combined: dict[int, float] = {}
    for pid, base in base_scores.items():
        d = id_to_depth.get(pid)
        if d is not None:
            weight = math.exp(-lam * abs(d - query_depth))
        else:
            weight = 1.0
        combined[pid] = base * weight

    ranked = sorted(combined.items(), key=lambda x: -x[1])
    return ranked[:limit]


def depth_band_fuse(
    results_a: list,
    results_b: list,
    query_depth: float,
    id_to_depth: dict[int, float],
    bandwidth: float = 0.5,
    alpha: float = 0.5,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Alpha fusion with hard depth-band pre-filter.

    Only results within [query_depth - bandwidth, query_depth + bandwidth]
    are considered. Then standard alpha fusion on survivors.
    """
    lo = query_depth - bandwidth
    hi = query_depth + bandwidth

    def _in_band(r):
        pid = r.id if hasattr(r, "id") else r
        d = id_to_depth.get(pid)
        return d is not None and lo <= d <= hi

    filtered_a = [r for r in results_a if _in_band(r)]
    filtered_b = [r for r in results_b if _in_band(r)]

    return linear_alpha_fuse(filtered_a, filtered_b, alpha=alpha, limit=limit)


def busemann_depth_proximity_fuse(
    results_a: list,
    results_b: list,
    query_depth: float,
    id_to_depth: dict[int, float],
    alpha: float = 0.5,
    depth_weight: float = 1.0,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Busemann fusion with depth proximity weighting.

    Enhanced version: results at the same hierarchy depth as the query are
    weighted higher. Uses exponential decay based on Busemann depth distance.

    Args:
        results_a: first result set (e.g., cosine)
        results_b: second result set (e.g., Poincare)
        query_depth: Busemann depth of the query point
        id_to_depth: mapping from point ID to Busemann depth
        alpha: blend weight (0=all_b, 1=all_a)
        depth_weight: steepness of depth proximity decay
        limit: max results to return
    """
    ids_a, scores_a = _extract_scores(results_a)
    ids_b, scores_b = _extract_scores(results_b)

    norm_a = _min_max_normalize(scores_a)
    norm_b = _min_max_normalize(scores_b)

    score_map_a = dict(zip(ids_a, norm_a))
    score_map_b = dict(zip(ids_b, norm_b))

    all_ids = set(ids_a) | set(ids_b)
    fused = []
    for pid in all_ids:
        sa = score_map_a.get(pid, 0.0)
        sb = score_map_b.get(pid, 0.0)
        cand_depth = id_to_depth.get(pid, query_depth)
        depth_proximity = math.exp(-abs(cand_depth - query_depth) * depth_weight)
        score = alpha * sa + (1.0 - alpha) * sb * depth_proximity
        fused.append((pid, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:limit]


def horosphere_score(
    candidates: list[tuple[int, float]],
    query_depth: float,
    id_to_depth: dict[int, float],
    intent: str = "auto",
    steepness: float = 1.0,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Re-weight candidates by horosphere (hierarchy depth) preference.

    Horospheres are level sets of the Busemann function — hyperbolic "floors"
    at a given depth.

    Args:
        candidates: list of (point_id, fused_score) tuples
        query_depth: Busemann depth of the query point
        id_to_depth: mapping from point ID to Busemann depth
        intent: "ancestors" (prefer shallower), "descendants" (prefer deeper),
                "siblings" (prefer same depth), "auto" (defaults to siblings)
        steepness: how aggressively to penalize wrong-depth results
        limit: max results to return
    """
    rescored = []
    for pid, score in candidates:
        cand_depth = id_to_depth.get(pid, query_depth)
        depth_delta = cand_depth - query_depth

        if intent == "ancestors":
            horo_weight = math.exp(-max(depth_delta, 0.0) * steepness)
        elif intent == "descendants":
            horo_weight = math.exp(min(depth_delta, 0.0) * steepness)
        elif intent in ("siblings", "auto"):
            horo_weight = math.exp(-abs(depth_delta) * steepness)
        else:
            horo_weight = 1.0

        rescored.append((pid, score * horo_weight))

    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:limit]
