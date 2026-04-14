"""Three retrieval strategies for chatbot query optimization benchmarks.

Strategies:
  - cosine_strategy:    single dense-vector search with optional payload filters
  - hyperbolic_strategy: tangent-space search with optional RRF fusion over dense
  - multi_hop_strategy: sequential per-tier queries traversing parent/child links
"""

import sys
import time
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams, Filter, FieldCondition, MatchValue

sys.path.insert(0, str(Path(__file__).parent.parent))
from tangent import tangent_query
from fusion import rrf_fuse


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_result(points: list) -> dict:
    """Convert a list of Qdrant ScoredPoints to the standardised result dict.

    Extracts from each point's payload:
      - tier       → str (default "")
      - parent_ids → first element (default "")
      - domain     → used as the branch label (default "")
    """
    ids: list[int] = []
    tiers: list[str] = []
    parent_ids: list[str] = []
    branches: list[str] = []
    areas: list[str] = []

    for p in points:
        payload = p.payload or {}
        ids.append(p.id)
        tiers.append(payload.get("tier", ""))
        raw_parents = payload.get("parent_ids", [])
        parent_ids.append(raw_parents[0] if raw_parents else "")
        branches.append(payload.get("domain", ""))
        areas.append(payload.get("area", ""))

    return {
        "ids": ids,
        "tiers": tiers,
        "parent_ids": parent_ids,
        "branches": branches,
        "areas": areas,
    }


def _build_filter(
    tier_filter: str | None,
    exclude_id: int | None,
    parent_filter: str | None,
) -> Filter | None:
    """Build a Qdrant Filter from optional tier, parent, and exclusion params."""
    must: list = []
    must_not: list = []

    if tier_filter is not None:
        must.append(FieldCondition(key="tier", match=MatchValue(value=tier_filter)))

    if parent_filter is not None:
        must.append(FieldCondition(key="parent_ids", match=MatchValue(value=parent_filter)))

    if exclude_id is not None:
        must_not.append(FieldCondition(key="point_id", match=MatchValue(value=exclude_id)))

    if not must and not must_not:
        return None

    return Filter(
        must=must if must else None,
        must_not=must_not if must_not else None,
    )


# ---------------------------------------------------------------------------
# Strategy 1: Cosine (dense vector, single query)
# ---------------------------------------------------------------------------

def cosine_strategy(
    client: QdrantClient,
    collection: str,
    query_vec: np.ndarray,
    limit: int = 10,
    tier_filter: str | None = None,
    exclude_id: int | None = None,
    parent_filter: str | None = None,
) -> dict:
    """Retrieve using the 'dense' named vector with optional payload filters.

    Args:
        client:        Qdrant client instance.
        collection:    Name of the unified collection.
        query_vec:     Dense query embedding (numpy float32 array).
        limit:         Maximum number of results to return.
        tier_filter:   If set, restrict results to this tier value.
        exclude_id:    If set, exclude this point ID from results (self-exclusion).
        parent_filter: If set, restrict results to points with this parent_id.

    Returns:
        Standardised result dict with an additional 'query_count' and
        'latency_ms' key.
    """
    query_filter = _build_filter(tier_filter, exclude_id, parent_filter)

    t0 = time.perf_counter()
    points = client.query_points(
        collection_name=collection,
        query=query_vec.tolist(),
        using="dense",
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=128),
    ).points
    latency_ms = (time.perf_counter() - t0) * 1000.0

    result = _extract_result(points)
    result["query_count"] = 1
    result["latency_ms"] = latency_ms
    return result


# ---------------------------------------------------------------------------
# Strategy 2: Hyperbolic (tangent-space search, optional RRF fusion)
# ---------------------------------------------------------------------------

def hyperbolic_strategy(
    client: QdrantClient,
    collection: str,
    query_vec_dense: np.ndarray,
    query_vec_poincare: np.ndarray,
    curvature: float = 0.25,
    limit: int = 10,
    tier_filter: str | None = None,
    exclude_id: int | None = None,
    parent_filter: str | None = None,
    use_rrf: bool = True,
) -> dict:
    """Retrieve using the tangent-space 'tangent' named vector, optionally fused
    with a dense cosine search via RRF.

    The query Poincare vector is projected to the tangent space at the origin
    (centroid = zeros) before being submitted to the 'tangent' HNSW index.

    Args:
        client:            Qdrant client instance.
        collection:        Name of the unified collection.
        query_vec_dense:   Dense query embedding for the RRF dense arm.
        query_vec_poincare: Query vector in Poincare ball coordinates.
        curvature:         Poincare ball curvature (default 0.25).
        limit:             Maximum number of results to return.
        tier_filter:       Payload tier filter (optional).
        exclude_id:        Self-exclusion by point ID (optional).
        parent_filter:     Payload parent_ids filter (optional).
        use_rrf:           If True, also search 'dense' and fuse with RRF.

    Returns:
        Standardised result dict with 'query_count' (1 or 2) and 'latency_ms'.
    """
    centroid = np.zeros_like(query_vec_poincare, dtype=np.float32)
    tangent_vec = tangent_query(query_vec_poincare, centroid, curvature)

    query_filter = _build_filter(tier_filter, exclude_id, parent_filter)

    t0 = time.perf_counter()

    tangent_points = client.query_points(
        collection_name=collection,
        query=tangent_vec.tolist(),
        using="tangent",
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=128),
    ).points

    if use_rrf:
        dense_points = client.query_points(
            collection_name=collection,
            query=query_vec_dense.tolist(),
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128),
        ).points
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # RRF returns (point_id, score) tuples — we need payload for each id.
        fused_pairs = rrf_fuse(tangent_points, dense_points, k=60, limit=limit)

        # Build lookup for payload by point id
        payload_map: dict[int, dict] = {}
        for p in tangent_points:
            payload_map[p.id] = p.payload or {}
        for p in dense_points:
            if p.id not in payload_map:
                payload_map[p.id] = p.payload or {}

        ids: list[int] = []
        tiers: list[str] = []
        parent_ids: list[str] = []
        branches: list[str] = []

        for pid, _score in fused_pairs:
            payload = payload_map.get(pid, {})
            ids.append(pid)
            tiers.append(payload.get("tier", ""))
            raw_parents = payload.get("parent_ids", [])
            parent_ids.append(raw_parents[0] if raw_parents else "")
            branches.append(payload.get("domain", ""))

        result = {
            "ids": ids,
            "tiers": tiers,
            "parent_ids": parent_ids,
            "branches": branches,
            "query_count": 2,
            "latency_ms": latency_ms,
        }
    else:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = _extract_result(tangent_points)
        result["query_count"] = 1
        result["latency_ms"] = latency_ms

    return result


# ---------------------------------------------------------------------------
# Strategy 3: Multi-hop (sequential per-tier collection traversal)
# ---------------------------------------------------------------------------

def multi_hop_strategy(
    client: QdrantClient,
    tier_collections: dict[str, str],
    query_vec: np.ndarray,
    start_tier: str,
    direction: str = "up",
    limit: int = 10,
    exclude_id: int | None = None,
) -> dict:
    """Retrieve across separate per-tier collections by hopping along hierarchy links.

    Traversal order:
      direction="up":     content → story → narrative  (via parent_ids)
      direction="down":   narrative → story → content  (via child_ids)
      direction="single": only the start_tier collection

    After the first-tier search, the first result's vector is used as the proxy
    query for each subsequent hop.

    Args:
        client:           Qdrant client instance.
        tier_collections: Mapping of tier name → collection name, e.g.
                          {"narrative": "bgc_narratives", "story": "bgc_stories",
                           "content": "bgc_contents"}.
        query_vec:        Initial dense query embedding.
        start_tier:       The tier to begin traversal from.
        direction:        Traversal direction: "up", "down", or "single".
        limit:            Maximum results per tier hop; final results from last hop.
        exclude_id:       Self-exclusion point ID (applied only to first hop).

    Returns:
        Standardised result dict combining results from all hops, with
        'query_count' (total queries issued) and 'latency_ms' (total wall time).
    """
    # Determine traversal order based on direction
    ordered_tiers = _traversal_order(tier_collections, start_tier, direction)

    all_ids: list[int] = []
    all_tiers: list[str] = []
    all_parent_ids: list[str] = []
    all_branches: list[str] = []
    all_areas: list[str] = []
    query_count = 0
    current_vec = query_vec
    # IDs to look up in the next hop (from parent_ids or child_ids)
    next_hop_link_ids: list[str] = []

    t0 = time.perf_counter()
    link_key = "parent_ids" if direction == "up" else "child_ids"

    for hop_idx, tier in enumerate(ordered_tiers):
        coll = tier_collections.get(tier)
        if coll is None:
            continue

        is_first_hop = hop_idx == 0

        if is_first_hop:
            # First hop: vector similarity search
            hop_filter = _build_filter(None, exclude_id, None)
            points = client.query_points(
                collection_name=coll,
                query=current_vec.tolist(),
                using="dense",
                query_filter=hop_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,
                search_params=SearchParams(hnsw_ef=128),
            ).points
            query_count += 1
        elif next_hop_link_ids:
            # Subsequent hops: use parent_ids/child_ids to filter
            link_filter = Filter(should=[
                FieldCondition(key="point_id", match=MatchValue(value=lid))
                for lid in next_hop_link_ids[:20]  # cap to avoid huge filter
            ])
            points = client.query_points(
                collection_name=coll,
                query=current_vec.tolist(),
                using="dense",
                query_filter=link_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,
                search_params=SearchParams(hnsw_ef=128),
            ).points
            query_count += 1
        else:
            # No links found from previous hop — fall back to vector search
            points = client.query_points(
                collection_name=coll,
                query=current_vec.tolist(),
                using="dense",
                limit=limit,
                with_payload=True,
                with_vectors=True,
                search_params=SearchParams(hnsw_ef=128),
            ).points
            query_count += 1

        if not points:
            break

        # Accumulate results
        hop_result = _extract_result(points)
        all_ids.extend(hop_result["ids"])
        all_tiers.extend(hop_result["tiers"])
        all_parent_ids.extend(hop_result["parent_ids"])
        all_branches.extend(hop_result["branches"])
        all_areas.extend(hop_result.get("areas", []))

        # Collect link IDs for next hop
        next_hop_link_ids = []
        for p in points:
            payload = p.payload or {}
            next_hop_link_ids.extend(payload.get(link_key, []))

        # Update proxy vector for fallback
        top_point = points[0]
        vec = top_point.vector
        if isinstance(vec, dict):
            vec = vec.get("dense", next(iter(vec.values())))
        if vec is not None:
            current_vec = np.array(vec, dtype=np.float32)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "ids": all_ids,
        "tiers": all_tiers,
        "parent_ids": all_parent_ids,
        "branches": all_branches,
        "areas": all_areas,
        "query_count": query_count,
        "latency_ms": latency_ms,
    }


def _traversal_order(
    tier_collections: dict[str, str],
    start_tier: str,
    direction: str,
) -> list[str]:
    """Determine the ordered list of tiers to traverse.

    If the tier_collections keys follow a natural hierarchy order (narrative >
    story > content), this infers that order. Unknown tiers fall back to
    key-insertion order.
    """
    if direction == "single":
        return [start_tier]

    # Canonical hierarchy top-to-bottom
    canonical = ["narrative", "story", "content"]
    known = [t for t in canonical if t in tier_collections]
    remaining = [t for t in tier_collections if t not in canonical]
    ordered = known + remaining  # top → bottom

    if start_tier not in ordered:
        # Fallback: just use all tiers in insertion order
        ordered = list(tier_collections.keys())

    start_idx = ordered.index(start_tier)

    if direction == "up":
        # content (bottom) → narrative (top): reverse from start upward
        return list(reversed(ordered[: start_idx + 1]))
    elif direction == "down":
        # narrative (top) → content (bottom): forward from start downward
        return ordered[start_idx:]
    else:
        raise ValueError(f"Unknown direction '{direction}'; expected 'up', 'down', or 'single'.")
