#!/usr/bin/env python3
"""Chatbot query implementations for the testbench.

Five query patterns that mirror real Pythia/Lens chatbot usage:
1. drill_down   — vertical traversal (narrative → stories → content)
2. lateral      — horizontal search (same-tier related items)
3. cross_branch — structural similarity across domains
4. temporal     — time-windowed search within tier
5. depth_band   — Busemann depth-filtered search
"""
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Range, SearchParams,
)
from hyperbolic_math import alpha_precompute, fused_norms, poincare_distance_with_alpha
from tangent import tangent_query


def drill_down_query(client, collection, start_point_id, using="poincare", limit=10, ef=128):
    """Vertical traversal: start point → children (hop 1) → grandchildren (hop 2).

    Retrieves the start point, then searches for items whose parent_ids contains
    the start point's point_id. For the top-5 hop-1 results, performs a second
    hop searching for their children.

    Returns:
        dict with hop1_results, hop2_results, hop1_latency_ms, hop2_latency_ms,
        total_latency_ms — or {"error": str} on failure.
    """
    try:
        t_start = time.time()

        # Retrieve start point with vectors and payload
        retrieved = client.retrieve(
            collection_name=collection,
            ids=[start_point_id],
            with_vectors=True,
            with_payload=True,
        )
        if not retrieved:
            return {"error": f"Point {start_point_id} not found in collection {collection}"}

        start_point = retrieved[0]
        start_point_id_str = start_point.payload.get("point_id", str(start_point_id))

        # Extract the named vector for hop-1 search
        vec = start_point.vector
        if isinstance(vec, dict):
            query_vector = vec.get(using)
        else:
            query_vector = vec

        if query_vector is None:
            return {"error": f"Named vector '{using}' not found on point {start_point_id}"}

        # Convert numpy arrays to plain lists
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        # Hop 1: search for children (items whose parent_ids contains start_point_id_str)
        t_hop1_start = time.time()
        hop1_response = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=using,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_ids",
                        match=MatchValue(value=start_point_id_str),
                    )
                ]
            ),
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=False,
        )
        hop1_latency_ms = (time.time() - t_hop1_start) * 1000.0
        hop1_results = hop1_response.points

        # Hop 2: for top-5 hop-1 results, search for their children
        t_hop2_start = time.time()
        hop2_results = []
        for parent_point in hop1_results[:5]:
            parent_point_id_str = parent_point.payload.get("point_id", str(parent_point.id))

            hop2_response = client.query_points(
                collection_name=collection,
                query=query_vector,
                using=using,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_ids",
                            match=MatchValue(value=parent_point_id_str),
                        )
                    ]
                ),
                limit=limit,
                search_params=SearchParams(hnsw_ef=ef),
                with_payload=True,
                with_vectors=False,
            )
            hop2_results.extend(hop2_response.points)
        hop2_latency_ms = (time.time() - t_hop2_start) * 1000.0

        total_latency_ms = (time.time() - t_start) * 1000.0

        return {
            "hop1_results": hop1_results,
            "hop2_results": hop2_results,
            "hop1_latency_ms": hop1_latency_ms,
            "hop2_latency_ms": hop2_latency_ms,
            "total_latency_ms": total_latency_ms,
        }
    except Exception as e:
        return {"error": str(e)}


def lateral_query(client, collection, query_point_id, using="dense", limit=10, ef=128):
    """Horizontal search: find related items within the same tier.

    Retrieves the query point to determine its tier, then searches for other
    items at the same tier using the specified named vector, excluding self.

    Returns:
        dict with results, query_tier, query_domain, query_area, latency_ms
        — or {"error": str} on failure.
    """
    try:
        t_start = time.time()

        # Retrieve query point with vectors and payload
        retrieved = client.retrieve(
            collection_name=collection,
            ids=[query_point_id],
            with_vectors=True,
            with_payload=True,
        )
        if not retrieved:
            return {"error": f"Point {query_point_id} not found in collection {collection}"}

        query_point = retrieved[0]
        payload = query_point.payload or {}
        query_tier = payload.get("tier")
        query_domain = payload.get("domain")
        query_area = payload.get("area")

        # Extract the named vector
        vec = query_point.vector
        if isinstance(vec, dict):
            query_vector = vec.get(using)
        else:
            query_vector = vec

        if query_vector is None:
            return {"error": f"Named vector '{using}' not found on point {query_point_id}"}

        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        # Build filter: same tier, exclude self
        must_conditions = []
        if query_tier is not None:
            must_conditions.append(
                FieldCondition(key="tier", match=MatchValue(value=query_tier))
            )

        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=using,
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit + 1,  # fetch one extra to allow self-exclusion
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=False,
        )

        # Exclude the query point itself
        results = [r for r in response.points if r.id != query_point_id][:limit]

        latency_ms = (time.time() - t_start) * 1000.0

        return {
            "results": results,
            "query_tier": query_tier,
            "query_domain": query_domain,
            "query_area": query_area,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        return {"error": str(e)}


def cross_branch_query(client, collection, narrative_point_id, using="poincare", limit=10, ef=128):
    """Structural similarity across domains: find narratives in other domains.

    Retrieves the narrative point to get its domain, then searches for narrative-
    type items in different domains using must_not to exclude the source domain.

    Returns:
        dict with results, source_domain, latency_ms — or {"error": str} on failure.
    """
    try:
        t_start = time.time()

        # Retrieve narrative point with vectors and payload
        retrieved = client.retrieve(
            collection_name=collection,
            ids=[narrative_point_id],
            with_vectors=True,
            with_payload=True,
        )
        if not retrieved:
            return {"error": f"Point {narrative_point_id} not found in collection {collection}"}

        narrative_point = retrieved[0]
        payload = narrative_point.payload or {}
        source_domain = payload.get("domain")

        # Extract the named vector
        vec = narrative_point.vector
        if isinstance(vec, dict):
            query_vector = vec.get(using)
        else:
            query_vector = vec

        if query_vector is None:
            return {"error": f"Named vector '{using}' not found on point {narrative_point_id}"}

        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        # Filter: item_type == "narrative" AND domain != source_domain
        must_conditions = [
            FieldCondition(key="item_type", match=MatchValue(value="narrative"))
        ]
        must_not_conditions = []
        if source_domain is not None:
            must_not_conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=source_domain))
            )

        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=using,
            query_filter=Filter(
                must=must_conditions,
                must_not=must_not_conditions if must_not_conditions else None,
            ),
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=False,
        )

        latency_ms = (time.time() - t_start) * 1000.0

        return {
            "results": response.points,
            "source_domain": source_domain,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        return {"error": str(e)}


def temporal_query(client, collection, query_vector, date_gte, date_lt, tier=None, using="dense", limit=10, ef=128):
    """Time-windowed search within an optional tier.

    Searches for items within a date range [date_gte, date_lt), optionally
    restricted to a specific tier. query_vector is passed directly (not looked
    up from a point).

    Args:
        query_vector: vector to search with (list or numpy array)
        date_gte: lower bound for created_at (inclusive), ISO-8601 string or numeric
        date_lt:  upper bound for created_at (exclusive), ISO-8601 string or numeric
        tier:     optional tier value to filter by
        using:    named vector to search with

    Returns:
        dict with results, latency_ms — or {"error": str} on failure.
    """
    try:
        t_start = time.time()

        # Convert numpy arrays to plain lists
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        # Build filter conditions
        must_conditions = [
            FieldCondition(
                key="created_at",
                range=Range(gte=date_gte, lt=date_lt),
            )
        ]
        if tier is not None:
            must_conditions.append(
                FieldCondition(key="tier", match=MatchValue(value=tier))
            )

        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=using,
            query_filter=Filter(must=must_conditions),
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=False,
        )

        latency_ms = (time.time() - t_start) * 1000.0

        return {
            "results": response.points,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        return {"error": str(e)}


def depth_band_query(client, collection, query_vector, depth_min, depth_max, using="poincare", limit=10, ef=128):
    """Busemann depth-filtered search.

    Searches for items within a Busemann depth range [depth_min, depth_max]
    using the specified named vector. query_vector is passed directly.

    Args:
        query_vector: vector to search with (list or numpy array)
        depth_min:    lower bound for busemann_depth (inclusive)
        depth_max:    upper bound for busemann_depth (inclusive)
        using:        named vector to search with

    Returns:
        dict with results, latency_ms — or {"error": str} on failure.
    """
    try:
        t_start = time.time()

        # Convert numpy arrays to plain lists
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=using,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="busemann_depth",
                        range=Range(gte=depth_min, lte=depth_max),
                    )
                ]
            ),
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=False,
        )

        latency_ms = (time.time() - t_start) * 1000.0

        return {
            "results": response.points,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        return {"error": str(e)}


def klein_prefilter_query(client, collection, query_poincare, curvature=5.0,
                          prefetch_limit=200, klein_top=50, final_top=10,
                          using_prefetch="dense", ef=128):
    """Three-stage search: cosine prefetch → Klein pre-rank → Poincaré re-rank.

    Stage 1: Fast HNSW search using cosine named vector (server-side)
    Stage 2: Re-rank candidates by Klein chord distance (client-side, cheap)
    Stage 3: Re-rank survivors by exact Poincaré distance (client-side, expensive)
    """
    import numpy as np
    from hyperbolic_math import (poincare_to_klein, klein_chord_distance_sq,
                                  poincare_distance_with_alpha, alpha_precompute, fused_norms)
    try:
        t0 = time.time()

        # Convert query for prefetch
        query_list = query_poincare.tolist() if hasattr(query_poincare, "tolist") else list(query_poincare)

        # Stage 1: Cosine prefetch from Qdrant
        prefetch_results = client.query_points(
            collection_name=collection,
            query=query_list,
            using=using_prefetch,
            limit=prefetch_limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=True,
        ).points
        t1 = time.time()

        # Stage 2: Klein pre-rank
        query_klein = poincare_to_klein(np.array(query_poincare), c=curvature)
        scored = []
        for r in prefetch_results:
            vec = r.vector
            if isinstance(vec, dict):
                vec = vec.get("poincare", next(iter(vec.values())))
            if vec is None:
                continue
            poincare_vec = np.array(vec, dtype=np.float32)
            klein_vec = poincare_to_klein(poincare_vec, c=curvature)
            klein_dist = klein_chord_distance_sq(query_klein, klein_vec)
            scored.append((r, klein_dist, poincare_vec))
        scored.sort(key=lambda x: x[1])
        klein_survivors = scored[:klein_top]
        t2 = time.time()

        # Stage 3: Poincaré re-rank
        query_np = np.array(query_poincare, dtype=np.float32)
        query_alpha = alpha_precompute(query_np, c=curvature)
        poincare_scored = []
        for r, _, poincare_vec in klein_survivors:
            diff_sq, _, _ = fused_norms(query_np, poincare_vec)
            cand_alpha = alpha_precompute(poincare_vec, c=curvature)
            dist = poincare_distance_with_alpha(diff_sq, query_alpha, cand_alpha, c=curvature)
            poincare_scored.append((r, float(dist)))
        poincare_scored.sort(key=lambda x: x[1])
        final_results = poincare_scored[:final_top]
        t3 = time.time()

        return {
            "results": [r for r, _ in final_results],
            "distances": [d for _, d in final_results],
            "prefetch_count": len(prefetch_results),
            "klein_survivors": len(klein_survivors),
            "final_count": len(final_results),
            "stage1_latency_ms": (t1 - t0) * 1000,
            "stage2_latency_ms": (t2 - t1) * 1000,
            "stage3_latency_ms": (t3 - t2) * 1000,
            "total_latency_ms": (t3 - t0) * 1000,
        }
    except Exception as e:
        return {"error": str(e)}


def geometric_filtered_query(client, collection, query_vector, filters,
                              using="poincare", limit=200, final_top=10, ef=128,
                              curvature=5.0):
    """Search Qdrant, then apply geometric filters client-side.

    Args:
        filters: list of filter specs, each a dict:
            {"type": "inball", "center": ndarray, "radius": float}
            {"type": "incone", "axis": ndarray, "aperture": float}
            {"type": "depth_band", "depth_min": float, "depth_max": float}
    Filters compose with AND logic.
    """
    import numpy as np
    from hyperbolic_math import inball_filter, incone_filter

    try:
        t0 = time.time()
        query_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

        results = client.query_points(
            collection_name=collection,
            query=query_list,
            using=using,
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=True,
        ).points
        t1 = time.time()

        # Convert to candidate dicts
        candidates = []
        for r in results:
            vec = r.vector
            if isinstance(vec, dict):
                vec = vec.get("poincare", vec.get("dense", next(iter(vec.values()))))
            candidates.append({
                "vector": np.array(vec, dtype=np.float32),
                "point": r,
                "busemann_depth": r.payload.get("busemann_depth", 0),
            })

        # Apply filters (AND logic)
        filtered = candidates
        for f in filters:
            if f["type"] == "inball":
                filtered = inball_filter(filtered, f["center"], f["radius"], curvature)
            elif f["type"] == "incone":
                filtered = incone_filter(filtered, f["axis"], f["aperture"], f.get("origin"))
            elif f["type"] == "depth_band":
                filtered = [c for c in filtered
                           if f["depth_min"] <= c["busemann_depth"] <= f["depth_max"]]

        t2 = time.time()
        final = filtered[:final_top]

        return {
            "results": [c["point"] for c in final],
            "initial_count": len(candidates),
            "filtered_count": len(filtered),
            "final_count": len(final),
            "search_latency_ms": (t1 - t0) * 1000,
            "filter_latency_ms": (t2 - t1) * 1000,
            "total_latency_ms": (t2 - t0) * 1000,
        }
    except Exception as e:
        return {"error": str(e)}


def alpha_pipeline_query(
    client,
    collection,
    query_poincare,
    curvature=1.0,
    prefetch_limit=200,
    alpha_top=50,
    final_top=10,
    using_prefetch="dense",
    ef=128,
):
    """3-stage alpha pipeline: cosine prefetch -> alpha re-rank -> exact Poincare.

    Replaces Klein pipeline. Alpha proxy gives exact ordering (monotonic with
    Poincare distance), no model conversion needed.
    """
    t_start = time.time()

    # Stage 1: Cosine prefetch (server-side)
    t1 = time.time()
    prefetch_results = client.query_points(
        collection_name=collection,
        query=query_poincare.tolist(),
        using=using_prefetch,
        with_payload=True,
        with_vectors=["poincare"],
        limit=prefetch_limit,
        search_params=SearchParams(hnsw_ef=ef),
    ).points
    t1_end = time.time()

    if not prefetch_results:
        return {
            "results": [], "distances": [],
            "prefetch_count": 0, "alpha_survivors": 0, "final_count": 0,
            "stage1_latency_ms": 0, "stage2_latency_ms": 0,
            "stage3_latency_ms": 0, "total_latency_ms": 0,
        }

    # Stage 2: Alpha re-rank (client-side, no acosh)
    t2 = time.time()
    query_alpha = alpha_precompute(query_poincare, curvature)

    scored = []
    for pt in prefetch_results:
        cand_vec = np.array(pt.vector["poincare"])
        cand_alpha = pt.payload.get("alpha")
        if cand_alpha is None:
            cand_alpha = alpha_precompute(cand_vec, curvature)
        diff = query_poincare - cand_vec
        diff_sq = float(np.dot(diff, diff))
        proxy = diff_sq * query_alpha * cand_alpha
        scored.append((pt, proxy))

    scored.sort(key=lambda x: x[1])
    survivors = scored[:alpha_top]
    t2_end = time.time()

    # Stage 3: Exact Poincare (client-side, reuse proxy)
    t3 = time.time()
    exact = []
    for pt, proxy in survivors:
        sqrt_c = np.sqrt(curvature)
        arg = 1.0 + 2.0 * curvature * proxy
        dist = float(np.arccosh(max(arg, 1.0))) / sqrt_c
        exact.append((pt, dist))

    exact.sort(key=lambda x: x[1])
    final = exact[:final_top]
    t3_end = time.time()

    return {
        "results": [pt for pt, _ in final],
        "distances": [d for _, d in final],
        "prefetch_count": len(prefetch_results),
        "alpha_survivors": len(survivors),
        "final_count": len(final),
        "stage1_latency_ms": (t1_end - t1) * 1000,
        "stage2_latency_ms": (t2_end - t2) * 1000,
        "stage3_latency_ms": (t3_end - t3) * 1000,
        "total_latency_ms": (t3_end - t_start) * 1000,
    }


def tangent_pipeline_query(
    client,
    collection,
    query_poincare,
    centroid,
    curvature=1.0,
    prune_factor=10,
    final_top=10,
    ef=128,
):
    """2-stage tangent pipeline: tangent HNSW (server) -> exact Poincare (client).

    Uses Qdrant's native Euclidean HNSW on the "tangent" named vector for
    approximate hyperbolic nearest-neighbor search.
    """
    t_start = time.time()

    q_tangent = tangent_query(query_poincare, centroid, curvature)

    # Stage 1: Tangent HNSW search (server-side Euclidean)
    t1 = time.time()
    tangent_limit = final_top * prune_factor
    tangent_results = client.query_points(
        collection_name=collection,
        query=q_tangent.tolist(),
        using="tangent",
        with_payload=True,
        with_vectors=["poincare"],
        limit=tangent_limit,
        search_params=SearchParams(hnsw_ef=ef),
    ).points
    t1_end = time.time()

    if not tangent_results:
        return {
            "results": [], "distances": [],
            "tangent_candidates": 0, "final_count": 0,
            "stage1_latency_ms": 0, "stage2_latency_ms": 0,
            "total_latency_ms": 0,
        }

    # Stage 2: Exact Poincare re-rank (client-side)
    t2 = time.time()
    exact = []
    for pt in tangent_results:
        cand_vec = np.array(pt.vector["poincare"])
        diff_sq, u_sq, v_sq = fused_norms(query_poincare, cand_vec)
        denom_u = max(1.0 - curvature * u_sq, 1e-7)
        denom_v = max(1.0 - curvature * v_sq, 1e-7)
        arg = 1.0 + 2.0 * curvature * diff_sq / (denom_u * denom_v)
        dist = float(np.arccosh(max(arg, 1.0))) / np.sqrt(curvature)
        exact.append((pt, dist))

    exact.sort(key=lambda x: x[1])
    final = exact[:final_top]
    t2_end = time.time()

    return {
        "results": [pt for pt, _ in final],
        "distances": [d for _, d in final],
        "tangent_candidates": len(tangent_results),
        "final_count": len(final),
        "stage1_latency_ms": (t1_end - t1) * 1000,
        "stage2_latency_ms": (t2_end - t2) * 1000,
        "total_latency_ms": (t2_end - t_start) * 1000,
    }


def combined_pipeline_query(
    client,
    collection,
    query_poincare,
    centroid,
    curvature=1.0,
    cosine_limit=100,
    tangent_limit=100,
    alpha_top=50,
    final_top=10,
    ef=128,
):
    """Combined pipeline: cosine + tangent prefetch -> alpha re-rank -> exact Poincare.

    Merges two independent retrieval signals (semantic similarity from cosine,
    hierarchical proximity from tangent) before alpha re-ranking.
    """
    t_start = time.time()

    # Stage 1a: Cosine prefetch (server)
    cosine_results = client.query_points(
        collection_name=collection,
        query=query_poincare.tolist(),
        using="dense",
        with_payload=True,
        with_vectors=["poincare"],
        limit=cosine_limit,
        search_params=SearchParams(hnsw_ef=ef),
    ).points

    # Stage 1b: Tangent prefetch (server)
    q_tangent = tangent_query(query_poincare, centroid, curvature)
    tangent_results = client.query_points(
        collection_name=collection,
        query=q_tangent.tolist(),
        using="tangent",
        with_payload=True,
        with_vectors=["poincare"],
        limit=tangent_limit,
        search_params=SearchParams(hnsw_ef=ef),
    ).points
    t1_end = time.time()

    # Merge + deduplicate by point ID
    seen_ids = set()
    merged = []
    for pt in cosine_results + tangent_results:
        if pt.id not in seen_ids:
            seen_ids.add(pt.id)
            merged.append(pt)

    # Stage 2: Alpha re-rank (client, no acosh)
    t2 = time.time()
    query_alpha = alpha_precompute(query_poincare, curvature)
    scored = []
    for pt in merged:
        cand_vec = np.array(pt.vector["poincare"])
        cand_alpha = pt.payload.get("alpha")
        if cand_alpha is None:
            cand_alpha = alpha_precompute(cand_vec, curvature)
        diff = query_poincare - cand_vec
        diff_sq = float(np.dot(diff, diff))
        proxy = diff_sq * query_alpha * cand_alpha
        scored.append((pt, proxy))

    scored.sort(key=lambda x: x[1])
    survivors = scored[:alpha_top]
    t2_end = time.time()

    # Stage 3: Exact Poincare
    t3 = time.time()
    exact = []
    for pt, proxy in survivors:
        sqrt_c = np.sqrt(curvature)
        arg = 1.0 + 2.0 * curvature * proxy
        dist = float(np.arccosh(max(arg, 1.0))) / sqrt_c
        exact.append((pt, dist))

    exact.sort(key=lambda x: x[1])
    final = exact[:final_top]
    t3_end = time.time()

    return {
        "results": [pt for pt, _ in final],
        "distances": [d for _, d in final],
        "cosine_count": len(cosine_results),
        "tangent_count": len(tangent_results),
        "merged_count": len(merged),
        "alpha_survivors": len(survivors),
        "final_count": len(final),
        "stage1_latency_ms": (t1_end - t_start) * 1000,
        "stage2_latency_ms": (t2_end - t2) * 1000,
        "stage3_latency_ms": (t3_end - t3) * 1000,
        "total_latency_ms": (t3_end - t_start) * 1000,
    }
