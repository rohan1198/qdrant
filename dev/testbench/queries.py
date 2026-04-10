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
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Range, SearchParams,
)


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
