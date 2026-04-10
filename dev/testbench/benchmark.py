"""Benchmark script for hyperbolic (Poincare / Busemann) vector collections in Qdrant.

Runs 11 benchmark suites via BenchmarkRunner:
  1. Gromov Delta          — 4-point hyperbolicity measure
  2. Hierarchy Separation  — Busemann depth per tier + band overlap + Einstein midpoints
  3. Drill-Down Queries    — vertical traversal recall (chatbot pattern)
  4. Lateral Exploration   — horizontal same-tier search quality (chatbot pattern)
  5. Cross-Branch Discovery— structural similarity across domains (chatbot pattern)
  6. Depth-Band Filtering  — Busemann depth-filtered search precision
  7. Unified vs Separate   — compare unified and per-tier collections
  8. Latency               — throughput and p50/p95/p99 at multiple ef values
  9. Quantization Recall   — recall of quantized vs exact Poincaré search
 10. Klein Pre-Filter      — Klein pre-filter pipeline vs direct Poincaré search
 11. Geometric Filters     — geometric filter precision and selectivity

Auto-discovers dataset collections and adapts to whatever exists.

Usage:
  python benchmark.py [--qdrant-url http://localhost:6334] [--dataset bgc]
  python benchmark.py --suites gromov_delta hierarchy_separation latency
"""

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
from tqdm import tqdm

from fusion import rrf_fuse
from hyperbolic_math import (
    EPS,
    busemann_depth_single,
    compute_busemann_depths,
    compute_focal_direction,
    einstein_midpoint,
    gromov_delta,
)
from queries import (
    cross_branch_query,
    depth_band_query,
    drill_down_query,
    lateral_query,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EF_VALUES = [64, 128, 256]
DEFAULT_CURVATURE = 5.0

UNIFIED_COLLECTION = "wos_unified"
CONTENT_ID_OFFSET = 0
STORY_ID_OFFSET = 100_000
NARRATIVE_ID_OFFSET = 200_000


# ---------------------------------------------------------------------------
# Multi-label matching
# ---------------------------------------------------------------------------


def _get_paths(pt: dict) -> list[str]:
    """Get hierarchy paths from a point, with robust fallback.

    Priority: all_paths (if non-empty) > hierarchy_path > domain/area reconstruction.
    All returned values are guaranteed to be strings.
    """
    paths = pt.get("all_paths") or []
    if paths:
        return [str(p) for p in paths]

    hp = pt.get("hierarchy_path", "")
    if hp:
        return [str(hp)]

    # Last resort: reconstruct from domain/area
    domain = pt.get("domain", "")
    area = pt.get("area", "")
    if domain and area:
        return [f"{domain}/{area}"]
    elif domain:
        return [str(domain)]
    return []


def _paths_match_area(query_pt: dict, result_pt: dict) -> bool:
    """Check if result shares an L2 (area) category with query, considering all_paths."""
    q_paths = _get_paths(query_pt)
    r_paths = _get_paths(result_pt)

    # Extract L2 prefixes (first 2 components of each path)
    q_areas = set()
    for p in q_paths:
        parts = p.split("/")
        if len(parts) >= 2:
            q_areas.add("/".join(parts[:2]))
    r_areas = set()
    for p in r_paths:
        parts = p.split("/")
        if len(parts) >= 2:
            r_areas.add("/".join(parts[:2]))

    # Fallback to exact area match if no 2-component paths
    if not q_areas or not r_areas:
        return query_pt.get("area") == result_pt.get("area") and query_pt.get("area") not in (None, "", -1)

    return bool(q_areas & r_areas)


def _paths_match_domain(query_pt: dict, result_pt: dict) -> bool:
    """Check if result shares an L1 (domain) category with query, considering all_paths."""
    q_paths = _get_paths(query_pt)
    r_paths = _get_paths(result_pt)

    q_domains = {p.split("/")[0] for p in q_paths if p}
    r_domains = {p.split("/")[0] for p in r_paths if p}

    # Fallback to exact domain match
    if not q_domains or not r_domains:
        return query_pt.get("domain") == result_pt.get("domain") and query_pt.get("domain") not in (None, "", -1)

    return bool(q_domains & r_domains)


# ---------------------------------------------------------------------------
# Collection discovery
# ---------------------------------------------------------------------------


def discover_collections(client: QdrantClient, qdrant_url: str) -> list[dict]:
    """Auto-discover wos_* collections and detect their config.

    Uses the REST API for collection detail (the qdrant_client SDK cannot
    parse our custom Poincare distance type).

    Returns a list of dicts:
      {"name": str, "is_poincare": bool, "curvature": float | None, "strategy": str | None}
    """
    import requests

    existing = client.get_collections().collections
    collections = []

    KNOWN_PREFIXES = ["wos_", "bgc_", "scihtc_", "eurlex_"]

    for col_info in existing:
        name = col_info.name

        # Check if collection matches any known prefix
        matched_prefix = None
        for pfx in KNOWN_PREFIXES:
            if name.startswith(pfx):
                matched_prefix = pfx.rstrip("_")
                break
        if matched_prefix is None:
            continue

        # Use REST API to get distance type (SDK chokes on Poincare)
        try:
            resp = requests.get(f"{qdrant_url}/collections/{name}")
            col_json = resp.json().get("result", {})
            vectors = col_json.get("config", {}).get("params", {}).get("vectors", {})
            distance = vectors.get("distance", "unknown")
        except Exception:
            distance = "unknown"

        is_poincare = "poincare" in distance.lower()

        # Detect unified collection (named vectors) — any prefix
        if name == f"{matched_prefix}_unified":
            is_poincare = True
            strategy = "unified"
            curvature = DEFAULT_CURVATURE

        # Detect curvature sweep collections ({prefix}_unified_c10, etc.)
        elif name.startswith(f"{matched_prefix}_unified_c"):
            is_poincare = True
            strategy = "unified"
            suffix = name.replace(f"{matched_prefix}_unified_c", "")
            try:
                curvature = int(suffix) / 10.0
            except ValueError:
                curvature = DEFAULT_CURVATURE

        # Cosine baseline — any prefix
        elif name == f"{matched_prefix}_cosine":
            strategy = None
            curvature = None
        elif "_uniform" in name:
            strategy = "uniform"
            curvature = DEFAULT_CURVATURE
        elif "_static" in name:
            strategy = "static"
            curvature = DEFAULT_CURVATURE
        elif "_einstein" in name and "_spread" not in name:
            strategy = "einstein"
            curvature = DEFAULT_CURVATURE
        elif "_spread" in name:
            strategy = "einstein_spread"
            curvature = DEFAULT_CURVATURE
        elif is_poincare:
            # Legacy collections like wos_c05, wos_c50
            strategy = "uniform"
            suffix = name.replace("wos_c", "")
            try:
                curvature = int(suffix) / 10.0
            except ValueError:
                curvature = DEFAULT_CURVATURE
        else:
            strategy = None
            curvature = None

        collections.append({
            "name": name,
            "is_poincare": is_poincare,
            "curvature": curvature,
            "strategy": strategy,
        })

    # Sort: cosine first, then by strategy name
    collections.sort(key=lambda c: (c["is_poincare"], c["strategy"] or ""))
    return collections


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def scroll_all(
    client: QdrantClient,
    collection: str,
    vector_name: str | None = None,
) -> list[dict]:
    """Scroll all points from a collection, returning list of dicts.

    For unified collections, specify vector_name ('dense' or 'poincare')
    to extract that named vector into 'vector'. If None, extracts the first/only vector.

    When with_vectors=True returns a dict of named vectors, ALL named vectors
    are stored under 'vectors' (dict) for multi-space use. The 'vector' field
    always contains the single vector specified by vector_name.
    """
    points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        for pt in batch:
            raw_vec = pt.vector
            all_vectors = None

            if isinstance(raw_vec, dict):
                # Store all named vectors for multi-space use
                all_vectors = {k: np.array(v, dtype=np.float32) for k, v in raw_vec.items()}
                if vector_name and vector_name in raw_vec:
                    raw_vec = raw_vec[vector_name]
                else:
                    raw_vec = next(iter(raw_vec.values()))

            entry = {
                "id": pt.id,
                "vector": np.array(raw_vec, dtype=np.float32),
                "tier": pt.payload.get("tier", "leaf"),
                "item_type": pt.payload.get("item_type", pt.payload.get("tier", "content")),
                "domain": pt.payload.get("domain", -1),
                "area": pt.payload.get("area", -1),
                "busemann_depth": pt.payload.get("busemann_depth"),
                "all_paths": pt.payload.get("all_paths", []),
                "hierarchy_path": pt.payload.get("hierarchy_path", ""),
                "point_id": pt.payload.get("point_id", str(pt.id)),
                "parent_ids": pt.payload.get("parent_ids", []),
                "child_ids": pt.payload.get("child_ids", pt.payload.get("source_ids", [])),
            }
            if all_vectors is not None:
                entry["vectors"] = all_vectors
            points.append(entry)
        if next_offset is None:
            break
        offset = next_offset
    return points


# ---------------------------------------------------------------------------
# Search helper
# ---------------------------------------------------------------------------


def _search(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    limit: int,
    search_params: SearchParams | None = None,
) -> list:
    """Run nearest-neighbour search."""
    vec = query_vector.tolist()
    kwargs: dict = dict(collection_name=collection, limit=limit)
    if search_params:
        kwargs["search_params"] = search_params

    try:
        results = client.query_points(query=vec, **kwargs)
        return results.points
    except (AttributeError, TypeError):
        pass

    try:
        return client.search(query_vector=vec, **kwargs)
    except Exception:
        return []


def _search_named(
    client: QdrantClient,
    collection: str,
    vector_name: str,
    query_vector: np.ndarray,
    limit: int,
    tier_filter: str | None = None,
) -> list:
    """Search a named vector in the unified collection."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    vec = query_vector.tolist()

    query_filter = None
    if tier_filter:
        query_filter = Filter(
            must=[FieldCondition(key="tier", match=MatchValue(value=tier_filter))]
        )

    try:
        results = client.query_points(
            collection_name=collection,
            query=vec,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )
        return results.points
    except (AttributeError, TypeError):
        pass

    try:
        return client.search(
            collection_name=collection,
            query_vector=(vector_name, vec),
            limit=limit,
            query_filter=query_filter,
        )
    except Exception:
        return []


def _search_named_depth_range(
    client: QdrantClient,
    collection: str,
    vector_name: str,
    query_vector: np.ndarray,
    limit: int,
    depth_lo: float,
    depth_hi: float,
) -> list:
    """Search named vector with Busemann depth range filter."""
    from qdrant_client.models import FieldCondition, Filter, Range

    vec = query_vector.tolist()
    query_filter = Filter(
        must=[
            FieldCondition(
                key="busemann_depth",
                range=Range(gte=depth_lo, lte=depth_hi),
            ),
        ],
    )

    try:
        results = client.query_points(
            collection_name=collection,
            query=vec,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )
        return results.points
    except (AttributeError, TypeError):
        pass

    try:
        return client.search(
            collection_name=collection,
            query_vector=(vector_name, vec),
            limit=limit,
            query_filter=query_filter,
        )
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Benchmark 1 — Hierarchy Separation
# ---------------------------------------------------------------------------


def benchmark_hierarchy_separation(points: list[dict], curvature: float) -> dict:
    """Compute Busemann depth per tier, separation metrics, and band overlap."""
    depths = compute_busemann_depths(points, c=curvature)

    # Normalize tier names: unified collection uses narrative/story/content
    tier_map = {"narrative": "root", "story": "mid", "content": "leaf"}
    tier_depths: dict[str, list[float]] = {"root": [], "mid": [], "leaf": []}
    for pt, d in zip(points, depths):
        tier = tier_map.get(pt["tier"], pt["tier"])
        if tier in tier_depths:
            tier_depths[tier].append(d)

    results: dict = {}
    for tier, ds in tier_depths.items():
        if ds:
            results[f"avg_depth_{tier}"] = float(np.mean(ds))
            results[f"std_depth_{tier}"] = float(np.std(ds))
        else:
            results[f"avg_depth_{tier}"] = None
            results[f"std_depth_{tier}"] = None

    # --- Sep ratio (all points, original metric) ---
    all_depths = depths
    std_all = float(np.std(all_depths)) if all_depths else EPS
    avg_leaf = results["avg_depth_leaf"]
    avg_root = results["avg_depth_root"]

    if avg_leaf is not None and avg_root is not None and std_all > EPS:
        sep_ratio = abs(avg_leaf - avg_root) / std_all
    else:
        sep_ratio = None
    results["separation_ratio"] = sep_ratio

    # --- Sep ratio (content-only, comparable to Round 2) ---
    leaf_depths = tier_depths["leaf"]
    if leaf_depths:
        std_content = float(np.std(leaf_depths)) if len(leaf_depths) > 1 else EPS
        if avg_leaf is not None and avg_root is not None and std_content > EPS:
            results["separation_ratio_content"] = abs(avg_leaf - avg_root) / std_content
        else:
            results["separation_ratio_content"] = None
    else:
        results["separation_ratio_content"] = None

    # --- Cross-tier separation metric ---
    avg_mid = results.get("avg_depth_mid")
    std_leaf = results.get("std_depth_leaf") or EPS
    std_mid = results.get("std_depth_mid") or EPS
    std_root = results.get("std_depth_root") or EPS

    if avg_leaf is not None and avg_mid is not None and avg_root is not None:
        gap_content_story = abs(avg_leaf - avg_mid)
        gap_story_narrative = abs(avg_mid - avg_root)
        worst_std = max(std_leaf, std_mid, std_root)
        if worst_std > EPS:
            results["cross_tier_separation"] = min(gap_content_story, gap_story_narrative) / worst_std
        else:
            results["cross_tier_separation"] = None
    else:
        results["cross_tier_separation"] = None

    results["num_points"] = len(points)

    # --- 3-way tier classification accuracy ---
    if avg_leaf is not None and avg_mid is not None and avg_root is not None:
        sorted_avgs = sorted([(avg_root, "root"), (avg_mid, "mid"), (avg_leaf, "leaf")])
        thresh_lo = (sorted_avgs[0][0] + sorted_avgs[1][0]) / 2.0
        thresh_hi = (sorted_avgs[1][0] + sorted_avgs[2][0]) / 2.0

        correct = 0
        for pt, d in zip(points, depths):
            normalized_tier = tier_map.get(pt["tier"], pt["tier"])
            if d < thresh_lo:
                predicted = sorted_avgs[0][1]
            elif d < thresh_hi:
                predicted = sorted_avgs[1][1]
            else:
                predicted = sorted_avgs[2][1]
            if normalized_tier == predicted:
                correct += 1
        results["tier_classification_accuracy"] = correct / len(points)
    elif all_depths:
        median_d = float(np.median(all_depths))
        correct = 0
        for pt, d in zip(points, depths):
            predicted = "leaf" if d >= median_d else "root"
            normalized_tier = tier_map.get(pt["tier"], pt["tier"])
            if normalized_tier == predicted or (normalized_tier == "mid" and predicted == "root"):
                correct += 1
        results["tier_classification_accuracy"] = correct / len(points)
    else:
        results["tier_classification_accuracy"] = None

    # Band overlap
    results["band_overlap"] = _compute_band_overlap(tier_depths)

    return results


def _compute_band_overlap(tier_depths: dict[str, list[float]]) -> float | None:
    """Compute fraction of vectors that fall outside their tier's Busemann band.

    Bands defined by midpoint thresholds between adjacent tier means:
      threshold_root_mid = (avg_root + avg_mid) / 2
      threshold_mid_leaf = (avg_mid + avg_leaf) / 2
    A vector is "misplaced" if its depth crosses into a different tier's band.
    """
    avgs = {}
    for tier in ("root", "mid", "leaf"):
        if not tier_depths[tier]:
            return None
        avgs[tier] = float(np.mean(tier_depths[tier]))

    # Ensure the thresholds make sense (root < mid < leaf in depth)
    sorted_tiers = sorted(avgs.keys(), key=lambda t: avgs[t])

    if len(sorted_tiers) < 3:
        return None

    lo_tier, mid_tier, hi_tier = sorted_tiers
    thresh_lo_mid = (avgs[lo_tier] + avgs[mid_tier]) / 2.0
    thresh_mid_hi = (avgs[mid_tier] + avgs[hi_tier]) / 2.0

    tier_to_band = {
        lo_tier: (-np.inf, thresh_lo_mid),
        mid_tier: (thresh_lo_mid, thresh_mid_hi),
        hi_tier: (thresh_mid_hi, np.inf),
    }

    total = 0
    misplaced = 0
    for tier, ds in tier_depths.items():
        lo, hi = tier_to_band[tier]
        for d in ds:
            total += 1
            if d < lo or d >= hi:
                misplaced += 1

    return misplaced / total if total > 0 else None


# ---------------------------------------------------------------------------
# Benchmark 2 — Retrieval Quality
# ---------------------------------------------------------------------------


def benchmark_retrieval_quality(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    k: int = 10,
    num_queries: int = 500,
    vector_name: str | None = None,
) -> dict:
    """Sample leaf documents and evaluate recall@10 within hierarchy."""
    leaf_points = [pt for pt in points if pt["tier"] in ("leaf", "content")]
    if not leaf_points:
        return {"error": "no leaf/content points found"}

    query_sample = random.sample(leaf_points, min(num_queries, len(leaf_points)))

    area_recalls: list[float] = []
    domain_recalls: list[float] = []
    h_precisions: list[float] = []

    id_to_pt = {pt["id"]: pt for pt in points}

    for query_pt in query_sample:
        try:
            if vector_name:
                results = _search_named(client, collection, vector_name, query_pt["vector"], limit=k + 1)
            else:
                results = _search(client, collection, query_pt["vector"], limit=k + 1)
        except Exception:
            continue

        query_id = query_pt["id"]
        neighbors = [r for r in results if r.id != query_id][:k]

        if not neighbors:
            continue

        same_area = 0
        same_domain = 0
        h_prec = 0.0

        for nb in neighbors:
            nb_pt = id_to_pt.get(nb.id)
            if nb_pt is None:
                continue

            if _paths_match_area(query_pt, nb_pt):
                same_area += 1
                h_prec += 1.0
            elif _paths_match_domain(query_pt, nb_pt):
                same_domain += 1
                h_prec += 0.5

        area_recalls.append(same_area / k)
        domain_recalls.append((same_area + same_domain) / k)
        h_precisions.append(h_prec / k)

    def _safe_mean(lst: list[float]) -> float | None:
        return float(np.mean(lst)) if lst else None

    return {
        "num_queries": len(area_recalls),
        "area_recall_at_10": _safe_mean(area_recalls),
        "domain_recall_at_10": _safe_mean(domain_recalls),
        "hierarchical_precision": _safe_mean(h_precisions),
    }


def benchmark_retrieval_modes(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    k: int = 10,
    num_queries: int = 500,
) -> dict:
    """Run retrieval quality in multiple filter modes for unified collection.

    Modes: unfiltered, tier-filtered, depth-range, cosine+tier-filtered.
    """
    content_points = [p for p in points if p["tier"] in ("leaf", "content")]
    if not content_points:
        return {"error": "no content points"}

    query_sample = random.sample(content_points, min(num_queries, len(content_points)))
    id_to_pt = {p["id"]: p for p in points}

    # Compute depth stats for content tier (for depth-range band)
    content_depths = [p["busemann_depth"] for p in content_points if p.get("busemann_depth") is not None]
    if content_depths:
        depth_mean = float(np.mean(content_depths))
        depth_std = float(np.std(content_depths))
        depth_lo = depth_mean - 2 * depth_std
        depth_hi = depth_mean + 2 * depth_std
    else:
        depth_lo, depth_hi = 0.0, 10.0

    # Each mode: (vector_key, search_fn_factory)
    modes = {
        "unfiltered": ("poincare", lambda q: _search_named(client, collection, "poincare", q, limit=k + 1)),
        "tier_filtered": ("poincare", lambda q: _search_named(client, collection, "poincare", q, limit=k + 1, tier_filter="content")),
        "depth_range": ("poincare", lambda q: _search_named_depth_range(client, collection, "poincare", q, limit=k + 1, depth_lo=depth_lo, depth_hi=depth_hi)),
        "cosine_tier_filtered": ("dense", lambda q: _search_named(client, collection, "dense", q, limit=k + 1, tier_filter="content")),
    }

    mode_results: dict[str, dict] = {}

    for mode_name, (vec_key, search_fn) in modes.items():
        area_recalls = []
        domain_recalls = []
        h_precisions = []

        for qpt in query_sample:
            # Use the correct vector for this mode's space
            vectors = qpt.get("vectors", {})
            query_vec = vectors.get(vec_key, qpt["vector"])
            try:
                results = search_fn(query_vec)
            except Exception:
                continue

            query_id = qpt["id"]
            neighbors = [r for r in results if r.id != query_id][:k]
            if not neighbors:
                continue

            same_area = 0
            same_domain = 0
            h_prec = 0.0
            for nb in neighbors:
                nb_pt = id_to_pt.get(nb.id)
                if nb_pt is None:
                    continue
                if _paths_match_area(qpt, nb_pt):
                    same_area += 1
                    h_prec += 1.0
                elif _paths_match_domain(qpt, nb_pt):
                    same_domain += 1
                    h_prec += 0.5

            area_recalls.append(same_area / k)
            domain_recalls.append((same_area + same_domain) / k)
            h_precisions.append(h_prec / k)

        def _safe_mean(lst):
            return float(np.mean(lst)) if lst else None

        mode_results[mode_name] = {
            "num_queries": len(area_recalls),
            "area_recall_at_10": _safe_mean(area_recalls),
            "domain_recall_at_10": _safe_mean(domain_recalls),
            "hierarchical_precision": _safe_mean(h_precisions),
        }

    mode_results["depth_range_band"] = {"lo": depth_lo, "hi": depth_hi}
    return mode_results


# ---------------------------------------------------------------------------
# Benchmark 3 — Cross-Tier Retrieval
# ---------------------------------------------------------------------------


def benchmark_cross_tier(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    k: int = 5,
) -> dict:
    """Evaluate cross-tier retrieval: can we find parent/child points?

    - parent_recall: query a content point, check if its parent area/domain appears
    - child_recall: query a narrative/story point, check if child points appear
    """
    content_points = [p for p in points if p["tier"] == "content"]
    story_points = [p for p in points if p["tier"] == "story"]
    narrative_points = [p for p in points if p["tier"] == "narrative"]

    if not story_points or not narrative_points:
        return {"error": "need story + narrative tiers in collection"}

    # Map area -> story ID, domain -> narrative ID
    area_to_story_id = {}
    for sp in story_points:
        area_to_story_id[sp["area"]] = sp["id"]

    domain_to_narrative_id = {}
    for np_ in narrative_points:
        domain_to_narrative_id[np_["domain"]] = np_["id"]

    # --- Parent recall: query content, find story/narrative in results ---
    sample_content = random.sample(content_points, min(200, len(content_points)))
    parent_story_hits = 0
    parent_narrative_hits = 0

    for qpt in sample_content:
        results = _search_named(client, collection, "poincare", qpt["vector"], limit=k + 1)
        result_ids = {r.id for r in results if r.id != qpt["id"]}

        expected_story_id = area_to_story_id.get(qpt["area"])
        expected_narrative_id = domain_to_narrative_id.get(qpt["domain"])

        if expected_story_id in result_ids:
            parent_story_hits += 1
        if expected_narrative_id in result_ids:
            parent_narrative_hits += 1

    n_content = len(sample_content)

    # --- Child recall: query narratives, find stories in results ---
    child_story_hits = 0
    child_total = 0

    for npt in narrative_points:
        results = _search_named(client, collection, "poincare", npt["vector"], limit=20)
        result_ids = {r.id for r in results if r.id != npt["id"]}

        expected_stories = [
            sp["id"] for sp in story_points if sp["domain"] == npt["domain"]
        ]
        for sid in expected_stories:
            child_total += 1
            if sid in result_ids:
                child_story_hits += 1

    return {
        "parent_story_recall": parent_story_hits / n_content if n_content > 0 else None,
        "parent_narrative_recall": parent_narrative_hits / n_content if n_content > 0 else None,
        "child_story_recall": child_story_hits / child_total if child_total > 0 else None,
        "num_content_queries": n_content,
        "num_narrative_queries": len(narrative_points),
    }


# ---------------------------------------------------------------------------
# Benchmark 4 — Dual-Space Comparison
# ---------------------------------------------------------------------------


def benchmark_dual_space(
    client: QdrantClient,
    unified_collection: str,
    cosine_collection: str,
    points_unified: list[dict],
    points_cosine: list[dict],
    k: int = 10,
    num_queries: int = 500,
) -> dict:
    """Compare fusion strategies: RRF, linear alpha, Busemann-weighted, depth-band."""
    from fusion import (
        busemann_weighted_fuse,
        depth_band_fuse,
        linear_alpha_fuse,
        rrf_fuse,
    )

    content_points = [p for p in points_unified if p["tier"] == "content"]
    if not content_points:
        return {"error": "no content points in unified collection"}

    query_sample = random.sample(content_points, min(num_queries, len(content_points)))

    cosine_id_to_pt = {p["id"]: p for p in points_cosine}
    unified_id_to_pt = {p["id"]: p for p in points_unified}

    # Build depth lookup for Busemann-weighted fusion
    id_to_depth = {p["id"]: p["busemann_depth"] for p in points_unified if p.get("busemann_depth") is not None}

    # Compute content depth stats for depth-band
    content_depths = [p["busemann_depth"] for p in content_points if p.get("busemann_depth") is not None]
    depth_mean = float(np.mean(content_depths)) if content_depths else 1.0
    depth_std = float(np.std(content_depths)) if content_depths else 0.1

    over_fetch = k * 3  # More candidates for fusion

    # Define all strategy configs to sweep
    strategies: dict[str, dict] = {
        "cosine_only": {},
        "poincare_only": {},
        "rrf_k60": {},
    }
    for a in [0.3, 0.5, 0.7, 0.9]:
        strategies[f"alpha_{a}"] = {"alpha": a}
    for a in [0.5, 0.7]:
        for lam in [0.5, 1.0, 2.0]:
            strategies[f"buse_a{a}_l{lam}"] = {"alpha": a, "lam": lam}
    for a in [0.5, 0.7]:
        for bw in [0.2, 0.5]:
            strategies[f"band_a{a}_bw{bw}"] = {"alpha": a, "bandwidth": bw}

    # Accumulate recalls per strategy
    strategy_area_recalls: dict[str, list[float]] = {s: [] for s in strategies}
    strategy_domain_recalls: dict[str, list[float]] = {s: [] for s in strategies}

    for qpt in query_sample:
        query_id = qpt["id"]
        query_area = qpt["area"]
        query_domain = qpt["domain"]
        query_depth = qpt.get("busemann_depth", depth_mean)

        # Get the correct vector for each space
        vectors = qpt.get("vectors", {})
        cosine_vec = vectors.get("dense", vectors.get("cosine", qpt["vector"]))
        poincare_vec = vectors.get("poincare", qpt["vector"])

        # Fetch results from both spaces using correct vectors
        try:
            cos_results = _search(client, cosine_collection, cosine_vec, limit=k + 1)
        except Exception:
            continue

        try:
            cos_unified = _search_named(client, unified_collection, "dense", cosine_vec, limit=over_fetch, tier_filter="content")
        except Exception:
            cos_unified = []

        try:
            poincare_unified = _search_named(client, unified_collection, "poincare", poincare_vec, limit=over_fetch, tier_filter="content")
        except Exception:
            poincare_unified = []

        def _score(neighbors, id_to_pt, use_tuple=False):
            same_area = 0
            same_domain = 0
            for nb in neighbors:
                nb_id = nb[0] if use_tuple else (nb.id if hasattr(nb, "id") else nb)
                nb_pt = id_to_pt.get(nb_id)
                if nb_pt is None:
                    continue
                if _paths_match_area(qpt, nb_pt):
                    same_area += 1
                elif _paths_match_domain(qpt, nb_pt):
                    same_domain += 1
            return same_area / k, (same_area + same_domain) / k

        # --- Cosine-only baseline ---
        cos_neighbors = [r for r in cos_results if r.id != query_id][:k]
        ar, dr = _score(cos_neighbors, cosine_id_to_pt)
        strategy_area_recalls["cosine_only"].append(ar)
        strategy_domain_recalls["cosine_only"].append(dr)

        # --- Poincare-only ---
        poin_neighbors = [r for r in poincare_unified if r.id != query_id][:k]
        ar, dr = _score(poin_neighbors, unified_id_to_pt)
        strategy_area_recalls["poincare_only"].append(ar)
        strategy_domain_recalls["poincare_only"].append(dr)

        # --- RRF ---
        fused = rrf_fuse(cos_unified, poincare_unified, k=60, limit=k)
        fused = [(pid, s) for pid, s in fused if pid != query_id][:k]
        ar, dr = _score(fused, unified_id_to_pt, use_tuple=True)
        strategy_area_recalls["rrf_k60"].append(ar)
        strategy_domain_recalls["rrf_k60"].append(dr)

        # --- Linear alpha sweep ---
        for a in [0.3, 0.5, 0.7, 0.9]:
            key = f"alpha_{a}"
            fused = linear_alpha_fuse(cos_unified, poincare_unified, alpha=a, limit=k)
            fused = [(pid, s) for pid, s in fused if pid != query_id][:k]
            ar, dr = _score(fused, unified_id_to_pt, use_tuple=True)
            strategy_area_recalls[key].append(ar)
            strategy_domain_recalls[key].append(dr)

        # --- Busemann-weighted sweep ---
        for a in [0.5, 0.7]:
            for lam in [0.5, 1.0, 2.0]:
                key = f"buse_a{a}_l{lam}"
                fused = busemann_weighted_fuse(cos_unified, poincare_unified, query_depth, id_to_depth, alpha=a, lam=lam, limit=k)
                fused = [(pid, s) for pid, s in fused if pid != query_id][:k]
                ar, dr = _score(fused, unified_id_to_pt, use_tuple=True)
                strategy_area_recalls[key].append(ar)
                strategy_domain_recalls[key].append(dr)

        # --- Depth-band sweep ---
        for a in [0.5, 0.7]:
            for bw in [0.2, 0.5]:
                key = f"band_a{a}_bw{bw}"
                fused = depth_band_fuse(cos_unified, poincare_unified, query_depth, id_to_depth, bandwidth=bw, alpha=a, limit=k)
                fused = [(pid, s) for pid, s in fused if pid != query_id][:k]
                ar, dr = _score(fused, unified_id_to_pt, use_tuple=True)
                strategy_area_recalls[key].append(ar)
                strategy_domain_recalls[key].append(dr)

    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else None

    results = {"num_queries": len(strategy_area_recalls.get("cosine_only", []))}
    for key in strategies:
        results[key] = {
            "area_recall_at_10": _safe_mean(strategy_area_recalls[key]),
            "domain_recall_at_10": _safe_mean(strategy_domain_recalls[key]),
        }

    return results


# ---------------------------------------------------------------------------
# Benchmark 5 — Performance & Latency
# ---------------------------------------------------------------------------


def benchmark_latency(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    num_queries: int = 1000,
    vector_name: str | None = None,
) -> dict:
    """Run sequential search at multiple ef values, measure QPS and latency."""
    leaf_points = [pt for pt in points if pt["tier"] in ("leaf", "content")]
    if not leaf_points:
        return {"error": "no leaf/content points found"}

    query_sample = random.sample(leaf_points, min(num_queries, len(leaf_points)))
    ef_results: dict = {}

    for ef in EF_VALUES:
        search_params = SearchParams(hnsw_ef=ef)
        latencies_ms: list[float] = []

        for query_pt in query_sample:
            t0 = time.perf_counter()
            try:
                if vector_name:
                    _search_named(
                        client, collection, vector_name, query_pt["vector"],
                        limit=10,
                    )
                else:
                    _search(
                        client, collection, query_pt["vector"],
                        limit=10, search_params=search_params,
                    )
            except Exception:
                pass
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        if latencies_ms:
            total_s = sum(latencies_ms) / 1000.0
            ef_results[f"ef_{ef}"] = {
                "qps": len(latencies_ms) / total_s if total_s > 0 else None,
                "p50_ms": float(np.percentile(latencies_ms, 50)),
                "p95_ms": float(np.percentile(latencies_ms, 95)),
                "p99_ms": float(np.percentile(latencies_ms, 99)),
                "num_queries": len(latencies_ms),
            }
        else:
            ef_results[f"ef_{ef}"] = {"error": "no results"}

    return ef_results


# ---------------------------------------------------------------------------
# Comparison table (Benchmark 3)
# ---------------------------------------------------------------------------


def print_comparison_table(
    all_results: dict, collection_configs: list[dict]
) -> None:
    """Print a side-by-side comparison table across all collections."""
    header = (
        f"{'Collection':<20} {'Strategy':>12} {'SepRatio':>10} "
        f"{'BandOvlp':>10} {'AreaRec@10':>11} {'DomRec@10':>10} "
        f"{'H-Prec':>8} {'p95@ef128':>10}"
    )
    print("\n" + "=" * len(header))
    print("Projection Strategy Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for col_cfg in collection_configs:
        col = col_cfg["name"]
        strategy = col_cfg.get("strategy") or "cosine"

        res = all_results.get(col, {})

        sep = res.get("hierarchy_separation", {})
        sep_ratio = sep.get("separation_ratio")
        sep_str = f"{sep_ratio:.3f}" if sep_ratio is not None else "N/A"

        band_ovlp = sep.get("band_overlap")
        band_str = f"{band_ovlp:.3f}" if band_ovlp is not None else "N/A"

        rq = res.get("retrieval_quality", {})
        area_rec = rq.get("area_recall_at_10")
        dom_rec = rq.get("domain_recall_at_10")
        h_prec = rq.get("hierarchical_precision")
        area_str = f"{area_rec:.3f}" if area_rec is not None else "N/A"
        dom_str = f"{dom_rec:.3f}" if dom_rec is not None else "N/A"
        h_str = f"{h_prec:.3f}" if h_prec is not None else "N/A"

        lat = res.get("latency", {})
        p95 = lat.get("ef_128", {}).get("p95_ms")
        p95_str = f"{p95:.1f}ms" if p95 is not None else "N/A"

        print(
            f"{col:<20} {strategy:>12} {sep_str:>10} "
            f"{band_str:>10} {area_str:>11} {dom_str:>10} "
            f"{h_str:>8} {p95_str:>10}"
        )

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# BenchmarkRunner — unified engine with 8 suites
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Unified benchmark engine with 11 suites.

    Suites:
      1. gromov_delta          — 4-point hyperbolicity measure
      2. hierarchy_separation  — Busemann depth per tier + Einstein midpoints
      3. drill_down            — vertical traversal recall (chatbot pattern)
      4. lateral               — horizontal same-tier search (chatbot pattern)
      5. cross_branch          — structural similarity across domains (chatbot)
      6. depth_band            — Busemann depth-filtered search precision
      7. unified_vs_separate   — compare unified and per-tier collections
      8. latency               — throughput and p50/p95/p99
      9. quantization_recall   — recall of quantized vs exact Poincaré search
     10. klein_prefilter       — Klein pre-filter pipeline vs direct Poincaré
     11. geometric_filters     — geometric filter precision and selectivity
    """

    def __init__(self, client: QdrantClient, qdrant_url: str, dataset: str, curvature: float = 5.0):
        self.client = client
        self.qdrant_url = qdrant_url
        self.dataset = dataset
        self.curvature = curvature
        self.results: dict = {}

        # Core collection names
        self.unified_collection = f"{dataset}_unified"
        self.cosine_collection = f"{dataset}_cosine"

        # Lazy-loaded point caches
        self._points_poincare: list[dict] | None = None
        self._points_cosine: list[dict] | None = None

    # ------------------------------------------------------------------
    # Point caching
    # ------------------------------------------------------------------

    def _load_unified_points(self, vector_name: str = "poincare") -> list[dict]:
        """Load and cache points from the unified collection."""
        if self._points_poincare is None:
            print(f"  Loading points from {self.unified_collection} (vector={vector_name})...")
            try:
                self._points_poincare = scroll_all(self.client, self.unified_collection, vector_name=vector_name)
                print(f"  Loaded {len(self._points_poincare)} points")
            except Exception as e:
                print(f"  WARNING: Could not load {self.unified_collection}: {e}")
                self._points_poincare = []
        return self._points_poincare

    def _load_cosine_points(self) -> list[dict]:
        """Load and cache points from the cosine baseline collection."""
        if self._points_cosine is None:
            print(f"  Loading points from {self.cosine_collection}...")
            try:
                self._points_cosine = scroll_all(self.client, self.cosine_collection)
                print(f"  Loaded {len(self._points_cosine)} points")
            except Exception as e:
                print(f"  WARNING: Could not load {self.cosine_collection}: {e}")
                self._points_cosine = []
        return self._points_cosine

    def _get_points_by_tier(self, points: list[dict], tier: str) -> list[dict]:
        """Filter points by tier (handles both unified and legacy tier names)."""
        return [p for p in points if p["tier"] == tier]

    # ------------------------------------------------------------------
    # run_all
    # ------------------------------------------------------------------

    def run_all(self) -> dict:
        """Run all 11 benchmark suites in order."""
        print(f"\n{'='*60}")
        print(f"Benchmarking dataset: {self.dataset}")
        print(f"Curvature: {self.curvature}")
        print(f"{'='*60}")

        self.suite_gromov_delta()
        self.suite_hierarchy_separation()
        self.suite_drill_down()
        self.suite_lateral()
        self.suite_cross_branch()
        self.suite_depth_band()
        self.suite_unified_vs_separate()
        self.suite_latency()
        self.suite_quantization_recall()
        self.suite_klein_prefilter()
        self.suite_geometric_filters()

        return self.results

    # ------------------------------------------------------------------
    # Suite 1: Gromov Delta
    # ------------------------------------------------------------------

    def suite_gromov_delta(self) -> dict:
        """Compute Gromov delta-hyperbolicity of the poincare vectors."""
        print(f"\n--- Suite 1: Gromov Delta ---")
        try:
            points = self._load_unified_points()
            if not points:
                result = {"error": "no points loaded"}
                self.results["gromov_delta"] = result
                return result

            # Extract poincare vectors
            vectors = []
            for pt in points:
                vec = pt.get("vectors", {}).get("poincare", pt["vector"])
                vectors.append(np.array(vec, dtype=np.float32))

            if len(vectors) < 4:
                result = {"error": "need at least 4 vectors"}
                self.results["gromov_delta"] = result
                return result

            vectors_array = np.array(vectors)
            num_samples = min(2000, len(vectors) * (len(vectors) - 1))
            delta, recommendation = gromov_delta(vectors_array, num_samples=num_samples)

            result = {
                "delta": float(delta),
                "recommendation": recommendation,
                "num_vectors": len(vectors),
                "num_samples": num_samples,
            }
            self.results["gromov_delta"] = result

            print(f"  delta = {delta:.4f}")
            print(f"  recommendation = {recommendation}")
            print(f"  num_vectors = {len(vectors)}")

        except Exception as e:
            result = {"error": str(e)}
            self.results["gromov_delta"] = result
            print(f"  ERROR: {e}")

        return self.results["gromov_delta"]

    # ------------------------------------------------------------------
    # Suite 2: Hierarchy Separation
    # ------------------------------------------------------------------

    def suite_hierarchy_separation(self) -> dict:
        """Compute Busemann depth per tier, separation metrics, and Einstein midpoint quality."""
        print(f"\n--- Suite 2: Hierarchy Separation ---")
        try:
            points = self._load_unified_points()
            if not points:
                result = {"error": "no points loaded"}
                self.results["hierarchy_separation"] = result
                return result

            # Reuse the existing benchmark function
            sep_res = benchmark_hierarchy_separation(points, curvature=self.curvature)

            # Add Einstein midpoint quality comparison
            tier_map = {"narrative": "root", "story": "mid", "content": "leaf"}
            tier_vectors: dict[str, list[np.ndarray]] = {"root": [], "mid": [], "leaf": []}
            for pt in points:
                tier = tier_map.get(pt["tier"], pt["tier"])
                if tier in tier_vectors:
                    vec = pt.get("vectors", {}).get("poincare", pt["vector"])
                    tier_vectors[tier].append(np.array(vec, dtype=np.float32))

            # Compute Einstein midpoints for each tier
            midpoints = {}
            for tier, vecs in tier_vectors.items():
                if len(vecs) >= 2:
                    try:
                        mp = einstein_midpoint(vecs, c=self.curvature)
                        midpoints[tier] = mp
                    except Exception:
                        pass

            # Compute pairwise midpoint distances if we have all three
            if len(midpoints) >= 2:
                midpoint_distances = {}
                tier_names = list(midpoints.keys())
                for i, t1 in enumerate(tier_names):
                    for t2 in tier_names[i + 1:]:
                        dist = float(np.linalg.norm(midpoints[t1] - midpoints[t2]))
                        midpoint_distances[f"{t1}_vs_{t2}"] = dist
                sep_res["einstein_midpoint_distances"] = midpoint_distances

                # Midpoint norms (distance from origin = depth proxy)
                midpoint_norms = {}
                for tier, mp in midpoints.items():
                    midpoint_norms[tier] = float(np.linalg.norm(mp))
                sep_res["einstein_midpoint_norms"] = midpoint_norms

            self.results["hierarchy_separation"] = sep_res

            # Print summary
            sep_ratio = sep_res.get("separation_ratio")
            cross_sep = sep_res.get("cross_tier_separation")
            band_ovlp = sep_res.get("band_overlap")
            tier_acc = sep_res.get("tier_classification_accuracy")
            print(f"  sep_ratio = {sep_ratio:.4f}" if sep_ratio else "  sep_ratio = N/A")
            print(f"  cross_tier_sep = {cross_sep:.4f}" if cross_sep else "  cross_tier_sep = N/A")
            print(f"  band_overlap = {band_ovlp:.4f}" if band_ovlp is not None else "  band_overlap = N/A")
            print(f"  tier_accuracy = {tier_acc:.4f}" if tier_acc else "  tier_accuracy = N/A")
            if "einstein_midpoint_distances" in sep_res:
                for pair, dist in sep_res["einstein_midpoint_distances"].items():
                    print(f"  einstein_midpoint_{pair} = {dist:.4f}")

        except Exception as e:
            result = {"error": str(e)}
            self.results["hierarchy_separation"] = result
            print(f"  ERROR: {e}")

        return self.results["hierarchy_separation"]

    # ------------------------------------------------------------------
    # Suite 3: Drill-Down Queries
    # ------------------------------------------------------------------

    def suite_drill_down(self) -> dict:
        """Evaluate vertical traversal: narrative -> stories -> content."""
        print(f"\n--- Suite 3: Drill-Down Queries ---")
        try:
            points = self._load_unified_points()
            if not points:
                result = {"error": "no points loaded"}
                self.results["drill_down"] = result
                return result

            # Select narrative-tier points
            narrative_points = [p for p in points if p["tier"] == "narrative"]
            if not narrative_points:
                result = {"error": "no narrative-tier points found"}
                self.results["drill_down"] = result
                print(f"  {result['error']}")
                return result

            sample = narrative_points[:100]
            print(f"  Testing {len(sample)} narrative points...")

            # Build child_ids lookup from payload
            id_to_pt = {p["id"]: p for p in points}

            methods = ["poincare", "dense"]
            method_results: dict[str, dict] = {}

            for method in methods:
                hop1_recalls = []
                precisions = []
                latencies = []
                errors = 0

                for pt in sample:
                    result = drill_down_query(
                        self.client, self.unified_collection,
                        start_point_id=pt["id"], using=method, limit=10, ef=128,
                    )

                    if "error" in result:
                        errors += 1
                        continue

                    latencies.append(result["total_latency_ms"])

                    # Measure hop1 recall: what fraction of results are actual children?
                    # We check by looking at parent_ids of returned points
                    hop1_results = result.get("hop1_results", [])
                    if hop1_results:
                        # Count results whose parent_ids match our start point
                        point_id_str = pt.get("point_id", str(pt["id"]))
                        true_children = 0
                        for r in hop1_results:
                            r_payload = r.payload if hasattr(r, "payload") and r.payload else {}
                            r_parent_ids = r_payload.get("parent_ids", [])
                            if point_id_str in r_parent_ids:
                                true_children += 1
                        hop1_recalls.append(true_children / len(hop1_results))

                        # Precision: fraction sharing domain with query
                        same_domain = 0
                        for r in hop1_results:
                            r_payload = r.payload if hasattr(r, "payload") and r.payload else {}
                            if r_payload.get("domain") == pt.get("domain"):
                                same_domain += 1
                        precisions.append(same_domain / len(hop1_results))

                def _safe_mean(lst):
                    return float(np.mean(lst)) if lst else None

                method_results[method] = {
                    "num_queries": len(sample) - errors,
                    "errors": errors,
                    "hop1_recall": _safe_mean(hop1_recalls),
                    "precision": _safe_mean(precisions),
                    "avg_latency_ms": _safe_mean(latencies),
                    "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
                }

                print(f"  {method}: hop1_recall={_safe_mean(hop1_recalls):.4f}" if hop1_recalls else f"  {method}: hop1_recall=N/A", end="")
                print(f", precision={_safe_mean(precisions):.4f}" if precisions else ", precision=N/A", end="")
                print(f", avg_latency={_safe_mean(latencies):.1f}ms" if latencies else ", avg_latency=N/A")

            self.results["drill_down"] = method_results

        except Exception as e:
            result = {"error": str(e)}
            self.results["drill_down"] = result
            print(f"  ERROR: {e}")

        return self.results["drill_down"]

    # ------------------------------------------------------------------
    # Suite 4: Lateral Exploration
    # ------------------------------------------------------------------

    def suite_lateral(self) -> dict:
        """Evaluate horizontal search within the same tier."""
        print(f"\n--- Suite 4: Lateral Exploration ---")
        try:
            points = self._load_unified_points()
            if not points:
                result = {"error": "no points loaded"}
                self.results["lateral"] = result
                return result

            # Sample random points
            sample_size = min(200, len(points))
            sample = random.sample(points, sample_size)
            print(f"  Testing {sample_size} random points...")

            id_to_pt = {p["id"]: p for p in points}
            methods = ["dense", "poincare"]
            method_results: dict[str, dict] = {}

            for method in methods:
                area_recalls = []
                domain_recalls = []
                diversities = []
                latencies = []
                errors = 0

                for pt in sample:
                    result = lateral_query(
                        self.client, self.unified_collection,
                        query_point_id=pt["id"], using=method, limit=10, ef=128,
                    )

                    if "error" in result:
                        errors += 1
                        continue

                    latencies.append(result["latency_ms"])
                    results_list = result.get("results", [])

                    if not results_list:
                        continue

                    # Measure area recall: same area as query
                    same_area = 0
                    same_domain = 0
                    unique_areas = set()

                    for r in results_list:
                        r_payload = r.payload if hasattr(r, "payload") and r.payload else {}
                        r_pt = {"area": r_payload.get("area"), "domain": r_payload.get("domain"),
                                "all_paths": r_payload.get("all_paths", []),
                                "hierarchy_path": r_payload.get("hierarchy_path", "")}

                        if _paths_match_area(pt, r_pt):
                            same_area += 1
                        if _paths_match_domain(pt, r_pt):
                            same_domain += 1

                        area_val = r_payload.get("area")
                        if area_val is not None:
                            unique_areas.add(area_val)

                    n = len(results_list)
                    area_recalls.append(same_area / n)
                    domain_recalls.append(same_domain / n)
                    diversities.append(len(unique_areas))

                def _safe_mean(lst):
                    return float(np.mean(lst)) if lst else None

                method_results[method] = {
                    "num_queries": len(sample) - errors,
                    "errors": errors,
                    "area_recall": _safe_mean(area_recalls),
                    "domain_recall": _safe_mean(domain_recalls),
                    "avg_diversity": _safe_mean(diversities),
                    "avg_latency_ms": _safe_mean(latencies),
                }

                print(f"  {method}: area_recall={_safe_mean(area_recalls):.4f}" if area_recalls else f"  {method}: area_recall=N/A", end="")
                print(f", domain_recall={_safe_mean(domain_recalls):.4f}" if domain_recalls else ", domain_recall=N/A", end="")
                print(f", diversity={_safe_mean(diversities):.1f}" if diversities else ", diversity=N/A")

            self.results["lateral"] = method_results

        except Exception as e:
            result = {"error": str(e)}
            self.results["lateral"] = result
            print(f"  ERROR: {e}")

        return self.results["lateral"]

    # ------------------------------------------------------------------
    # Suite 5: Cross-Branch Discovery
    # ------------------------------------------------------------------

    def suite_cross_branch(self) -> dict:
        """Evaluate structural similarity search across domains."""
        print(f"\n--- Suite 5: Cross-Branch Discovery ---")
        try:
            points = self._load_unified_points()
            if not points:
                result = {"error": "no points loaded"}
                self.results["cross_branch"] = result
                return result

            # Select narrative points
            narrative_points = [p for p in points if p["tier"] == "narrative"]
            if not narrative_points:
                result = {"error": "no narrative-tier points found"}
                self.results["cross_branch"] = result
                print(f"  {result['error']}")
                return result

            sample = narrative_points[:50]
            print(f"  Testing {len(sample)} narrative points...")

            methods = ["poincare", "dense"]
            method_results: dict[str, dict] = {}

            for method in methods:
                unique_domains_counts = []
                depth_similarities = []
                latencies = []
                errors = 0

                for pt in sample:
                    result = cross_branch_query(
                        self.client, self.unified_collection,
                        narrative_point_id=pt["id"], using=method, limit=10, ef=128,
                    )

                    if "error" in result:
                        errors += 1
                        continue

                    latencies.append(result["latency_ms"])
                    results_list = result.get("results", [])

                    if not results_list:
                        continue

                    # Count unique domains in results
                    domains = set()
                    depths = []
                    for r in results_list:
                        r_payload = r.payload if hasattr(r, "payload") and r.payload else {}
                        domain = r_payload.get("domain")
                        if domain is not None:
                            domains.add(domain)
                        bd = r_payload.get("busemann_depth")
                        if bd is not None:
                            depths.append(bd)

                    unique_domains_counts.append(len(domains))

                    # Structural depth similarity: how close are result depths to query depth?
                    query_depth = pt.get("busemann_depth")
                    if query_depth is not None and depths:
                        depth_diffs = [abs(d - query_depth) for d in depths]
                        depth_similarities.append(float(np.mean(depth_diffs)))

                def _safe_mean(lst):
                    return float(np.mean(lst)) if lst else None

                method_results[method] = {
                    "num_queries": len(sample) - errors,
                    "errors": errors,
                    "avg_unique_domains": _safe_mean(unique_domains_counts),
                    "avg_depth_similarity": _safe_mean(depth_similarities),
                    "avg_latency_ms": _safe_mean(latencies),
                }

                print(f"  {method}: unique_domains={_safe_mean(unique_domains_counts):.1f}" if unique_domains_counts else f"  {method}: unique_domains=N/A", end="")
                print(f", depth_sim={_safe_mean(depth_similarities):.4f}" if depth_similarities else ", depth_sim=N/A")

            self.results["cross_branch"] = method_results

        except Exception as e:
            result = {"error": str(e)}
            self.results["cross_branch"] = result
            print(f"  ERROR: {e}")

        return self.results["cross_branch"]

    # ------------------------------------------------------------------
    # Suite 6: Depth-Band Filtering
    # ------------------------------------------------------------------

    def suite_depth_band(self) -> dict:
        """Evaluate Busemann depth-band filtered search precision."""
        print(f"\n--- Suite 6: Depth-Band Filtering ---")
        try:
            points = self._load_unified_points()
            if not points:
                result = {"error": "no points loaded"}
                self.results["depth_band"] = result
                return result

            # Compute depth stats from hierarchy separation or from points
            tier_map = {"narrative": "root", "story": "mid", "content": "leaf"}
            tier_depths: dict[str, list[float]] = {"root": [], "mid": [], "leaf": []}
            for pt in points:
                bd = pt.get("busemann_depth")
                if bd is not None:
                    tier = tier_map.get(pt["tier"], pt["tier"])
                    if tier in tier_depths:
                        tier_depths[tier].append(bd)

            # Define depth bands from tier statistics
            bands: list[dict] = []
            for tier in ("root", "mid", "leaf"):
                ds = tier_depths[tier]
                if ds:
                    mean_d = float(np.mean(ds))
                    std_d = float(np.std(ds))
                    bands.append({
                        "name": tier,
                        "depth_min": mean_d - 2 * std_d,
                        "depth_max": mean_d + 2 * std_d,
                        "expected_tier": tier,
                    })

            if not bands:
                # Fallback: use default bands if no busemann_depth in payload
                bands = [
                    {"name": "shallow", "depth_min": -2.0, "depth_max": 0.0, "expected_tier": "root"},
                    {"name": "middle", "depth_min": 0.0, "depth_max": 1.0, "expected_tier": "mid"},
                    {"name": "deep", "depth_min": 1.0, "depth_max": 3.0, "expected_tier": "leaf"},
                ]

            print(f"  Defined {len(bands)} depth bands")

            # Sample query vectors from content tier
            content_points = [p for p in points if p["tier"] in ("leaf", "content")]
            if not content_points:
                content_points = points
            query_sample = random.sample(content_points, min(100, len(content_points)))

            band_results: dict[str, dict] = {}
            for band in bands:
                precisions = []
                counts = []
                latencies = []

                for qpt in query_sample:
                    query_vec = qpt.get("vectors", {}).get("poincare", qpt["vector"])
                    if hasattr(query_vec, "tolist"):
                        query_vec_list = query_vec.tolist()
                    else:
                        query_vec_list = list(query_vec)

                    result = depth_band_query(
                        self.client, self.unified_collection,
                        query_vector=query_vec_list,
                        depth_min=band["depth_min"],
                        depth_max=band["depth_max"],
                        using="poincare", limit=10, ef=128,
                    )

                    if "error" in result:
                        continue

                    latencies.append(result["latency_ms"])
                    results_list = result.get("results", [])
                    counts.append(len(results_list))

                    if results_list:
                        # Precision: fraction of results actually in the depth band
                        in_band = 0
                        for r in results_list:
                            r_payload = r.payload if hasattr(r, "payload") and r.payload else {}
                            bd = r_payload.get("busemann_depth")
                            if bd is not None and band["depth_min"] <= bd <= band["depth_max"]:
                                in_band += 1
                        precisions.append(in_band / len(results_list))

                def _safe_mean(lst):
                    return float(np.mean(lst)) if lst else None

                band_results[band["name"]] = {
                    "depth_min": band["depth_min"],
                    "depth_max": band["depth_max"],
                    "precision": _safe_mean(precisions),
                    "avg_result_count": _safe_mean(counts),
                    "avg_latency_ms": _safe_mean(latencies),
                    "num_queries": len(precisions),
                }

                print(f"  band={band['name']}: precision={_safe_mean(precisions):.4f}" if precisions else f"  band={band['name']}: precision=N/A", end="")
                print(f", avg_count={_safe_mean(counts):.1f}" if counts else ", avg_count=N/A")

            self.results["depth_band"] = band_results

        except Exception as e:
            result = {"error": str(e)}
            self.results["depth_band"] = result
            print(f"  ERROR: {e}")

        return self.results["depth_band"]

    # ------------------------------------------------------------------
    # Suite 7: Unified vs Separate
    # ------------------------------------------------------------------

    def suite_unified_vs_separate(self) -> dict:
        """Compare unified collection vs separate per-tier collections."""
        print(f"\n--- Suite 7: Unified vs Separate ---")
        try:
            # Check which separate collections exist
            separate_collections = {
                "narratives": f"{self.dataset}_narratives",
                "stories": f"{self.dataset}_stories",
                "content": f"{self.dataset}_content",
            }

            existing_separate: dict[str, str] = {}
            try:
                all_collections = {c.name for c in self.client.get_collections().collections}
            except Exception:
                all_collections = set()

            for tier_name, col_name in separate_collections.items():
                if col_name in all_collections:
                    existing_separate[tier_name] = col_name

            if not existing_separate:
                result = {"note": "no separate tier collections found, skipping comparison"}
                self.results["unified_vs_separate"] = result
                print(f"  No separate collections found ({list(separate_collections.values())})")
                print("  Skipping comparison")
                return result

            print(f"  Found separate collections: {list(existing_separate.values())}")

            # Load unified points
            points_unified = self._load_unified_points()
            if not points_unified:
                result = {"error": "no unified points loaded"}
                self.results["unified_vs_separate"] = result
                return result

            unified_id_to_pt = {p["id"]: p for p in points_unified}

            comparison = {}

            # Run drill-down on unified
            narrative_points = [p for p in points_unified if p["tier"] == "narrative"]
            drill_sample = narrative_points[:50]

            if drill_sample:
                # Unified drill-down
                unified_latencies = []
                unified_recalls = []
                for pt in drill_sample:
                    res = drill_down_query(
                        self.client, self.unified_collection,
                        start_point_id=pt["id"], using="poincare", limit=10, ef=128,
                    )
                    if "error" not in res:
                        unified_latencies.append(res["total_latency_ms"])
                        hop1 = res.get("hop1_results", [])
                        if hop1:
                            point_id_str = str(pt["id"])
                            children = sum(
                                1 for r in hop1
                                if point_id_str in (r.payload or {}).get("parent_ids", [])
                                or pt["id"] in (r.payload or {}).get("parent_ids", [])
                            )
                            unified_recalls.append(children / len(hop1))

                def _safe_mean(lst):
                    return float(np.mean(lst)) if lst else None

                comparison["unified_drill_down"] = {
                    "avg_latency_ms": _safe_mean(unified_latencies),
                    "hop1_recall": _safe_mean(unified_recalls),
                    "num_queries": len(drill_sample),
                }

            # Lateral on unified
            lateral_sample = random.sample(points_unified, min(100, len(points_unified)))
            unified_lat_latencies = []
            unified_lat_area_recalls = []
            for pt in lateral_sample:
                res = lateral_query(
                    self.client, self.unified_collection,
                    query_point_id=pt["id"], using="dense", limit=10, ef=128,
                )
                if "error" not in res:
                    unified_lat_latencies.append(res["latency_ms"])
                    results_list = res.get("results", [])
                    if results_list:
                        same_area = sum(
                            1 for r in results_list
                            if _paths_match_area(
                                pt,
                                {"area": (r.payload or {}).get("area"),
                                 "domain": (r.payload or {}).get("domain"),
                                 "all_paths": (r.payload or {}).get("all_paths", []),
                                 "hierarchy_path": (r.payload or {}).get("hierarchy_path", "")},
                            )
                        )
                        unified_lat_area_recalls.append(same_area / len(results_list))

            comparison["unified_lateral"] = {
                "avg_latency_ms": _safe_mean(unified_lat_latencies),
                "area_recall": _safe_mean(unified_lat_area_recalls),
                "num_queries": len(lateral_sample),
            }

            # Try separate collections for comparison
            for tier_name, col_name in existing_separate.items():
                try:
                    sep_points = scroll_all(self.client, col_name)
                    if not sep_points:
                        continue
                    sep_sample = random.sample(sep_points, min(50, len(sep_points)))

                    sep_latencies = []
                    sep_area_recalls = []
                    sep_id_to_pt = {p["id"]: p for p in sep_points}

                    for qpt in sep_sample:
                        try:
                            results = _search(self.client, col_name, qpt["vector"], limit=11)
                            neighbors = [r for r in results if r.id != qpt["id"]][:10]
                            if not neighbors:
                                continue
                            same_area = sum(
                                1 for nb in neighbors
                                if nb.id in sep_id_to_pt
                                and _paths_match_area(qpt, sep_id_to_pt[nb.id])
                            )
                            sep_area_recalls.append(same_area / len(neighbors))
                        except Exception:
                            continue

                    comparison[f"separate_{tier_name}"] = {
                        "collection": col_name,
                        "num_points": len(sep_points),
                        "area_recall": _safe_mean(sep_area_recalls),
                        "num_queries": len(sep_area_recalls),
                    }
                    print(f"  {col_name}: area_recall={_safe_mean(sep_area_recalls):.4f}" if sep_area_recalls else f"  {col_name}: area_recall=N/A")
                except Exception as e:
                    comparison[f"separate_{tier_name}"] = {"error": str(e)}

            # Print unified summary
            ud = comparison.get("unified_drill_down", {})
            ul = comparison.get("unified_lateral", {})
            print(f"  unified drill_down: recall={ud.get('hop1_recall', 'N/A')}, latency={ud.get('avg_latency_ms', 'N/A')}")
            print(f"  unified lateral: area_recall={ul.get('area_recall', 'N/A')}, latency={ul.get('avg_latency_ms', 'N/A')}")

            self.results["unified_vs_separate"] = comparison

        except Exception as e:
            result = {"error": str(e)}
            self.results["unified_vs_separate"] = result
            print(f"  ERROR: {e}")

        return self.results["unified_vs_separate"]

    # ------------------------------------------------------------------
    # Suite 8: Latency
    # ------------------------------------------------------------------

    def suite_latency(self) -> dict:
        """Measure throughput and latency at multiple ef values."""
        print(f"\n--- Suite 8: Latency ---")
        try:
            points = self._load_unified_points()
            if not points:
                result = {"error": "no points loaded"}
                self.results["latency"] = result
                return result

            lat_res = benchmark_latency(
                self.client, self.unified_collection, points,
                num_queries=1000, vector_name="poincare",
            )
            self.results["latency"] = lat_res

            for ef in EF_VALUES:
                ef_data = lat_res.get(f"ef_{ef}", {})
                qps = ef_data.get("qps")
                p50 = ef_data.get("p50_ms")
                p95 = ef_data.get("p95_ms")
                p99 = ef_data.get("p99_ms")
                if qps and p95:
                    print(f"  ef={ef}: qps={qps:.1f}, p50={p50:.1f}ms, p95={p95:.1f}ms, p99={p99:.1f}ms")

        except Exception as e:
            result = {"error": str(e)}
            self.results["latency"] = result
            print(f"  ERROR: {e}")

        return self.results["latency"]

    # ------------------------------------------------------------------
    # Suite 9: Quantization Recall
    # ------------------------------------------------------------------

    def suite_quantization_recall(self):
        """Measure recall of quantized vs exact Poincaré search."""
        print(f"\n--- Suite 9: Quantization Recall ---")
        try:
            points = self._load_unified_points()
            sample = random.sample(points, min(200, len(points)))
            recalls = []

            for pt in tqdm(sample, desc="  quantization recall"):
                vec = pt.get("vectors", {}).get("poincare", pt["vector"])
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()

                try:
                    quantized = self.client.query_points(
                        collection_name=self.unified_collection,
                        query=vec, using="poincare", limit=10,
                        search_params=SearchParams(hnsw_ef=128),
                        with_payload=False,
                    ).points
                except Exception:
                    continue

                try:
                    exact = self.client.query_points(
                        collection_name=self.unified_collection,
                        query=vec, using="poincare", limit=10,
                        search_params=SearchParams(hnsw_ef=128, exact=True),
                        with_payload=False,
                    ).points
                except Exception:
                    continue

                q_ids = {r.id for r in quantized}
                e_ids = {r.id for r in exact}
                if e_ids:
                    recalls.append(len(q_ids & e_ids) / len(e_ids))

            result = {
                "recall_at_10": float(np.mean(recalls)) if recalls else None,
                "recall_min": float(np.min(recalls)) if recalls else None,
                "recall_max": float(np.max(recalls)) if recalls else None,
                "num_queries": len(recalls),
            }
            self.results["quantization_recall"] = result
            if recalls:
                print(f"  recall@10: {np.mean(recalls):.4f} (min={np.min(recalls):.4f}, max={np.max(recalls):.4f})")
            else:
                print("  No valid recall measurements")
        except Exception as e:
            self.results["quantization_recall"] = {"error": str(e)}
            print(f"  ERROR: {e}")

    # ------------------------------------------------------------------
    # Suite 10: Klein Pre-Filter
    # ------------------------------------------------------------------

    def suite_klein_prefilter(self):
        """Compare Klein pre-filter pipeline vs direct Poincaré search."""
        print(f"\n--- Suite 10: Klein Pre-Filter ---")
        try:
            from queries import klein_prefilter_query
            points = self._load_unified_points()
            sample = random.sample(points, min(100, len(points)))

            klein_latencies = []
            recall_vs_direct = []

            for pt in tqdm(sample, desc="  klein prefilter"):
                poincare_vec = pt.get("vectors", {}).get("poincare", pt["vector"])

                # Direct Poincaré search baseline
                try:
                    vec_list = poincare_vec.tolist() if hasattr(poincare_vec, "tolist") else poincare_vec
                    direct = self.client.query_points(
                        collection_name=self.unified_collection,
                        query=vec_list, using="poincare", limit=10,
                        search_params=SearchParams(hnsw_ef=128),
                        with_payload=False,
                    ).points
                    direct_ids = {r.id for r in direct}
                except Exception:
                    continue

                # Klein pre-filter pipeline
                result = klein_prefilter_query(
                    self.client, self.unified_collection,
                    poincare_vec, curvature=self.curvature,
                    prefetch_limit=200, klein_top=50, final_top=10,
                )
                if "error" in result:
                    continue

                klein_ids = {r.id for r in result["results"]}
                if direct_ids:
                    recall_vs_direct.append(len(klein_ids & direct_ids) / len(direct_ids))
                klein_latencies.append(result["total_latency_ms"])

            self.results["klein_prefilter"] = {
                "recall_vs_direct": float(np.mean(recall_vs_direct)) if recall_vs_direct else None,
                "avg_klein_latency_ms": float(np.mean(klein_latencies)) if klein_latencies else None,
                "num_queries": len(recall_vs_direct),
            }
            if recall_vs_direct:
                print(f"  recall vs direct: {np.mean(recall_vs_direct):.4f}")
                print(f"  avg Klein latency: {np.mean(klein_latencies):.1f}ms")
            else:
                print("  No valid measurements")
        except Exception as e:
            self.results["klein_prefilter"] = {"error": str(e)}
            print(f"  ERROR: {e}")

    # ------------------------------------------------------------------
    # Suite 11: Geometric Filters
    # ------------------------------------------------------------------

    def suite_geometric_filters(self):
        """Validate geometric filter precision and selectivity."""
        print(f"\n--- Suite 11: Geometric Filters ---")
        try:
            from queries import geometric_filtered_query
            points = self._load_unified_points()
            content_points = [p for p in points if p.get("item_type", p.get("tier")) == "content"]
            sample = random.sample(content_points, min(100, len(content_points)))

            # Test InBall
            inball_selectivities = []
            for pt in tqdm(sample[:50], desc="  inball"):
                result = geometric_filtered_query(
                    self.client, self.unified_collection, pt["vector"],
                    filters=[{"type": "inball", "center": pt["vector"], "radius": 1.5}],
                    using="poincare", limit=100, final_top=50, curvature=self.curvature,
                )
                if "error" not in result and result["initial_count"] > 0:
                    inball_selectivities.append(result["filtered_count"] / result["initial_count"])

            # Test InCone
            incone_selectivities = []
            for pt in tqdm(sample[:50], desc="  incone"):
                result = geometric_filtered_query(
                    self.client, self.unified_collection, pt["vector"],
                    filters=[{"type": "incone", "axis": pt["vector"], "aperture": 0.5}],
                    using="poincare", limit=100, final_top=50, curvature=self.curvature,
                )
                if "error" not in result and result["initial_count"] > 0:
                    incone_selectivities.append(result["filtered_count"] / result["initial_count"])

            # Test composition: InBall + DepthBand
            composed_selectivities = []
            for pt in tqdm(sample[:50], desc="  composed"):
                depth = pt.get("busemann_depth", 1.0)
                result = geometric_filtered_query(
                    self.client, self.unified_collection, pt["vector"],
                    filters=[
                        {"type": "inball", "center": pt["vector"], "radius": 2.0},
                        {"type": "depth_band", "depth_min": depth - 0.2, "depth_max": depth + 0.2},
                    ],
                    using="poincare", limit=100, final_top=50, curvature=self.curvature,
                )
                if "error" not in result and result["initial_count"] > 0:
                    composed_selectivities.append(result["filtered_count"] / result["initial_count"])

            self.results["geometric_filters"] = {
                "inball_selectivity": float(np.mean(inball_selectivities)) if inball_selectivities else None,
                "incone_selectivity": float(np.mean(incone_selectivities)) if incone_selectivities else None,
                "composed_selectivity": float(np.mean(composed_selectivities)) if composed_selectivities else None,
                "num_queries": len(inball_selectivities),
            }
            if inball_selectivities:
                print(f"  inball selectivity: {np.mean(inball_selectivities):.4f}")
            if incone_selectivities:
                print(f"  incone selectivity: {np.mean(incone_selectivities):.4f}")
            if composed_selectivities:
                print(f"  composed selectivity: {np.mean(composed_selectivities):.4f}")
        except Exception as e:
            self.results["geometric_filters"] = {"error": str(e)}
            print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark hyperbolic vector collections in Qdrant"
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6334",
        help="Qdrant gRPC/HTTP URL (default: http://localhost:6334)",
    )
    parser.add_argument(
        "--dataset",
        default="bgc",
        choices=["bgc", "hwv", "wos", "eurlex"],
        help="Which dataset to benchmark (default: bgc)",
    )
    parser.add_argument(
        "--suites",
        nargs="*",
        default=None,
        help="Run specific suites (default: all). "
             "Options: gromov_delta, hierarchy_separation, drill_down, lateral, "
             "cross_branch, depth_band, unified_vs_separate, latency, "
             "quantization_recall, klein_prefilter, geometric_filters",
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=5.0,
        help="Poincare ball curvature (default: 5.0)",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    client = QdrantClient(url=args.qdrant_url, timeout=60)

    runner = BenchmarkRunner(client, args.qdrant_url, args.dataset, curvature=args.curvature)

    if args.suites:
        for suite_name in args.suites:
            method = getattr(runner, f"suite_{suite_name}", None)
            if method:
                method()
            else:
                print(f"Unknown suite: {suite_name}")
                print("  Available: gromov_delta, hierarchy_separation, drill_down, "
                      "lateral, cross_branch, depth_band, unified_vs_separate, latency, "
                      "quantization_recall, klein_prefilter, geometric_filters")
    else:
        runner.run_all()

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = results_dir / f"benchmark_{args.dataset}_{timestamp}.json"
    with open(output, "w") as f:
        json.dump(runner.results, f, indent=2, default=str)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
