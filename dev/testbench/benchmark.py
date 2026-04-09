"""Benchmark script for hyperbolic (Poincare / Busemann) vector collections in Qdrant.

Runs 4 benchmark suites:
  1. Hierarchy Separation  — Busemann depth per tier + band overlap
  2. Retrieval Quality     — recall@10 within hierarchy
  3. Curvature Comparison  — summary table (printed after 1, 2, 4)
  4. Performance & Latency — throughput and p50/p95/p99

Auto-discovers wos_* collections and adapts to whatever exists.

Usage:
  python benchmark.py [--qdrant-url http://localhost:6334]
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams

from fusion import rrf_fuse
from hyperbolic_math import (
    EPS,
    busemann_depth_single,
    compute_busemann_depths,
    compute_focal_direction,
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

    For unified collections, specify vector_name ('cosine' or 'poincare')
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
                "domain": pt.payload.get("domain", -1),
                "area": pt.payload.get("area", -1),
                "busemann_depth": pt.payload.get("busemann_depth"),
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

            if nb_pt["area"] == query_pt["area"]:
                same_area += 1
                h_prec += 1.0
            elif nb_pt["domain"] == query_pt["domain"]:
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
        "cosine_tier_filtered": ("cosine", lambda q: _search_named(client, collection, "cosine", q, limit=k + 1, tier_filter="content")),
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
                if nb_pt["area"] == qpt["area"]:
                    same_area += 1
                    h_prec += 1.0
                elif nb_pt["domain"] == qpt["domain"]:
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
        cosine_vec = vectors.get("cosine", qpt["vector"])
        poincare_vec = vectors.get("poincare", qpt["vector"])

        # Fetch results from both spaces using correct vectors
        try:
            cos_results = _search(client, cosine_collection, cosine_vec, limit=k + 1)
        except Exception:
            continue

        try:
            cos_unified = _search_named(client, unified_collection, "cosine", cosine_vec, limit=over_fetch, tier_filter="content")
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
                if nb_pt["area"] == query_area:
                    same_area += 1
                elif nb_pt["domain"] == query_domain:
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
        "--suite",
        default="all",
        choices=["all", "hierarchy", "retrieval", "cross-tier", "dual-space", "latency"],
        help="Which benchmark suite to run (default: all)",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    client = QdrantClient(url=args.qdrant_url)

    collection_configs = discover_collections(client, args.qdrant_url)
    if not collection_configs:
        print("No wos_* collections found. Run embed.py first.")
        return

    print(f"Discovered collections: {[c['name'] for c in collection_configs]}")

    all_results: dict = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"benchmark_{timestamp}.json"

    # Find unified + cosine collection pairs grouped by prefix (wos, bgc, scihtc)
    # Each prefix can have at most one unified and one cosine collection
    dataset_pairs: list[tuple[str, str]] = []  # (unified_name, cosine_name)
    prefix_unified: dict[str, str] = {}
    prefix_cosine: dict[str, str] = {}
    for c in collection_configs:
        # Extract prefix from name (e.g., "wos" from "wos_unified")
        name = c["name"]
        for pfx in ["wos", "bgc", "scihtc", "eurlex"]:
            if name.startswith(pfx + "_"):
                if c.get("strategy") == "unified" and "_c" not in name.replace(pfx + "_unified", ""):
                    prefix_unified[pfx] = name
                if c.get("strategy") is None and name == pfx + "_cosine":
                    prefix_cosine[pfx] = name
                break

    for pfx in sorted(set(prefix_unified) & set(prefix_cosine)):
        dataset_pairs.append((prefix_unified[pfx], prefix_cosine[pfx]))

    # --- Standard per-collection benchmarks ---
    for col_cfg in collection_configs:
        collection = col_cfg["name"]
        curvature = col_cfg["curvature"]
        strategy = col_cfg.get("strategy") or "cosine"

        print(f"\n{'='*60}")
        print(f"Collection: {collection}  (strategy={strategy}, curvature={curvature})")
        print(f"{'='*60}")

        vector_name = "poincare" if strategy == "unified" else None
        print("  Loading points...")
        points = scroll_all(client, collection, vector_name=vector_name)
        print(f"  Loaded {len(points)} points")

        col_results: dict = {}

        # --- Benchmark 1: Hierarchy Separation ---
        if args.suite in ("all", "hierarchy"):
            if col_cfg["is_poincare"] and curvature is not None:
                print("  [1] Hierarchy Separation...")
                sep_res = benchmark_hierarchy_separation(points, curvature=curvature)
                col_results["hierarchy_separation"] = sep_res
                sep_ratio = sep_res.get("separation_ratio")
                sep_content = sep_res.get("separation_ratio_content")
                cross_sep = sep_res.get("cross_tier_separation")
                band_ovlp = sep_res.get("band_overlap")
                tier_acc = sep_res.get("tier_classification_accuracy")
                print(f"      sep_ratio_all={sep_ratio:.4f}" if sep_ratio else "      sep_ratio_all=N/A")
                print(f"      sep_ratio_content={sep_content:.4f}" if sep_content else "      sep_ratio_content=N/A")
                print(f"      cross_tier_sep={cross_sep:.4f}" if cross_sep else "      cross_tier_sep=N/A")
                print(f"      band_overlap={band_ovlp:.4f}" if band_ovlp else "      band_overlap=N/A")
                print(f"      tier_accuracy={tier_acc:.4f}" if tier_acc else "      tier_accuracy=N/A")
            else:
                col_results["hierarchy_separation"] = {}

        # --- Benchmark 2: Retrieval Quality ---
        if args.suite in ("all", "retrieval"):
            print("  [2] Retrieval Quality (500 queries)...")
            rq_res = benchmark_retrieval_quality(client, collection, points, k=10, num_queries=500, vector_name=vector_name)
            col_results["retrieval_quality"] = rq_res
            ar = rq_res.get("area_recall_at_10")
            dr = rq_res.get("domain_recall_at_10")
            hp = rq_res.get("hierarchical_precision")
            print(f"      area_recall@10={ar:.4f}" if ar else "      area_recall@10=N/A")
            print(f"      domain_recall@10={dr:.4f}" if dr else "      domain_recall@10=N/A")
            print(f"      h_precision={hp:.4f}" if hp else "      h_precision=N/A")

        # --- Benchmark 5: Latency ---
        if args.suite in ("all", "latency"):
            print("  [5] Latency (1000 queries)...")
            lat_res = benchmark_latency(client, collection, points, num_queries=1000, vector_name=vector_name)
            col_results["latency"] = lat_res
            for ef in EF_VALUES:
                ef_data = lat_res.get(f"ef_{ef}", {})
                qps = ef_data.get("qps")
                p95 = ef_data.get("p95_ms")
                if qps and p95:
                    print(f"      ef={ef}: qps={qps:.1f}, p95={p95:.1f}ms")

        all_results[collection] = col_results

    # --- Unified-specific benchmarks (per dataset pair) ---
    for unified_name, cosine_name in dataset_pairs:
        print(f"\n  === Dataset pair: unified={unified_name}, cosine={cosine_name} ===")

        # --- Benchmark 3: Cross-Tier Retrieval ---
        if args.suite in ("all", "cross-tier"):
            print(f"\n{'='*60}")
            print(f"Cross-Tier Retrieval ({unified_name}, poincare)")
            print(f"{'='*60}")
            points_unified = scroll_all(client, unified_name, vector_name="poincare")
            ct_res = benchmark_cross_tier(client, unified_name, points_unified, k=5)
            all_results.setdefault(unified_name, {})["cross_tier"] = ct_res
            print(f"  parent_story_recall:     {ct_res.get('parent_story_recall')}")
            print(f"  parent_narrative_recall:  {ct_res.get('parent_narrative_recall')}")
            print(f"  child_story_recall:       {ct_res.get('child_story_recall')}")

        # --- Benchmark 2b: Multi-Mode Retrieval ---
        if args.suite in ("all", "retrieval"):
            print(f"\n{'='*60}")
            print(f"Multi-Mode Retrieval Quality ({unified_name})")
            print(f"{'='*60}")
            points_unified_poincare = scroll_all(client, unified_name, vector_name="poincare")
            mr_res = benchmark_retrieval_modes(client, unified_name, points_unified_poincare, k=10, num_queries=500)
            all_results.setdefault(unified_name, {})["retrieval_modes"] = mr_res

            print(f"\n  {'Mode':<24} {'AreaRec@10':>12} {'DomRec@10':>12} {'H-Prec':>8}")
            print(f"  {'-'*56}")
            for mode in ("unfiltered", "tier_filtered", "depth_range", "cosine_tier_filtered"):
                m = mr_res.get(mode, {})
                ar = m.get("area_recall_at_10")
                dr = m.get("domain_recall_at_10")
                hp = m.get("hierarchical_precision")
                ar_s = f"{ar:.4f}" if ar is not None else "N/A"
                dr_s = f"{dr:.4f}" if dr is not None else "N/A"
                hp_s = f"{hp:.3f}" if hp is not None else "N/A"
                print(f"  {mode:<24} {ar_s:>12} {dr_s:>12} {hp_s:>8}")
            band = mr_res.get("depth_range_band", {})
            print(f"  depth_range band: [{band.get('lo', '?'):.3f}, {band.get('hi', '?'):.3f}]")

        # --- Benchmark 4: Fusion Strategy Comparison ---
        if args.suite in ("all", "dual-space"):
            print(f"\n{'='*60}")
            print("Fusion Strategy Comparison")
            print(f"{'='*60}")
            points_unified_cosine = scroll_all(client, unified_name, vector_name="cosine")
            points_cosine_baseline = scroll_all(client, cosine_name)
            ds_res = benchmark_dual_space(
                client, unified_name, cosine_name,
                points_unified_cosine, points_cosine_baseline,
                k=10, num_queries=500,
            )
            all_results.setdefault(unified_name, {})["dual_space"] = ds_res

            # Sort strategies by area_recall descending
            strat_scores = []
            for key, val in ds_res.items():
                if isinstance(val, dict) and "area_recall_at_10" in val:
                    strat_scores.append((key, val))
            strat_scores.sort(key=lambda x: -(x[1].get("area_recall_at_10") or 0))

            print(f"\n  {'Strategy':<24} {'AreaRec@10':>12} {'DomRec@10':>12}")
            print(f"  {'-'*48}")
            for key, val in strat_scores:
                ar = val.get("area_recall_at_10")
                dr = val.get("domain_recall_at_10")
                ar_s = f"{ar:.4f}" if ar is not None else "N/A"
                dr_s = f"{dr:.4f}" if dr is not None else "N/A"
                marker = " <-- best" if strat_scores[0][0] == key else ""
                print(f"  {key:<24} {ar_s:>12} {dr_s:>12}{marker}")

    # --- Comparison table ---
    print_comparison_table(all_results, collection_configs)

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
