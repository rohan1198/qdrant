# Phase 7: Chatbot Query Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark 6 chatbot query patterns × 3 retrieval strategies × 2 datasets to validate whether hyperbolic embeddings produce hierarchy-aware retrieval that outperforms flat cosine.

**Architecture:** New `dev/testbench/chatbot/` package with 5 modules. Imports shared code from parent testbench (scroll_all, tangent/fusion/queries). Runs on existing BGC unified collection + new HWV unified collection + new per-tier collections for multi-hop baseline.

**Tech Stack:** Python 3.12, qdrant-client, numpy, scipy (entropy)

**Spec:** `docs/superpowers/specs/2026-04-14-phase7-chatbot-query-optimization-design.md`

---

### Task 1: Create chatbot package scaffold + metrics module

**Files:**
- Create: `dev/testbench/chatbot/__init__.py`
- Create: `dev/testbench/chatbot/metrics.py`

- [ ] **Step 1: Create package directory and __init__.py**

```bash
mkdir -p dev/testbench/chatbot/results
```

```python
# dev/testbench/chatbot/__init__.py
"""Phase 7: Chatbot query optimization benchmarks."""
```

- [ ] **Step 2: Write metrics.py with all 10 hierarchy-awareness metrics**

```python
# dev/testbench/chatbot/metrics.py
"""Hierarchy-awareness metrics for chatbot query evaluation."""

import math
from collections import Counter


def ancestor_precision(result_ids: list, ancestor_ids: set) -> float:
    """Fraction of results that are actual hierarchy ancestors of the query."""
    if not result_ids:
        return 0.0
    return sum(1 for rid in result_ids if rid in ancestor_ids) / len(result_ids)


def subtree_coherence(result_ids: list, subtree_ids: set) -> float:
    """Fraction of results belonging to the query's subtree."""
    if not result_ids:
        return 0.0
    return sum(1 for rid in result_ids if rid in subtree_ids) / len(result_ids)


def depth_coverage(result_tiers: list[str], all_tiers: set[str]) -> float:
    """Fraction of hierarchy levels represented in results."""
    if not all_tiers:
        return 0.0
    return len(set(result_tiers) & all_tiers) / len(all_tiers)


def depth_distribution_entropy(result_tiers: list[str]) -> float:
    """Shannon entropy of tier distribution in results. Higher = better spread."""
    if not result_tiers:
        return 0.0
    counts = Counter(result_tiers)
    total = len(result_tiers)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def sibling_recall(result_ids: list, sibling_ids: set, k: int = 10) -> float:
    """Fraction of true siblings in top-k results."""
    top_k = result_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for rid in top_k if rid in sibling_ids) / len(top_k)


def intruder_rate(result_parent_ids: list[str], query_parent_id: str) -> float:
    """Fraction of results from a different parent branch."""
    if not result_parent_ids:
        return 0.0
    return sum(1 for pid in result_parent_ids if pid != query_parent_id) / len(result_parent_ids)


def depth_precision(result_tiers: list[str], target_tier: str) -> float:
    """Fraction of results at the target tier."""
    if not result_tiers:
        return 0.0
    return sum(1 for t in result_tiers if t == target_tier) / len(result_tiers)


def branch_diversity(result_branches: list[str]) -> int:
    """Number of distinct narrative branches in results."""
    return len(set(result_branches))


def branch_coverage(result_branches: list[str], all_branches: set[str]) -> float:
    """Fraction of relevant branches represented."""
    if not all_branches:
        return 0.0
    return len(set(result_branches) & all_branches) / len(all_branches)


def redundancy_ratio(result_branches: list[str]) -> float:
    """Fraction of results sharing a branch with another result. Lower is better."""
    if len(result_branches) <= 1:
        return 0.0
    counts = Counter(result_branches)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    return duplicated / len(result_branches)
```

- [ ] **Step 3: Verify the module imports**

Run: `cd dev/testbench && python -c "from chatbot.metrics import ancestor_precision, subtree_coherence, depth_distribution_entropy, branch_diversity; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add dev/testbench/chatbot/__init__.py dev/testbench/chatbot/metrics.py
git commit -m "feat(chatbot): Phase 7 scaffold + hierarchy-awareness metrics"
```

---

### Task 2: Implement the three retrieval strategies (baselines.py)

**Files:**
- Create: `dev/testbench/chatbot/baselines.py`

This module provides three functions, one per strategy. Each takes a Qdrant client, collection name(s), query vector, and pattern-specific params, and returns a standardized result dict: `{"ids": [...], "tiers": [...], "parent_ids": [...], "branches": [...], "query_count": int, "latency_ms": float}`.

- [ ] **Step 1: Write baselines.py**

```python
# dev/testbench/chatbot/baselines.py
"""Three retrieval strategies for chatbot query evaluation."""

import sys
import time
import numpy as np
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams, Filter, FieldCondition, MatchValue, Range

# Add parent testbench to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tangent import tangent_query
from fusion import rrf_fuse


def _extract_result(points: list, id_to_meta: dict) -> dict:
    """Convert Qdrant ScoredPoints to standardized result dict."""
    ids = [p.id for p in points]
    tiers = []
    parent_ids = []
    branches = []
    for p in points:
        payload = p.payload or {}
        tiers.append(payload.get("tier", "content"))
        pids = payload.get("parent_ids", [])
        parent_ids.append(pids[0] if pids else "")
        branches.append(payload.get("domain", ""))
    return {"ids": ids, "tiers": tiers, "parent_ids": parent_ids, "branches": branches}


def cosine_strategy(
    client: QdrantClient,
    collection: str,
    query_vec: np.ndarray,
    limit: int = 10,
    tier_filter: str | None = None,
    exclude_id: int | None = None,
    parent_filter: str | None = None,
) -> dict:
    """Strategy 1: Cosine-only on unified collection."""
    t_start = time.time()

    must_conditions = []
    if tier_filter:
        must_conditions.append(FieldCondition(key="tier", match=MatchValue(value=tier_filter)))
    if parent_filter:
        must_conditions.append(FieldCondition(key="parent_ids", match=MatchValue(value=parent_filter)))

    query_filter = Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=collection,
        query=query_vec.tolist(),
        using="dense",
        with_payload=True,
        limit=limit + (1 if exclude_id else 0),
        query_filter=query_filter,
        search_params=SearchParams(hnsw_ef=128),
    ).points

    # Exclude self
    if exclude_id is not None:
        results = [r for r in results if r.id != exclude_id]
    results = results[:limit]

    latency = (time.time() - t_start) * 1000
    extracted = _extract_result(results, {})
    extracted["query_count"] = 1
    extracted["latency_ms"] = latency
    return extracted


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
    """Strategy 2: Hyperbolic (tangent HNSW + optional RRF with dense)."""
    t_start = time.time()
    query_count = 0

    must_conditions = []
    if tier_filter:
        must_conditions.append(FieldCondition(key="tier", match=MatchValue(value=tier_filter)))
    if parent_filter:
        must_conditions.append(FieldCondition(key="parent_ids", match=MatchValue(value=parent_filter)))
    query_filter = Filter(must=must_conditions) if must_conditions else None

    # Tangent HNSW search
    centroid = np.zeros(len(query_vec_poincare))
    q_tangent = tangent_query(query_vec_poincare, centroid, curvature)

    fetch_limit = limit * 5 if use_rrf else limit + (1 if exclude_id else 0)
    tangent_results = client.query_points(
        collection_name=collection,
        query=q_tangent.tolist(),
        using="tangent",
        with_payload=True,
        limit=fetch_limit,
        query_filter=query_filter,
        search_params=SearchParams(hnsw_ef=128),
    ).points
    query_count += 1

    if use_rrf:
        # Also get dense results for RRF fusion
        dense_results = client.query_points(
            collection_name=collection,
            query=query_vec_dense.tolist(),
            using="dense",
            with_payload=True,
            limit=fetch_limit,
            query_filter=query_filter,
            search_params=SearchParams(hnsw_ef=128),
        ).points
        query_count += 1

        fused = rrf_fuse(tangent_results, dense_results, limit=limit + (1 if exclude_id else 0))
        fused_ids = {pid for pid, _ in fused}

        # Rebuild results list in fused order with payloads
        id_to_point = {p.id: p for p in tangent_results + dense_results}
        results = [id_to_point[pid] for pid, _ in fused if pid in id_to_point]
    else:
        results = tangent_results

    # Exclude self
    if exclude_id is not None:
        results = [r for r in results if r.id != exclude_id]
    results = results[:limit]

    latency = (time.time() - t_start) * 1000
    extracted = _extract_result(results, {})
    extracted["query_count"] = query_count
    extracted["latency_ms"] = latency
    return extracted


def multi_hop_strategy(
    client: QdrantClient,
    tier_collections: dict[str, str],
    query_vec: np.ndarray,
    start_tier: str,
    direction: str = "up",
    limit: int = 10,
    exclude_id: int | None = None,
) -> dict:
    """Strategy 3: Multi-hop across per-tier collections.

    Args:
        tier_collections: {"narrative": "bgc_narratives", "story": "bgc_stories", "content": "bgc_content"}
        start_tier: which tier to query first
        direction: "up" (content→narrative), "down" (narrative→content), or "single" (one tier only)
    """
    t_start = time.time()
    query_count = 0
    all_results = []

    # Tier ordering for traversal
    tier_order = ["content", "story", "narrative"]
    if direction == "up":
        start_idx = tier_order.index(start_tier) if start_tier in tier_order else 0
        tiers_to_visit = tier_order[start_idx:]
    elif direction == "down":
        start_idx = tier_order.index(start_tier) if start_tier in tier_order else len(tier_order) - 1
        tiers_to_visit = list(reversed(tier_order[:start_idx + 1]))
    else:
        tiers_to_visit = [start_tier]

    current_query = query_vec
    visited_ids = set()

    for tier in tiers_to_visit:
        coll = tier_collections.get(tier)
        if not coll:
            continue

        results = client.query_points(
            collection_name=coll,
            query=current_query.tolist(),
            using="dense",
            with_payload=True,
            limit=limit,
            search_params=SearchParams(hnsw_ef=128),
        ).points
        query_count += 1

        for r in results:
            if r.id not in visited_ids and r.id != exclude_id:
                all_results.append(r)
                visited_ids.add(r.id)

        # For traversal, use IDs from this tier to inform next query
        if results and direction in ("up", "down"):
            key = "parent_ids" if direction == "up" else "child_ids"
            next_ids = []
            for r in results[:3]:
                next_ids.extend(r.payload.get(key, []))
            # Use first result's vector as proxy for next tier query
            if results:
                vec = results[0].vector
                if isinstance(vec, dict):
                    vec = vec.get("dense", next(iter(vec.values())))
                current_query = np.array(vec, dtype=np.float32)

    all_results = all_results[:limit]
    latency = (time.time() - t_start) * 1000
    extracted = _extract_result(all_results, {})
    extracted["query_count"] = query_count
    extracted["latency_ms"] = latency
    return extracted
```

- [ ] **Step 2: Verify imports**

Run: `cd dev/testbench && python -c "from chatbot.baselines import cosine_strategy, hyperbolic_strategy, multi_hop_strategy; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add dev/testbench/chatbot/baselines.py
git commit -m "feat(chatbot): three retrieval strategies (cosine, hyperbolic, multi-hop)"
```

---

### Task 3: Implement the 6 query patterns (query_patterns.py)

**Files:**
- Create: `dev/testbench/chatbot/query_patterns.py`

Each pattern function takes the loaded point data + Qdrant client + strategy functions, runs 50 queries, computes per-query metrics, and returns aggregated results.

- [ ] **Step 1: Write query_patterns.py**

```python
# dev/testbench/chatbot/query_patterns.py
"""Six chatbot query patterns for hierarchy-aware retrieval evaluation."""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from hyperbolic_math import einstein_midpoint

from chatbot.metrics import (
    ancestor_precision,
    subtree_coherence,
    depth_coverage,
    depth_distribution_entropy,
    sibling_recall,
    intruder_rate,
    depth_precision,
    branch_diversity,
    branch_coverage,
    redundancy_ratio,
)
from chatbot.baselines import cosine_strategy, hyperbolic_strategy, multi_hop_strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_hierarchy_index(points: list[dict]) -> dict:
    """Build lookup structures from loaded points.

    Returns dict with:
      id_to_point: {point_id: point_dict}
      tier_to_ids: {"content": set(...), "story": set(...), ...}
      domain_to_ids: {"Literature": set(...), ...}
      parent_to_children: {"area_Fiction": set(child_ids)}
      id_to_ancestors: {point_id: set(ancestor_ids)}
      id_to_subtree: {point_id: set(all descendant ids including self)}
      all_tiers: set of all tier names
      domain_to_branches: {"Literature": set(area names)}
    """
    id_to_point = {p["id"]: p for p in points}
    tier_to_ids = {}
    domain_to_ids = {}
    parent_to_children = {}
    all_tiers = set()

    for p in points:
        tier = p["tier"]
        all_tiers.add(tier)
        tier_to_ids.setdefault(tier, set()).add(p["id"])
        domain = p.get("domain", "")
        if domain:
            domain_to_ids.setdefault(domain, set()).add(p["id"])
        for pid in p.get("parent_ids", []):
            parent_to_children.setdefault(pid, set()).add(p["id"])

    # Build ancestor map from hierarchy_path
    id_to_ancestors = {}
    point_id_to_id = {}
    for p in points:
        point_id_str = p.get("point_id", "")
        if point_id_str:
            point_id_to_id[point_id_str] = p["id"]

    for p in points:
        ancestors = set()
        for pid_str in p.get("parent_ids", []):
            if pid_str in point_id_to_id:
                ancestors.add(point_id_to_id[pid_str])
                # Walk up
                parent_point = id_to_point.get(point_id_to_id[pid_str])
                if parent_point:
                    for gpid_str in parent_point.get("parent_ids", []):
                        if gpid_str in point_id_to_id:
                            ancestors.add(point_id_to_id[gpid_str])
        id_to_ancestors[p["id"]] = ancestors

    # Build subtree map
    id_to_subtree = {}
    for p in points:
        domain = p.get("domain", "")
        area = p.get("area", "")
        subtree = {p["id"]}
        # Add all points with same domain (for narrative-tier)
        if p["tier"] in ("narrative", "root"):
            subtree = domain_to_ids.get(domain, {p["id"]})
        # Add all points with same area (for story-tier)
        elif p["tier"] in ("story", "mid"):
            for other in points:
                if other.get("area") == area:
                    subtree.add(other["id"])
        id_to_subtree[p["id"]] = subtree

    # Domain to branches (unique areas per domain)
    domain_to_branches = {}
    for p in points:
        domain = p.get("domain", "")
        area = p.get("area", "")
        if domain and area:
            domain_to_branches.setdefault(domain, set()).add(area)

    return {
        "id_to_point": id_to_point,
        "tier_to_ids": tier_to_ids,
        "domain_to_ids": domain_to_ids,
        "parent_to_children": parent_to_children,
        "id_to_ancestors": id_to_ancestors,
        "id_to_subtree": id_to_subtree,
        "all_tiers": all_tiers,
        "domain_to_branches": domain_to_branches,
        "point_id_to_id": point_id_to_id,
    }


def _run_strategy(strategy_name, strategy_fn, queries, metric_fn) -> dict:
    """Run a strategy across all queries, collect metrics."""
    all_metrics = []
    total_query_count = 0
    total_latency = 0.0

    for q in tqdm(queries, desc=f"  {strategy_name}"):
        result = strategy_fn(q)
        metrics = metric_fn(q, result)
        metrics["query_count"] = result["query_count"]
        metrics["latency_ms"] = result["latency_ms"]
        total_query_count += result["query_count"]
        total_latency += result["latency_ms"]
        all_metrics.append(metrics)

    # Aggregate
    agg = {}
    if all_metrics:
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            agg[key] = float(np.mean(vals))
    agg["total_query_count"] = total_query_count
    agg["total_latency_ms"] = total_latency
    agg["num_queries"] = len(queries)
    return agg


# ---------------------------------------------------------------------------
# Pattern 1: Drill-Up
# ---------------------------------------------------------------------------

def pattern_drill_up(client, collection, tier_collections, points, idx, curvature=0.25, n=50):
    """Pattern 1: Given a leaf doc, find its ancestors."""
    print("\n=== Pattern 1: Drill-Up ===")
    leaf_ids = [p for p in points if p["tier"] in ("content", "leaf")]
    rng = np.random.default_rng(42)
    sample = [leaf_ids[i] for i in rng.choice(len(leaf_ids), min(n, len(leaf_ids)), replace=False)]

    def metric_fn(query_point, result):
        gt_ancestors = idx["id_to_ancestors"].get(query_point["id"], set())
        return {
            "ancestor_precision": ancestor_precision(result["ids"], gt_ancestors),
        }

    strategies = {
        "cosine": lambda q: cosine_strategy(client, collection, q["vector"], limit=10, exclude_id=q["id"]),
        "hyperbolic": lambda q: hyperbolic_strategy(
            client, collection, q["vector"],
            q.get("vectors", {}).get("poincare", q["vector"]),
            curvature=curvature, limit=10, exclude_id=q["id"],
        ),
        "multi_hop": lambda q: multi_hop_strategy(
            client, tier_collections, q["vector"],
            start_tier="content", direction="up", limit=10, exclude_id=q["id"],
        ),
    }

    results = {}
    for name, fn in strategies.items():
        results[name] = _run_strategy(name, fn, sample, metric_fn)
    return results


# ---------------------------------------------------------------------------
# Pattern 2: Context Assembly
# ---------------------------------------------------------------------------

def pattern_context_assembly(client, collection, tier_collections, points, idx, curvature=0.25, n=50):
    """Pattern 2: Given a narrative, return its full subtree."""
    print("\n=== Pattern 2: Context Assembly ===")
    narrative_points = [p for p in points if p["tier"] in ("narrative", "root")]
    rng = np.random.default_rng(43)
    sample = [narrative_points[i] for i in rng.choice(len(narrative_points), min(n, len(narrative_points)), replace=False)]

    def metric_fn(query_point, result):
        gt_subtree = idx["id_to_subtree"].get(query_point["id"], set())
        return {
            "subtree_coherence": subtree_coherence(result["ids"], gt_subtree),
            "depth_coverage": depth_coverage(result["tiers"], idx["all_tiers"]),
        }

    strategies = {
        "cosine": lambda q: cosine_strategy(client, collection, q["vector"], limit=20, exclude_id=q["id"]),
        "hyperbolic": lambda q: hyperbolic_strategy(
            client, collection, q["vector"],
            q.get("vectors", {}).get("poincare", q["vector"]),
            curvature=curvature, limit=20, exclude_id=q["id"],
        ),
        "multi_hop": lambda q: multi_hop_strategy(
            client, tier_collections, q["vector"],
            start_tier="narrative", direction="down", limit=20, exclude_id=q["id"],
        ),
    }

    results = {}
    for name, fn in strategies.items():
        results[name] = _run_strategy(name, fn, sample, metric_fn)
    return results


# ---------------------------------------------------------------------------
# Pattern 3: Hierarchy-Aware Similarity
# ---------------------------------------------------------------------------

def pattern_hierarchy_similarity(client, collection, tier_collections, points, idx, curvature=0.25, n=50):
    """Pattern 3: Find siblings (same parent, semantically similar)."""
    print("\n=== Pattern 3: Hierarchy-Aware Similarity ===")
    # Pick leaf docs that have a parent
    candidates = [p for p in points if p["tier"] in ("content", "leaf") and p.get("parent_ids")]
    rng = np.random.default_rng(44)
    sample = [candidates[i] for i in rng.choice(len(candidates), min(n, len(candidates)), replace=False)]

    def metric_fn(query_point, result):
        parent_str = query_point["parent_ids"][0]
        gt_siblings = idx["parent_to_children"].get(parent_str, set()) - {query_point["id"]}
        parent_ids_of_results = result["parent_ids"]
        return {
            "sibling_recall": sibling_recall(result["ids"], gt_siblings),
            "intruder_rate": intruder_rate(parent_ids_of_results, parent_str),
        }

    strategies = {
        "cosine": lambda q: cosine_strategy(
            client, collection, q["vector"], limit=10,
            exclude_id=q["id"], parent_filter=q["parent_ids"][0],
        ),
        "hyperbolic": lambda q: hyperbolic_strategy(
            client, collection, q["vector"],
            q.get("vectors", {}).get("poincare", q["vector"]),
            curvature=curvature, limit=10, exclude_id=q["id"],
            parent_filter=q["parent_ids"][0],
        ),
        "multi_hop": lambda q: multi_hop_strategy(
            client, tier_collections, q["vector"],
            start_tier="content", direction="single", limit=10, exclude_id=q["id"],
        ),
    }

    results = {}
    for name, fn in strategies.items():
        results[name] = _run_strategy(name, fn, sample, metric_fn)
    return results


# ---------------------------------------------------------------------------
# Pattern 4: Multi-Level Briefing
# ---------------------------------------------------------------------------

def pattern_multi_level_briefing(client, collection, tier_collections, points, idx, curvature=0.25, n=50):
    """Pattern 4: Depth-diverse results for a domain."""
    print("\n=== Pattern 4: Multi-Level Briefing ===")
    domains = list(idx["domain_to_ids"].keys())
    rng = np.random.default_rng(45)
    selected_domains = [domains[i] for i in rng.choice(len(domains), min(n, len(domains)), replace=False)]

    # Compute domain centroids
    domain_queries = []
    for domain in selected_domains:
        domain_pts = [p for p in points if p.get("domain") == domain]
        if not domain_pts:
            continue
        centroid = np.mean([p["vector"] for p in domain_pts], axis=0)
        poincare_vecs = [p.get("vectors", {}).get("poincare") for p in domain_pts if p.get("vectors", {}).get("poincare") is not None]
        poincare_centroid = np.mean(poincare_vecs, axis=0) if poincare_vecs else centroid
        domain_queries.append({
            "id": None,
            "vector": centroid,
            "vectors": {"poincare": poincare_centroid},
            "domain": domain,
            "tier": "query",
        })

    def metric_fn(query_point, result):
        domain = query_point["domain"]
        gt_subtree = idx["domain_to_ids"].get(domain, set())
        return {
            "depth_entropy": depth_distribution_entropy(result["tiers"]),
            "subtree_precision": subtree_coherence(result["ids"], gt_subtree),
            "depth_coverage": depth_coverage(result["tiers"], idx["all_tiers"]),
        }

    strategies = {
        "cosine": lambda q: cosine_strategy(client, collection, q["vector"], limit=20),
        "hyperbolic": lambda q: hyperbolic_strategy(
            client, collection, q["vector"],
            q.get("vectors", {}).get("poincare", q["vector"]),
            curvature=curvature, limit=20,
        ),
        "multi_hop": lambda q: multi_hop_strategy(
            client, tier_collections, q["vector"],
            start_tier="narrative", direction="down", limit=20,
        ),
    }

    results = {}
    for name, fn in strategies.items():
        results[name] = _run_strategy(name, fn, domain_queries, metric_fn)
    return results


# ---------------------------------------------------------------------------
# Pattern 5: Cross-Branch Discovery
# ---------------------------------------------------------------------------

def pattern_cross_branch(client, collection, tier_collections, points, idx, curvature=0.25, n=50):
    """Pattern 5: Find narratives from different branches."""
    print("\n=== Pattern 5: Cross-Branch Discovery ===")
    narrative_points = [p for p in points if p["tier"] in ("narrative", "root")]
    rng = np.random.default_rng(46)
    sample = [narrative_points[i] for i in rng.choice(len(narrative_points), min(n, len(narrative_points)), replace=False)]

    def metric_fn(query_point, result):
        return {
            "depth_precision": depth_precision(result["tiers"], query_point["tier"]),
            "branch_diversity": branch_diversity(result["branches"]),
        }

    strategies = {
        "cosine": lambda q: cosine_strategy(client, collection, q["vector"], limit=10, exclude_id=q["id"]),
        "hyperbolic": lambda q: hyperbolic_strategy(
            client, collection, q["vector"],
            q.get("vectors", {}).get("poincare", q["vector"]),
            curvature=curvature, limit=10, exclude_id=q["id"], use_rrf=False,
        ),
        "multi_hop": lambda q: multi_hop_strategy(
            client, tier_collections, q["vector"],
            start_tier="narrative", direction="single", limit=10, exclude_id=q["id"],
        ),
    }

    results = {}
    for name, fn in strategies.items():
        results[name] = _run_strategy(name, fn, sample, metric_fn)
    return results


# ---------------------------------------------------------------------------
# Pattern 6: Narrative Landscape
# ---------------------------------------------------------------------------

def pattern_narrative_landscape(client, collection, tier_collections, points, idx, curvature=0.25, n=50):
    """Pattern 6: Surface distinct narrative branches for a broad topic."""
    print("\n=== Pattern 6: Narrative Landscape ===")
    domains = list(idx["domain_to_ids"].keys())
    rng = np.random.default_rng(47)
    selected_domains = [domains[i] for i in rng.choice(len(domains), min(n, len(domains)), replace=False)]

    # One representative doc per domain as the "broad query"
    domain_queries = []
    for domain in selected_domains:
        domain_pts = [p for p in points if p.get("domain") == domain]
        if not domain_pts:
            continue
        rep = domain_pts[rng.integers(len(domain_pts))]
        rep_copy = dict(rep)
        rep_copy["domain"] = domain
        domain_queries.append(rep_copy)

    def metric_fn(query_point, result):
        domain = query_point["domain"]
        gt_branches = idx["domain_to_branches"].get(domain, set())
        return {
            "branch_diversity": branch_diversity(result["branches"]),
            "branch_coverage": branch_coverage(result["branches"], gt_branches),
            "redundancy_ratio": redundancy_ratio(result["branches"]),
        }

    strategies = {
        "cosine": lambda q: cosine_strategy(client, collection, q["vector"], limit=20, exclude_id=q["id"]),
        "hyperbolic": lambda q: hyperbolic_strategy(
            client, collection, q["vector"],
            q.get("vectors", {}).get("poincare", q["vector"]),
            curvature=curvature, limit=20, exclude_id=q["id"],
        ),
        "multi_hop": lambda q: multi_hop_strategy(
            client, tier_collections, q["vector"],
            start_tier="narrative", direction="single", limit=20, exclude_id=q["id"],
        ),
    }

    results = {}
    for name, fn in strategies.items():
        results[name] = _run_strategy(name, fn, domain_queries, metric_fn)
    return results
```

- [ ] **Step 2: Verify imports**

Run: `cd dev/testbench && python -c "from chatbot.query_patterns import pattern_drill_up, pattern_narrative_landscape; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add dev/testbench/chatbot/query_patterns.py
git commit -m "feat(chatbot): 6 query patterns (drill-up, context, similarity, briefing, cross-branch, landscape)"
```

---

### Task 4: Implement the main evaluator (eval.py)

**Files:**
- Create: `dev/testbench/chatbot/eval.py`

The evaluator loads points, builds hierarchy index, creates per-tier collections for multi-hop baseline, runs all 6 patterns × 3 strategies, and saves results to JSON.

- [ ] **Step 1: Write eval.py**

```python
# dev/testbench/chatbot/eval.py
"""Phase 7: Chatbot query optimization evaluator.

Usage:
  cd dev/testbench
  python -m chatbot.eval --dataset bgc
  python -m chatbot.eval --dataset hwv
  python -m chatbot.eval --dataset bgc --patterns drill_up multi_level_briefing
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from qdrant_client import QdrantClient

# Add parent testbench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.query_patterns import (
    pattern_drill_up,
    pattern_context_assembly,
    pattern_hierarchy_similarity,
    pattern_multi_level_briefing,
    pattern_cross_branch,
    pattern_narrative_landscape,
    _build_hierarchy_index,
)


PATTERNS = {
    "drill_up": pattern_drill_up,
    "context_assembly": pattern_context_assembly,
    "hierarchy_similarity": pattern_hierarchy_similarity,
    "multi_level_briefing": pattern_multi_level_briefing,
    "cross_branch": pattern_cross_branch,
    "narrative_landscape": pattern_narrative_landscape,
}


def scroll_all_points(client: QdrantClient, collection: str) -> list[dict]:
    """Scroll all points with all vectors and payloads."""
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
            all_vectors = {}
            if isinstance(raw_vec, dict):
                all_vectors = {k: np.array(v, dtype=np.float32) for k, v in raw_vec.items()}
                primary = all_vectors.get("dense", next(iter(all_vectors.values())))
            else:
                primary = np.array(raw_vec, dtype=np.float32)

            points.append({
                "id": pt.id,
                "vector": primary,
                "vectors": all_vectors,
                "tier": pt.payload.get("tier", "content"),
                "domain": pt.payload.get("domain", ""),
                "area": pt.payload.get("area", ""),
                "hierarchy_path": pt.payload.get("hierarchy_path", ""),
                "parent_ids": pt.payload.get("parent_ids", []),
                "child_ids": pt.payload.get("child_ids", []),
                "busemann_depth": pt.payload.get("busemann_depth", 0.0),
                "point_id": pt.payload.get("point_id", str(pt.id)),
            })
        if next_offset is None:
            break
        offset = next_offset
    return points


def create_tier_collections(client: QdrantClient, qdrant_url: str, dataset: str, points: list[dict]):
    """Create per-tier collections for multi-hop baseline."""
    tier_map = {"narrative": [], "story": [], "content": []}
    # Map non-standard tier names
    tier_aliases = {
        "root": "narrative", "narrative": "narrative",
        "mid": "story", "story": "story", "sub_story": "story",
        "leaf": "content", "content": "content", "tier_L4": "content",
    }

    for p in points:
        bucket = tier_aliases.get(p["tier"], "content")
        tier_map[bucket].append(p)

    tier_collections = {}
    for tier, tier_points in tier_map.items():
        coll_name = f"{dataset}_{tier}s"  # bgc_narratives, bgc_stories, bgc_contents
        tier_collections[tier] = coll_name

        if not tier_points:
            print(f"  Skipping {coll_name} (0 points)")
            continue

        # Delete if exists
        try:
            requests.delete(f"{qdrant_url}/collections/{coll_name}")
        except Exception:
            pass

        dim = len(tier_points[0]["vector"])
        body = {
            "vectors": {"dense": {"size": dim, "distance": "Cosine"}},
        }
        resp = requests.put(f"{qdrant_url}/collections/{coll_name}", json=body)
        print(f"  Created {coll_name} ({len(tier_points)} points) — status {resp.status_code}")

        # Upsert in batches
        from qdrant_client.models import PointStruct
        batch_size = 500
        for i in range(0, len(tier_points), batch_size):
            batch = tier_points[i:i + batch_size]
            qdrant_points = [
                PointStruct(
                    id=p["id"],
                    vector={"dense": p["vector"].tolist()},
                    payload={
                        "tier": p["tier"],
                        "domain": p["domain"],
                        "area": p["area"],
                        "parent_ids": p["parent_ids"],
                        "child_ids": p["child_ids"],
                        "point_id": p["point_id"],
                        "busemann_depth": p["busemann_depth"],
                    },
                )
                for p in batch
            ]
            client.upsert(collection_name=coll_name, points=qdrant_points)

        # Create payload indexes
        for field, schema in [("parent_ids", "keyword"), ("child_ids", "keyword"), ("tier", "keyword")]:
            try:
                client.create_payload_index(collection_name=coll_name, field_name=field, field_schema=schema)
            except Exception:
                pass

    return tier_collections


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Chatbot query optimization benchmarks")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--dataset", default="bgc", choices=["bgc", "hwv"])
    parser.add_argument("--patterns", nargs="*", default=None,
                        help="Run specific patterns (default: all)")
    parser.add_argument("--curvature", type=float, default=0.25)
    parser.add_argument("--num-queries", type=int, default=50)
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url, timeout=60)
    unified_collection = f"{args.dataset}_unified"

    print(f"\n{'='*60}")
    print(f"Phase 7: Chatbot Query Optimization")
    print(f"Dataset: {args.dataset}")
    print(f"Curvature: {args.curvature}")
    print(f"{'='*60}")

    # Load all points from unified collection
    print(f"\nLoading points from {unified_collection}...")
    points = scroll_all_points(client, unified_collection)
    print(f"  Loaded {len(points)} points")
    tiers = {}
    for p in points:
        tiers[p["tier"]] = tiers.get(p["tier"], 0) + 1
    print(f"  Tier distribution: {tiers}")

    # Build hierarchy index
    print("Building hierarchy index...")
    idx = _build_hierarchy_index(points)
    print(f"  Domains: {len(idx['domain_to_ids'])}, Tiers: {idx['all_tiers']}")

    # Create per-tier collections for multi-hop baseline
    print("\nCreating per-tier collections for multi-hop baseline...")
    tier_collections = create_tier_collections(client, args.qdrant_url, args.dataset, points)
    print(f"  Collections: {tier_collections}")

    # Run patterns
    patterns_to_run = args.patterns or list(PATTERNS.keys())
    all_results = {}

    for pattern_name in patterns_to_run:
        fn = PATTERNS.get(pattern_name)
        if not fn:
            print(f"Unknown pattern: {pattern_name}")
            continue
        result = fn(
            client, unified_collection, tier_collections,
            points, idx, curvature=args.curvature, n=args.num_queries,
        )
        all_results[pattern_name] = result

        # Print summary
        for strategy, metrics in result.items():
            metric_strs = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in metrics.items() if k not in ("total_query_count", "total_latency_ms", "num_queries")]
            print(f"    {strategy}: {', '.join(metric_strs[:4])}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"chatbot_{args.dataset}_{timestamp}.json"

    output = {
        "dataset": args.dataset,
        "curvature": args.curvature,
        "num_points": len(points),
        "tier_distribution": tiers,
        "timestamp": datetime.now().isoformat(),
        "patterns": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the evaluator runs help**

Run: `cd dev/testbench && python -m chatbot.eval --help`
Expected: Shows argparse help with --dataset, --patterns, --curvature options

- [ ] **Step 3: Commit**

```bash
git add dev/testbench/chatbot/eval.py
git commit -m "feat(chatbot): main evaluator — loads points, builds index, runs patterns, saves JSON"
```

---

### Task 5: Run evaluation on BGC

This is a manual step — the user runs the commands.

- [ ] **Step 1: Ensure Qdrant is running with BGC data**

Run: `curl -s http://localhost:6333/healthz`
Expected: `healthz check passed`

If not running: `cd dev/testbench && docker compose up -d`

- [ ] **Step 2: Run the full evaluation on BGC**

Run: `cd dev/testbench && python -m chatbot.eval --dataset bgc`

Expected: All 6 patterns run with 3 strategies each. JSON results saved to `dev/testbench/chatbot/results/`.

- [ ] **Step 3: Review results and commit**

```bash
git add dev/testbench/chatbot/
git commit -m "feat(chatbot): Phase 7 complete — 6 patterns × 3 strategies on BGC"
```

---

### Task 6: Embed HWV at c=0.25 and run evaluation

HWV needs to be embedded with tangent vectors like BGC.

- [ ] **Step 1: Embed HWV dataset**

Run: `cd dev/testbench && python embed.py --dataset hwv --skip-embed`

This re-projects HWV at c=0.25 (using the CURVATURE constant updated in Phase 6), computes tangent vectors, and populates `hwv_unified` with dense + poincaré + tangent named vectors.

If the HWV embeddings aren't cached, run without `--skip-embed`:
Run: `cd dev/testbench && python embed.py --dataset hwv`

- [ ] **Step 2: Run the full evaluation on HWV**

Run: `cd dev/testbench && python -m chatbot.eval --dataset hwv`

- [ ] **Step 3: Commit**

```bash
git add dev/testbench/chatbot/results/
git commit -m "feat(chatbot): Phase 7 HWV results — 6-level hierarchy validation"
```

---

### Task 7: Fix issues and iterate

After running on both datasets, there will likely be edge cases, missing data, or unexpected results. This task is for iteration.

- [ ] **Step 1: Compare BGC and HWV results**

Read both JSON result files. Check:
- Do hyperbolic metrics beat cosine on at least 4/6 patterns?
- Is HWV (deeper hierarchy) showing stronger hyperbolic advantage than BGC?
- Are there any patterns where multi-hop baseline beats hyperbolic?
- Are there NaN or zero values indicating missing data?

- [ ] **Step 2: Fix any issues found**

Common fixes:
- Adjust query generation if a tier has too few points for sampling
- Fix tier name mapping if HWV uses different tier labels than BGC
- Adjust limit parameter if results are sparse

- [ ] **Step 3: Re-run and commit final results**

```bash
cd dev/testbench && python -m chatbot.eval --dataset bgc
cd dev/testbench && python -m chatbot.eval --dataset hwv
git add dev/testbench/chatbot/
git commit -m "feat(chatbot): Phase 7 final results — BGC + HWV validated"
```
