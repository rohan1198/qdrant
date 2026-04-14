"""
Six chatbot query pattern benchmarks for hierarchy-aware retrieval evaluation.

Each pattern:
  1. Selects query points from loaded data
  2. Runs all 3 strategies (cosine, hyperbolic, multi_hop)
  3. Computes pattern-specific metrics per query
  4. Returns aggregated results per strategy
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from hyperbolic_math import einstein_midpoint

from chatbot.metrics import (
    ancestor_precision,
    ancestor_hit_rate,
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
# Shared helpers
# ---------------------------------------------------------------------------

def _build_hierarchy_index(points: list[dict]) -> dict:
    """Build lookup structures from a list of loaded point dicts.

    Returns a dict with keys:
        id_to_point        {int_id: point_dict}
        tier_to_ids        {"content": set, "story": set, ...}
        domain_to_ids      {"Literature": set, ...}
        parent_to_children {"area_Fiction": set(child_int_ids)}
        id_to_ancestors    {int_id: set(ancestor_int_ids)}
        id_to_subtree      {int_id: set(descendant_int_ids + self)}
        all_tiers          set of tier names
        domain_to_branches {"Literature": set(area_names)}
        point_id_to_id     {"point_id_string": int_id}
    """
    id_to_point: dict[int, dict] = {}
    tier_to_ids: dict[str, set] = {}
    domain_to_ids: dict[str, set] = {}
    parent_to_children: dict[str, set] = {}
    point_id_to_id: dict[str, int] = {}
    domain_to_branches: dict[str, set] = {}

    for p in points:
        pid: int = p["id"]
        id_to_point[pid] = p

        # point_id string → numeric id
        str_pid = p.get("point_id", "")
        if str_pid:
            point_id_to_id[str_pid] = pid

        # tier index
        tier = p.get("tier", "")
        tier_to_ids.setdefault(tier, set()).add(pid)

        # domain index
        domain = p.get("domain", "")
        if domain:
            domain_to_ids.setdefault(domain, set()).add(pid)

        # domain → branches (areas)
        area = p.get("area", "")
        if domain and area:
            domain_to_branches.setdefault(domain, set()).add(area)

        # parent → children (using parent_ids strings as keys)
        for parent_str in p.get("parent_ids", []):
            parent_to_children.setdefault(parent_str, set()).add(pid)

    all_tiers: set = set(tier_to_ids.keys())

    # Build id_to_ancestors: walk up via parent_ids strings
    id_to_ancestors: dict[int, set] = {}
    for p in points:
        pid: int = p["id"]
        ancestors: set = set()
        queue = list(p.get("parent_ids", []))
        visited_str: set = set()
        while queue:
            parent_str = queue.pop(0)
            if parent_str in visited_str:
                continue
            visited_str.add(parent_str)
            parent_int = point_id_to_id.get(parent_str)
            if parent_int is None:
                continue
            ancestors.add(parent_int)
            grandparent_point = id_to_point.get(parent_int)
            if grandparent_point is not None:
                queue.extend(grandparent_point.get("parent_ids", []))
        id_to_ancestors[pid] = ancestors

    # Pre-build area_to_ids for O(n) subtree construction
    area_to_ids: dict[str, set] = {}
    for p in points:
        area = p.get("area", "")
        if area:
            area_to_ids.setdefault(area, set()).add(p["id"])

    # Build id_to_subtree using parent_to_children for proper traversal.
    # For narrative: self + all stories with parent=self + all content with parent=those stories
    # For story: self + all content with parent=self
    # For content: just self
    id_to_subtree: dict[int, set] = {}
    for p in points:
        pid: int = p["id"]
        tier = p.get("tier", "")
        point_id_str = p.get("point_id", "")

        if tier == "narrative":
            subtree = {pid}
            # Direct children (stories)
            children = parent_to_children.get(point_id_str, set())
            subtree.update(children)
            # Grandchildren (content of those stories)
            for child_id in children:
                child_point = id_to_point.get(child_id)
                if child_point:
                    child_str = child_point.get("point_id", "")
                    grandchildren = parent_to_children.get(child_str, set())
                    subtree.update(grandchildren)
            # Fallback: also include all same-domain points if subtree is tiny
            if len(subtree) < 5:
                domain = p.get("domain", "")
                subtree.update(domain_to_ids.get(domain, set()))
            id_to_subtree[pid] = subtree
        elif tier == "story":
            subtree = {pid}
            children = parent_to_children.get(point_id_str, set())
            subtree.update(children)
            # Fallback: also include same-area points
            if len(subtree) < 5:
                area = p.get("area", "")
                subtree.update(area_to_ids.get(area, set()))
            id_to_subtree[pid] = subtree
        else:
            id_to_subtree[pid] = {pid}

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


def _run_strategy(strategy_name: str, strategy_fn, queries: list[dict], metric_fn) -> dict:
    """Run strategy_fn across all queries, collecting per-query metrics.

    strategy_fn(query_dict) -> result dict
    metric_fn(result, query_dict) -> dict of metric_name: value

    Returns averaged metrics plus summed total_query_count and total_latency_ms.
    """
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    total_query_count = 0
    total_latency_ms = 0.0

    for q in tqdm(queries, desc=strategy_name, leave=False):
        result = strategy_fn(q)
        metrics = metric_fn(result, q)
        for k, v in metrics.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            metric_counts[k] = metric_counts.get(k, 0) + 1
        total_query_count += result.get("query_count", 0)
        total_latency_ms += result.get("latency_ms", 0.0)

    out: dict = {}
    for k in metric_sums:
        out[k] = metric_sums[k] / metric_counts[k] if metric_counts[k] else 0.0
    out["total_query_count"] = total_query_count
    out["total_latency_ms"] = total_latency_ms
    return out


# ---------------------------------------------------------------------------
# Pattern 1: Drill-up — given a leaf doc, find its ancestors
# ---------------------------------------------------------------------------

def pattern_drill_up(
    client,
    collection: str,
    tier_collections: dict,
    points: list[dict],
    idx: dict,
    curvature: float = 0.25,
    n: int = 50,
) -> dict:
    """Pattern 1: Given a leaf document, find its hierarchy ancestors."""
    rng = np.random.default_rng(42)

    # Determine leaf tier (deepest tier by convention: "content")
    leaf_tier = "content"
    leaf_ids = list(idx["tier_to_ids"].get(leaf_tier, set()))
    if not leaf_ids:
        # Fall back: pick any tier that is not narrative
        non_root = [t for t in idx["all_tiers"] if t != "narrative"]
        for t in non_root:
            leaf_ids = list(idx["tier_to_ids"].get(t, set()))
            if leaf_ids:
                leaf_tier = t
                break

    chosen_ids = rng.choice(leaf_ids, size=min(n, len(leaf_ids)), replace=False)
    sampled = [idx["id_to_point"][i] for i in chosen_ids]

    def _metric(result, q):
        ancestor_ids = idx["id_to_ancestors"].get(q["_pid"], set())
        return {
            "ancestor_precision": ancestor_precision(result["ids"], ancestor_ids),
            "ancestor_hit_rate": ancestor_hit_rate(result["ids"], ancestor_ids),
        }

    def _cosine(q):
        return cosine_strategy(
            client, collection,
            query_vec=q["_dense"],
            limit=10,
            exclude_id=q["_point_id_str"],
        )

    def _hyperbolic(q):
        return hyperbolic_strategy(
            client, collection,
            query_vec_dense=q["_dense"],
            query_vec_poincare=q["_poincare"],
            curvature=curvature,
            limit=10,
            exclude_id=q["_point_id_str"],
            use_rrf=True,
        )

    def _multi_hop(q):
        return multi_hop_strategy(
            client, tier_collections,
            query_vec=q["_dense"],
            start_tier="content",
            direction="up",
            limit=10,
            exclude_id=q["_point_id_str"],
        )

    queries = _prepare_queries(sampled)

    return {
        "cosine":     _run_strategy("cosine",     _cosine,     queries, _metric),
        "hyperbolic": _run_strategy("hyperbolic", _hyperbolic, queries, _metric),
        "multi_hop":  _run_strategy("multi_hop",  _multi_hop,  queries, _metric),
    }


# ---------------------------------------------------------------------------
# Pattern 2: Context assembly — given a narrative, return its subtree
# ---------------------------------------------------------------------------

def pattern_context_assembly(
    client,
    collection: str,
    tier_collections: dict,
    points: list[dict],
    idx: dict,
    curvature: float = 0.25,
    n: int = 50,
) -> dict:
    """Pattern 2: Given a narrative/root document, return its subtree."""
    rng = np.random.default_rng(43)

    narrative_ids = list(idx["tier_to_ids"].get("narrative", set()))
    if not narrative_ids:
        # Try root-like tiers
        for t in idx["all_tiers"]:
            candidate = list(idx["tier_to_ids"].get(t, set()))
            if candidate:
                narrative_ids = candidate
                break

    chosen_ids = rng.choice(narrative_ids, size=min(n, len(narrative_ids)), replace=False)
    sampled = [idx["id_to_point"][i] for i in chosen_ids]

    def _metric(result, q):
        subtree_ids = idx["id_to_subtree"].get(q["_pid"], set())
        res_tiers = result.get("tiers", [])
        return {
            "subtree_coherence": subtree_coherence(result["ids"], subtree_ids),
            "depth_coverage":    depth_coverage(res_tiers, idx["all_tiers"]),
        }

    def _cosine(q):
        return cosine_strategy(
            client, collection,
            query_vec=q["_dense"],
            limit=20,
            exclude_id=q["_point_id_str"],
        )

    def _hyperbolic(q):
        return hyperbolic_strategy(
            client, collection,
            query_vec_dense=q["_dense"],
            query_vec_poincare=q["_poincare"],
            curvature=curvature,
            limit=20,
            exclude_id=q["_point_id_str"],
            use_rrf=True,
        )

    def _multi_hop(q):
        return multi_hop_strategy(
            client, tier_collections,
            query_vec=q["_dense"],
            start_tier="narrative",
            direction="down",
            limit=20,
            exclude_id=q["_point_id_str"],
        )

    queries = _prepare_queries(sampled)

    return {
        "cosine":     _run_strategy("cosine",     _cosine,     queries, _metric),
        "hyperbolic": _run_strategy("hyperbolic", _hyperbolic, queries, _metric),
        "multi_hop":  _run_strategy("multi_hop",  _multi_hop,  queries, _metric),
    }


# ---------------------------------------------------------------------------
# Pattern 3: Hierarchy similarity — find siblings under the same parent
# ---------------------------------------------------------------------------

def pattern_hierarchy_similarity(
    client,
    collection: str,
    tier_collections: dict,
    points: list[dict],
    idx: dict,
    curvature: float = 0.25,
    n: int = 50,
) -> dict:
    """Pattern 3: Given a leaf doc with a parent, find its siblings."""
    rng = np.random.default_rng(44)

    # Leaf docs that have at least one parent_id
    candidates = [
        p for p in points
        if p.get("tier", "") == "content" and p.get("parent_ids")
    ]
    if not candidates:
        # Any doc with parent_ids
        candidates = [p for p in points if p.get("parent_ids")]

    chosen = rng.choice(
        np.arange(len(candidates)),
        size=min(n, len(candidates)),
        replace=False,
    )
    sampled = [candidates[i] for i in chosen]

    def _metric(result, q):
        parent_str = q["_parent_str"]
        sibling_ids = idx["parent_to_children"].get(parent_str, set()) - {q["_pid"]}
        return {
            "sibling_recall": sibling_recall(result["ids"], sibling_ids, k=10),
            "intruder_rate":  intruder_rate(result.get("parent_ids", []), parent_str),
        }

    def _cosine(q):
        return cosine_strategy(
            client, collection,
            query_vec=q["_dense"],
            limit=10,
            parent_filter=q["_parent_str"],
            exclude_id=q["_point_id_str"],
        )

    def _hyperbolic(q):
        # No parent_filter — test whether geometry alone clusters siblings
        return hyperbolic_strategy(
            client, collection,
            query_vec_dense=q["_dense"],
            query_vec_poincare=q["_poincare"],
            curvature=curvature,
            limit=10,
            exclude_id=q["_point_id_str"],
            use_rrf=True,
        )

    def _multi_hop(q):
        return cosine_strategy(
            client, tier_collections.get("content", collection),
            query_vec=q["_dense"],
            limit=10,
            parent_filter=q["_parent_str"],
            exclude_id=q["_point_id_str"],
        )

    queries = _prepare_queries(sampled)

    return {
        "cosine":     _run_strategy("cosine",     _cosine,     queries, _metric),
        "hyperbolic": _run_strategy("hyperbolic", _hyperbolic, queries, _metric),
        "multi_hop":  _run_strategy("multi_hop",  _multi_hop,  queries, _metric),
    }


# ---------------------------------------------------------------------------
# Pattern 4: Multi-level briefing — depth-diverse results for a domain
# ---------------------------------------------------------------------------

def pattern_multi_level_briefing(
    client,
    collection: str,
    tier_collections: dict,
    points: list[dict],
    idx: dict,
    curvature: float = 0.25,
    n: int = 50,
) -> dict:
    """Pattern 4: Domain centroid query expecting depth-diverse results."""
    rng = np.random.default_rng(45)

    all_domains = list(idx["domain_to_ids"].keys())
    chosen_domains = list(rng.choice(all_domains, size=min(n, len(all_domains)), replace=False))

    # Build synthetic centroid queries (no real point, id=None)
    queries = []
    for domain in chosen_domains:
        domain_point_ids = list(idx["domain_to_ids"].get(domain, set()))
        if not domain_point_ids:
            continue
        domain_points = [idx["id_to_point"][i] for i in domain_point_ids]

        # Dense centroid (mean)
        dense_vecs = [p["vectors"]["dense"] for p in domain_points if "vectors" in p]
        if not dense_vecs:
            dense_vecs = [p.get("vector", np.zeros(1024, dtype=np.float32)) for p in domain_points]
        dense_centroid = np.mean(dense_vecs, axis=0).astype(np.float32)

        # Poincare centroid via Einstein midpoint
        poincare_vecs = [p["vectors"]["poincare"] for p in domain_points if "vectors" in p and "poincare" in p["vectors"]]
        if poincare_vecs:
            try:
                poincare_centroid = einstein_midpoint(poincare_vecs, c=curvature)
            except Exception:
                poincare_centroid = np.mean(poincare_vecs, axis=0).astype(np.float32)
        else:
            poincare_centroid = dense_centroid.copy()

        queries.append({
            "_pid": None,
            "_point_id_str": None,
            "_dense": dense_centroid,
            "_poincare": poincare_centroid,
            "_parent_str": None,
            "_domain": domain,
        })

    def _metric(result, q):
        domain = q["_domain"]
        domain_ids = idx["domain_to_ids"].get(domain, set())
        res_tiers = result.get("tiers", [])
        return {
            "depth_entropy":      depth_distribution_entropy(res_tiers),
            "subtree_precision":  subtree_coherence(result["ids"], domain_ids),
            "depth_coverage":     depth_coverage(res_tiers, idx["all_tiers"]),
        }

    def _cosine(q):
        return cosine_strategy(
            client, collection,
            query_vec=q["_dense"],
            limit=20,
            exclude_id=None,
        )

    def _hyperbolic(q):
        return hyperbolic_strategy(
            client, collection,
            query_vec_dense=q["_dense"],
            query_vec_poincare=q["_poincare"],
            curvature=curvature,
            limit=20,
            exclude_id=None,
            use_rrf=True,
        )

    def _multi_hop(q):
        return multi_hop_strategy(
            client, tier_collections,
            query_vec=q["_dense"],
            start_tier="narrative",
            direction="down",
            limit=20,
            exclude_id=None,
        )

    return {
        "cosine":     _run_strategy("cosine",     _cosine,     queries, _metric),
        "hyperbolic": _run_strategy("hyperbolic", _hyperbolic, queries, _metric),
        "multi_hop":  _run_strategy("multi_hop",  _multi_hop,  queries, _metric),
    }


# ---------------------------------------------------------------------------
# Pattern 5: Cross-branch — find narratives from different branches
# ---------------------------------------------------------------------------

def pattern_cross_branch(
    client,
    collection: str,
    tier_collections: dict,
    points: list[dict],
    idx: dict,
    curvature: float = 0.25,
    n: int = 50,
) -> dict:
    """Pattern 5: Given a narrative, find narratives from different branches."""
    rng = np.random.default_rng(46)

    narrative_ids = list(idx["tier_to_ids"].get("narrative", set()))
    if not narrative_ids:
        for t in idx["all_tiers"]:
            candidate = list(idx["tier_to_ids"].get(t, set()))
            if candidate:
                narrative_ids = candidate
                break

    chosen_ids = rng.choice(narrative_ids, size=min(n, len(narrative_ids)), replace=False)
    sampled = [idx["id_to_point"][i] for i in chosen_ids]

    def _metric(result, q):
        target_tier = q.get("tier", "narrative")
        res_tiers = result.get("tiers", [])
        res_branches = result.get("branches", [])
        return {
            "depth_precision": depth_precision(res_tiers, target_tier),
            "branch_diversity": float(branch_diversity(res_branches)),
        }

    def _cosine(q):
        return cosine_strategy(
            client, collection,
            query_vec=q["_dense"],
            limit=10,
            exclude_id=q["_point_id_str"],
        )

    def _hyperbolic(q):
        return hyperbolic_strategy(
            client, collection,
            query_vec_dense=q["_dense"],
            query_vec_poincare=q["_poincare"],
            curvature=curvature,
            limit=10,
            exclude_id=q["_point_id_str"],
            use_rrf=False,  # tangent only for branch structure
        )

    def _multi_hop(q):
        return multi_hop_strategy(
            client, tier_collections,
            query_vec=q["_dense"],
            start_tier="narrative",
            direction="single",
            limit=10,
            exclude_id=q["_point_id_str"],
        )

    queries = _prepare_queries(sampled)

    return {
        "cosine":     _run_strategy("cosine",     _cosine,     queries, _metric),
        "hyperbolic": _run_strategy("hyperbolic", _hyperbolic, queries, _metric),
        "multi_hop":  _run_strategy("multi_hop",  _multi_hop,  queries, _metric),
    }


# ---------------------------------------------------------------------------
# Pattern 6: Narrative landscape — surface distinct branches for broad topic
# ---------------------------------------------------------------------------

def pattern_narrative_landscape(
    client,
    collection: str,
    tier_collections: dict,
    points: list[dict],
    idx: dict,
    curvature: float = 0.25,
    n: int = 50,
) -> dict:
    """Pattern 6: 1 representative doc per domain, test branch diversity."""
    rng = np.random.default_rng(47)

    all_domains = list(idx["domain_to_ids"].keys())
    chosen_domains = list(rng.choice(all_domains, size=min(n, len(all_domains)), replace=False))

    # Pick one representative point per domain (first by stable sort of id)
    sampled = []
    for domain in chosen_domains:
        domain_point_ids = sorted(idx["domain_to_ids"].get(domain, set()))
        if not domain_point_ids:
            continue
        rep_id = domain_point_ids[0]
        sampled.append(idx["id_to_point"][rep_id])

    def _metric(result, q):
        domain = q.get("domain", "")
        all_branches_for_domain = idx["domain_to_branches"].get(domain, set())
        # Use areas (not domains) for branch comparison — matches domain_to_branches
        res_areas = result.get("areas", [])
        res_branches = result.get("branches", [])
        return {
            "branch_diversity":  float(branch_diversity(res_branches)),
            "branch_coverage":   branch_coverage(res_areas, all_branches_for_domain),
            "redundancy_ratio":  redundancy_ratio(res_branches),
        }

    def _cosine(q):
        return cosine_strategy(
            client, collection,
            query_vec=q["_dense"],
            limit=20,
            exclude_id=q["_point_id_str"],
        )

    def _hyperbolic(q):
        return hyperbolic_strategy(
            client, collection,
            query_vec_dense=q["_dense"],
            query_vec_poincare=q["_poincare"],
            curvature=curvature,
            limit=20,
            exclude_id=q["_point_id_str"],
            use_rrf=True,
        )

    def _multi_hop(q):
        return multi_hop_strategy(
            client, tier_collections,
            query_vec=q["_dense"],
            start_tier="narrative",
            direction="single",
            limit=20,
            exclude_id=q["_point_id_str"],
        )

    queries = _prepare_queries(sampled)

    return {
        "cosine":     _run_strategy("cosine",     _cosine,     queries, _metric),
        "hyperbolic": _run_strategy("hyperbolic", _hyperbolic, queries, _metric),
        "multi_hop":  _run_strategy("multi_hop",  _multi_hop,  queries, _metric),
    }


# ---------------------------------------------------------------------------
# Internal helper: normalise sampled point dicts into query dicts
# ---------------------------------------------------------------------------

def _prepare_queries(sampled: list[dict]) -> list[dict]:
    """Add strategy-facing keys to each sampled point dict.

    Adds:
        _pid          int Qdrant ID
        _point_id_str str point_id payload (for exclude_id)
        _dense        np.ndarray dense vector
        _poincare     np.ndarray poincare vector
        _parent_str   first parent_id string (or None)
    """
    out = []
    for p in sampled:
        vectors = p.get("vectors", {})
        dense = vectors.get("dense", p.get("vector", np.zeros(1024, dtype=np.float32)))
        poincare = vectors.get("poincare", dense)
        if isinstance(dense, list):
            dense = np.array(dense, dtype=np.float32)
        if isinstance(poincare, list):
            poincare = np.array(poincare, dtype=np.float32)

        parent_ids = p.get("parent_ids", [])
        q = dict(p)  # shallow copy to preserve tier, domain, area, etc.
        q["_pid"] = p["id"]
        q["_point_id_str"] = p.get("point_id", None)
        q["_dense"] = dense
        q["_poincare"] = poincare
        q["_parent_str"] = parent_ids[0] if parent_ids else None
        out.append(q)
    return out
