"""Phase 7: Main evaluator entry point for chatbot query optimization benchmarks.

Usage:
    python -m chatbot.eval --dataset bgc --curvature 0.25 --num-queries 50
    python dev/testbench/chatbot/eval.py --help
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

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

# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

PATTERNS = {
    "drill_up": pattern_drill_up,
    "context_assembly": pattern_context_assembly,
    "hierarchy_similarity": pattern_hierarchy_similarity,
    "multi_level_briefing": pattern_multi_level_briefing,
    "cross_branch": pattern_cross_branch,
    "narrative_landscape": pattern_narrative_landscape,
}

# ---------------------------------------------------------------------------
# Tier alias mapping
# ---------------------------------------------------------------------------

TIER_ALIASES = {
    "root": "narrative",
    "narrative": "narrative",
    "mid": "story",
    "story": "story",
    "sub_story": "story",
    "leaf": "content",
    "content": "content",
    "tier_L4": "content",
}


# ---------------------------------------------------------------------------
# scroll_all_points
# ---------------------------------------------------------------------------

def scroll_all_points(client: QdrantClient, collection: str) -> list[dict]:
    """Scroll all points from *collection* with vectors and payload.

    For each point:
    - If the vector is a dict (named vectors): extract all as np.float32 arrays
      into a ``vectors`` dict, and use ``"dense"`` as the primary ``vector``.
    - If the vector is a plain list: store it as-is under ``vector``.

    Payload fields extracted: tier, domain, area, hierarchy_path, parent_ids,
    child_ids, busemann_depth, point_id.

    Returns:
        List of point dicts compatible with the schema expected by
        query_patterns.py.
    """
    points: list[dict] = []
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=500,
            with_vectors=True,
            with_payload=True,
        )

        for p in batch:
            payload = p.payload or {}

            # --- Vector handling ---
            raw_vec = p.vector
            if isinstance(raw_vec, dict):
                vectors = {name: np.array(v, dtype=np.float32) for name, v in raw_vec.items()}
                vector = vectors.get("dense", next(iter(vectors.values())) if vectors else None)
            else:
                raw_list = raw_vec if raw_vec is not None else []
                vector = np.array(raw_list, dtype=np.float32)
                vectors = {"dense": vector}

            # --- Payload extraction ---
            tier_raw = payload.get("tier", "")
            tier = TIER_ALIASES.get(tier_raw, tier_raw)

            point_dict = {
                "id": p.id,
                "vector": vector,
                "vectors": vectors,
                # normalised tier name
                "tier": tier,
                "tier_raw": tier_raw,
                "domain": payload.get("domain", ""),
                "area": payload.get("area", ""),
                "hierarchy_path": payload.get("hierarchy_path", ""),
                "parent_ids": payload.get("parent_ids", []),
                "child_ids": payload.get("child_ids", []),
                "busemann_depth": payload.get("busemann_depth", 0.0),
                "point_id": payload.get("point_id", p.id),
            }
            points.append(point_dict)

        if next_offset is None:
            break
        offset = next_offset

    return points


# ---------------------------------------------------------------------------
# create_tier_collections
# ---------------------------------------------------------------------------

def create_tier_collections(
    client: QdrantClient,
    qdrant_url: str,
    dataset: str,
    points: list[dict],
) -> dict[str, str]:
    """Create per-tier Qdrant collections for the multi-hop baseline.

    Collections created:
        {dataset}_narratives — narrative/root tier docs
        {dataset}_stories    — story/mid tier docs
        {dataset}_contents   — content/leaf tier docs

    Each collection uses a ``{"vectors": {"dense": {"size": dim,
    "distance": "Cosine"}}}`` schema and payload indexes on ``parent_ids``,
    ``child_ids``, and ``tier``.

    Returns:
        Mapping of canonical tier name → collection name, e.g.
        ``{"narrative": "bgc_narratives", "story": "bgc_stories",
           "content": "bgc_contents"}``.
    """
    tier_to_colname = {
        "narrative": f"{dataset}_narratives",
        "story": f"{dataset}_stories",
        "content": f"{dataset}_contents",
    }

    # Infer vector dimension from the first point that has a dense vector
    dim = 768
    for pt in points:
        v = pt.get("vector")
        if v is not None and hasattr(v, "__len__") and len(v) > 0:
            dim = int(len(v))
            break

    headers = {"Content-Type": "application/json"}

    for canonical_tier, coll_name in tier_to_colname.items():
        # Delete if exists
        del_url = f"{qdrant_url}/collections/{coll_name}"
        try:
            resp = requests.delete(del_url, timeout=30)
            # 200 = deleted, 404 = not found — both acceptable
        except requests.RequestException:
            pass

        # Create collection
        create_url = f"{qdrant_url}/collections/{coll_name}"
        payload_body = {
            "vectors": {
                "dense": {
                    "size": dim,
                    "distance": "Cosine",
                }
            }
        }
        resp = requests.put(create_url, json=payload_body, headers=headers, timeout=60)
        resp.raise_for_status()

        # Filter points for this tier
        tier_points = [p for p in points if p.get("tier") == canonical_tier]
        if not tier_points:
            print(f"  [warn] no points for tier '{canonical_tier}' — skipping {coll_name}")
            continue

        # Upsert in batches of 500
        batch_size = 500
        for start in range(0, len(tier_points), batch_size):
            batch = tier_points[start : start + batch_size]
            structs = []
            for pt in batch:
                vec = pt.get("vector")
                vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                structs.append(
                    PointStruct(
                        id=pt["id"],
                        vector={"dense": vec_list},
                        payload={
                            "tier": pt["tier"],
                            "tier_raw": pt.get("tier_raw", pt["tier"]),
                            "domain": pt.get("domain", ""),
                            "area": pt.get("area", ""),
                            "hierarchy_path": pt.get("hierarchy_path", ""),
                            "parent_ids": pt.get("parent_ids", []),
                            "child_ids": pt.get("child_ids", []),
                            "busemann_depth": pt.get("busemann_depth", 0.0),
                            "point_id": pt.get("point_id", pt["id"]),
                        },
                    )
                )
            client.upsert(collection_name=coll_name, points=structs)

        # Create payload indexes
        index_url = f"{qdrant_url}/collections/{coll_name}/index"
        for field_name, field_schema in [
            ("parent_ids", "keyword"),
            ("child_ids", "keyword"),
            ("tier", "keyword"),
        ]:
            resp = requests.put(
                index_url,
                json={"field_name": field_name, "field_schema": field_schema},
                headers=headers,
                timeout=60,
            )
            # Non-fatal: index creation may fail if field doesn't exist in data
            if resp.status_code not in (200, 201, 409):
                print(
                    f"  [warn] index on {coll_name}.{field_name} returned {resp.status_code}"
                )

        print(f"  created {coll_name}: {len(tier_points)} points")

    return {canonical: colname for canonical, colname in tier_to_colname.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: chatbot query pattern benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant HTTP URL",
    )
    parser.add_argument(
        "--dataset",
        default="bgc",
        choices=["bgc", "hwv"],
        help="Dataset name (selects {dataset}_unified collection)",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        metavar="PATTERN",
        help=(
            "Which patterns to run (space-separated). "
            f"Available: {', '.join(PATTERNS)}. "
            "Omit to run all."
        ),
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=0.25,
        help="Poincare ball curvature c",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=50,
        help="Number of query points to sample per pattern",
    )
    args = parser.parse_args()

    # --- Validate requested patterns ---
    if args.patterns:
        unknown = [p for p in args.patterns if p not in PATTERNS]
        if unknown:
            parser.error(f"Unknown patterns: {unknown}. Choose from: {list(PATTERNS)}")
        selected_patterns = {k: PATTERNS[k] for k in args.patterns}
    else:
        selected_patterns = PATTERNS

    # --- Header ---
    print("=" * 60)
    print(f"Phase 7 chatbot benchmark")
    print(f"  dataset   : {args.dataset}")
    print(f"  curvature : {args.curvature}")
    print(f"  num_queries: {args.num_queries}")
    print(f"  patterns  : {list(selected_patterns)}")
    print(f"  qdrant_url: {args.qdrant_url}")
    print("=" * 60)

    # --- Connect ---
    client = QdrantClient(url=args.qdrant_url, timeout=120)

    # --- Load points ---
    collection = f"{args.dataset}_unified"
    print(f"\nLoading points from '{collection}' ...")
    t0 = time.perf_counter()
    points = scroll_all_points(client, collection)
    load_secs = time.perf_counter() - t0
    print(f"  loaded {len(points)} points in {load_secs:.1f}s")

    # --- Tier distribution ---
    tier_dist = dict(Counter(p["tier"] for p in points))
    print(f"  tier distribution: {tier_dist}")

    # --- Build hierarchy index ---
    print("\nBuilding hierarchy index ...")
    hierarchy_index = _build_hierarchy_index(points)
    print(f"  hierarchy index built")

    # --- Create per-tier collections ---
    print(f"\nCreating per-tier collections for multi-hop baseline ...")
    tier_collections = create_tier_collections(client, args.qdrant_url, args.dataset, points)
    print(f"  tier collections: {tier_collections}")

    # --- Run patterns ---
    results_by_pattern: dict[str, dict] = {}

    for pattern_name, pattern_fn in selected_patterns.items():
        print(f"\n--- Pattern: {pattern_name} ---")
        t_pat = time.perf_counter()
        try:
            pattern_results = pattern_fn(
                client=client,
                collection=collection,
                tier_collections=tier_collections,
                points=points,
                idx=hierarchy_index,
                curvature=args.curvature,
                n=args.num_queries,
            )
        except Exception as exc:
            print(f"  [error] {pattern_name} failed: {exc}")
            results_by_pattern[pattern_name] = {"error": str(exc)}
            continue

        pat_secs = time.perf_counter() - t_pat

        # Print per-strategy summary
        for strategy_name, strategy_metrics in pattern_results.items():
            if isinstance(strategy_metrics, dict) and "error" not in strategy_metrics:
                # Print key scalar metrics
                summary_keys = [
                    "ancestor_precision", "subtree_coherence", "depth_coverage",
                    "sibling_recall", "intruder_rate", "depth_precision",
                    "branch_diversity", "branch_coverage", "redundancy_ratio",
                    "query_count", "latency_ms",
                ]
                summary = {k: strategy_metrics[k] for k in summary_keys if k in strategy_metrics}
                print(f"  {strategy_name}: {summary}")
            else:
                print(f"  {strategy_name}: {strategy_metrics}")

        print(f"  pattern completed in {pat_secs:.1f}s")
        results_by_pattern[pattern_name] = pattern_results

    # --- Save results ---
    timestamp = datetime.utcnow().isoformat()
    output = {
        "dataset": args.dataset,
        "curvature": args.curvature,
        "num_points": len(points),
        "tier_distribution": tier_dist,
        "timestamp": timestamp,
        "patterns": results_by_pattern,
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ts_slug = timestamp[:19].replace(":", "-").replace("T", "_")
    out_path = results_dir / f"chatbot_{args.dataset}_{ts_slug}.json"

    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=_json_default)

    print(f"\nResults saved to {out_path}")
    print("Done.")


def _json_default(obj):
    """JSON serializer for types not handled by default encoder."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
