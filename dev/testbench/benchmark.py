"""Benchmark script for hyperbolic (Poincaré / Busemann) vector collections in Qdrant.

Runs 4 benchmark suites:
  1. Hierarchy Separation  — Busemann depth per tier
  2. Retrieval Quality     — recall@10 within hierarchy
  3. Curvature Comparison  — summary table (printed after 1, 2, 4)
  4. Performance & Latency — throughput and p50/p95/p99

Collections expected:
  wos_cosine, wos_c05, wos_c10, wos_c20, wos_c50

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTIONS = ["wos_cosine", "wos_c05", "wos_c10", "wos_c20", "wos_c50"]
POINCARE_COLLECTIONS = ["wos_c05", "wos_c10", "wos_c20", "wos_c50"]

CURVATURE_MAP = {
    "wos_cosine": None,
    "wos_c05": 0.5,
    "wos_c10": 1.0,
    "wos_c20": 2.0,
    "wos_c50": 5.0,
}

EF_VALUES = [64, 128, 256]
EPS = 1e-5

# ---------------------------------------------------------------------------
# Busemann math
# ---------------------------------------------------------------------------


def poincare_to_lorentz(p: np.ndarray, c: float = 1.0) -> np.ndarray:
    norm_sq = float(np.sum(p ** 2))
    denom = max(1.0 - c * norm_sq, EPS)
    x0 = (1.0 + c * norm_sq) / denom
    spatial = 2.0 * np.sqrt(c) * p / denom
    return np.concatenate([[x0], spatial])


def lorentz_inner(x: np.ndarray, y: np.ndarray) -> float:
    return float(-x[0] * y[0] + np.dot(x[1:], y[1:]))


def busemann_score(x_lorentz: np.ndarray, focal: np.ndarray) -> float:
    inner = lorentz_inner(x_lorentz, focal)
    return float(np.log(max(-inner, EPS)))


def compute_focal_direction(points: list[np.ndarray], c: float = 1.0) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def scroll_all(client: QdrantClient, collection: str) -> list[dict]:
    """Scroll all points from a collection, returning list of dicts."""
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
            # pt.vector may be a list or dict (named vectors); normalise to ndarray
            raw_vec = pt.vector
            if isinstance(raw_vec, dict):
                # pick the first (and only) named vector
                raw_vec = next(iter(raw_vec.values()))
            points.append({
                "id": pt.id,
                "vector": np.array(raw_vec, dtype=np.float32),
                "tier": pt.payload.get("tier", "leaf"),
                "domain": pt.payload.get("domain", -1),
                "area": pt.payload.get("area", -1),
            })
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
    """Run nearest-neighbour search, trying query_points then falling back to search."""
    vec = query_vector.tolist()
    kwargs: dict = dict(collection_name=collection, limit=limit)
    if search_params:
        kwargs["search_params"] = search_params

    try:
        results = client.query_points(
            query=vec,
            **kwargs,
        )
        # query_points returns a QueryResponse with a .points attribute
        return results.points
    except (AttributeError, TypeError):
        pass

    try:
        return client.search(
            query_vector=vec,
            **kwargs,
        )
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Benchmark 1 — Hierarchy Separation
# ---------------------------------------------------------------------------


def benchmark_hierarchy_separation(points: list[dict], curvature: float) -> dict:
    """Compute Busemann depth per tier and separation metrics."""
    depths = compute_busemann_depths(points, c=curvature)

    tier_depths: dict[str, list[float]] = {"root": [], "mid": [], "leaf": []}
    for pt, d in zip(points, depths):
        tier = pt["tier"]
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

    # Separation ratio = |avg_leaf - avg_root| / std_all
    all_depths = depths
    std_all = float(np.std(all_depths)) if all_depths else EPS
    avg_leaf = results["avg_depth_leaf"]
    avg_root = results["avg_depth_root"]

    if avg_leaf is not None and avg_root is not None and std_all > EPS:
        sep_ratio = abs(avg_leaf - avg_root) / std_all
    else:
        sep_ratio = None

    results["separation_ratio"] = sep_ratio
    results["num_points"] = len(points)

    # Tier classification accuracy via median threshold
    # Classify: depth < median → "root/mid", depth >= median → "leaf"
    if all_depths:
        median_d = float(np.median(all_depths))
        correct = 0
        for pt, d in zip(points, depths):
            predicted = "leaf" if d >= median_d else "root"
            if pt["tier"] == predicted or (pt["tier"] == "mid" and predicted == "root"):
                correct += 1
        results["tier_classification_accuracy"] = correct / len(points)
    else:
        results["tier_classification_accuracy"] = None

    return results


# ---------------------------------------------------------------------------
# Benchmark 2 — Retrieval Quality
# ---------------------------------------------------------------------------


def benchmark_retrieval_quality(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    k: int = 10,
    num_queries: int = 500,
) -> dict:
    """Sample leaf documents and evaluate recall@10 within hierarchy."""
    leaf_points = [pt for pt in points if pt["tier"] == "leaf"]
    if not leaf_points:
        return {"error": "no leaf points found"}

    query_sample = random.sample(leaf_points, min(num_queries, len(leaf_points)))

    area_recalls: list[float] = []
    domain_recalls: list[float] = []
    h_precisions: list[float] = []

    for query_pt in query_sample:
        try:
            results = _search(client, collection, query_pt["vector"], limit=k + 1)
        except Exception:
            continue

        # Filter out the query point itself
        query_id = query_pt["id"]
        neighbors = [r for r in results if r.id != query_id][:k]

        if not neighbors:
            continue

        # Resolve neighbor payloads from local points index for speed
        id_to_pt = {pt["id"]: pt for pt in points}

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


# ---------------------------------------------------------------------------
# Benchmark 4 — Performance & Latency
# ---------------------------------------------------------------------------


def benchmark_latency(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    num_queries: int = 1000,
) -> dict:
    """Run sequential search at multiple ef values, measure QPS and latency percentiles."""
    leaf_points = [pt for pt in points if pt["tier"] == "leaf"]
    if not leaf_points:
        return {"error": "no leaf points found"}

    query_sample = random.sample(leaf_points, min(num_queries, len(leaf_points)))
    ef_results: dict = {}

    for ef in EF_VALUES:
        search_params = SearchParams(hnsw_ef=ef)
        latencies_ms: list[float] = []

        for query_pt in query_sample:
            t0 = time.perf_counter()
            try:
                _search(
                    client,
                    collection,
                    query_pt["vector"],
                    limit=10,
                    search_params=search_params,
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
# Curvature comparison table (Benchmark 3)
# ---------------------------------------------------------------------------


def print_curvature_comparison(all_results: dict) -> None:
    """Print a side-by-side comparison table across all collections."""
    header = (
        f"{'Collection':<15} {'Curvature':>9} {'SepRatio':>10} "
        f"{'AreaRec@10':>11} {'DomRec@10':>10} {'H-Prec':>8} {'p95@ef128':>10}"
    )
    print("\n" + "=" * len(header))
    print("Curvature Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for col in COLLECTIONS:
        c_val = CURVATURE_MAP.get(col)
        c_str = f"{c_val:.2f}" if c_val is not None else "cosine"

        res = all_results.get(col, {})

        sep = res.get("hierarchy_separation", {})
        sep_ratio = sep.get("separation_ratio")
        sep_str = f"{sep_ratio:.3f}" if sep_ratio is not None else "N/A"

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
            f"{col:<15} {c_str:>9} {sep_str:>10} "
            f"{area_str:>11} {dom_str:>10} {h_str:>8} {p95_str:>10}"
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
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url)

    # Determine which collections actually exist
    existing = {c.name for c in client.get_collections().collections}
    collections_to_run = [c for c in COLLECTIONS if c in existing]

    if not collections_to_run:
        print(f"No benchmark collections found. Expected one of: {COLLECTIONS}")
        return

    print(f"Found collections: {collections_to_run}")

    all_results: dict = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"benchmark_{timestamp}.json"

    for collection in collections_to_run:
        c_val = CURVATURE_MAP.get(collection)
        print(f"\n{'='*60}")
        print(f"Collection: {collection}  (curvature={c_val})")
        print(f"{'='*60}")

        print("  Loading points...")
        points = scroll_all(client, collection)
        print(f"  Loaded {len(points)} points")

        col_results: dict = {}

        # --- Benchmark 1: Hierarchy Separation (Poincaré only) ---
        if collection in POINCARE_COLLECTIONS and c_val is not None:
            print("  [1/3] Hierarchy Separation...")
            sep_res = benchmark_hierarchy_separation(points, curvature=c_val)
            col_results["hierarchy_separation"] = sep_res
            sep_ratio = sep_res.get("separation_ratio")
            print(
                f"        sep_ratio={sep_ratio:.4f}" if sep_ratio is not None
                else "        sep_ratio=N/A"
            )
            for tier in ("root", "mid", "leaf"):
                avg = sep_res.get(f"avg_depth_{tier}")
                if avg is not None:
                    print(f"        avg_depth_{tier}={avg:.4f}")
        else:
            print("  [1/3] Hierarchy Separation — skipped (cosine collection)")
            col_results["hierarchy_separation"] = {}

        # --- Benchmark 2: Retrieval Quality ---
        print("  [2/3] Retrieval Quality (500 queries)...")
        rq_res = benchmark_retrieval_quality(
            client, collection, points, k=10, num_queries=500
        )
        col_results["retrieval_quality"] = rq_res
        print(
            f"        area_recall@10={rq_res.get('area_recall_at_10'):.4f}"
            if rq_res.get("area_recall_at_10") is not None else "        area_recall@10=N/A"
        )
        print(
            f"        domain_recall@10={rq_res.get('domain_recall_at_10'):.4f}"
            if rq_res.get("domain_recall_at_10") is not None else "        domain_recall@10=N/A"
        )
        print(
            f"        h_precision={rq_res.get('hierarchical_precision'):.4f}"
            if rq_res.get("hierarchical_precision") is not None else "        h_precision=N/A"
        )

        # --- Benchmark 4: Latency ---
        print("  [3/3] Latency (1000 queries, ef=64/128/256)...")
        lat_res = benchmark_latency(client, collection, points, num_queries=1000)
        col_results["latency"] = lat_res
        for ef in EF_VALUES:
            ef_data = lat_res.get(f"ef_{ef}", {})
            qps = ef_data.get("qps")
            p95 = ef_data.get("p95_ms")
            print(
                f"        ef={ef}: qps={qps:.1f}, p95={p95:.1f}ms"
                if qps is not None and p95 is not None else f"        ef={ef}: N/A"
            )

        all_results[collection] = col_results

        # Save intermediate results after each collection
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # --- Benchmark 3: Curvature Comparison (summary table) ---
    print_curvature_comparison(all_results)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
