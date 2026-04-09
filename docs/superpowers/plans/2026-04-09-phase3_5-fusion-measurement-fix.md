# Phase 3.5: Fusion Overhaul, Measurement Fix, Tier-Filtered Retrieval

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three Round 3 issues — broken fusion, measurement artifacts, and missing tier-filtered retrieval — then re-sweep curvatures with improved metrics.

**Architecture:** All changes are Python-side (testbench). No Rust changes. Three new fusion strategies in `fusion.py`, improved hierarchy metrics and multi-mode retrieval benchmarks in `benchmark.py`, and payload index creation in `embed.py`. Curvature re-sweep reuses existing sweep infrastructure.

**Tech Stack:** Python (qdrant-client, numpy), Qdrant REST API (payload indices)

**Spec:** `docs/superpowers/specs/2026-04-09-phase3_5-fusion-measurement-fix-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `dev/testbench/fusion.py` | Modify | Add `linear_alpha_fuse`, `busemann_weighted_fuse`, `depth_band_fuse` |
| `dev/testbench/benchmark.py` | Modify | Content-only sep_ratio, cross-tier sep metric, multi-mode retrieval, fusion strategy comparison |
| `dev/testbench/embed.py` | Modify | Add payload index creation after unified collection upsert |

---

### Task 1: Add New Fusion Strategies to `fusion.py`

**Files:**
- Modify: `dev/testbench/fusion.py`

- [ ] **Step 1: Add score normalization helper**

Add after the `rrf_fuse` function (after line 41):

```python


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
```

- [ ] **Step 2: Add linear alpha fusion**

Add after `_extract_scores`:

```python


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
```

- [ ] **Step 3: Add Busemann depth-weighted fusion**

Add after `linear_alpha_fuse`:

```python


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
```

- [ ] **Step 4: Add depth-band pre-filter fusion**

Add after `busemann_weighted_fuse`:

```python


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
```

- [ ] **Step 5: Update module docstring**

Replace line 1-8 of `fusion.py`:

```python
"""Fusion strategies for dual-space search.

Combines results from cosine and poincare search into a single ranking.
Four strategies available:
  - rrf_fuse: Reciprocal Rank Fusion (rank-based, baseline)
  - linear_alpha_fuse: Score-normalized alpha blending
  - busemann_weighted_fuse: Alpha blend weighted by Busemann depth proximity
  - depth_band_fuse: Hard depth-band filter + alpha blend
"""
```

- [ ] **Step 6: Verify syntax**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "import ast; ast.parse(open('fusion.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 2: Fix Hierarchy Separation Metrics in `benchmark.py`

**Files:**
- Modify: `dev/testbench/benchmark.py:251-345`

- [ ] **Step 1: Add content-only sep_ratio and cross-tier metric**

Replace the entire `benchmark_hierarchy_separation` function (lines 251-303) with:

```python
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
        # Within-content separation using WOS root/mid/leaf sub-tiers
        # For content-only, recompute from just the leaf depths
        std_content = float(np.std(leaf_depths)) if len(leaf_depths) > 1 else EPS
        # Use all-tier avg_leaf vs avg_root for the numerator
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
        # Thresholds between adjacent tier means
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
        # Fallback: 2-way classification
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
```

- [ ] **Step 2: Update hierarchy separation printing in `main()`**

Find the hierarchy separation printing block (around line 791-796) and replace:

```python
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
```

- [ ] **Step 3: Verify syntax**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "import ast; ast.parse(open('benchmark.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 3: Add Payload Indices to `embed.py`

**Files:**
- Modify: `dev/testbench/embed.py:785-792`

- [ ] **Step 1: Add payload index creation after unified upsert**

Find the unified collection upsert block (around line 785-792). After the `upsert_unified(...)` call, add payload index creation. The block should become:

```python
        # Unified collection
        print("\nCreating unified collection ...")
        create_unified_collection(qdrant_url, UNIFIED_COLLECTION, size=VECTOR_DIM, curvature=CURVATURE)
        upsert_unified(
            client, UNIFIED_COLLECTION,
            docs, embeddings_128, poincare_vectors, busemann_depths,
            area_entries, domain_entries,
        )

        # Create payload indices for server-side filtering
        print("Creating payload indices ...")
        client.create_payload_index(
            collection_name=UNIFIED_COLLECTION,
            field_name="busemann_depth",
            field_schema="float",
        )
        client.create_payload_index(
            collection_name=UNIFIED_COLLECTION,
            field_name="tier",
            field_schema="keyword",
        )
        print("  Created indices on 'busemann_depth' (float) and 'tier' (keyword)")
```

- [ ] **Step 2: Verify syntax**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "import ast; ast.parse(open('embed.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 4: Add Multi-Mode Retrieval Quality to `benchmark.py`

**Files:**
- Modify: `dev/testbench/benchmark.py`

- [ ] **Step 1: Add depth-range search helper**

Add after the `_search_named` function (around line 243):

```python
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
```

- [ ] **Step 2: Add multi-mode retrieval benchmark function**

Add after `benchmark_retrieval_quality` (after the function ends, around line 415):

```python
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

    modes = {
        "unfiltered": lambda q: _search_named(client, collection, "poincare", q, limit=k + 1),
        "tier_filtered": lambda q: _search_named(client, collection, "poincare", q, limit=k + 1, tier_filter="content"),
        "depth_range": lambda q: _search_named_depth_range(client, collection, "poincare", q, limit=k + 1, depth_lo=depth_lo, depth_hi=depth_hi),
        "cosine_tier_filtered": lambda q: _search_named(client, collection, "cosine", q, limit=k + 1, tier_filter="content"),
    }

    mode_results: dict[str, dict] = {}

    for mode_name, search_fn in modes.items():
        area_recalls = []
        domain_recalls = []
        h_precisions = []

        for qpt in query_sample:
            try:
                results = search_fn(qpt["vector"])
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
```

- [ ] **Step 3: Wire multi-mode retrieval into `main()`**

In the unified-specific benchmarks section (around line 826), after the cross-tier block and before the dual-space block, add:

```python
        # --- Benchmark 2b: Multi-Mode Retrieval ---
        if args.suite in ("all", "retrieval"):
            print(f"\n{'='*60}")
            print("Multi-Mode Retrieval Quality (wos_unified)")
            print(f"{'='*60}")
            points_unified_poincare = scroll_all(client, UNIFIED_COLLECTION, vector_name="poincare")
            mr_res = benchmark_retrieval_modes(client, UNIFIED_COLLECTION, points_unified_poincare, k=10, num_queries=500)
            all_results.setdefault(UNIFIED_COLLECTION, {})["retrieval_modes"] = mr_res

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
```

- [ ] **Step 4: Verify syntax**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "import ast; ast.parse(open('benchmark.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 5: Replace Dual-Space Benchmark with Fusion Strategy Comparison

**Files:**
- Modify: `dev/testbench/benchmark.py`

- [ ] **Step 1: Replace `benchmark_dual_space` function**

Replace the entire `benchmark_dual_space` function (lines 501-613) with a new version that tests all fusion strategies:

```python
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

        # Fetch results from both spaces
        try:
            cos_results = _search(client, cosine_collection, qpt["vector"], limit=k + 1)
        except Exception:
            continue

        try:
            cos_unified = _search_named(client, unified_collection, "cosine", qpt["vector"], limit=over_fetch, tier_filter="content")
        except Exception:
            cos_unified = []

        try:
            poincare_unified = _search_named(client, unified_collection, "poincare", qpt["vector"], limit=over_fetch, tier_filter="content")
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
```

- [ ] **Step 2: Update dual-space printing in `main()`**

Find the dual-space printing block (around line 840-860) and replace with:

```python
        # --- Benchmark 4: Fusion Strategy Comparison ---
        if args.suite in ("all", "dual-space"):
            print(f"\n{'='*60}")
            print("Fusion Strategy Comparison")
            print(f"{'='*60}")
            points_unified_cosine = scroll_all(client, UNIFIED_COLLECTION, vector_name="cosine")
            points_cosine_baseline = scroll_all(client, "wos_cosine")
            ds_res = benchmark_dual_space(
                client, UNIFIED_COLLECTION, "wos_cosine",
                points_unified_cosine, points_cosine_baseline,
                k=10, num_queries=500,
            )
            all_results.setdefault(UNIFIED_COLLECTION, {})["dual_space"] = ds_res

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
```

- [ ] **Step 3: Verify syntax**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "import ast; ast.parse(open('benchmark.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 6: Run Benchmarks and Capture Results

- [ ] **Step 1: Re-create payload indices**

Since we already have data in `wos_unified`, we just need to add indices. Run:

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='http://localhost:6334')
client.create_payload_index('wos_unified', 'busemann_depth', 'float')
client.create_payload_index('wos_unified', 'tier', 'keyword')
print('Indices created')
"
```

- [ ] **Step 2: Run benchmarks**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 benchmark.py --qdrant-url http://localhost:6334
```

Expected output includes:
- Hierarchy separation with `sep_ratio_content` and `cross_tier_sep` metrics
- Multi-mode retrieval table (unfiltered vs tier-filtered vs depth-range vs cosine)
- Fusion strategy comparison table (sorted by area_recall)

- [ ] **Step 3: Record results in analysis doc**

Save output to `dev/testbench/results/analysis_round3_5.md` with comparison tables.

---

### Task 7: Commit All Phase 3 + 3.5 Changes

- [ ] **Step 1: Run final checks**

Verify Rust tests still pass:
```bash
cd /home/rohan/projects/qdrant
cargo test --features hyperbolic -p segment -- --nocapture 2>&1 | tail -10
```

Verify all Python files parse:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
for f in fusion.py embed.py benchmark.py hyperbolic_math.py; do
    python3 -c "import ast; ast.parse(open('$f').read()); print('$f OK')"
done
```

- [ ] **Step 2: Stage and commit**

```bash
cd /home/rohan/projects/qdrant
git add lib/segment/src/spaces/hyperbolic/poincare_math.rs
git add dev/testbench/fusion.py dev/testbench/embed.py dev/testbench/benchmark.py
git add dev/testbench/hyperbolic_math.py dev/testbench/run.sh
git add dev/testbench/results/
git add docs/superpowers/specs/ docs/superpowers/plans/
git commit -m "feat(hyperbolic): Phase 3+3.5 — unified collection, fusion overhaul, measurement fixes

Phase 3:
- Fix lorentz_to_poincare: add sqrt(c) divisor for configurable curvature
- Create wos_unified collection with cosine + poincare named vectors
- Generate tier centroids via Einstein midpoint (areas + domains)
- Add Busemann depth to payload, RRF fusion module
- Cross-tier retrieval: 42.5% parent story recall

Phase 3.5:
- Replace RRF with linear alpha + Busemann depth-weighted fusion
- Add content-only sep_ratio and cross-tier separation metrics
- Add multi-mode retrieval (unfiltered, tier-filtered, depth-range)
- Add payload indices on busemann_depth and tier
- Fusion strategy sweep across alpha, lambda, bandwidth parameters
- Clean up legacy per-strategy collections"
```
