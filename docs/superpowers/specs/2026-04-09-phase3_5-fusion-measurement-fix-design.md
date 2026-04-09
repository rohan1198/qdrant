# Phase 3.5: Fusion Overhaul, Measurement Fix, Tier-Filtered Retrieval

**Date:** 2026-04-09
**Branch:** `feat/hyperbolic-vector-support`
**Status:** Approved
**Prerequisite:** Phase 3 complete (unified collection `wos_unified` with named vectors + Busemann depth payload)

---

## Overview

Round 3 proved the unified collection concept works (hierarchy encoding, cross-tier retrieval) but revealed three problems:

1. **RRF fusion underperforms** — 0.200 area_recall vs 0.220 cosine-only. Rank-based fusion averages correlated rankings.
2. **Poincare flat retrieval dropped** — 0.178 vs 0.216 in Round 2. HNSW graph distortion from centroid nodes + no tier filtering in benchmark.
3. **Sep_ratio dropped** — 9.5 vs 20.4. Measurement artifact from including centroid points in the calculation.

This design fixes all three, then runs a curvature re-sweep with the improved metrics.

## Design Decisions

- **Approach B (Testbench + Light Qdrant Payload Optimization):** All fixes in Python testbench, plus Qdrant payload indices on `busemann_depth` and `tier` for server-side filtering.
- **Fusion strategy:** Linear alpha fusion with Busemann depth weighting. Replaces RRF as default.
- **No Rust changes** this round. Klein chord routing deferred to Phase 4.
- **Curvature re-sweep** now benefits from better metrics (content-only sep_ratio, tier-filtered retrieval, multiple fusion strategies).

---

## Section 1: Fix Measurement

### Content-Only Sep_Ratio

Compute sep_ratio using only content-tier points (same population as Round 2):

```
sep_ratio_content = |avg_depth(content, tier=leaf) - avg_depth(content, tier=root)| / std_depth(content)
```

The existing all-points calculation becomes `sep_ratio_all` for comparison.

### Cross-Tier Separation Metric

New metric measuring how well Busemann depth separates the three tiers:

```
cross_tier_sep = min(
    |mean_depth(content) - mean_depth(story)|,
    |mean_depth(story) - mean_depth(narrative)|
) / max(std_depth(content), std_depth(story), std_depth(narrative))
```

Measures the weakest link between adjacent tiers, normalized by worst intra-tier spread.

### 3-Way Tier Classification

Update tier classification accuracy to use thresholds between adjacent tier means instead of simple median binary classification.

**Files:** `benchmark.py` (benchmark_hierarchy_separation function).

---

## Section 2: Fusion Overhaul

### Three New Fusion Strategies in `fusion.py`

**Strategy 1: Linear alpha fusion**

```
score(d) = alpha * sim_cosine_norm(d) + (1 - alpha) * sim_poincare_norm(d)
```

Score normalization (critical):
- Cosine results: Qdrant returns similarity (higher=better). Min-max normalize to [0,1].
- Poincare results: Qdrant returns negative distance (higher=closer). Min-max normalize to [0,1].

Sweep alpha in {0.3, 0.5, 0.7, 0.9}.

**Strategy 2: Busemann depth-weighted alpha fusion**

```
depth_weight(d) = exp(-lambda * |busemann_depth(d) - busemann_depth(query)|)
score(d) = depth_weight(d) * (alpha * sim_cosine_norm(d) + (1 - alpha) * sim_poincare_norm(d))
```

Points at similar depth to query get boosted. Lambda controls sharpness.
Sweep lambda in {0.5, 1.0, 2.0} x alpha in {0.3, 0.5, 0.7, 0.9}.

**Strategy 3: Depth-band pre-filter + alpha fusion**

```
band = [query_depth - bandwidth, query_depth + bandwidth]
candidates = qdrant_search(filter: busemann_depth in band)
score = alpha * sim_cosine_norm + (1 - alpha) * sim_poincare_norm
```

Server-side depth range filter via Qdrant Range payload filter before fusion.
Sweep bandwidth in {0.2, 0.5, 1.0} x alpha in {0.3, 0.5, 0.7, 0.9}.

### Benchmark Comparison

Run all strategies on same 500-query set. Report best config for each:

| Strategy | Parameters | AreaRec@10 | DomRec@10 |
|----------|-----------|-----------|-----------|
| RRF (k=60) | baseline | ... | ... |
| Linear alpha | best alpha | ... | ... |
| Busemann-weighted | best alpha x lambda | ... | ... |
| Depth-band + alpha | best bandwidth x alpha | ... | ... |

**Files:** `fusion.py` (add three strategies), `benchmark.py` (extend dual-space suite).

---

## Section 3: Tier-Filtered Retrieval + Payload Indices

### Payload Indices

Create at collection setup time in `embed.py`:

- `busemann_depth` — float index (enables Range filter)
- `tier` — keyword index (enables exact-match filter, per-tier HNSW sub-graphs)

### Extended Retrieval Quality Benchmark

Run retrieval quality in four modes for unified collection:

| Mode | Filter | Purpose |
|------|--------|---------|
| Unfiltered | None | Current behavior |
| Tier-filtered | tier == "content" | Fair comparison to Round 2 |
| Depth-range | busemann_depth in [content_mean - 2*std, content_mean + 2*std] | Content band via payload index |
| Cosine + tier-filtered | tier == "content", using cosine vector | Cosine within unified |

**Expected:** Tier-filtered Poincare recovers to ~0.21+ (Round 2 level). If not, the problem is HNSW graph structure, not centroid contamination.

**Files:** `embed.py` (add payload index creation), `benchmark.py` (retrieval modes).

---

## Section 4: Curvature Re-sweep

### Methodology

Sweep c in {1.0, 2.0, 5.0, 10.0} with improved metrics:

For each curvature:
1. Re-project content via einstein_spread
2. Recompute tier centroids via Einstein midpoint
3. Recompute Busemann depths
4. Create temporary wos_unified_c{X} with payload indices
5. Run: content-only sep_ratio, cross-tier separation, tier-filtered retrieval, best fusion
6. Drop temporary collection

### Key Question

Does optimal curvature differ for hierarchy quality vs flat retrieval?

### Output

Comparison table: all curvatures x all metrics. Saved to `results/analysis_round3_5.md`.

**Files:** `embed.py` (update sweep to add payload indices), `benchmark.py` (sweep runner uses new metrics).

---

## File Changes Summary

| File | Change |
|------|--------|
| `dev/testbench/fusion.py` | Add linear_alpha_fuse, busemann_weighted_fuse, depth_band_fuse |
| `dev/testbench/benchmark.py` | Content-only sep_ratio, cross-tier metric, 3-way classification, retrieval filter modes, fusion strategy comparison, curvature sweep with new metrics |
| `dev/testbench/embed.py` | Add payload index creation after upsert, update sweep function |
| `dev/testbench/results/analysis_round3_5.md` | Results output |

## Success Criteria

1. **Fusion beats cosine baseline:** At least one fusion strategy produces area_recall@10 > 0.209 (cosine baseline)
2. **Tier-filtered Poincare recovers:** Tier-filtered Poincare area_recall@10 >= 0.200 (close to Round 2's 0.216)
3. **Content-only sep_ratio stable:** sep_ratio_content ~20 (matching Round 2)
4. **Cross-tier separation positive:** cross_tier_sep > 3.0
5. **Best curvature identified:** Clear winner across metrics, or documented trade-offs
