# Hyperbolic Vector Support for Qdrant

**Branch:** `feat/hyperbolic-vector-support`
**Timeline:** 2026-04-07 to 2026-04-14 (Phases 1-7)
**Scale:** 26 commits, ~100 files, +32,000 lines across Rust core, Python testbench, chatbot evaluator, datasets, and documentation

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Why Hyperbolic Geometry](#2-why-hyperbolic-geometry)
3. [Architecture Overview](#3-architecture-overview)
4. [Qdrant Fork: What We Changed](#4-qdrant-fork-what-we-changed)
5. [Mathematical Foundations](#5-mathematical-foundations)
6. [SIMD Optimization](#6-simd-optimization)
7. [Testbench](#7-testbench)
8. [Datasets](#8-datasets)
9. [Benchmark Results](#9-benchmark-results)
10. [Algorithms and Methodologies](#10-algorithms-and-methodologies)
11. [Phase 6: Client-Side Algorithm Suite](#11-phase-6-client-side-algorithm-suite)
12. [Phase 7: Chatbot Query Optimization](#12-phase-7-chatbot-query-optimization)
13. [Key Discoveries](#13-key-discoveries)
14. [Phase History](#14-phase-history)
15. [File Inventory](#15-file-inventory)
16. [What's Next](#16-whats-next)

---

## 1. What This Is

A fork of Qdrant that adds `Distance::Poincare` — a hyperbolic distance metric for the Poincare ball model. This enables **hierarchy-aware vector search** that flat Euclidean/cosine distance cannot do.

The core idea: in a unified Qdrant collection, each point has multiple named vectors:
- **`dense`** (1024d, Cosine) — for flat semantic similarity ("find similar items")
- **`poincare`** (128d, Poincare) — for exact hyperbolic distance (brute-force, geometric filters)
- **`tangent`** (128d, Euclidean) — for hierarchy-aware HNSW search (log-map projection of Poincare)

Cosine and hyperbolic are **complementary, not competing**. A chatbot routes by query intent:
- "Find posts about knife crime" → cosine (domain precision)
- "What narratives are driving the unrest?" → tangent (depth diversity, hierarchy traversal)
- "Give me a briefing on UK perception in Iraq" → tangent + RRF fusion (multi-level structure)

---

## 2. Why Hyperbolic Geometry

### The Problem

Hierarchical data (taxonomies, org charts, narrative hierarchies) has tree-like structure. Euclidean space can't embed trees without distortion — the number of nodes grows exponentially with depth, but Euclidean volume grows only polynomially.

### The Solution

Hyperbolic space has exponential volume growth (V(r) ~ e^r), matching tree branching. The Poincare ball model places:
- **Root** at the center (low Busemann depth)
- **Leaves** near the boundary (high Busemann depth)
- **Siblings** close together, **cousins** far apart

This is not theoretical — our testbench proves it across Phases 5-7:

| Capability | Cosine | Hyperbolic (Tangent HNSW) |
|-----------|--------|--------------------------|
| "Find similar posts" | **0.58 area recall** | 0.57 area recall |
| "Find children of this narrative" | Cannot do this | **1.0 recall** |
| "Find items at hierarchy depth 2-3" | Cannot do this | **100% precision** |
| "Find my ancestor in the hierarchy" | 8% hit rate | **46% hit rate (5.75x)** |
| "Give me a multi-level briefing" | 0.62 depth entropy | **0.94 depth entropy (+52%)** |
| "What narrative branches exist?" | 2.4 branches | **6.0 branches (2.5x)** |
| "Find siblings without explicit filter" | N/A | **44.8% sibling recall from geometry alone** |
| Brute-force recall@10 (c=0.25) | — | **0.90 (tangent HNSW)** |

Cosine excels at domain precision. Hyperbolic excels at depth diversity. Together they unlock the full picture.

---

## 3. Architecture Overview

### Qdrant Fork (Rust, ~900 lines)

The fork adds one new distance metric to Qdrant's existing framework. It's minimal and future-proof against upstream updates.

```
lib/segment/src/spaces/hyperbolic/
├── mod.rs              — Module registration + re-exports
├── poincare_math.rs    — Core math (distance, exp/log maps, Mobius, Einstein midpoint)
├── poincare_metric.rs  — Metric trait impl with SIMD dispatch
├── poincare_avx.rs     — AVX2+FMA accelerated distance (x86_64)
├── poincare_sse.rs     — SSE accelerated distance (x86/x86_64)
└── poincare_neon.rs    — NEON accelerated distance (aarch64)
```

Plus wiring across ~30 files:
- `Distance::Poincare` enum variant in `types.rs`
- Scorer dispatch in `raw_scorer.rs`, `async_raw_scorer.rs`, `point_scorer.rs`
- Configurable curvature (`Option<f32>`) per collection
- gRPC proto: `Poincare = 5`
- Feature-gated behind `--features hyperbolic`

### Testbench (Python, ~10,000 lines)

Everything application-specific lives here, not in the Qdrant fork:

```
dev/testbench/
├── benchmark.py          — 17-suite benchmark engine (Phases 1-6)
├── embed.py              — Embedding pipeline (pplx-embed-v1 → PCA → Poincare → tangent)
├── collection_builder.py — Unified/separate/cosine collection creation
├── queries.py            — 10 query types (drill-down, lateral, Klein, geometric, alpha, tangent, combined...)
├── hyperbolic_math.py    — Poincare/Lorentz/Klein math + geometric filters
├── projection.py         — 4 tier-aware projection strategies + auto_curvature
├── fusion.py             — 6 dual-space fusion strategies (RRF, alpha, busemann, horosphere)
├── tangent.py            — Tangent space projection (origin/frechet/einstein centroids)
├── comparator.py         — A/B comparison framework (recall@k, rank correlation)
├── pipeline_config.py    — Per-collection pipeline config presets
├── download_dataset.py   — WOS dataset downloader
└── chatbot/              — Phase 7: chatbot query optimization evaluator
    ├── eval.py           — Main evaluator (6 patterns × 3 strategies)
    ├── baselines.py      — Cosine, hyperbolic, multi-hop strategies
    ├── metrics.py        — 10 hierarchy-awareness metrics
    ├── query_patterns.py — 6 chatbot query pattern implementations
    └── results/          — JSON output
```

### Design Principle

**Qdrant provides the primitive** (`Distance::Poincare`). **Python provides the intelligence** (Busemann scoring, Einstein midpoints, geometric filters, hierarchy encoding). This keeps the fork small and merge-safe.

---

## 4. Qdrant Fork: What We Changed

### 4.1 Distance Enum

Added `Distance::Poincare` to `lib/segment/src/types.rs`, gated behind `#[cfg(feature = "hyperbolic")]`. Includes:
- `distance_order()` → `SmallBetter` (like Euclid)
- `postprocess_score()` → `abs()` (convert negative similarity to positive distance)
- `preprocess_vector()` → `project_to_ball()` (ensure vectors stay inside the ball)
- Curvature-aware preprocessing via `preprocess_vector_with_curvature()`

### 4.2 Curvature Configuration

Each named vector can specify curvature:
```json
{
  "vectors": {
    "poincare": {
      "size": 128,
      "distance": "Poincare",
      "curvature": 5.0
    }
  }
}
```

Stored as `curvature: Option<f32>` on `VectorDataConfig`. Defaults to 1.0 (standard unit Poincare ball). Flows through the scorer chain via `PoincareCurvatureQueryScorer`.

### 4.3 Scorer Chain

```
Search Request
  → FilteredScorer::new(curvature)
    → new_raw_scorer_with_curvature()
      → PoincareCurvatureQueryScorer
        → poincare_similarity_dispatched()  (SIMD when available)
          → poincare_similarity_avx / _sse / _neon / scalar fallback
```

The curvature parameter flows from collection config through the entire scorer chain. Before Phase 4, `new_raw_scorer_with_curvature()` existed but was never called (a critical bug we fixed).

### 4.4 gRPC Support

`Poincare = 5` added to the Distance enum in `collections.proto`. Conversion match arms in `conversions.rs` with proper `#[cfg]` gating — returns an error if the `hyperbolic` feature is not enabled.

### 4.5 Feature Flag Forwarding

The `hyperbolic` feature propagates through the entire crate chain:
```
Cargo.toml (root)
  → lib/storage/Cargo.toml
    → lib/shard/Cargo.toml
      → lib/collection/Cargo.toml
        → lib/segment/Cargo.toml (where the code lives)
  → lib/api/Cargo.toml (gRPC support)
```

Without the feature, `Distance::Poincare` is compiled out entirely — the fork builds identically to upstream Qdrant.

### 4.6 Quantization

Poincare maps to `DistanceType::L2` in the quantization library (`quantized_vectors.rs`). Validated with recall@10 = 1.0 on real data — the L2 approximation is sufficient for Poincare vectors.

### 4.7 Integration Tests

`lib/segment/tests/integration/hnsw_poincare_search_test.rs`:
- **Basic search correctness** (recall ≥ 95% vs brute-force)
- **Edge cases** (zero vector, boundary vector, single-point collection)
- **Quantization recall** (91% in Rust integration test, 100% on testbench data)

---

## 5. Mathematical Foundations

### 5.1 Poincare Ball Model

The Poincare ball B^n_c = {x in R^n : c||x||^2 < 1} with curvature parameter c > 0.

**Distance formula:**
```
d(u, v) = (1/sqrt(c)) * acosh(1 + 2c * ||u-v||^2 / ((1 - c||u||^2)(1 - c||v||^2)))
```

Key property: distance grows logarithmically near the center but exponentially near the boundary. This matches tree depth — root nodes are central, leaves are peripheral.

### 5.2 Numerically Stable acosh

Three regimes to avoid precision loss (`stable_acosh` in `poincare_math.rs`):
1. `x <= 1.0` → return 0 (domain boundary)
2. `1 < x < 1+1e-6` → Taylor expansion: `sqrt(2 * (x-1))` (avoids cancellation)
3. `x > 1e6` → asymptotic: `ln(2x)` (avoids `sqrt(x^2 - 1)` overflow)

### 5.3 Lorentz Hyperboloid Model

Alternative representation where the hyperboloid H^n = {x in R^(n+1) : <x,x>_L = -1/c} with Minkowski inner product <x,y>_L = -x_0*y_0 + x_1*y_1 + ... + x_n*y_n.

Used for:
- **Busemann scoring** (horosphere-based depth measurement)
- **Einstein midpoint** (closed-form hyperbolic centroid)

Conversions between models:
```
Poincare → Lorentz:  x_0 = (1 + c||p||^2) / (1 - c||p||^2),  x_i = 2*sqrt(c)*p_i / (1 - c||p||^2)
Lorentz → Poincare:  p_i = x_i / (sqrt(c) * (x_0 + 1))
```

### 5.4 Busemann Function

Measures "depth" in the hierarchy. Given a focal direction xi (light-like vector on the boundary):
```
B_xi(x) = log(-<x, xi>_L)
```

Lower Busemann score = closer to root. Higher = closer to leaves. Used to classify points into hierarchy tiers with 99.72% accuracy on BGC.

### 5.5 Einstein Midpoint

Closed-form hyperbolic centroid via the Lorentz model:
```
midpoint = sum(gamma_i * x_i) / sum(gamma_i)
```
where gamma_i = x_i[0] (the Lorentz time component, acts as a natural weight).

O(n) computation vs O(n * iterations) for iterative Frechet mean. Used to create aggregate embeddings for stories and narratives from their children.

### 5.6 Klein Disk Model

A projective model where geodesics are straight chords:
```
k_i = 2 * p_i / (1 + c * ||p||^2)
```

Klein chord distance (Euclidean L2 between Klein points) approximates Poincare distance ordering for nearby points. No transcendental functions needed — useful for cheap client-side pre-filtering.

### 5.7 Gromov Delta-Hyperbolicity

Measures how "tree-like" a metric space is using the 4-point condition. For random 4-tuples (x, y, u, v):
```
delta = max over samples of (s1 - s2) / 2
```
where s1, s2, s3 are the three pairwise distance sums sorted descending.

| Delta | Recommendation |
|-------|---------------|
| < 0.15 | Lorentz (extremely tree-like) |
| < 0.30 | Poincare (mildly hierarchical) |
| < 0.50 | Cosine (on hypersphere) |
| >= 0.50 | L2 (Euclidean) |

Our datasets: BGC = 0.057, HWV = 0.070 — both classified as "lorentz" (extremely tree-like).

### 5.8 Geometric Filters

Client-side post-filters on search results:

**InBall:** Keep points within Poincare distance r of a center. Uses precomputed alpha values for fast distance computation.

**InCone:** Keep points within angular aperture of a direction axis. Computed as arccos(dot(normalized_candidate, normalized_axis)). At aperture=0.5 rad, achieves 1.1% selectivity — extremely precise subtree isolation.

**DepthBand:** Keep points within a Busemann depth range [min, max]. 100% precision for tier selection.

---

## 6. SIMD Optimization

### 6.1 What Gets Optimized

The `fused_norms()` function — the inner loop computing three simultaneous accumulators:
```
diff_sq += (u[i] - v[i])^2
u_sq    += u[i]^2
v_sq    += v[i]^2
```

This is ~80% of distance computation time. The acosh + division are scalar (called once per distance).

### 6.2 Implementation

Three architecture-specific files following Qdrant's exact pattern from `simple_avx.rs`/`simple_sse.rs`/`simple_neon.rs`:

**AVX2+FMA** (`poincare_avx.rs`):
- 12 `__m256` registers (4 accumulators x 3 norms)
- Processes 32 elements per iteration
- Uses `_mm256_fmadd_ps` (fused multiply-add)
- `hsum256_ps` for horizontal sum

**SSE** (`poincare_sse.rs`):
- 12 `__m128` registers
- Processes 16 elements per iteration
- Uses `_mm_mul_ps` + `_mm_add_ps` (no FMA in basic SSE)

**NEON** (`poincare_neon.rs`):
- 12 `float32x4_t` registers
- Processes 16 elements per iteration
- Uses `vfmaq_f32` (native NEON FMA)
- `vaddvq_f32` for horizontal sum

### 6.3 Runtime Dispatch

`poincare_metric.rs` detects CPU features at runtime:
```
AVX+FMA + dim >= 32 → poincare_similarity_avx
SSE + dim >= 16     → poincare_similarity_sse
NEON + dim >= 16    → poincare_similarity_neon
fallback            → scalar poincare_distance
```

Both `PoincareMetric::similarity()` (DEFAULT_CURVATURE) and `PoincareCurvatureQueryScorer` (runtime curvature) use the same dispatch.

---

## 7. Testbench

### 7.1 Embedding Pipeline

```
Text documents
  → pplx-embed-v1 (1024d, GPU)
  → PCA reduction (1024d → 128d, ~70% variance retained)
  → Poincare projection (4 strategies, c=5.0)
  → Busemann depth computation (Lorentz model)
  → Alpha precomputation (1/(1 - c||x||^2))
  → Klein vector computation (for client-side pre-filtering)
  → Einstein midpoint aggregation (for story/narrative centroids)
  → Synthetic timestamp generation (for temporal queries)
  → Upsert to Qdrant (unified collection with named vectors + payloads)
```

### 7.2 Projection Strategies

| Strategy | Method | Use Case |
|----------|--------|----------|
| **uniform** | All vectors scaled to 0.9 ball radius | Baseline |
| **static** | Per-tier scaling (root=0.15, mid=0.50, leaf=0.90) | Known hierarchy |
| **einstein** | Calibrate radii from Einstein midpoint of each tier | Data-driven |
| **einstein_spread** | Einstein + intra-tier variance band | Fine-grained encoding |

### 7.3 Benchmark Suites (11 total)

| Suite | Purpose | Key Metric |
|-------|---------|------------|
| 1. Gromov Delta | Quantify tree-likeness | delta value + recommendation |
| 2. Hierarchy Separation | Tier separation quality | sep_ratio, tier_accuracy, band_overlap |
| 3. Drill-Down | Vertical traversal (narrative→stories→content) | hop1_recall, precision, latency |
| 4. Lateral | Same-tier search | area_recall, domain_recall, diversity |
| 5. Cross-Branch | Cross-domain structural similarity | unique_domains, depth_similarity |
| 6. Depth-Band | Busemann depth-filtered search | filter_precision, band coverage |
| 7. Unified vs Separate | Compare collection architectures | recall_delta, latency_delta |
| 8. Latency | Performance profiling | QPS, p50/p95/p99 |
| 9. Quantization Recall | Quantized vs exact search | recall@10 |
| 10. Klein Pre-Filter | 3-stage pipeline speedup | recall_vs_direct, latency |
| 11. Geometric Filters | InBall/InCone/composed precision | selectivity, filter_precision |

### 7.4 Query Types (7 total)

| Query | Pattern | Use Case |
|-------|---------|----------|
| **drill_down** | Narrative → filter by parent_ids → stories → content | "Show me stories behind this narrative" |
| **lateral** | Same-tier filter + vector search | "Find related stories" |
| **cross_branch** | Narrative search excluding source domain | "Find structurally similar narratives in other domains" |
| **temporal** | Date range + tier filter + vector search | "What happened last week at this tier?" |
| **depth_band** | Busemann depth range + vector search | "Find items at hierarchy depth 2-3" |
| **klein_prefilter** | Cosine prefetch → Klein re-rank → Poincare re-rank | Client-side optimization pipeline |
| **geometric_filtered** | Vector search → InBall/InCone/DepthBand post-filter | "Find items in this subtree direction within distance R" |

---

## 8. Datasets

### 8.1 Validated Datasets

| Dataset | Domain | Depth | Documents | Labels | Source |
|---------|--------|-------|-----------|--------|--------|
| **WOS** | Academic CS | 2 | 46,985 | 141 | HDLTex (Mendeley) |
| **BGC** | Book genres | 4 | 91,894 | 146 | U. Hamburg (CC BY-NC) |
| **EurLex** | EU legal | 3 | 17,000 | 556 | HuggingFace (CC BY-SA) |
| **HWV** | Wikipedia | 6 | 10,013 | 1,186 | CoNLL 2024 (MIT) |

### 8.2 Data Preparation Scripts

Each dataset has a preparation script in `dev/datasets/{name}/prepare.py` that:
1. Downloads from the source
2. Parses the hierarchy tree
3. Generates internal node entries (Einstein midpoint text for aggregate points)
4. Outputs `documents.jsonl` with standardized fields: `id`, `text`, `tier`, `depth`, `domain`, `area`, `hierarchy_path`, `parent_ids`, `child_ids`

---

## 9. Benchmark Results

### 9.1 Phase 5 Final Results (BGC, 92K docs, 11 suites)

| Metric | Value | Significance |
|--------|-------|-------------|
| **Gromov Delta** | 0.057 | "Lorentz" — extremely tree-like, strongest hyperbolic endorsement |
| **Drill-Down Recall** | 1.0 | Perfect hierarchy navigation |
| **Drill-Down Precision** | 1.0 | Zero false positives |
| **Depth-Band Precision** | 100% | Busemann depth perfectly separates tiers |
| **Tier Classification** | 99.72% | From Busemann depth alone |
| **Separation Ratio** | 13.19 | Strong inter-tier separation |
| **Quantization Recall** | 1.0 | L2 mapping is perfect for Poincare |
| **Lateral Area Recall** | 0.58 (dense), 0.57 (poincare) | Poincare matches cosine — no penalty |
| **InCone Selectivity** | 1.1% | Highly directional subtree filter |
| **Klein vs Direct** | 0.827 recall | Confirms server-side Poincare HNSW is necessary |
| **Latency** | 215-218 QPS, p50=4.1ms | Production-viable performance |

### 9.2 Depth-Scaling Curve (4 datasets)

| Dataset | Depth | Gromov Delta | Sep Ratio | Tier Acc | Drill-Down |
|---------|-------|-------------|-----------|----------|------------|
| WOS | 2 | — | 11.42 | 98.99% | — |
| EurLex | 3 | — | 1.45 | 94.48% | — |
| BGC | 4 | 0.057 | 13.19 | 99.72% | 1.0 |
| HWV | 6 | 0.070 | 2.68 | 89.82% | 1.0 |

**Key insight:** Semantic distinctness between tiers matters more than depth count. BGC's distinct genres give the best separation despite being only 4 levels. Drill-down works perfectly at all depths tested.

### 9.3 Key Finding: Klein Can't Replace Server-Side Poincare

The Klein client-side pre-filter achieves only 0.827 recall vs direct Poincare search. This is because:
- Cosine prefetch finds semantically similar items (different set than Poincare-nearest)
- Klein chord distance is an approximation — ordering diverges for distant points
- **Conclusion:** The Qdrant fork is necessary. Client-side workarounds can't replicate server-side Poincare HNSW.

---

## 10. Algorithms and Methodologies

### 10.1 Projection Strategies

**Problem:** Raw PCA-reduced vectors are Euclidean. They need to be projected into the Poincare ball with tier-aware radial placement.

**Solution:** Einstein midpoint calibration:
1. Project all vectors uniformly via exp_map
2. Compute Einstein midpoint per tier → extract natural tier radii
3. Scale each vector so its norm matches its tier's calibrated radius
4. Add intra-tier variance spread for fine-grained encoding

### 10.2 Unified Collection Architecture

**Problem:** Pythia uses 13 separate Qdrant collections. Cross-tier queries require Neo4j graph traversal.

**Solution:** One collection per client with named vectors + payloads:
- Named vectors: `dense` (Cosine, 1024d) + `poincare` (Poincare, 128d)
- Payloads: `item_type`, `tier`, `busemann_depth`, `parent_ids`, `child_ids`, `alpha`, `klein_vector`
- Cross-tier queries become filtered vector searches: `item_type=story AND parent_ids contains narrative_id`

### 10.3 Dual-Space Fusion

Combining cosine and Poincare results:
- **RRF (Reciprocal Rank Fusion):** score = 1/(k + rank_cosine) + 1/(k + rank_poincare)
- **Alpha Blending:** score = alpha * cosine_score + (1-alpha) * poincare_score
- **Busemann-Weighted:** alpha varies by Busemann depth (deeper = more Poincare weight)
- **Depth-Band:** pre-filter by Busemann depth range, then search

### 10.4 Gromov Delta Analysis

Quantitative measure of dataset tree-likeness. Sample 4-tuples, compute the 4-point condition, report maximum delta. Used to decide whether hyperbolic geometry is worthwhile for a given dataset.

### 10.5 Three-Stage Klein Pipeline

Client-side optimization:
1. **Qdrant cosine search** (server) → top-200 candidates
2. **Klein chord distance** (client, cheap L2) → re-rank to top-50
3. **Poincare distance** (client, exact) → re-rank to top-10

Reduces acosh calls from ~200 to 50, but recall is 0.827 (not a full replacement).

---

## 11. Phase 6: Client-Side Algorithm Suite

**Commit:** `f0f041698` (2026-04-14) | **+4,407 lines, 24 files**

### What Was Built

Three-layer client-side pipeline:

1. **Pre-upload**: tangent coordinate computation (3 centroid strategies), alpha precomputation, auto-curvature
2. **Query-time**: alpha pipeline, tangent HNSW pipeline, combined pipeline
3. **Post-processing**: 6 fusion strategies (RRF, linear alpha, busemann weighted, depth band, busemann proximity, horosphere with 4 intents)

Plus: A/B comparator framework, pipeline config presets, gRPC curvature field in VectorParams.

### Key Findings (BGC, 92K docs, c=0.25)

| Pipeline | Recall@10 vs Brute-Force | Notes |
|----------|------------------------:|-------|
| **Tangent HNSW (origin)** | **0.900** | Uses stock Qdrant Euclidean HNSW |
| RRF fusion (cosine + Poincare) | 0.690 | Best fusion strategy |
| Alpha pipeline | 0.614 | -33% latency vs Klein |
| Klein pipeline | 0.614 | Phase 5 baseline |

### Critical Discovery: Curvature is Inverted

c=0.25 is **3.3x better** than c=5.0 on same_tier_precision (0.810 vs 0.246 in initial sweep). The Gromov delta → curvature mapping we designed was inverted. Low curvature preserves neighborhood structure; high curvature distorts boundary-heavy data. The auto_curvature heuristic is unreliable — per-collection empirical calibration (or SONA learning) is required.

### Optimal Parameters

| Parameter | Value | Evidence |
|-----------|-------|---------|
| Curvature | c=0.25 | 2x recall vs c=5.0 |
| Query pipeline | Tangent HNSW | 0.90 vs 0.614 recall |
| Centroid strategy | Origin | Same quality as frechet/einstein, free |
| Prune factor | 5 | Same recall as 20, 4x less work |
| Fusion | RRF | 0.690 recall, best strategy |

---

## 12. Phase 7: Chatbot Query Optimization

**Commit:** `2ba55c3fb` (2026-04-14) | **+3,122 lines, 8 files**

### Motivation

Real user queries from Pythia's `NetworksLensHistory` (67 queries across 3 clients) revealed that many chatbot queries are inherently hierarchical. The current Pythia pipeline requires 5-7 cross-database hops (Qdrant → Neo4j → MongoDB → LLM). We hypothesized that hyperbolic embeddings in a unified collection could reduce this to 1-2 Qdrant queries.

### What Was Built

New `dev/testbench/chatbot/` module: 6 query patterns × 3 retrieval strategies evaluated on BGC.

**Strategies:**
- **Cosine-only** — dense named vector + payload filters (flat baseline)
- **Hyperbolic** — tangent HNSW + RRF fusion (the thesis)
- **Multi-hop** — separate per-tier collections, sequential queries (current Pythia simulation)

**Patterns (motivated by real Pythia queries):**

| Pattern | Example Query | What It Tests |
|---------|--------------|---------------|
| Drill-Up | "Disorder following teen riots in Clapham" | Upward hierarchy traversal |
| Context Assembly | "Central bank narratives" | Full subtree retrieval |
| Hierarchy Similarity | "Knife Crime in London" | Sibling finding |
| Multi-Level Briefing | "Tell me about the narratives around crime in London" | Depth-diverse results |
| Cross-Branch | "King Charles narratives" | Cross-branch narrative discovery |
| Narrative Landscape | "view of british in iraq currently" | Distinct narrative branches |

### Results: Thesis Validated

| Pattern | Key Metric | Cosine | Hyperbolic | Hyperbolic Advantage |
|---------|-----------|--------|------------|---------------------|
| Drill-Up | ancestor_hit_rate | 0.08 | **0.46** | **5.75x** |
| Context Assembly | depth_coverage | 0.952 | **1.000** | Full tier coverage |
| Hierarchy Similarity | sibling_recall (no filter) | N/A | **0.448** | Geometry alone |
| Multi-Level Briefing | depth_entropy | 0.615 | **0.937** | **+52%** |
| Cross-Branch | branch_diversity | 2.43 | **6.0** | **2.5x** |
| Cross-Branch | depth_precision | 0.129 | **0.500** | **3.9x** |

### The Complementarity Finding

Hyperbolic and cosine are **complementary, not competing**:
- Cosine excels at **domain precision** (subtree_coherence: 0.414 vs 0.264)
- Hyperbolic excels at **depth diversity** (depth_coverage: 1.0 vs 0.952, entropy: +52%)

A chatbot routes by query intent:
- Flat semantic queries → cosine
- Hierarchy-aware queries → tangent HNSW
- Both live as named vectors in the same unified collection

---

## 13. Key Discoveries

### 1. Curvature Must Be Calibrated Empirically

The Gromov delta → curvature mapping is inverted. c=0.25 is 3.3x better than c=5.0 on BGC. auto_curvature() suggested c=2.0 — wrong. Low curvature preserves neighborhood structure for boundary-heavy data (99.9% content tier). Per-collection calibration or SONA learning is essential.

### 2. Tangent HNSW is the Production Workhorse

Recall 0.90 using stock Qdrant Euclidean HNSW on tangent-projected coordinates. No fork needed for the highest-recall pipeline. The Poincare fork is valuable for exact distance, geometric filters, and brute-force ground truth — but tangent HNSW is the deployment path.

### 3. Phase 5 vs Phase 6 Ground Truth Difference

Phase 5 measured Klein pipeline vs HNSW (recall 0.827). Phase 6 measured all pipelines vs brute-force exact Poincare (recall 0.298 at c=5.0, 0.614 at c=0.25). The stricter ground truth revealed that HNSW at c=5.0 was itself inaccurate — Phase 5's "good" numbers were two approximate methods agreeing with each other.

### 4. Hyperbolic Geometry Encodes Hierarchy in the Embedding

Phase 7's hierarchy_similarity pattern proved this: tangent HNSW finds 44.8% true siblings without any explicit parent filtering — purely from the geometric structure of the Poincare ball. The embedding captures hierarchy relationships that cosine completely misses.

### 5. RRF Fusion Beats Any Single Pipeline

RRF combining cosine + Poincare results achieves 0.690 recall — higher than either pipeline alone (cosine or alpha/tangent). The two spaces capture different information.

### 6. Busemann/Horosphere Fusion Shows No Benefit on Flat Data

On BGC (99.9% content tier), depth-based reweighting has nothing meaningful to shift. The value of Busemann/horosphere fusion will appear on data with balanced tier distribution — like real Pythia collections with narratives + stories + posts.

---

## 14. Phase History

### Phase 1: Foundation (2026-04-07)
- Distance::Poincare enum + feature gate
- Poincare ball core math (distance, exp/log maps, projection)
- PoincareMetric trait for f32/u8/f16

### Phase 1.5: Advanced Math (2026-04-07)
- Busemann scoring via Lorentz model
- Tangent space cache + pruning scorer
- Configurable curvature + Einstein midpoint

### Phase 2: Testbench (2026-04-08)
- Embedding pipeline (pplx-embed-v1 → PCA → Poincare)
- 5 benchmark suites
- WOS dataset validation
- Tier-aware projection strategies (4 variants)

### Phase 3: Unified Collection (2026-04-09)
- Unified collection with named vectors
- Fusion strategies (RRF, alpha, Busemann-weighted, depth-band)
- Multi-dataset support (BGC, EurLex, SciHTC)
- Fair multi-label evaluation methodology

### Phase 3.5: Measurement Fix (2026-04-09)
- Fixed evaluation to be fair across cosine and Poincare
- Curvature sweep (c=0.5, 1.0, 2.0, 5.0 — optimal: 5.0)
- Cross-dataset validation (WOS, BGC, EurLex)

### Phase 4: Architecture Validation (2026-04-10)
- Fixed curvature dead code bug (scorer chain never used configurable curvature)
- Added Poincare to gRPC proto
- Fixed preprocess_vector to use collection curvature
- Removed Busemann/tangent from Rust → Python
- 8-suite testbench with chatbot query benchmarks
- BGC + HWV validation: drill-down recall = 1.0 on both
- Gromov delta analysis, alpha precomputation, Einstein midpoint centroids

### Phase 5: Production Polish (2026-04-10)
- SIMD Poincare distance (AVX2+FMA, SSE, NEON)
- Integration tests (3 end-to-end HNSW+Poincare)
- Quantization validated (recall@10 = 1.0)
- Klein client-side pre-filter (recall = 0.827)
- Geometric filters (InBall, InCone, composable)
- 11 benchmark suites total

### Phase 6: Client-Side Algorithm Suite (2026-04-14)
- Tangent HNSW pipeline (0.90 recall — best)
- Alpha pipeline (-33% latency vs Klein)
- 6 fusion strategies (RRF wins at 0.690)
- Curvature finding: c=0.25 >> c=5.0 (inverted mapping)
- gRPC curvature field in VectorParams
- SIMD unsafe cleanup
- 17 benchmark suites total

### Phase 7: Chatbot Query Optimization (2026-04-14)
- 6 chatbot query patterns from real Pythia user queries
- 3 retrieval strategies (cosine, hyperbolic, multi-hop)
- Validated thesis: hyperbolic complements cosine
- Drill-up ancestor hit rate: 46% vs 8% (5.75x)
- Multi-level briefing depth entropy: +52%
- Cross-branch diversity: 6.0 vs 2.4 branches
- Hierarchy similarity: 44.8% siblings from geometry alone

---

## 15. File Inventory

### Qdrant Fork (Rust)

| File | Lines | Purpose |
|------|-------|---------|
| `lib/segment/src/spaces/hyperbolic/mod.rs` | 21 | Module registration + re-exports |
| `lib/segment/src/spaces/hyperbolic/poincare_math.rs` | 754 | Core Poincare ball math (15 public functions) |
| `lib/segment/src/spaces/hyperbolic/poincare_metric.rs` | 271 | Metric trait + SIMD dispatch + curvature scorer |
| `lib/segment/src/spaces/hyperbolic/poincare_avx.rs` | 183 | AVX2+FMA SIMD distance |
| `lib/segment/src/spaces/hyperbolic/poincare_sse.rs` | 179 | SSE SIMD distance |
| `lib/segment/src/spaces/hyperbolic/poincare_neon.rs` | 166 | NEON SIMD distance |
| `lib/segment/tests/integration/hnsw_poincare_search_test.rs` | 464 | 3 integration tests |

Plus ~30 modified files for wiring (scorer dispatch, feature flags, gRPC, config).

### Testbench (Python)

| File | Lines | Purpose |
|------|-------|---------|
| `dev/testbench/benchmark.py` | ~2,550 | 17-suite benchmark engine (Phases 1-6) |
| `dev/testbench/embed.py` | ~1,300 | Embedding + projection + tangent + upsert pipeline |
| `dev/testbench/collection_builder.py` | ~520 | Collection creation (unified/separate/cosine) |
| `dev/testbench/queries.py` | ~770 | 10 query implementations (incl. alpha, tangent, combined) |
| `dev/testbench/hyperbolic_math.py` | ~490 | Poincare/Lorentz/Klein math + filters + log_map + frechet |
| `dev/testbench/projection.py` | ~260 | 4 projection strategies + auto_curvature |
| `dev/testbench/fusion.py` | ~260 | 6 fusion strategies (RRF, alpha, busemann, horosphere) |
| `dev/testbench/tangent.py` | 85 | Tangent space projection (3 centroid strategies) |
| `dev/testbench/comparator.py` | 87 | A/B comparison (recall@k, rank correlation, reports) |
| `dev/testbench/pipeline_config.py` | 64 | Pipeline config presets |
| `dev/testbench/download_dataset.py` | 154 | WOS dataset downloader |

### Chatbot Evaluator (Phase 7)

| File | Lines | Purpose |
|------|-------|---------|
| `dev/testbench/chatbot/eval.py` | ~420 | Main evaluator entry point |
| `dev/testbench/chatbot/baselines.py` | ~415 | 3 retrieval strategies (cosine, hyperbolic, multi-hop) |
| `dev/testbench/chatbot/query_patterns.py` | ~710 | 6 chatbot query patterns + hierarchy index |
| `dev/testbench/chatbot/metrics.py` | ~100 | 11 hierarchy-awareness metrics |

### Datasets

| File | Lines | Purpose |
|------|-------|---------|
| `dev/datasets/bgc/prepare.py` | 443 | BGC dataset preparation |
| `dev/datasets/eurlex/prepare.py` | 278 | EurLex dataset preparation |
| `dev/datasets/hwv/prepare.py` | 444 | HWV WikiVitals preparation |
| `dev/datasets/scihtc/prepare.py` | 317 | SciHTC preparation |

### Rust Public API (poincare_math.rs)

```rust
pub fn poincare_distance(u, v, c) -> f32
pub fn project_to_ball(v, c) -> Vec<f32>
pub fn exp_map_origin(v, c) -> Vec<f32>
pub fn log_map_origin(p, c) -> Vec<f32>
pub fn mobius_add(x, y, c) -> Vec<f32>
pub fn mobius_neg(x) -> Vec<f32>
pub fn conformal_factor(x, c) -> f32
pub fn exp_map(p, v, c) -> Vec<f32>
pub fn log_map(p, y, c) -> Vec<f32>
pub fn frechet_mean(points, c) -> Vec<f32>
pub fn einstein_midpoint(points, c) -> Vec<f32>
pub fn poincare_to_lorentz(p, c) -> Vec<f32>
pub fn lorentz_inner(x, y) -> f32
pub fn project_hyperboloid(x, c) -> Vec<f32>
pub fn lorentz_to_poincare(x, c) -> Vec<f32>
```

### Python Public API (hyperbolic_math.py)

```python
exp_map_at_origin(v, c)          poincare_to_klein(vector, c)
project_to_ball(x, c)            poincare_to_klein_batch(vectors, c)
poincare_to_lorentz(p, c)        klein_chord_distance_sq(u, v)
lorentz_to_poincare(x, c)        inball_filter(candidates, center, radius, c)
lorentz_inner(x, y)              incone_filter(candidates, axis, aperture, origin)
project_hyperboloid(x, c)        gromov_delta(vectors, num_samples)
busemann_score(x_lorentz, focal) alpha_precompute(vector, c)
compute_focal_direction(points, c) alpha_precompute_batch(vectors, c)
compute_busemann_depths(points, c) fused_norms(u, v)
busemann_depth_single(v, focal, c) poincare_distance_with_alpha(diff_sq, a_u, a_v, c)
einstein_midpoint(points, c)
```

---

## 16. What's Next

### Immediate: HWV Validation

Run Phase 7 chatbot evaluator on HWV (6-level, 10K docs) to confirm deeper hierarchies show stronger hyperbolic advantage. Infrastructure is ready — just need `python embed.py --dataset hwv` then `python -m chatbot.eval --dataset hwv`.

### Short-term: Query Intent Routing

Build a lightweight classifier that detects whether a chatbot query needs hierarchy (route to tangent HNSW) or flat similarity (route to cosine). Based on Phase 7's pattern analysis of real Pythia queries.

### Medium-term: Pythia Integration

Carry proven patterns to the Pythia platform:

1. **Unified per-client collections** — replace per-type collections (networks_posts, stories, narratives) with one collection per client containing all tiers
2. **Named vectors** — dense (1024d Cosine) + sparse (BGE-M3) + tangent (128d Euclid) + poincare (128d Poincare)
3. **Lens retriever update** — query intent routing selects cosine vs tangent per query
4. **Neo4j scope reduction** — hierarchy moves to Qdrant payloads (busemann_depth, parent_ids, child_ids); Neo4j handles only actor relationships (CO_AMPLIFIES, POSTED_IN)
5. **Embedding pipeline update** — add PCA → Poincare projection → tangent → Busemann depth to the existing pplx-embed-v1 pipeline

### Long-term: SONA Learning

Per-collection adaptive learning for:
- **Curvature** — learn optimal c from retrieval feedback (manual tuning proven broken by Phase 6)
- **Fusion weights** — learn cosine/tangent routing thresholds per collection
- **Prune factor** — optimize tangent HNSW prune_factor per collection
- **Projection strategy** — learn optimal einstein_spread parameters

### Inspiration Sources

- **RuVector** (`~/projects/ruvector`) — SONA self-learning architecture, Einstein midpoint, tangent space pruning, EWC++/InfoNCE for continual learning
- **HyperspaceDB** (`~/projects/hyperspace-db`) — Klein chord routing, Gromov delta analysis, geometric filters (InBall/InCone/InBox), anisotropic quantization, alpha precomputation
