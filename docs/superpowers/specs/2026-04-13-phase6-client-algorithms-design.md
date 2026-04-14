# Phase 6: Client-Side Algorithm Suite — Design Spec

**Date:** 2026-04-13
**Branch:** `feat/hyperbolic-vector-support`
**Scope:** Alpha precomputation, tangent space pruning, Busemann-weighted fusion, horosphere scoring, Gromov-guided curvature, per-collection config, 6 new benchmark suites
**Server-side change:** One gRPC proto addition (curvature field in VectorParams)

---

## 1. Motivation

Phase 5 delivered a production-quality Poincare distance metric in the Qdrant fork: SIMD-optimized, integration-tested, with quantization validated at recall=1.0. The server-side implementation is complete and minimal — 900 lines of Rust, easy to maintain against upstream.

But the client-side pipeline has room to grow. The existing Klein pre-filter (recall=0.827) uses an approximate distance proxy. Fusion is basic RRF or linear alpha blending. Curvature is hardcoded at c=5.0. There's no server-side approximate hyperbolic search — all Poincare queries go through exact distance computation.

Phase 6 keeps the server minimal and builds a rich **client-side algorithm suite** that:
- Replaces Klein with **alpha precomputation** (exact ordering, no acosh, monotonically equivalent to Poincare distance)
- Adds **tangent space HNSW** (server-side approximate hyperbolic search using Qdrant's native Euclidean HNSW)
- Introduces **hierarchy-aware post-processing** (Busemann-weighted fusion, horosphere scoring, depth-band filtering)
- Auto-selects **curvature per collection** from Gromov delta analysis
- Structures all parameters as **per-collection config** for future SONA adaptive learning

### Design Principles

1. **Minimal server-side changes** — One proto field addition. Everything else is client-side Python.
2. **Benchmark everything** — 6 new suites comparing every algorithm against Phase 5 baselines on BGC and HWV.
3. **Parameters are config, not constants** — Every tunable value lives in a per-collection config dataclass, ready for future learned optimization.
4. **Composable layers** — Pre-upload, query-time, and post-processing layers are independent and testable in isolation.

---

## 2. Architecture Overview

```
SERVER-SIDE (minimal)
  One change: add curvature field to gRPC VectorParams proto

CLIENT-SIDE ALGORITHM SUITE (dev/testbench/)

  Layer 1: PRE-UPLOAD
    Gromov delta → auto-curvature per collection
    Enhanced projection (Gromov-guided radii)
    Alpha precomputation → stored as payload
    Busemann depth → stored as payload
    Tangent coordinates → stored as named vector

  Layer 2: QUERY-TIME
    Path A: Alpha pipeline (replaces Klein stage 2)
    Path B: Tangent HNSW (server-side Euclidean)
    Path A+B: Combined (merge candidates from both)
    All: exact Poincare for final re-ranking

  Layer 3: POST-PROCESSING
    Busemann-weighted fusion (cosine + Poincare scores)
    Horosphere scoring (ancestor/descendant/sibling bias)
    Depth-band filtering (server-side via payload index)

  TESTBENCH
    6 new benchmark suites (12-17)
    A/B comparison framework
    Per-collection config system
```

### Collection Schema

Per-client collection (e.g., "iraq_networks"):

```
Named vectors:
  "dense"    → 1024d, Cosine    (semantic similarity)
  "poincare" → 128d,  Poincare  (exact hyperbolic distance)
  "tangent"  → 128d,  Euclid    (approximate hyperbolic via tangent space)

Payload fields:
  tier             (keyword)  — content | stories | narratives
  domain           (keyword)  — topic domain
  area             (keyword)  — geographic/thematic area
  hierarchy_path   (keyword)  — full path string
  parent_ids       (keyword[]) — parent point IDs
  child_ids        (keyword[]) — child point IDs
  busemann_depth   (float, indexed) — hierarchy depth score
  alpha            (float)    — precomputed 1/(1 - c*||x||^2)
  tangent_centroid (keyword)  — which centroid strategy was used
```

### Per-Collection Config

```python
@dataclass
class PipelineConfig:
    # Layer 1: Pre-upload
    curvature: float = 1.0
    projection_strategy: str = "einstein_spread"
    tangent_centroid: str = "origin"  # "origin" | "frechet" | "einstein"

    # Layer 2: Query-time
    query_pipeline: str = "alpha"    # "alpha" | "tangent" | "combined"
    prune_factor: int = 10
    stage_sizes: tuple = (200, 50)

    # Layer 3: Post-processing
    fusion_strategy: str = "busemann"  # "busemann" | "rrf" | "linear_alpha"
    fusion_alpha: float = 0.5
    depth_weight: float = 1.0
    horo_steepness: float = 1.0
    depth_band_width: float | None = None
```

All values are manually set in Phase 6, benchmarked on BGC/HWV. Designed to be replaced by SONA-learned values per collection in future Pythia integration.

---

## 3. Server-Side Change: gRPC Curvature Field

### What

Add an optional `curvature` field to `VectorParams` in the gRPC proto definition.

### Where

`lib/api/src/grpc/proto/collections.proto`, in the `VectorParams` message:

```proto
message VectorParams {
  uint64 size = 1;
  Distance distance = 2;
  optional HnswConfigDiff hnsw_config = 3;
  optional QuantizationConfig quantization_config = 4;
  optional bool on_disk = 5;
  optional Datatype datatype = 6;
  optional MultiVectorConfig multivec_config = 7;
  optional float curvature = 8;  // NEW: Poincare ball curvature, default 1.0
}
```

### Conversion Wiring

In `lib/collection/src/config.rs` (line ~618), the proto-to-internal conversion currently hardcodes `curvature: None`. Change this to read the proto field:

```rust
#[cfg(feature = "hyperbolic")]
curvature: vector_params.curvature,
```

The internal `VectorDataConfig.curvature` already exists as `Option<f32>` in `lib/segment/src/types.rs` (line 1698), feature-gated behind `#[cfg(feature = "hyperbolic")]`.

### Why gRPC

Production inference uses gRPC for protobuf serialization, streaming, and service contracts. The REST API workaround (Phase 4) works for development but is insufficient for Pythia's production pipeline.

---

## 4. Layer 1: Pre-Upload

### 4.1 Gromov Delta Auto-Curvature

Existing `gromov_delta()` in `hyperbolic_math.py` computes dataset tree-likeness. New: use it to auto-select curvature per collection.

**Algorithm:**

```
Input: raw embeddings (1024d dense, pre-PCA)
  → PCA to 128d
  → Gromov delta analysis (~1000 samples)
  → Curvature selection (testable thresholds):
      delta < 0.10  →  c = 2.0  (strongly tree-like)
      delta < 0.20  →  c = 1.0  (moderately tree-like)
      delta < 0.35  →  c = 0.5  (weakly tree-like)
      delta >= 0.35 →  c = 0.25 (barely tree-like)
```

Thresholds are initial guesses — Suite 15 (Curvature Sweep) will empirically determine optimal breakpoints on BGC and HWV.

**New function in projection.py:**

```python
def auto_curvature(gromov_delta: float) -> float:
    """Select curvature from Gromov delta. Thresholds are benchmarked, not final."""
    if gromov_delta < 0.10: return 2.0
    if gromov_delta < 0.20: return 1.0
    if gromov_delta < 0.35: return 0.5
    return 0.25
```

### 4.2 Enhanced Projection

Existing `project_einstein_spread()` remains the projection strategy. Enhancements:
- Accept curvature from auto-selection instead of hardcoded value
- Feed Gromov delta into radius calibration: lower delta (more tree-like) → more spread between tiers

No new projection strategies needed.

### 4.3 Alpha and Busemann Precomputation

After projection, before upload, compute two payload fields per point:

```python
# Alpha: conformal factor (enables acosh-free distance ordering)
# Math: alpha_x = 1 / (1 - c * ||x||^2)
# Verification: identical formula in HyperspaceDB (vector.rs:33) and our poincare_math.rs
alpha = alpha_precompute(poincare_vector, curvature)

# Busemann: hierarchy depth score (enables server-side depth filtering)
# Math: B_xi(x) = log(-<x, xi>_L) where xi is light-like focal direction
# Requires Lorentz coordinates (d+1 dims) — convert first
lorentz = poincare_to_lorentz(poincare_vector, curvature)
busemann = busemann_score(lorentz, focal_direction)
```

Both functions already exist in `hyperbolic_math.py`. The change is storing them as payload fields during collection upload.

### 4.4 Tangent Coordinate Computation

New step. Compute tangent space coordinates for each point — a Euclidean representation that approximates Poincare distance.

**Mathematical basis (verified against RuVector tangent.rs):**
- `log_map(centroid, point, c)` projects a Poincare ball point into the tangent space at `centroid`
- In tangent space, the Riemannian metric is locally Euclidean
- Euclidean distance between tangent vectors approximates Poincare distance with error O(||u-c||^3 + ||v-c||^3)
- Approximation degrades far from centroid — compensated by `prune_factor` inflation + exact re-ranking

**Three centroid strategies to benchmark:**

```python
# Strategy A: Origin (zero vector)
# Cheapest. log_map_origin is ~3x faster (no Mobius ops).
# Good fit: our projections center roots at origin.
# Formula: log_0(p) = (2/sqrt(c)) * arctanh(sqrt(c)*||p||) * p/||p||
centroid = np.zeros(dim)
tangent_coords = [log_map_origin(p, c) for p in poincare_vectors]

# Strategy B: Frechet mean (iterative, 100 iters, lr=0.1, tol=1e-6)
# Most accurate centroid. Expensive to compute, but one-time cost.
# Query-time uses general log_map (expensive).
centroid = frechet_mean(all_points, c)
tangent_coords = [log_map(centroid, p, c) for p in poincare_vectors]

# Strategy C: Einstein midpoint (closed-form, O(n*d))
# Nearly as accurate as Frechet (<0.1 L2 distance, verified in test).
# Same query-time cost as Frechet.
centroid = einstein_midpoint(all_points, c)
tangent_coords = [log_map(centroid, p, c) for p in poincare_vectors]
```

Tangent coordinates are uploaded as the `"tangent"` named vector with Euclidean distance. Qdrant's native HNSW indexes them automatically — no fork changes needed.

### 4.5 Collection Creation

Collection builder creates the unified collection via gRPC (with new curvature field):

```python
CreateCollection("iraq_networks"):
  vectors:
    "dense":    VectorParams(size=1024, distance=Cosine)
    "poincare": VectorParams(size=128, distance=Poincare, curvature=auto_c)
    "tangent":  VectorParams(size=128, distance=Euclid)
  payload_indices:
    "busemann_depth": float range index   # enables server-side depth-band filtering
    "tier": keyword index
```

---

## 5. Layer 2: Query-Time

Two independent query paths, plus a combined mode. All share exact Poincare re-ranking as the final step.

### 5.1 Path A: Alpha Pipeline

Replaces the existing Klein 3-stage pipeline.

```
Stage 1: Cosine Prefetch (SERVER)
  Search "dense" named vector via HNSW
  Return top-200 candidates with payloads (includes alpha, poincare vector)

Stage 2: Alpha Re-rank (CLIENT, no acosh)
  query_alpha = 1 / (1 - c * ||q||^2)
  For each candidate:
    diff_sq = ||q - candidate_poincare||^2
    proxy = diff_sq * query_alpha * candidate_alpha
  Sort by proxy ascending
  Keep top-50

Stage 3: Exact Poincare (CLIENT)
  d(q,v) = (1/sqrt(c)) * acosh(1 + 2*c * proxy)
  Sort by exact distance
  Return top-k
```

**Why alpha replaces Klein:**
- Klein chord distance is an approximate ordering proxy
- Alpha proxy (`||u-v||^2 * alpha_u * alpha_v`) gives **exact** ordering (monotonic with Poincare distance, since acosh is strictly increasing on [1, inf))
- Alpha is cheaper: no model conversion step (Klein requires `2*x / (1 + c*||x||^2)` per vector)
- Alpha values are precomputed and stored as payload — zero computation for candidate alphas

**Stage 3 optimization:** The `proxy` value computed in Stage 2 IS the argument to acosh (minus the `1 + 2c*` wrapper). Stage 3 just applies `acosh(1 + 2*c*proxy) / sqrt(c)` — no recomputation needed.

### 5.2 Path B: Tangent HNSW

Uses Qdrant's native Euclidean HNSW for approximate hyperbolic search.

```
Pre-query: Tangent Projection (CLIENT)
  query_tangent = log_map(centroid, query, c)
  (If centroid=origin: use log_map_origin — ~3x cheaper)

Stage 1: Tangent HNSW Search (SERVER)
  Search "tangent" named vector (Euclidean HNSW)
  query = query_tangent
  Return top-(k * prune_factor) candidates
  e.g., prune_factor=10, k=10 → top-100

Stage 2: Exact Poincare Re-rank (CLIENT)
  Fetch raw "poincare" vectors for candidates
  d(q,v) = poincare_distance(q, v, c)
  Sort by exact distance
  Return top-k
```

**Why tangent HNSW:**
- Qdrant's battle-tested Euclidean HNSW does the heavy lifting server-side
- No cosine detour — searches in a space that directly approximates hyperbolic geometry
- Only `k * prune_factor` exact distance computations (e.g., 100 vs 200 in alpha pipeline)
- `prune_factor` controls the recall/compute trade-off

**Configurable knobs:**
- `prune_factor`: 5, 10, 20 (benchmarked in Suite 14)
- Centroid strategy: origin, frechet, einstein (benchmarked in Suite 13)
- Optional: combine with depth-band payload filter for server-side hierarchy pre-filtering

### 5.3 Path A+B: Combined Pipeline

Merges candidates from both cosine and tangent search.

```
              Query
             /     \
Search "dense"     Search "tangent"
(Cosine HNSW)      (Euclidean HNSW)
top-100             top-100
     \               /
      Merge + deduplicate by point ID
                |
        Alpha re-rank (no acosh)
        Keep top-50
                |
        Exact Poincare re-rank
        Return top-k
```

Two independent retrieval signals — semantic similarity (cosine) and hierarchical proximity (tangent) — merged and refined. Richest pipeline but most expensive (two server queries + client-side processing).

---

## 6. Layer 3: Post-Processing

Operates on candidates returned by the query-time layer. Re-scores using hierarchy-aware signals.

### 6.1 Busemann-Weighted Fusion

Enhanced version of existing `busemann_weighted_fuse()` in `fusion.py`.

**Algorithm:**

```
Inputs:
  cosine_results:   [(id, cosine_score), ...]
  poincare_results: [(id, poincare_distance), ...]
  query_busemann:   float (Busemann depth of query point)
  config:           PipelineConfig

1. Normalize scores to [0,1]:
   cosine_norm   = (score - min) / (max - min)
   poincare_norm = 1 - (dist - min) / (max - min)   # invert distance

2. Compute depth proximity weight per candidate:
   depth_proximity = exp(-|cand_busemann - query_busemann| * config.depth_weight)

3. Fused score:
   fused = config.fusion_alpha * cosine_norm
         + (1 - config.fusion_alpha) * poincare_norm * depth_proximity

4. Sort by fused score descending, return top-k
```

**What this captures:** Results that are both semantically similar (cosine) AND at a similar hierarchy depth (Busemann proximity) rank higher. A narrative-level query won't be polluted by content-level results even if they're semantically close.

### 6.2 Horosphere Scoring

New algorithm inspired by RuVector's Lorentz Cascade Attention. A horosphere is a level set of the Busemann function — a hyperbolic "floor" at a given depth.

**Algorithm:**

```
Inputs:
  candidates: [(id, fused_score, busemann_depth), ...]
  query_busemann: float
  intent: "ancestors" | "descendants" | "siblings" | "auto"
  config: PipelineConfig

For each candidate:
  depth_delta = candidate_busemann - query_busemann

  if intent == "ancestors":
    horo_weight = exp(-max(depth_delta, 0) * config.horo_steepness)

  elif intent == "descendants":
    horo_weight = exp(min(depth_delta, 0) * config.horo_steepness)

  elif intent == "siblings":
    horo_weight = exp(-|depth_delta| * config.horo_steepness)

  elif intent == "auto":
    horo_weight = exp(-|depth_delta| * config.horo_steepness)

  final_score = fused_score * horo_weight

Sort by final_score descending, return top-k
```

**Use cases:**
- "Find the parent narrative of this story" → intent="ancestors"
- "Find sub-stories of this narrative" → intent="descendants"
- "Find related stories at the same level" → intent="siblings"
- Default search → intent="auto" (siblings behavior)

### 6.3 Depth-Band Filtering (Server-Side)

Runs server-side using Qdrant's native payload filtering. No client-side post-processing.

```python
filter = {
    "must": [{
        "key": "busemann_depth",
        "range": {
            "gte": query_busemann - config.depth_band_width,
            "lte": query_busemann + config.depth_band_width
        }
    }]
}
```

Pre-filters candidates at the HNSW level. Combined with tangent HNSW (Path B):

```
Search "tangent" (Euclidean HNSW)
  + filter: busemann_depth in [query_depth +/- band_width]
  → Server returns only hierarchy-relevant candidates
  → Client does exact Poincare re-rank
```

`depth_band_width` is per-collection config. Set to `None` to disable.

### 6.4 Composition

```
Query-time results (Path A, B, or combined)
    |
    v
Depth-band filter (server-side, optional, coarse)
    |
    v
Busemann-weighted fusion (default blender)
    |
    v
Horosphere scoring (optional, intent-driven, fine)
    |
    v
Final top-k results
```

---

## 7. Testbench Additions

### 7.1 New Modules

```
dev/testbench/
  tangent.py          NEW — tangent space computation + centroid strategies
  pipeline_config.py  NEW — PipelineConfig dataclass + defaults
  comparator.py       NEW — A/B comparison framework
```

**tangent.py:**
- `compute_centroid(points, strategy, curvature)` — origin / frechet / einstein
- `compute_tangent_coords(points, centroid, curvature)` — batch log_map
- `tangent_query(query_poincare, centroid, curvature)` — single log_map for query

**pipeline_config.py:**
- `PipelineConfig` dataclass with all Layer 1/2/3 parameters
- Default configs: `PHASE5_BASELINE`, `ALPHA_PIPELINE`, `TANGENT_PIPELINE`, `COMBINED_PIPELINE`

**comparator.py:**
- `run_comparison(client, collection, queries, configs)` — runs queries through multiple configs
- `compare_report(results)` — side-by-side table, per-metric best performer

### 7.2 Modifications to Existing Modules

**collection_builder.py:**
- `create_unified_collection()`: add "tangent" named vector (Euclid, 128d)
- `populate_collection()`: accept tangent_coords, alpha, busemann arrays
- Store alpha and busemann_depth as payload fields
- Create payload index on busemann_depth (range)

**hyperbolic_math.py:**
- Add `log_map_origin(p, c)`: fast-path origin log_map (no Mobius ops)
- Add `tangent_distance_sq(u, v)`: plain Euclidean L2 squared

**queries.py:**
- `alpha_pipeline_query()`: replaces Klein stage 2 with alpha re-rank
- `tangent_pipeline_query()`: search tangent vector, re-rank with exact Poincare
- `combined_pipeline_query()`: merge cosine + tangent, alpha re-rank, exact Poincare

**fusion.py:**
- Enhanced `busemann_weighted_fuse()`: depth_proximity weighting
- New `horosphere_score()`: ancestor/descendant/sibling weighting

**projection.py:**
- `auto_curvature(gromov_delta)`: curvature from delta mapping

### 7.3 New Benchmark Suites

| # | Suite | What it tests | Key metrics |
|---|---|---|---|
| 12 | Alpha vs Klein | Alpha pipeline vs existing Klein pipeline | Recall@10, stage 2 latency, Spearman rank correlation vs exact |
| 13 | Tangent Centroid Comparison | Origin vs Frechet vs Einstein centroids | Recall@10 per centroid, rank correlation, centroid computation time |
| 14 | Tangent Prune Factor Sweep | prune_factor = 5, 10, 20 with best centroid | Recall@10 vs exact-distance compute count |
| 15 | Curvature Sweep | c = 0.25, 0.5, 1.0, 2.0, 5.0 | Drill-down recall, lateral recall, Gromov delta correlation |
| 16 | Fusion Comparison | Busemann vs RRF vs linear alpha vs horosphere | Drill-down, lateral, cross-branch recall |
| 17 | End-to-End Pipeline | Full best-config pipeline vs Phase 5 baseline | All query types, total latency, per-stage breakdown |

### 7.4 Suite Details

**Suite 12 — Alpha vs Klein (50 queries):**
- Stage 1: Shared cosine prefetch top-200
- Path A (Klein): poincare_to_klein → klein_chord_distance_sq → top-50 → exact Poincare → top-10
- Path B (Alpha): diff_sq * alpha_q * alpha_v → top-50 → exact Poincare → top-10
- Ground truth: brute-force exact Poincare over all points
- Report: Recall@10, stage 2 latency, Spearman rho of stage-2 ordering vs exact

**Suite 13 — Tangent Centroid Comparison (50 queries):**
- Upload tangent coords with 3 centroid strategies as separate named vectors
- For each: search Euclidean HNSW top-100, exact Poincare re-rank → top-10
- Ground truth: brute-force exact Poincare
- Report: Recall@10 per centroid, rank correlation, centroid computation time

**Suite 14 — Tangent Prune Factor Sweep (50 queries):**
- Best centroid from Suite 13, prune_factor = 5, 10, 20
- Report: Recall@10 vs exact-distance compute count (the trade-off curve)

**Suite 15 — Curvature Sweep (50 queries):**
- For each c in [0.25, 0.5, 1.0, 2.0, 5.0]: re-project, build collection, run queries
- Compute Gromov delta on raw PCA vectors (curvature-independent)
- Report: Recall per curvature, optimal c for each dataset's delta, sensitivity curve

**Suite 16 — Fusion Comparison (50 queries):**
- Same query-time results, different post-processing: RRF, linear alpha, Busemann, horosphere (all intents)
- Report: Per-query-type recall comparison

**Suite 17 — End-to-End Pipeline (all query types):**
- Best config from suites 12-16 vs Phase 5 baseline (Klein, c=5.0, RRF)
- Run drill-down, lateral, cross-branch, temporal, depth-band queries
- Report: Side-by-side comparison, per-query-type improvement/regression, total latency

### 7.5 Execution

```bash
python benchmark.py --dataset bgc --suites 12,13,14,15,16,17
python benchmark.py --dataset hwv --suites 12,13,14,15,16,17
```

Total: ~1,600 query evaluations per dataset. Results → `dev/testbench/results/benchmark_phase6_<timestamp>.json`

---

## 8. Mathematical Foundations (Verified)

All formulas verified against HyperspaceDB and RuVector implementations. Cross-referenced file paths and line numbers documented below.

### 8.1 Alpha Precomputation — Exact Ordering

```
Poincare distance: d(u,v) = (1/sqrt(c)) * acosh(1 + 2c * ||u-v||^2 / ((1-c*||u||^2)(1-c*||v||^2)))

Define: alpha_x = 1 / (1 - c * ||x||^2)

Then: d(u,v) = (1/sqrt(c)) * acosh(1 + 2c * ||u-v||^2 * alpha_u * alpha_v)

Since acosh is strictly monotonically increasing on [1, inf):
  comparing d(u,v) ⟺ comparing ||u-v||^2 * alpha_u * alpha_v
```

Verified in: HyperspaceDB vector.rs:33,83-96 | Qdrant hyperbolic_math.py:218-264 | Qdrant poincare_math.rs:59-73

### 8.2 Tangent Space — Euclidean Approximation

```
log_map at origin: log_0(p) = (2/sqrt(c)) * arctanh(sqrt(c)*||p||) * p/||p||
  ~3x cheaper than general log_map (no Mobius ops)

General log_map: log_p(y) = (2/(sqrt(c)*lambda_p)) * arctanh(sqrt(c)*||-p +_c y||) * (-p +_c y) / ||-p +_c y||
  where lambda_p = 2/(1 - c*||p||^2)  (conformal factor)
  and +_c is Mobius addition

Approximation quality: ||log_c(u) - log_c(v)||_E ≈ d_Poincare(u,v) + O(||u-c||^3 + ||v-c||^3)
```

Verified in: RuVector tangent.rs:23-272 | Qdrant poincare_math.rs:106-207 | RuVector poincare.rs:404-456

### 8.3 Busemann Function — Requires Lorentz

```
B_xi(x) = log(-<x, xi>_L)
  where xi is light-like: <xi, xi>_L = 0
  and <x,y>_L = -x_0*y_0 + x_1*y_1 + ... + x_n*y_n  (Minkowski metric)

Focal direction: xi = (1, -u_normalized) where u is unit spatial direction from centroid
  Light-like verification: <xi,xi>_L = -1*1 + u.u = -1 + 1 = 0

IMPORTANT: Requires Lorentz coordinates (d+1 dims). Must convert via poincare_to_lorentz() first.
```

Verified in: Qdrant hyperbolic_math.py:89-107 | RuVector lorentz_cascade.rs:32-92

### 8.4 Gromov Delta — Sampling-Based

```
For 4-point tuples (x,y,u,v):
  s1 = d(x,y) + d(u,v)
  s2 = d(x,u) + d(y,v)
  s3 = d(x,v) + d(y,u)
  delta = (max(s1,s2,s3) - second_max(s1,s2,s3)) / 2

Global delta = max over all sampled tuples
  1000 samples is standard (used in both Qdrant and HyperspaceDB)
```

Verified in: Qdrant hyperbolic_math.py:171-210 | HyperspaceDB gromov.rs:11-74

---

## 9. Future: SONA Integration Surface

All tunable parameters in PipelineConfig are designed to be replaced by learned values. The mapping:

| Parameter | Current source | Future source |
|---|---|---|
| curvature | Gromov delta threshold | SONA curvature optimizer / ReasoningBank |
| prune_factor | Fixed (benchmarked) | Experience Replay (from past query recall) |
| fusion_alpha | Fixed (benchmarked) | InfoNCE contrastive loss (per query type) |
| depth_weight | Fixed (benchmarked) | InfoNCE (depth-weighted contrastive) |
| horo_steepness | Fixed (benchmarked) | InfoNCE (directional preference learning) |
| projection_radii | Einstein spread heuristic | ReasoningBank hierarchy profiles |
| stage_sizes | Fixed (benchmarked) | Experience Replay optimization |
| depth_band_width | Fixed (benchmarked) | Adaptive from recall monitoring |

**No architectural changes needed** — SONA writes to the same PipelineConfig structure. The manual tuning phase (Phase 6) determines the shape of the parameter space; SONA automates navigation of it per collection.

**EWC++ (Elastic Weight Consolidation)** prevents catastrophic forgetting when collection data distributions shift over time (e.g., narrative tree deepening during a crisis).

---

## 10. Datasets and Baselines

### Datasets

| Dataset | Docs | Levels | Gromov delta | Notes |
|---|---|---|---|---|
| BGC | 92K | 4 | 0.057 | Primary benchmark, strongly tree-like |
| HWV | 11K | 6 | ~0.06 | Secondary, deeper hierarchy, smaller |

### Phase 5 Baselines (to beat or match)

| Metric | BGC Phase 5 | HWV Phase 5 |
|---|---|---|
| Drill-down recall | 1.0 | 1.0 |
| Quantization recall | 1.0 | 1.0 |
| Klein pre-filter recall | 0.827 | ~0.83 |
| InCone selectivity | 1.1% | ~1% |
| Depth-band precision | 100% | 100% |
| Gromov delta | 0.057 | ~0.06 |

### Success Criteria

- Alpha pipeline recall >= Klein pipeline recall (0.827) — expected since ordering is exact
- Tangent HNSW recall@10 >= 0.90 at prune_factor=10
- Busemann fusion drill-down recall >= RRF drill-down recall
- Curvature sweep identifies optimal c for each dataset
- End-to-end pipeline matches or exceeds Phase 5 baselines across all query types

---

## 11. File Inventory

### New Files

| File | Purpose | Estimated lines |
|---|---|---|
| `dev/testbench/tangent.py` | Tangent space computation + centroid strategies | ~120 |
| `dev/testbench/pipeline_config.py` | PipelineConfig dataclass + preset configs | ~80 |
| `dev/testbench/comparator.py` | A/B comparison framework | ~150 |

### Modified Files

| File | Changes |
|---|---|
| `lib/api/src/grpc/proto/collections.proto` | Add curvature field to VectorParams |
| Proto conversion layer | Map curvature to VectorDataConfig.curvature |
| `dev/testbench/collection_builder.py` | Add tangent vector, alpha/busemann payloads, busemann index |
| `dev/testbench/hyperbolic_math.py` | Add log_map_origin, tangent_distance_sq |
| `dev/testbench/queries.py` | Add alpha_pipeline_query, tangent_pipeline_query, combined_pipeline_query |
| `dev/testbench/fusion.py` | Enhance busemann_weighted_fuse, add horosphere_score |
| `dev/testbench/projection.py` | Add auto_curvature function |
| `dev/testbench/benchmark.py` | Add suites 12-17 |
