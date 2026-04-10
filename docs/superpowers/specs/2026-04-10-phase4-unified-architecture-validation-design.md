# Phase 4: Unified Architecture Validation — Design Spec

**Date:** 2026-04-10
**Branch:** `feat/hyperbolic-vector-support`
**Scope:** Qdrant fork fixes + testbench overhaul + HWV dataset + chatbot query benchmarks

---

## 1. Motivation

Phases 1-3.5 validated that hyperbolic geometry enables hierarchy-aware queries (90.9% child recall on BGC, 99.72% tier classification accuracy). The core finding is confirmed: Poincaré adds capabilities cosine cannot provide, and both coexist via named vectors.

Phase 4 bridges testbench validation to Pythia production architecture. The end goal is a per-client unified Qdrant collection (narratives + stories + content + profiles in one collection) that powers a chatbot capable of complex hierarchical queries. Before touching Pythia, we validate the full architecture in the testbench.

A thorough audit of the Qdrant fork revealed critical bugs (curvature dead code, missing gRPC support) that must be fixed, and identified patterns from RuVector and HyperspaceDB that belong on the client side (Pythia), not in the Qdrant fork.

### Design Principles

1. **Minimal Qdrant fork** — Only `Distance::Poincare` + curvature config + core math. Everything application-specific is Pythia-side. The fork must survive upstream Qdrant updates with trivial rebasing.
2. **Testbench validates before Pythia** — Every pattern (Einstein midpoint, Busemann scoring, geometric filters, unified collections) is proven in the testbench before carrying to Pythia.
3. **Chatbot-driven benchmarks** — Benchmark suites mirror real query patterns: drill-down, lateral exploration, cross-branch discovery, temporal evolution.
4. **Iterative delivery** — Each piece is independently valuable. We don't need everything to ship something useful.

---

## 2. Qdrant Fork: Surgical Fixes

### 2.1 Fix Curvature Plumbing (Bug Fix)

**Problem:** `new_raw_scorer_with_curvature()` in `raw_scorer.rs:123-156` exists but is never called. All Poincaré scoring hardcodes `DEFAULT_CURVATURE = 1.0` regardless of collection config. Our testbench c=5.0 results were actually computed at c=1.0 — hierarchy separation came from projection strategy, not the distance metric's curvature parameter.

**Fix:** Wire curvature through the scorer creation chain.

**Files:**
- `lib/segment/src/index/hnsw_index/point_scorer.rs` — `FilteredScorer::new()` must accept and pass curvature when distance is Poincaré
- `lib/segment/src/vector_storage/raw_scorer.rs` — When distance is `Distance::Poincare`, call `new_raw_scorer_with_curvature()` instead of `new_raw_scorer()`

**Current flow:**
```
HNSW search → FilteredScorer::new()
            → new_raw_scorer()
            → PoincareMetric::similarity()
            → DEFAULT_CURVATURE (1.0)  ← WRONG
```

**Fixed flow:**
```
HNSW search → FilteredScorer::new(curvature)
            → new_raw_scorer_with_curvature(curvature)
            → PoincareCurvatureQueryScorer
            → config curvature (e.g. 5.0)  ← CORRECT
```

**Scope:** ~20 lines changed.

### 2.2 Add Poincaré to gRPC Proto (Feature Gap)

**Problem:** `Distance::Poincare` is missing from the gRPC protobuf enum (`collections.proto:137-143`). REST API works by accident via serde string deserialization, but gRPC clients cannot create Poincaré collections.

**Fix:**
```protobuf
// lib/api/src/grpc/proto/collections.proto
enum Distance {
  UnknownDistance = 0;
  Cosine = 1;
  Euclid = 2;
  Dot = 3;
  Manhattan = 4;
  Poincare = 5;
}
```

Plus conversion mapping in `lib/api/src/grpc/conversions.rs` — add `Poincare ↔ Distance::Poincare` match arms in `from_grpc_dist()` and the reverse conversion.

**Scope:** ~10 lines.

### 2.3 Fix preprocess_vector Curvature (Bug Fix)

**Problem:** `preprocess_vector()` in `lib/segment/src/types.rs:349-351` calls `project_to_ball()` with `DEFAULT_CURVATURE`, ignoring the collection's configured curvature. Vectors are projected to the wrong ball radius at non-default curvatures.

**Fix:** Pass curvature from `VectorDataConfig::curvature()` into the preprocessing call.

**Scope:** ~5 lines.

### 2.4 Remove Busemann + Tangent Space (Cleanup)

**Rationale:** Busemann scoring is computed at embedding time by Pythia and stored as a payload. Tangent space pruning is a premature optimization. Both move to Pythia-side Python.

**Delete:**
- `lib/segment/src/spaces/hyperbolic/busemann.rs` (130 lines)
- `lib/segment/src/spaces/hyperbolic/tangent_cache.rs` (207 lines)
- `lib/segment/src/spaces/hyperbolic/tangent_scorer.rs` (219 lines)

**Update:** `lib/segment/src/spaces/hyperbolic/mod.rs` — remove re-exports for deleted modules.

**Also trim** `poincare_math.rs` — remove functions only used by tangent space operations. Keep all functions that `poincare_distance()`, `einstein_midpoint()`, or the Metric trait depend on: `poincare_distance()`, `project_to_ball()`, `exp_map_at_origin()`, `log_map_at_origin()`, `mobius_add()`, `einstein_midpoint()`, `poincare_to_lorentz()`, `lorentz_to_poincare()` (needed by Einstein midpoint), `stable_acosh()`, `fused_norms()`, and all their tests. Only tangent-space-specific helpers (used exclusively by the deleted tangent_cache/tangent_scorer) are removed.

**Net:** -556 lines removed.

### 2.5 Post-Fix Fork Inventory

After all changes, the hyperbolic module is:

```
lib/segment/src/spaces/hyperbolic/
├── mod.rs              (~10 lines)
├── poincare_math.rs    (~300 lines, trimmed)
└── poincare_metric.rs  (~200 lines, unchanged)
```

Wiring across the codebase:
- `types.rs` — `Distance::Poincare` variant + `curvature: Option<f32>` on `VectorDataConfig`
- `raw_scorer.rs` — curvature-aware dispatch for Poincaré (~20 lines)
- `async_raw_scorer.rs` — same pattern (~10 lines)
- `quantized_scorer_builder.rs` — Poincaré match arm (~5 lines)
- `collections.proto` + `conversions.rs` — gRPC support (~10 lines)
- Feature flag forwarding in Cargo.toml chain (segment → collection → shard → storage → root)

**Total fork size: ~550 lines across ~10 files.** All additive — no modifications to existing Qdrant logic except adding match arms.

---

## 3. Testbench: Unified Collection Architecture

### 3.1 Collection Configurations

For each dataset, create **3 collection variants** to compare:

#### Variant 1: `{dataset}_cosine` (baseline)
- Single dense vector: 1024d, Cosine
- All tiers mixed together
- Payload: `item_type`, `tier`, `domain`, `area`, `parent_ids`, `child_ids`
- No Poincaré, no hierarchy encoding

#### Variant 2: `{dataset}_unified` (target architecture)
- Named vectors:
  - `dense`: 1024d, Cosine (flat semantic search)
  - `poincare`: 128d, Poincare with c=5.0 (hierarchical search)
  - `sparse`: BGE-M3 inverted index (keyword search)
- All tiers in one collection, discriminated by payloads
- Payloads:
  - `item_type`: `"narrative"` | `"story"` | `"content"` (keyword, indexed)
  - `tier`: int 0/1/2 (integer, indexed)
  - `busemann_depth`: float (float, indexed, range-queryable)
  - `alpha`: float (precomputed `1/(1-||x||²)`, for client-side distance)
  - `parent_ids`: list[str] (keyword, indexed)
  - `child_ids`: list[str] (keyword, indexed)
  - `domain`: str (top-level category, keyword, indexed)
  - `area`: str (second-level category, keyword, indexed)
  - `created_at`: datetime (synthetic timestamps, datetime, indexed)
  - `point_id`: str (unique identifier)

#### Variant 3: `{dataset}_separate_{tier}` (current Pythia pattern)
- Separate collections per tier: `{dataset}_narratives`, `{dataset}_stories`, `{dataset}_content`
- Each with: dense 1024d Cosine + poincare 128d Poincare
- Same payloads minus `item_type` and `tier` (implicit from collection)

Variant 2 vs Variant 3 answers: **does unification degrade performance?**

### 3.2 Benchmark Suites

#### Suite 1: Gromov Delta Analysis
- **Purpose:** Quantify how "tree-like" each dataset is before benchmarking
- **Method:** Sample 1000 random 4-tuples, compute delta-hyperbolicity via the 4-point condition
- **Output:** `delta` (float), `recommendation` (lorentz/poincare/cosine/l2)
- **Run:** Once per dataset at setup time
- **Source pattern:** HyperspaceDB `gromov.rs`

#### Suite 2: Hierarchy Separation (upgraded)
- **Purpose:** Validate Poincaré tier separation in unified collection context
- **Metrics:**
  - `sep_ratio`: |avg_leaf_depth - avg_root_depth| / std(all_depths)
  - `cross_tier_sep`: gap between adjacent tiers
  - `tier_accuracy`: classify tier from Busemann depth alone
  - `band_overlap`: fraction of points outside their tier's depth band
  - NEW: `einstein_vs_frechet`: quality comparison of Einstein midpoint centroids vs iterative Fréchet mean (MSE of centroid-to-children distances)
  - NEW: `alpha_speedup`: wall-clock comparison of distance computation with/without precomputed alpha

#### Suite 3: Drill-Down Queries (chatbot pattern: vertical traversal)
- **Purpose:** Validate "start at narrative, find stories, then content" pattern
- **Method:**
  1. Select 100 random narrative points
  2. For each, search `item_type=story` + `parent_ids contains narrative_id` using poincare vector, top-10
  3. For each story result, search `item_type=content` + `parent_ids contains story_id` using poincare vector, top-10
- **Metrics:**
  - `hop1_recall@10`: fraction of true story children found in first hop
  - `hop2_recall@10`: fraction of true content children found in second hop
  - `drill_down_precision`: fraction of results that are true descendants
  - `drill_down_latency_ms`: end-to-end for the 2-hop chain
- **Compare across:** poincare vs cosine vs parent_id-filter-only (no vector search)
- **Run on:** unified and separate collections

#### Suite 4: Lateral Exploration (chatbot pattern: horizontal search)
- **Purpose:** "Find related items at the same hierarchy level"
- **Method:**
  1. Select 200 random query points (mix of tiers)
  2. Search with `tier == query_tier` filter + vector search, top-10
- **Metrics:**
  - `lateral_area_recall@10`: fraction of results in same domain/area as query
  - `lateral_domain_recall@10`: fraction in same top-level domain
  - `lateral_diversity`: number of distinct subtrees (areas) in results
- **Compare:** cosine-only vs poincare-only vs RRF fusion
- **Run on:** unified and separate collections

#### Suite 5: Cross-Branch Discovery (chatbot pattern: structural similarity)
- **Purpose:** Find narratives with similar hierarchical structure across different semantic domains
- **Method:**
  1. Select 50 narrative points
  2. For each, search `item_type=narrative` + `domain != source_domain` using poincare vector, top-10
  3. For each result, measure structural similarity: subtree depth, child count, depth distribution
- **Metrics:**
  - `structural_similarity`: correlation of depth distributions between query and result subtrees
  - `cross_branch_depth_match`: how similar are the subtree depths?
  - `semantic_distance`: cosine distance between query and results (should be high — different domains)
- **Compare:** poincare (captures structure) vs cosine (captures semantics). Poincaré should outperform on structural metrics while cosine should outperform on semantic metrics. This proves they're complementary.
- **Run on:** unified collection only

#### Suite 6: Depth-Band Filtering (upgraded)
- **Purpose:** Validate `busemann_depth` range queries as a search strategy
- **Method:**
  1. Define depth bands per tier from hierarchy separation data
  2. Search with `busemann_depth` range filter + poincare/cosine vector, top-10
  3. Also test without depth filter as baseline
- **Metrics:**
  - `depth_band_recall@10`: same-tier recall with depth band filtering
  - `filter_precision`: fraction of results in correct depth band
  - `filter_speedup`: latency reduction vs unfiltered search
- **Run on:** unified and separate collections

#### Suite 7: Unified vs Separate Overhead
- **Purpose:** Directly compare unified collection performance against separate per-tier collections
- **Method:** Run Suites 3-6 on both `{dataset}_unified` and `{dataset}_separate_{tier}` collections
- **Metrics:**
  - `recall_delta`: recall(unified) - recall(separate) for each suite
  - `latency_delta`: latency(unified) - latency(separate)
  - `qps_ratio`: QPS(unified) / QPS(separate)
- **Expected:** Small overhead for unified (shared HNSW graph is larger, but payload filtering compensates)

#### Suite 8: Latency
- **Purpose:** Performance profiling
- **Method:** 1000 sequential queries at ef = {64, 128, 256}
- **Metrics:** QPS, p50_ms, p95_ms, p99_ms
- **Run on:** all three collection variants
- **Compare:** cosine search vs poincare search vs hybrid (both vectors)

---

## 4. Datasets

### 4.1 BGC — Blurb Genre Collection (existing, reinterpreted)

- **Source:** Already embedded in `dev/testbench/data/bgc/`
- **Size:** 91,894 book blurbs, 4-level genre hierarchy, 146 labels
- **Tier mapping:**
  - L0 (4 roots) → Themes (deferred, optional)
  - L1 (46 genres) → Narratives
  - L2 (77 sub-genres) → Stories
  - L3/leaf docs → Content
- **Aggregate points:** Create synthetic narrative/story points via Einstein midpoint of their children's embeddings. ~123 aggregate points (46 narratives + 77 stories).
- **Total unified collection:** ~92,017 points across 3 tiers
- **Ground truth:** Deterministic parent-child from genre tree
- **Temporal data:** Synthetic `created_at` spread across 30-day window. Content gets random dates; stories get median of children; narratives get earliest of children.

### 4.2 HWV — Hierarchical WikiVitals (new)

- **Source:** https://github.com/RomanPlaud/revisitingHTC
- **License:** MIT
- **Size:** 10,013 Wikipedia article abstracts, 1,186 category nodes, 6 levels
- **Key properties:**
  - Variable depth: paths range from 2 to 6 levels
  - Extreme class imbalance: many categories have <10 examples
  - Single-path leaf (SPL): one leaf per document
- **Tier mapping:**
  - Level 0 (11 nodes) → Themes
  - Level 1 (109 nodes) → Narratives
  - Level 2 (381 nodes) → Stories
  - Levels 3-5 (685 nodes) → Content (at varying depths)
  - Leaf documents → Content (at their actual depth)
- **Aggregate points:** Einstein midpoint for each internal node. ~1,186 aggregate points.
- **Total unified collection:** ~11,199 points across 4-6 tiers
- **Ground truth:** Full tree structure with parent-child edges
- **Temporal data:** Same synthetic approach as BGC

### 4.3 Embedding Pipeline

For both datasets:

1. **Text embedding:** pplx-embed-v1 (1024d, GPU) — existing for BGC, new for HWV
2. **PCA reduction:** 1024d → 128d — existing pipeline
3. **Poincaré projection:** Einstein midpoint + tier-aware scaling (static strategy with per-tier radii calibrated from data)
4. **Aggregate computation:** Einstein midpoint of children for each internal tree node
5. **Busemann depth:** Computed per-point Python-side via Lorentz model
6. **Alpha precomputation:** `1/(1 - c * ||x||²)` stored as payload
7. **Gromov delta:** Computed once per dataset at setup
8. **Synthetic timestamps:** Generated per the rules above

### 4.4 Data Prep: New File `dev/datasets/hwv/prepare.py`

Following the pattern of `dev/datasets/bgc/prepare.py`:
1. Download HWV from GitHub
2. Parse hierarchy tree and document-to-leaf assignments
3. Build parent-child adjacency maps
4. Output: `documents.jsonl` with fields: `id`, `text`, `labels`, `label_path`, `depth`, `parent_ids`, `child_ids`, `domain`, `area`

---

## 5. Testbench Code Structure

### 5.1 File Layout

```
dev/testbench/
├── benchmark.py          — REWRITE: 8-suite benchmark engine
├── embed.py              — EXTEND: HWV pipeline, Einstein midpoint aggregation
├── projection.py         — KEEP: existing 4 projection strategies
├── hyperbolic_math.py    — EXTEND: Gromov delta, alpha precomputation,
│                           Busemann scoring (moved from Rust), fused norms,
│                           Einstein midpoint (for aggregate points)
├── fusion.py             — KEEP: existing fusion strategies
├── download_dataset.py   — EXTEND: add HWV download
├── collections.py        — NEW: unified/separate/cosine collection creation
│                           with named vectors, payload schemas, parent/child wiring
├── queries.py            — NEW: chatbot query implementations
│                           (drill_down, lateral, cross_branch, temporal, depth_band)
├── run.sh                — UPDATE: add hwv subcommands
├── Dockerfile            — KEEP (already builds with --features hyperbolic)
├── docker-compose.yml    — KEEP
├── requirements.txt      — KEEP
└── results/              — benchmark outputs (JSON + analysis markdown)
```

### 5.2 New File: `collections.py`

Responsibilities:
- Create all 3 collection variants per dataset with proper vector configs
- Define payload index schemas
- Populate collections from embedded data
- Build parent/child linkage from hierarchy tree
- Compute and store alpha, Busemann depth, synthetic timestamps

### 5.3 New File: `queries.py`

Implements the 4 chatbot query patterns as reusable functions:

```python
def drill_down_query(client, collection, start_id, target_tier, using="poincare"):
    """Multi-hop: narrative → stories → content. Returns results at each hop."""

def lateral_query(client, collection, query_id, using="dense"):
    """Same-tier search: find related items at query's tier level."""

def cross_branch_query(client, collection, narrative_id, exclude_domain, using="poincare"):
    """Cross-domain: find structurally similar narratives in other domains."""

def temporal_query(client, collection, query_vector, date_range, tier, using="dense"):
    """Time-windowed: search within tier filtered by date range."""

def depth_band_query(client, collection, query_vector, depth_min, depth_max, using="poincare"):
    """Depth-filtered: search within Busemann depth band."""
```

### 5.4 Extended: `hyperbolic_math.py`

New functions (moved from Qdrant Rust or adopted from RuVector/HyperspaceDB):

```python
def gromov_delta(vectors, num_samples=1000):
    """4-point condition delta-hyperbolicity. Returns (delta, recommendation)."""

def alpha_precompute(vector, curvature=1.0):
    """Precompute 1/(1 - c*||x||²) for fast distance."""

def busemann_score(x_lorentz, focal_direction):
    """Busemann function: log(-<x, xi>_L). Moved from Qdrant Rust."""

def einstein_midpoint(points, curvature=1.0):
    """Closed-form hyperbolic centroid. O(1) vs 50+ iteration Fréchet mean."""

def fused_norms(u, v):
    """Single-pass ||u-v||², ||u||², ||v||². 30% speedup."""

def poincare_distance_with_alpha(diff_sq, alpha_u, alpha_v, curvature=1.0):
    """Distance from precomputed norms and alphas. Avoids redundant computation."""
```

### 5.5 Extended: `embed.py`

New in the embedding pipeline:
- `compute_aggregate_embeddings()` — Einstein midpoint for internal tree nodes
- `compute_busemann_depths()` — Busemann via Lorentz model (Python-side)
- `compute_alpha_values()` — Alpha precomputation for all points
- `generate_synthetic_timestamps()` — created_at for temporal benchmarks
- `run_gromov_analysis()` — Delta-hyperbolicity at setup time

### 5.6 Updated: `benchmark.py`

Complete rewrite organized as:

```python
class BenchmarkRunner:
    def __init__(self, client, dataset, collections):
        ...

    def run_all(self):
        self.suite_gromov_delta()
        self.suite_hierarchy_separation()
        self.suite_drill_down()
        self.suite_lateral()
        self.suite_cross_branch()
        self.suite_depth_band()
        self.suite_unified_vs_separate()
        self.suite_latency()

    def suite_gromov_delta(self): ...
    def suite_hierarchy_separation(self): ...
    def suite_drill_down(self): ...
    def suite_lateral(self): ...
    def suite_cross_branch(self): ...
    def suite_depth_band(self): ...
    def suite_unified_vs_separate(self): ...
    def suite_latency(self): ...
```

Each suite writes results to a structured JSON, keyed by `{dataset}.{collection_variant}.{suite_name}`.

---

## 6. Execution Order

### Step 1: Qdrant Fork Fixes
1. Fix curvature plumbing (wire scorer chain)
2. Add Poincare to gRPC proto + conversions
3. Fix preprocess_vector curvature
4. Remove Busemann + tangent from Rust
5. Run existing 740 Rust tests to verify no regressions
6. Rebuild Docker image

### Step 2: Testbench Math Extensions
1. Add Gromov delta to `hyperbolic_math.py`
2. Add alpha precomputation
3. Add fused norms
4. Move Busemann scoring from Rust to Python
5. Verify Einstein midpoint (already exists, validate against Fréchet)

### Step 3: HWV Dataset Preparation
1. Create `dev/datasets/hwv/prepare.py`
2. Download and parse HWV
3. Embed with pplx-embed-v1
4. Build hierarchy tree with parent/child maps

### Step 4: Collection Architecture
1. Write `collections.py` — unified/separate/cosine creation
2. Populate BGC collections (all 3 variants)
3. Populate HWV collections (all 3 variants)
4. Verify payload indices and named vector configs

### Step 5: Chatbot Query Implementation
1. Write `queries.py` — 5 query functions
2. Unit test each query pattern against known BGC ground truth

### Step 6: Benchmark Suites
1. Rewrite `benchmark.py` with 8 suites
2. Run on BGC first (fast iteration, known baseline)
3. Run on HWV (variable-depth validation)
4. Generate analysis reports

### Step 7: Analysis
1. Compare unified vs separate collection performance
2. Build depth-scaling curve (BGC 4-level vs HWV 6-level, with WOS 2-level and EurLex 3-level from prior rounds)
3. Identify which chatbot query patterns benefit most from Poincaré
4. Document findings for Pythia integration planning

---

## 7. Success Criteria

- [ ] Qdrant fork: 740 existing tests pass with curvature fix applied
- [ ] Qdrant fork: gRPC client can create Poincaré collections
- [ ] Gromov delta: BGC delta < 0.3 (confirms hyperbolic is appropriate)
- [ ] Drill-down recall@10 > 80% on BGC (matching prior child_recall of 90.9%)
- [ ] Unified vs separate: recall delta < 5% (unification doesn't significantly hurt)
- [ ] Unified vs separate: latency delta < 20% (acceptable overhead)
- [ ] Cross-branch: Poincaré outperforms cosine on structural similarity metrics
- [ ] HWV: variable-depth drill-down handles missing tiers gracefully
- [ ] Depth-scaling curve: monotonic or stable trend from 2 to 6 levels
- [ ] Einstein midpoint: quality within 5% of Fréchet mean at 100x speed

---

## 8. Non-Goals (Deferred)

- Klein chord routing in Qdrant HNSW (Phase LATER — needs HNSW layer-aware metric dispatch)
- SIMD Poincaré distance (Phase LATER — performance, not correctness)
- Anisotropic quantization (Phase LATER — test standard INT8 first)
- SONA integration (Phase NEXT — needs unified collections working first)
- Pythia code changes (Phase NEXT — testbench validates architecture first)
- SciHTC dataset (deferred — ACM access issues)
- PubMed/MeSH 13-level dataset (deferred — massive scale, needs infrastructure)
- Themes tier (deferred — Content→Stories→Narratives is the focus)
