# Hyperbolic Vector Support for Qdrant

**Date:** 2026-04-07
**Status:** Design approved, pending implementation
**Scope:** Qdrant fork with native Poincare distance, tangent space pruning, and Busemann scoring
**Reference implementations:** ruvector (Rust hyperbolic vector DB), PageIndex (hierarchical retrieval), Pythia (intelligence platform consumer)

---

## 1. Problem Statement

Qdrant operates exclusively in Euclidean/cosine space. Its four distance metrics (Cosine, Euclid, Dot, Manhattan) treat all vectors as flat peers with no notion of hierarchy. This creates a gap for applications that need hierarchy-aware retrieval — where data has a natural tree structure (e.g., Content > Stories > Narratives > Themes) and queries need to retrieve at specific abstraction levels.

### 1.1 The Pythia Use Case

Pythia is a social media intelligence platform with a 4-tier information hierarchy:

| Tier | Definition | Stability | Volume |
|------|-----------|-----------|--------|
| **Content** | Individual posts | Hours–days | Millions |
| **Stories** | Event-specific claims with concrete actors | Days–weeks | Hundreds/day |
| **Narratives** | Strategic framings across multiple stories | Weeks–months | Tens/day |
| **Themes** | Persistent ideological beliefs | Years | 5–15 per environment |

Currently, Pythia uses three databases:
- **MongoDB**: Source of truth (full documents)
- **Qdrant**: Semantic search (1024d cosine, flat space)
- **Neo4j**: Hierarchies and relationships (BELONGS_TO edges, evolution tracking)

The gap: hierarchical queries like "find all event-level stories under this narrative" require Qdrant (vector search) + Neo4j (graph traversal) + MongoDB (data fetch) round-trips. Qdrant can't express hierarchy in its vector space.

### 1.2 Why Hyperbolic Geometry

Hyperbolic space naturally encodes tree-like hierarchies. A tree with branching factor `b` and depth `d` requires `O(b^d)` dimensions for distortion-free Euclidean embedding but only `O(d)` dimensions in hyperbolic space. The Poincare ball model maps abstract concepts (narratives) near the origin and concrete instances (posts) near the boundary, with geodesic distance reflecting both semantic similarity and hierarchical relationship.

Busemann scoring — a lightweight O(d) dot-product-based function — measures "depth" in the hierarchy relative to a focal direction, enabling queries like "find vectors near X at abstraction level Y."

### 1.3 Euclidean Embeddings Are Not a Blocker

Standard embedding models (BGE-M3, pplx-embed-v1) are trained in Euclidean space. Post-hoc projection to hyperbolic space via exponential map at origin is well-established (Nickel & Kiela, 2017) and already implemented in both ruvector and Pythia's narratives pipeline. The tangent space at the Poincare ball origin IS Euclidean space, so the projection preserves relative distances for nearby points. Dual-space search (cosine + Poincare with RRF fusion) provides a safety net against projection artifacts.

---

## 2. Design Overview

### 2.1 Approach: Smart Integration (Approach 2)

Four implementation layers in Qdrant's `lib/segment` crate, plus Pythia-side collection and pipeline changes:

| Layer | What | Complexity | Dependency |
|-------|------|-----------|------------|
| **A** | `Distance::Poincare` metric + curvature config | ~500 lines | None |
| **B** | Tangent cache + pruning scorer (5-10x speedup) | ~800 lines | Layer A |
| **C** | Query-time Busemann scoring (hierarchy depth) | ~400 lines | Layer A |
| **D** | Server-side projection config (deferred) | ~300 lines | Layer A |

### 2.2 Architecture

All new code lives in a new `spaces/hyperbolic/` module. Existing files receive only match-arm additions (~90 lines total across existing files).

```
lib/segment/src/
  spaces/
    metric.rs                    # Existing Metric<T> trait (UNCHANGED)
    simple.rs                    # Existing metrics (UNCHANGED)
    hyperbolic/                  # NEW MODULE
      mod.rs                     # Public exports
      poincare_metric.rs         # Layer A: PoincareMetric impl
      poincare_math.rs           # Poincare ball operations
      tangent_cache.rs           # Layer B: TangentCache + Frechet mean
      tangent_scorer.rs          # Layer B: Tangent-pruned scorer wrapper
      busemann.rs                # Layer C: Lorentz conversion + Busemann scoring
      projection.rs              # Layer D: PCA + exp_map (deferred)

  types.rs                       # MODIFIED: Distance::Poincare, curvature config
  vector_storage/
    raw_scorer.rs                # MODIFIED: add Poincare match arms
    quantized/
      quantized_scorer_builder.rs  # MODIFIED: add Poincare match arms
  index/
    hnsw_index/
      hnsw.rs                    # MODIFIED: integrate tangent scorer (Layer B)
```

### 2.3 What Doesn't Change

- `GraphLayers` / `GraphLayersBuilder` — HNSW graph construction is metric-agnostic
- `VectorStorage` / `DenseVectorStorage` — vectors are still `Vec<f32>`
- `FilteredScorer` — still wraps `RawScorer`, unchanged interface
- All existing distance metrics — no modifications to Cosine/Euclid/Dot/Manhattan

---

## 3. Layer A: Poincare Distance Metric

### 3.1 Distance Enum Extension

```rust
pub enum Distance {
    Cosine,
    Euclid,
    Dot,
    Manhattan,
    Poincare,    // NEW
}
```

| Property | Value | Rationale |
|----------|-------|-----------|
| `distance_order()` | `SmallBetter` | Like Euclid — closer = more similar |
| `postprocess_score()` | Scale by `1/sqrt(c)` | Adjusts default c=1.0 distance to actual curvature |
| `preprocess_vector()` | Project to ball | Ensures `\|\|x\|\| < 1/sqrt(c) - eps` |

### 3.2 PoincareMetric Implementation

```rust
pub struct PoincareMetric;

impl Metric<VectorElementType> for PoincareMetric {
    fn distance() -> Distance { Distance::Poincare }

    fn similarity(v1: &[f32], v2: &[f32]) -> ScoreType {
        // Negative Poincare distance (matches Qdrant convention: higher = better)
        -poincare_distance(v1, v2, DEFAULT_CURVATURE)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        project_to_ball(vector, DEFAULT_CURVATURE)
    }
}
```

Implements `Metric<f32>`, `Metric<f16>`, `Metric<u8>` for all element types.

### 3.3 Core Math (poincare_math.rs)

Ported from ruvector's `poincare.rs`:

**Poincare distance:**
```
d(u,v) = (1/sqrt(c)) * acosh(1 + 2c||u-v||^2 / ((1 - c||u||^2)(1 - c||v||^2)))
```
- Fused single-pass norm computation (`||u-v||^2`, `||u||^2`, `||v||^2` in one loop)
- Stable `acosh` with Taylor expansion for arguments near 1

**Ball projection:**
```
project_to_ball(x, c) -> x * min(1, (1/sqrt(c) - eps) / ||x||)
```
- Automatic on both insert and query (per ruvector pattern)
- Caller never needs to ensure ball constraint

**Exponential map at origin:**
```
exp_0(v) = tanh(sqrt(c) * ||v|| / 2) * v / ||v||
```
- Maps Euclidean tangent vector to Poincare ball point

**Logarithmic map at origin:**
```
log_0(p) = arctanh(sqrt(c) * ||p||) * p / (sqrt(c) * ||p||)
```
- Maps Poincare ball point back to tangent space

**Frechet mean:**
- Hyperbolic centroid via Riemannian gradient descent
- Uses exp/log maps, learning rate 0.1, max 100 iterations, tolerance 1e-6

**Numerical stability:**
- EPS = 1e-5 throughout
- Zero vector -> origin (valid Poincare point)
- Large norms -> tanh saturates, project_to_ball clamps

### 3.4 Curvature Handling

Curvature is per-collection, but `Metric<T>::similarity()` is a static method. Resolution:
- `PoincareMetric` uses `DEFAULT_CURVATURE = 1.0` internally
- Actual curvature stored in `VectorDataConfig`
- Postprocessing applies `1/sqrt(c)` scaling factor (exact, not approximate)
- This keeps the `Metric<T>` trait unchanged

### 3.5 Configuration

```rust
// In types.rs, VectorDataConfig extension
pub struct VectorDataConfig {
    // ... existing fields ...
    pub curvature: Option<f32>,  // Poincare ball curvature (default: 1.0)
}
```

### 3.6 Dispatch Sites

Add `Distance::Poincare => PoincareMetric` match arms in:
- `raw_scorer.rs` — `raw_scorer_impl()` (~6 match arms)
- `quantized_scorer_builder.rs` — `build()` (~6 match arms)
- `types.rs` — `postprocess_score()`, `preprocess_vector()`, `distance_order()`

Total: ~90 lines across existing files.

---

## 4. Layer B: Tangent Space Pruning

### 4.1 Problem

Poincare distance involves `acosh`, division, and square root — roughly 3-5x more expensive per pair than cosine. At scale, this makes hyperbolic HNSW search unacceptably slow.

### 4.2 Solution

At any point `p` on the Poincare ball, the tangent space `T_pM` is flat Euclidean space. For points near `p`, Euclidean distance in tangent space closely approximates geodesic distance.

Pre-compute `log_p(x)` for all stored vectors at segment build time, then use cheap Euclidean distance to prune candidates before computing exact Poincare distance on survivors.

### 4.3 TangentCache

```rust
pub struct TangentCache {
    centroid: Vec<f32>,              // Frechet mean of all vectors in segment
    tangent_coords: Vec<Vec<f32>>,   // Pre-computed log_centroid(x) per vector
    conformal_factor: f32,           // Cached conformal factor at centroid
    curvature: f32,
}
```

**Built at:** Segment optimization time (same phase as HNSW graph construction).

**Persisted:** Alongside HNSW graph. Rebuilt on segment merge.

**Memory cost:** `4 bytes * dim * num_vectors` per segment. For 1M vectors at 128d: ~512MB. Acceptable because hyperbolic collections use lower dimensionality than cosine collections.

### 4.4 TangentScorer

```rust
pub struct TangentScorer<'a> {
    tangent_cache: &'a TangentCache,
    query_tangent: Vec<f32>,          // Pre-computed log_centroid(query)
    exact_scorer: Box<dyn RawScorer + 'a>,
    prune_factor: usize,              // Default: 10
}

impl RawScorer for TangentScorer<'_> {
    fn score_points(&self, points: &[PointOffsetType], scores: &mut [ScoredPointOffset]) {
        // 1. Score ALL candidates with cheap tangent distance — O(d) Euclidean
        // 2. Keep top (k * prune_factor) by tangent score
        // 3. Re-score survivors with exact Poincare distance — O(d) acosh
        // 4. Return top k by exact score
    }
}
```

**Integration:** Injected in `hnsw.rs` when constructing search scorer for Poincare collections with tangent cache available. Zero overhead for non-Poincare collections.

**Expected speedup:** 5-10x (validated in ruvector benchmarks with prune_factor=10).

**Configuration:** `prune_factor` configurable per collection in `HnswConfig`, default 10.

---

## 5. Layer C: Busemann Scoring

### 5.1 What It Does

Given a focal direction xi (point at infinity representing hierarchy root), the Busemann function measures each point's "depth" in the hierarchy:
- Low Busemann depth -> abstract, near hierarchy root (narratives)
- High Busemann depth -> concrete, near leaves (posts)

Computation is O(d) — a single dot product + log in the Lorentz model.

### 5.2 Implementation

```rust
// spaces/hyperbolic/busemann.rs

/// Convert Poincare ball point to Lorentz hyperboloid
pub fn poincare_to_lorentz(p: &[f32], c: f32) -> Vec<f32> { ... }

/// Compute focal direction from a set of points
pub fn compute_focal_direction(points: &[&[f32]], c: f32) -> Vec<f32> { ... }

/// Busemann score: B_xi(x) = log(-<x, xi>_L)
pub fn busemann_score(x_lorentz: &[f32], focal: &[f32]) -> f32 {
    let inner = lorentz_inner(x_lorentz, focal);
    (-inner).max(EPS).ln()
}
```

### 5.3 Integration

Busemann is NOT a distance metric — it's a single-point function relative to a focal direction. It's computed at **query time** (not ingestion), as a post-scoring enrichment on the candidate set:

```
HNSW search (Poincare distance) -> top-k candidates
  -> BusemannScorer enriches each result with busemann_depth
  -> Client receives: { id, poincare_distance, busemann_depth }
```

The focal direction is provided per-query by the client, not stored in collection config. This allows different entity_keys/topics to have different hierarchy roots.

### 5.4 Query API

```json
{
  "vector": { "name": "hyperbolic", "vector": [0.1, 0.2, "..."] },
  "params": {
    "busemann_focal": [0.1, 0.2, "..."],
    "busemann_depth_range": [0.3, 0.7]
  }
}
```

Each result returns both `poincare_distance` and `busemann_depth`. Client-side fusion and filtering.

---

## 6. Collection Architecture

### 6.1 Per-Client Collections

Instead of multi-tenant collections, each client/region gets its own unified collection containing posts + stories + narratives:

```
uk_hyperbolic         — UK posts + stories + narratives
iraq_hyperbolic       — Iraq posts + stories + narratives
syria_hyperbolic      — Syria posts + stories + narratives
sudan_hyperbolic      — Sudan posts + stories + narratives
c40_hyperbolic        — C40 global posts + stories + narratives
```

**Rationale:**
- Hierarchies are client-specific (UK narratives unrelated to Iraq's)
- Frechet mean / tangent cache centroid is meaningful per-client
- Busemann focal direction is hierarchy-specific
- Curvature can be tuned per client
- Full `m=16` HNSW graphs (no multi-tenancy overhead)
- Clean data lifecycle (archive/delete per client independently)
- No `data_source`/`client_ids` payload filtering needed

### 6.2 Collection Schema

```python
vectors = {
    "dense": VectorParams(
        size=1024,
        distance=Distance.COSINE,
        quantization_config=ScalarQuantization(
            type=ScalarType.INT8, quantile=0.99, always_ram=True
        ),
        on_disk=True,
    ),
    "hyperbolic": VectorParams(
        size=128,                    # PCA-reduced, configurable
        distance=Distance.POINCARE,  # New in fork
        quantization_config=ScalarQuantization(
            type=ScalarType.INT8, quantile=0.99, always_ram=True
        ),
        on_disk=True,
        # curvature=1.0 (fork-specific config)
    ),
}

sparse_vectors = {
    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
}

hnsw_config = HnswConfigDiff(
    m=16,               # Full global graph (no payload_m multi-tenancy)
    ef_construct=200,
    on_disk=True,
)
```

### 6.3 Payload Schema

```python
{
    # Tier identity
    "tier": "post" | "story" | "narrative",
    "source_id": str,          # post_id, story_id, or narrative_id
    "source_collection": str,  # origin collection for traceability

    # Temporal
    "date": str,               # ISO date
    "created_at": str,         # ISO timestamp

    # Content
    "text_preview": str,       # 280 chars

    # Hierarchy linkage
    "story_ids": list[str],
    "narrative_ids": list[str],

    # Metrics (tier-dependent, optional)
    "engagement_total": int,
    "coherence_score": float,
    "quality_tier": str,
    "trajectory": str,
    "evolution_type": str,

    # Platform/network (posts only)
    "platform": str,
    "username": str,
    "network_id": str,
    "entity_keys": list[str],  # global pipeline
}
```

### 6.4 Payload Indexes

```python
"tier":           KeywordIndex
"date":           DatetimeIndex
"story_ids":      KeywordIndex
"narrative_ids":  KeywordIndex
"platform":       KeywordIndex
"entity_keys":    KeywordIndex
```

### 6.5 Data Volumes (Projected 1 Year)

| Collection | Points/year | Hyperbolic storage (128d + tangent) | Total with dense+sparse |
|------------|------------|-------------------------------------|------------------------|
| sudan_hyperbolic | ~2.2M | ~1.1GB | ~10GB |
| uk_hyperbolic | ~1.8M | ~0.9GB | ~8GB |
| syria_hyperbolic | ~1.1M | ~0.6GB | ~5GB |
| iraq_hyperbolic | ~550K | ~0.3GB | ~2.5GB |

### 6.6 Relationship to Existing Collections

Existing collections (`networks_posts`, `global_stories`, etc.) remain untouched. The new hyperbolic collections are additive. Both can coexist — existing pipelines continue using the original collections, new hierarchical queries use the hyperbolic collections.

---

## 7. Pythia-Side Changes

### 7.1 Hyperbolic Projector

New module: `pythia/embedding/services/hyperbolic_projector.py`

```python
class HyperbolicProjector:
    def __init__(self, pca_matrix: np.ndarray, target_dim: int = 128, curvature: float = 1.0):
        self.pca_matrix = pca_matrix    # (1024, 128)
        self.target_dim = target_dim
        self.curvature = curvature

    def project(self, dense_vectors: np.ndarray) -> np.ndarray:
        # 1. PCA reduce: (batch, 1024) @ (1024, 128) -> (batch, 128)
        # 2. L2 normalize
        # 3. Exponential map at origin -> Poincare ball
        ...

    @classmethod
    def fit_pca(cls, sample_vectors: np.ndarray, target_dim: int = 128) -> np.ndarray:
        # Fit PCA from stratified sample (posts + stories + narratives)
        # Store as models/hyperbolic_pca_128d.npy
        ...
```

### 7.2 PCA Matrix Lifecycle

- Fit once from a stratified sample (~10K posts + all available stories/narratives)
- Store as numpy artifact: `models/hyperbolic_pca_128d.npy`
- Refit periodically (monthly) or when embedding model changes
- Version tracked in collection payload

### 7.3 Embedding Pipeline Update

After dense + sparse encoding, add:
```
Text -> pplx-embed-v1 (1024d dense) + BGE-M3 (sparse)
  -> PCA reduce to 128d -> L2 normalize -> exp_map_origin -> Poincare vector
  -> Upsert to {client}_hyperbolic (dense + sparse + hyperbolic vectors)
```

### 7.4 Hierarchical Search Pattern

PageIndex-inspired coarse-to-fine navigation in a single Qdrant query:

```json
{
  "prefetch": [
    {
      "query": { "name": "hyperbolic", "vector": ["..."] },
      "params": {
        "busemann_focal": ["..."],
        "busemann_depth_range": [0.0, 0.3]
      },
      "limit": 10,
      "using": "hyperbolic"
    },
    {
      "query": { "name": "hyperbolic", "vector": ["..."] },
      "params": {
        "busemann_depth_range": [0.3, 0.7]
      },
      "limit": 50,
      "using": "hyperbolic"
    },
    {
      "query": { "name": "dense", "vector": ["..."] },
      "limit": 50,
      "using": "dense"
    }
  ],
  "query": { "fusion": "rrf" },
  "limit": 20
}
```

Replaces: Qdrant cosine search -> Neo4j BELONGS_TO traversal -> MongoDB fetch.

---

## 8. Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| INT8 quantization at ball boundary | Medium | Tangent pruning bypasses it; can disable quant for hyperbolic vector |
| PCA stability across tiers | Low-Medium | Stratified training sample; already validated in Pythia narratives |
| Focal direction per entity_key | Low | Query-time Busemann; client manages focal directions |
| HNSW graph quality in hyperbolic space | Low | Validated in ruvector; higher ef_construct as insurance |
| Post-dominated HNSW graph | Medium | Brute-force fallback for small tiers; tier payload filtering |
| Dual write consistency | Low | Hyperbolic collections are rebuild-able from existing data |
| Curvature tuning | Low | Configurable per collection; diagnostic tooling planned |
| exp_map numerical edge cases | Low | Epsilon clamping; proven in ruvector + Pythia |

No showstoppers identified. The two medium risks (quantization, post-dominated graph) are testable early and have clear fallback paths.

---

## 9. Testing Strategy

### 9.1 Mathematical Correctness (Unit Tests)

Port ruvector's property tests into `lib/segment/src/spaces/hyperbolic/tests/`:
- Distance: symmetry, identity, non-negativity, triangle inequality
- Mobius operations: right identity, inverse
- Exp/log maps: round-trip inverses
- Ball projection: all outputs inside ball
- Numerical stability: near-boundary ops, zero vectors

### 9.2 Integration Tests

In `lib/segment/tests/`:
- Raw scorer returns correct Poincare distances
- Quantized scorer approximate agreement with f32
- Tangent scorer recall@10 vs exact brute force
- HNSW insert + search: recall@10 > 0.95
- Collection creation with Poincare distance
- Mixed-tier insert + tier-filtered search
- Dual-space search with RRF (cosine + Poincare)
- Busemann depth range queries

### 9.3 Validation on Pythia Data (Acceptance Tests)

- **Hierarchy separation:** Embed posts + stories + narratives, verify Busemann depth separates tiers with 90%+ accuracy
- **Retrieval quality:** Compare Poincare search recall@10 vs Neo4j BELONGS_TO ground truth (target > 0.7)
- **Dual-space fusion:** Verify RRF(cosine, Poincare) >= max(cosine-only, Poincare-only)
- **Performance:** 1.5M vectors at 128d, search p95 < 50ms at ef=128
- **Quantization boundary:** Measure distance error (INT8 vs f32) as function of ball radius

---

## 10. Fork Management

### 10.1 Branch Structure

One commit per layer, each self-contained and squashable:

```
upstream/master ──────────────────────────────────────►
                    \                         \
                     \ rebase                  \ rebase
                      v                         v
fork/master ──[A]──[B]──[C]──────────────[A']──[B']──[C']──►
```

### 10.2 Feature Gate

All changes behind a Cargo feature flag:

```toml
[features]
default = []
hyperbolic = []
```

Match arms and module imports gated with `#[cfg(feature = "hyperbolic")]`. Zero runtime cost when disabled.

### 10.3 Minimal Diff in Existing Files

| Existing file | Change type | Lines |
|---------------|------------|-------|
| types.rs | Enum variant + config fields | ~30 |
| raw_scorer.rs | Match arms | ~20 |
| quantized_scorer_builder.rs | Match arms | ~15 |
| hnsw.rs | Tangent scorer injection | ~25 |
| **Total** | | **~90 lines** |

Remaining ~1700+ lines are all in new `spaces/hyperbolic/` module (never conflicts with upstream).

### 10.4 Rebase Cadence

- Track upstream Qdrant releases (~monthly)
- Rebase fork onto upstream tag
- Resolve conflicts (expected trivial — only ~90 lines in existing files)
- Run full test suite
- Tag: `v{upstream_version}-hyperbolic.{patch}`

### 10.5 Upstream Contribution Path

1. Layer A (Poincare metric only) as initial PR — smallest, most self-contained
2. Layers B/C as follow-up PRs once A is accepted
3. Feature-gated so maintainers can merge without affecting defaults
4. Accompanied by benchmarks and documentation

---

## 11. Deferred Items

| Item | Reason | Revisit When |
|------|--------|-------------|
| Layer D: Server-side projection | Client-side sufficient for now | Multiple consumers need projection |
| Temporal decay weighting | Pythia-side concern, no Qdrant changes needed | After initial validation |
| Rolling window / archival collections | Start with full history (option B) | Data growth becomes a problem |
| Comments and articles | Defer to reduce initial scope | Post + story + narrative validated |
| Curvature canary testing | Per-collection curvature sufficient initially | Multiple curvatures need A/B testing |
| Upstream Qdrant PR | Validate in fork first | Fork proven in production |
| SIMD-optimized Poincare distance | Scalar implementation first | Performance profiling shows need |
| Per-shard curvature registry | Single curvature per collection sufficient | Different data regions need different curvatures |

---

## 12. Implementation Sequence

### Phase 1: Qdrant Fork — Layer A (Poincare Metric)

1. Add `Distance::Poincare` to enum
2. Implement `poincare_math.rs` (distance, projection, exp/log maps)
3. Implement `PoincareMetric` with `Metric<f32>`, `Metric<f16>`, `Metric<u8>`
4. Add match arms in scorer dispatch sites
5. Add curvature to `VectorDataConfig`
6. Unit tests for mathematical correctness
7. Integration tests for scorer + HNSW

### Phase 2: Qdrant Fork — Layer B (Tangent Pruning)

1. Implement `TangentCache` (Frechet mean, tangent coordinate pre-computation)
2. Implement `TangentScorer` (wrapper with prune_factor)
3. Integrate into segment optimization pipeline
4. Wire into HNSW search path for Poincare collections
5. Benchmark: measure speedup vs exact scoring

### Phase 3: Qdrant Fork — Layer C (Busemann Scoring)

1. Implement Lorentz conversion (`poincare_to_lorentz`)
2. Implement `busemann_score` function
3. Implement `compute_focal_direction`
4. Add Busemann parameters to query API
5. Wire into search result enrichment pipeline

### Phase 4: Pythia Integration

1. Implement `HyperbolicProjector` (PCA + exp_map)
2. Fit PCA matrix from stratified Pythia data sample
3. Create per-client hyperbolic collections
4. Update embedding pipeline to produce hyperbolic vectors
5. Update stories/narratives pipelines to write to hyperbolic collections
6. Validation on real Pythia data (acceptance tests)

### Phase 5: Validation and Tuning

1. Run hierarchy separation test (Busemann depth vs tier)
2. Run retrieval quality test (vs Neo4j ground truth)
3. Run dual-space fusion test (cosine + Poincare RRF)
4. Performance benchmarking (latency, throughput)
5. Curvature tuning per client
6. Quantization boundary validation

---

## 13. References

### Academic
- Nickel, M. & Kiela, D. (2017). "Poincare Embeddings for Learning Hierarchical Representations." NeurIPS.
- Nickel, M. & Kiela, D. (2018). "Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry." ICML.
- Sala, F. et al. (2018). "Representation Tradeoffs for Hyperbolic Embeddings." ICML.

### Implementation References
- ruvector: Rust hyperbolic vector database (`~/projects/ruvector`)
  - `crates/ruvector-hyperbolic-hnsw/` — Poincare HNSW, tangent pruning, sharding
  - `crates/ruvector-attention/src/hyperbolic/lorentz_cascade.rs` — Busemann scoring
- PageIndex: Hierarchical retrieval via LLM reasoning (`~/projects/PageIndex`)
  - Conceptual reference for coarse-to-fine hierarchical navigation pattern
- Pythia: Intelligence platform consumer (`~/projects/pythia`)
  - `pythia/narratives/services/poincare.py` — Existing Poincare projection
  - `pythia/narratives/services/busemann.py` — Existing Busemann scoring
  - `docs/NARRATIVE_HIERARCHY_FRAMEWORK.md` — 4-tier hierarchy definition

### Qdrant Integration Points
- `lib/segment/src/types.rs:309-366` — Distance enum
- `lib/segment/src/spaces/metric.rs` — Metric<T> trait
- `lib/segment/src/spaces/simple.rs` — Existing metric implementations
- `lib/segment/src/vector_storage/raw_scorer.rs:200-305` — Scorer dispatch
- `lib/segment/src/vector_storage/quantized/quantized_scorer_builder.rs:56-87` — Quantized dispatch
- `lib/segment/src/index/hnsw_index/graph_layers.rs:108-596` — HNSW search
- `lib/segment/src/index/hnsw_index/hnsw.rs` — HNSW index entry point
