# Hyperbolic V2: Configurable Curvature, Einstein Midpoint & Stability Improvements

**Date:** 2026-04-08
**Status:** Design approved, pending implementation
**Scope:** Three targeted improvements to the Qdrant fork's hyperbolic vector support
**Depends on:** Phase 1 implementation (`docs/superpowers/specs/2026-04-07-hyperbolic-vector-support-design.md`)
**Reference implementations:** ruvector (`~/projects/ruvector`), Pythia narratives (`~/projects/pythia/pythia/narratives/services/`)

---

## 1. Problem Statement

The Phase 1 hyperbolic implementation (`feat/hyperbolic-vector-support`, now merged to master) provides a working Poincare distance metric, tangent pruning, and Busemann scoring. However, three limitations remain before the fork is ready for Pythia's multi-client deployment:

1. **Curvature is hardcoded to 1.0.** Different Pythia clients (UK, Iraq, Syria, Sudan) have structurally different hierarchies. UK political narratives are shallow and wide (many overlapping stories), Sudan conflict narratives are deep and narrow (clear factional lines). One curvature cannot optimally serve both. Higher curvature increases separation between hierarchy levels but distorts within-level nuance.

2. **Tangent cache centroid uses iterative Frechet mean.** The current `frechet_mean()` runs Riemannian gradient descent for up to 100 iterations (lr=0.1, tol=1e-6). With 500K+ vectors per segment, this is a bottleneck during index construction. RuVector's Einstein midpoint computes the same centroid in O(1) via a closed-form Lorentz-space average.

3. **Stable acosh has only two regimes.** The current implementation handles near-identity (Taylor expansion) and standard ranges, but lacks the asymptotic `ln(2x)` regime for large arguments. At high curvature, points on opposite sides of the Poincare ball produce very large acosh arguments where `sqrt(x^2 - 1)` risks precision loss.

### 1.1 Context: Pythia's Multi-Client Architecture

Pythia is an intelligence platform with a 4-tier information hierarchy (Content > Stories > Narratives > Themes). The planned collection architecture splits data per source:

- `uk_networks`, `iraq_networks`, `syria_networks`, `sudan_networks` (per data_source)
- `uk_global`, `iraq_global`, etc. (per client_id)

Each unified collection contains posts + profiles + stories + narratives differentiated by a `type` payload field, with all tiers sharing one Poincare ball. This requires per-collection curvature tuning to match each source's hierarchy shape.

The collections use a dual HNSW strategy:
- Dense vector (1024d, Cosine): `m=0` global, `payload_m=16` on `type` (within-type semantic search)
- Hyperbolic vector (128d, Poincare): `m=16` global (cross-type hierarchy traversal), NO quantization (128d raw is only 512B/point)
- Sparse vector: inverted index, in-RAM

Dense vectors use INT8 scalar quantization (significant savings at 1024d). Hyperbolic vectors are stored unquantized — the 192MB savings from quantizing 128d vectors across 500K points is not worth the boundary sensitivity risk (Poincare distance denominator `(1-c||u||^2)(1-c||v||^2)` approaches zero near the boundary, amplifying quantization error).

### 1.2 Why These Three Changes

These are the improvements from ruvector (`~/projects/ruvector`) that directly benefit Pythia's use case:

| Improvement | RuVector reference | Impact |
|---|---|---|
| Configurable curvature | `shard.rs:24-74` (ShardCurvature) | Enables per-client hierarchy tuning |
| Einstein midpoint | `lorentz_cascade.rs:146-167` | 10-100x faster centroid computation |
| Stable acosh 3rd regime | `poincare.rs:259-275` | Numerical stability at high curvature |

Other ruvector features were evaluated and deferred:
- **Canary testing for curvature:** Operational safety feature for Phase 4 production, not needed now.
- **Lorentz as first-class distance:** Our Busemann already uses Lorentz internally. Full `Distance::Lorentz` adds storage complexity (d+1 dimensions) without clear benefit.
- **Multi-curvature cascade:** Neural attention mechanism, not applicable to vector search.
- **Hierarchy metrics (Spearman, AUPRC):** Validation tools that belong in Pythia's Python code, not the Qdrant fork.
- **4-element loop unrolling in fused_norms:** LLVM auto-vectorizes `iter().zip()` well in Rust. Worth benchmarking before manual optimization.

---

## 2. Design

All changes are feature-gated behind `#[cfg(feature = "hyperbolic")]` and additive to the existing implementation. No existing behavior changes.

### 2.1 Per-Collection Configurable Curvature

**Storage:** Add an optional curvature field to the vector data config in `types.rs`:

```rust
#[cfg(feature = "hyperbolic")]
pub curvature: Option<f32>,  // None = DEFAULT_CURVATURE (1.0)
```

This is persisted with the segment config alongside the existing `distance: Distance` field.

**API surface:** Curvature is an optional field on vector params at collection creation:

```json
{
  "vectors": {
    "hyperbolic": {
      "size": 128,
      "distance": "Poincare",
      "curvature": 2.0
    }
  }
}
```

Omitting `curvature` defaults to 1.0. Specifying curvature with a non-Poincare distance type is ignored.

**Scorer threading:** The `Metric` trait is stateless (`similarity()` is a static method), so curvature cannot flow through it. Instead, at each scorer dispatch site where we match `Distance::Poincare`, the curvature value is read from the segment's vector config and passed directly to `poincare_distance(v1, v2, curvature)`, bypassing the generic `Metric<T>::similarity()` path.

This works cleanly because:
- The Poincare match arms already use the cfg-split function pattern (separate `#[cfg(feature)]` blocks with distinct where clauses)
- The segment's vector config is available at scorer creation time
- All downstream functions (`poincare_distance`, `project_to_ball`, `exp_map`, `log_map`, `mobius_add`) already accept curvature as a parameter

**Tangent cache and Busemann:** Both `TangentCache::build()` and the Busemann functions already accept curvature parameters. They receive it from the segment config instead of `DEFAULT_CURVATURE`.

**Backward compatibility:** Existing collections with `Distance::Poincare` and no curvature field deserialize with `None`, mapping to `DEFAULT_CURVATURE = 1.0`. No migration needed, all existing tests pass unchanged.

**Performance impact:** None. Curvature is loaded once at scorer creation (per query) and captured in the closure. A single `f32` parameter in a function that already computes `acosh`, `sqrt`, and dot products is unmeasurable.

**Curvature selection (Pythia-side, not fork-side):** The fork's job is to accept `curvature: f32` and use it. Finding the right value per client is a Pythia concern:

- **Grid search (Phase 3):** Embed a sample (~1K posts, ~50 stories, ~10 narratives). Project at curvatures 0.5, 1.0, 2.0, 5.0. Measure Busemann depth separation between tiers. Pick the curvature maximizing separation.
- **Learned (Phase 4):** Optimize curvature against a metric (Spearman correlation between Busemann depth and ground-truth tier labels). Automate as a calibration step.
- **Heuristic:** Higher curvature = more tier separation but more within-tier distortion. Deep narrow hierarchies (Sudan) want higher c. Shallow wide hierarchies (UK) want lower c.

### 2.2 Einstein Midpoint

**Algorithm:** Closed-form centroid computation via Lorentz model averaging. Reference: ruvector `lorentz_cascade.rs:146-167`.

1. Convert all Poincare points to Lorentz hyperboloid: `p -> (x_0, x_1...x_n)` where `x_0 = (1 + c||p||^2) / (1 - c||p||^2)`
2. Compute Lorentz factors: `gamma_i = x_0_i` (the time component)
3. Weighted average: `m = sum(gamma_i * x_i) / sum(gamma_i)` — points closer to boundary have higher gamma, contributing proportionally more
4. Project back to hyperboloid: normalize so `<m, m>_L = -1/c`
5. Convert back to Poincare: `p_i = x_i / (x_0 + 1)`

Single pass over the data, no iteration, no convergence tolerance, no learning rate. O(n*d) where n is point count and d is dimensionality.

**Location:** New public function `einstein_midpoint(points: &[&[f32]], c: f32) -> Vec<f32>` in `poincare_math.rs`, alongside existing `frechet_mean()`. Both are kept — Frechet mean is the exact Riemannian centroid, Einstein midpoint is the fast approximation. For tangent cache centroids and focal direction computation, the approximation is sufficient and the speed difference matters.

**Building blocks:** `poincare_to_lorentz()` and `lorentz_inner()` currently live in `busemann.rs`. Since Einstein midpoint needs them too, these Lorentz conversion utilities move to `poincare_math.rs` (the shared math module) and `busemann.rs` imports from there. This avoids circular dependencies and keeps all core math in one place.

**Integration points:**
- `TangentCache::build()` — switch centroid computation from `frechet_mean()` to `einstein_midpoint()`. This is the primary speedup: called per segment, potentially over hundreds of thousands of vectors.
- `busemann.rs::compute_focal_direction()` — currently computes Frechet mean then converts to Lorentz. Einstein midpoint does the Lorentz conversion as part of its computation, so the focal direction can be derived directly from the midpoint's Lorentz coordinates. Cleaner and faster.

**Expected speedup:** 10-100x for centroid computation depending on point count. Frechet mean: 100 iterations × n points × d dimensions. Einstein midpoint: 1 pass × n points × d dimensions.

### 2.3 Stable Acosh Third Regime

**Current implementation** (`poincare_math.rs:34-49`):

```
x <= 1.0        -> 0.0
delta < 1e-6    -> sqrt(2 * delta)    (Taylor expansion, near-identity)
else            -> f32::acosh(x)      (standard library)
```

**Addition:** Asymptotic regime for large arguments. Reference: ruvector `poincare.rs:259-275`.

```
x <= 1.0        -> 0.0
delta < 1e-6    -> sqrt(2 * delta)    (Taylor, near-identity)
x > 1e6         -> ln(2x)            (asymptotic, avoids sqrt(x^2-1) overflow risk)
else            -> f32::acosh(x)      (standard)
```

The asymptotic regime handles the case where two points are on opposite sides of the Poincare ball. At high curvature, the acosh argument `1 + 2c||u-v||^2 / ((1-c||u||^2)(1-c||v||^2))` can become very large. The standard formula computes `ln(x + sqrt(x^2 - 1))`, where `sqrt(x^2 - 1)` risks precision loss for large x. The asymptotic expansion `ln(2x)` is the first-order approximation and is both faster and more stable.

~5 lines of code, one additional branch. No performance impact — the vast majority of calls hit the standard regime, and branch prediction handles the rare cases.

---

## 3. Files Changed

### Modified files

| File | Change | ~Lines |
|---|---|---|
| `lib/segment/src/spaces/hyperbolic/poincare_math.rs` | Add `einstein_midpoint()`, add 3rd acosh regime | +40 |
| `lib/segment/src/spaces/hyperbolic/tangent_cache.rs` | Switch centroid from `frechet_mean` to `einstein_midpoint` | ~5 |
| `lib/segment/src/spaces/hyperbolic/tangent_scorer.rs` | Thread curvature through `poincare_distance` call in rescore phase | ~5 |
| `lib/segment/src/spaces/hyperbolic/busemann.rs` | Move `poincare_to_lorentz` + `lorentz_inner` to `poincare_math.rs` (shared by Einstein midpoint); derive focal direction from midpoint's Lorentz coords | ~15 |
| `lib/segment/src/types.rs` | Add `curvature: Option<f32>` to vector config (cfg-gated) | ~10 |
| `lib/segment/src/vector_storage/raw_scorer.rs` | Thread curvature through Poincare match arms | ~15 |
| `lib/segment/src/vector_storage/async_raw_scorer.rs` | Thread curvature through Poincare match arm | ~5 |
| `lib/segment/src/vector_storage/quantized/quantized_scorer_builder.rs` | Thread curvature through Poincare match arms | ~10 |

### No new files

All changes are additions to existing hyperbolic module files and existing scorer dispatch sites.

### Test additions (~200 lines)

Added to inline `#[cfg(test)]` modules in existing files:

**Einstein midpoint tests:**
- Single point returns that point
- Two symmetric points equidistant from origin produce midpoint on the geodesic between them
- N points clustered near origin produce midpoint near origin
- N points near boundary produce midpoint inside ball (never outside)
- Result matches `frechet_mean()` within tolerance for well-behaved inputs
- Reference: ruvector `math_tests.rs:219-247`

**Stable acosh tests:**
- All three regimes produce correct values against f64 reference computation
- No discontinuities at regime boundaries: values agree within f32 epsilon at transition points (1e-6 and 1e6)
- Monotonicity: input increases -> output increases across all regimes

**Configurable curvature tests:**
- Same points, different curvatures -> different distances (higher c = larger distances)
- `c=1.0` matches existing test results exactly (backward compatibility)
- Distance properties hold at non-default curvatures: symmetry, triangle inequality, identity

**Integration test — hierarchy separation:**
1. Use a real-world hierarchical dataset from Hugging Face (e.g., news topic dataset with article -> event -> topic hierarchy, or tweet topic dataset). Synthetic data as fallback for fast CI.
2. Embed with a standard model, PCA reduce to 128d, project to Poincare ball via exp_map
3. Insert ~100 "leaf" vectors (near boundary), ~10 "mid" vectors (intermediate), ~3 "root" vectors (near origin) into a collection with `Distance::Poincare`
4. Search from a mid-level vector -> verify roots rank closer than distant leaves (hierarchy-aware ordering)
5. Compute Busemann depths -> verify roots < mid < leaves with clear separation
6. Repeat at curvature 1.0 and 2.0 -> verify higher curvature produces greater tier separation
7. Benchmark Einstein midpoint vs Frechet mean: time to compute centroid over 100, 1K, 10K, 100K vectors at 128d

**Total estimated diff:** ~300 lines including tests, all feature-gated behind `hyperbolic`.

---

## 4. What This Does NOT Cover

These items are out of scope for this spec and will be addressed in subsequent phases:

| Item | When |
|---|---|
| Pythia collection architecture (unified per-source collections) | Phase 2 brainstorm |
| HyperbolicProjector module (PCA + exp_map pipeline) | Phase 2 |
| Per-client collection creation and payload schema | Phase 2 |
| Embedding pipeline dual-write (dense + hyperbolic) | Phase 2 |
| Profile actor-depth computation (Einstein midpoint of posts) | Phase 2 |
| Stories/narratives pipeline integration | Phase 2 |
| Curvature calibration per client | Phase 3 validation |
| Canary testing for curvature changes | Phase 4 production |
| Radius-aware quantization for hyperbolic vectors | Deferred (unquantized 128d is sufficient) |
| Lorentz as first-class distance metric | Deferred (Poincare ball with max_norm suffices) |
| SIMD-optimized Poincare distance | Deferred (benchmark first) |
| 4-element loop unrolling in fused_norms | Deferred (benchmark LLVM auto-vectorization first) |

---

## 5. References

- Phase 1 design spec: `docs/superpowers/specs/2026-04-07-hyperbolic-vector-support-design.md`
- Phase 1 implementation plan: `docs/superpowers/plans/2026-04-07-hyperbolic-vector-support.md`
- Next steps roadmap: `docs/superpowers/NEXT_STEPS.md`
- RuVector Einstein midpoint: `~/projects/ruvector/crates/ruvector-attention/src/hyperbolic/lorentz_cascade.rs:146-167`
- RuVector stable acosh: `~/projects/ruvector/crates/ruvector-hyperbolic-hnsw/src/poincare.rs:259-275`
- RuVector per-shard curvature: `~/projects/ruvector/crates/ruvector-hyperbolic-hnsw/src/shard.rs:24-74`
- Pythia Poincare services: `~/projects/pythia/pythia/narratives/services/poincare.py`
- Pythia Busemann services: `~/projects/pythia/pythia/narratives/services/busemann.py`
