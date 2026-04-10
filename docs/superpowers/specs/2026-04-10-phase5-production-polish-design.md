# Phase 5: Production Polish — Design Spec

**Date:** 2026-04-10
**Branch:** `feat/hyperbolic-vector-support`
**Scope:** SIMD Poincaré distance, quantization validation, integration tests, Klein client-side pre-filter, geometric filters

---

## 1. Motivation

Phase 4 validated the unified collection architecture with perfect drill-down recall and 100% depth-band precision across both BGC (4 levels) and HWV (6 levels). The Poincaré distance implementation is functionally correct, but lacks the performance optimizations and test coverage that every other Qdrant distance metric has.

Phase 5 brings the Poincaré implementation to production quality:
- **SIMD** — every other Qdrant metric (Cosine, Euclid, Dot, Manhattan) has AVX/SSE/NEON optimized paths. Poincaré is the only scalar-only metric.
- **Quantization validation** — Poincaré maps to L2 in the quantization library. We need to measure whether this produces acceptable recall before deploying at scale.
- **Integration tests** — 625 unit tests pass, but zero end-to-end HNSW+Poincaré tests exist.
- **Klein pre-filter** — client-side optimization that avoids expensive `acosh` calls by pre-ranking with cheap Euclidean Klein chord distance.
- **Geometric filters** — InBall, InCone spatial queries for the chatbot, composable with depth-band filtering.

### Design Principles

1. **Follow existing Qdrant patterns exactly** — SIMD files mirror `simple_avx.rs`/`simple_sse.rs`/`simple_neon.rs`. Integration tests mirror `hnsw_quantized_search_test.rs`. No novel patterns.
2. **Measure before fixing** — Quantization gets a recall test first. Only fix if recall is below threshold.
3. **Client-side where possible** — Klein routing and geometric filters are Python testbench additions, not Qdrant fork changes.
4. **Testbench validates everything** — New benchmark suites for Klein speedup, geometric filter precision, and quantization recall.

---

## 2. SIMD Poincaré Distance

### 2.1 What Gets SIMD-Optimized

The `fused_norms()` function in `poincare_math.rs` is the inner loop of every Poincaré distance computation. It iterates over all dimensions, computing three accumulators simultaneously:

```
diff_sq += (u[i] - v[i])²
u_sq    += u[i]²
v_sq    += v[i]²
```

This is ~80% of distance computation time. The remaining work (denominator, `acosh`, division by `√c`) is scalar — called once per distance, not per-dimension.

### 2.2 Implementation Pattern

Follow Qdrant's exact pattern from `simple.rs` / `simple_avx.rs` / `simple_sse.rs` / `simple_neon.rs`:

**Three new files:**
- `lib/segment/src/spaces/hyperbolic/poincare_avx.rs` — AVX2+FMA (32 floats/iteration)
- `lib/segment/src/spaces/hyperbolic/poincare_sse.rs` — SSE (16 floats/iteration)
- `lib/segment/src/spaces/hyperbolic/poincare_neon.rs` — NEON/aarch64 (16 floats/iteration)

Each implements one function:

```rust
#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
pub(crate) unsafe fn poincare_similarity_avx(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
    curvature: f32,
) -> ScoreType
```

**Algorithm (AVX example):**
1. Initialize 4 pairs of `__m256` accumulators: `diff_sq_1..4`, `u_sq_1..4`, `v_sq_1..4` (12 registers total)
2. Process 32 elements per iteration (4 × 8-element AVX vectors):
   - Load `u[i..i+8]` and `v[i..i+8]`
   - `sub = _mm256_sub_ps(u, v)`
   - `diff_sq += _mm256_fmadd_ps(sub, sub, diff_sq)` (FMA: sub*sub + acc)
   - `u_sq += _mm256_fmadd_ps(u, u, u_sq)`
   - `v_sq += _mm256_fmadd_ps(v, v, v_sq)`
3. Horizontal sum all accumulators → scalar `diff_sq`, `u_sq`, `v_sq`
4. Handle remainder with scalar loop
5. Compute: `denom_u = max(1.0 - c*u_sq, EPS)`, `denom_v = max(1.0 - c*v_sq, EPS)`
6. `arg = 1.0 + 2.0*c*diff_sq / (denom_u * denom_v)`
7. Return: `-stable_acosh(arg) / sqrt(c)` (negated for similarity semantics)

SSE and NEON follow the same pattern with 4-element vectors instead of 8-element.

### 2.3 Dispatch in poincare_metric.rs

Update `PoincareMetric::similarity()` to match `simple.rs` dispatch pattern:

```rust
fn similarity(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("fma")
            && v1.len() >= 32
        {
            return unsafe { poincare_similarity_avx(v1, v2, DEFAULT_CURVATURE) };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse") && v1.len() >= 16 {
            return unsafe { poincare_similarity_sse(v1, v2, DEFAULT_CURVATURE) };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && v1.len() >= 16 {
            return unsafe { poincare_similarity_neon(v1, v2, DEFAULT_CURVATURE) };
        }
    }
    -poincare_distance(v1, v2, DEFAULT_CURVATURE)
}
```

Also update `PoincareCurvatureQueryScorer` to use SIMD-dispatched distance. Since the curvature scorer calls `poincare_distance(v1, v2, curvature)` with a runtime curvature value (not `DEFAULT_CURVATURE`), the SIMD functions must accept a `curvature: f32` parameter. The curvature scorer's `score_stored()` method should dispatch to the same SIMD functions, passing its stored `self.curvature` value.

### 2.4 Module Registration

Update `lib/segment/src/spaces/hyperbolic/mod.rs` to declare the new modules with conditional compilation:

```rust
#[cfg(target_arch = "x86_64")]
pub mod poincare_avx;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod poincare_sse;
#[cfg(target_arch = "aarch64")]
pub mod poincare_neon;
```

### 2.5 Testing

Each SIMD file includes feature-gated tests that validate results match the scalar baseline:

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_poincare_avx_matches_scalar() {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            let v1: Vec<f32> = (0..128).map(|i| (i as f32 * 0.001).sin() * 0.3).collect();
            let v2: Vec<f32> = (0..128).map(|i| (i as f32 * 0.002).cos() * 0.3).collect();
            let simd = unsafe { poincare_similarity_avx(&v1, &v2, 5.0) };
            let scalar = -poincare_distance(&v1, &v2, 5.0);
            assert!((simd - scalar).abs() < 1e-4, "SIMD={simd}, scalar={scalar}");
        }
    }
}
```

### 2.6 Expected Speedup

At 128 dimensions (our testbench vector size):
- AVX processes 32 elements/iteration → 4 iterations for the inner loop (vs 128 scalar iterations)
- ~3-5x speedup on fused_norms, translating to ~2-3x overall distance computation speedup
- The `acosh` + division remains scalar (called once), limiting the theoretical maximum

---

## 3. Quantization Validation

### 3.1 Methodology

Follow both Qdrant's and HyperspaceDB's established testing patterns.

### 3.2 Rust Integration Test: Recall@K

**File:** `lib/segment/tests/integration/hnsw_poincare_search_test.rs`

Following `hnsw_quantized_search_test.rs` pattern:

1. Create 5000 random vectors in the Poincaré ball:
   - 1000 near origin (||x|| < 0.2/√c)
   - 2000 mid-ball (0.2/√c < ||x|| < 0.6/√c)
   - 2000 near boundary (0.6/√c < ||x|| < 0.95/√c)
2. Build HNSW index with INT8 scalar quantization (quantile=0.99, always_ram=true)
3. Run 100 search queries
4. For each query, run two searches:
   - Quantized: normal HNSW with quantization
   - Exact: `exact: true, quantization: { ignore: true }`
5. Compute `sames_count()` (top-10 ID intersection)
6. Assert: accuracy ≥ 70%

### 3.3 Rust Unit Test: Ordering Preservation

Following HyperspaceDB's `test_lorentz_quantized_distance_preserves_ordering` pattern:

1. Three vectors at known Poincaré distances: near (d≈0.5), mid (d≈1.5), far (d≈3.0)
2. Quantize all three with INT8
3. Compute quantized distances
4. Assert: `d_quantized(near) < d_quantized(mid) < d_quantized(far)`
5. Also test: self-distance of quantized vector < 0.05

### 3.4 Testbench Benchmark Suite: Real-World Recall

**New suite in `benchmark.py`:** `suite_quantization_recall`

1. For 200 random query points from the unified collection:
   - Search with quantization enabled (default Qdrant behavior)
   - Search with `exact: true` (bypass HNSW + quantization)
2. Compute `quantization_recall@10`: fraction of exact top-10 found in quantized top-10
3. Compute average relative distance error: `|d_quantized - d_exact| / d_exact`
4. Report per-tier breakdown (are boundary points worse than origin points?)

### 3.5 Decision Gate

| Rust recall@10 | Testbench recall@10 | Action |
|----------------|---------------------|--------|
| ≥ 70% | ≥ 90% | L2 mapping is acceptable. Document. |
| ≥ 70% | < 90% | Add oversampling recommendation (fetch 2x, rescore with exact) |
| < 70% | any | Add Poincaré-aware rescoring in quantized scorer builder |

Poincaré-aware rescoring: quantized L2 gets top-50 candidates → exact Poincaré re-ranks to top-10. This is a small change in the quantized scorer path (add a rescore flag for Poincaré, similar to how Binary quantization auto-rescores).

---

## 4. Integration Tests

### 4.1 Test File

`lib/segment/tests/integration/hnsw_poincare_search_test.rs`

Following the pattern of existing integration tests (`hnsw_quantized_search_test.rs`, `byte_storage_quantization_test.rs`).

### 4.2 Test Cases

**Test 1: HNSW search correctness (c=1.0)**
- 1000 vectors, stratified by radius (near/mid/boundary)
- Build HNSW index (m=16, ef_construct=128)
- 50 queries, compare HNSW top-10 vs brute-force exact top-10
- Assert: recall ≥ 95%

**Test 2: Curvature-aware search (c=5.0)**
- Same vector set, but create collection with curvature=5.0
- Verify search uses `PoincareCurvatureQueryScorer` (results differ from c=1.0)
- 50 queries, verify recall ≥ 95%
- Compare result ordering vs c=1.0 — must differ for at least some queries (proving curvature is active)

**Test 3: Named vectors (dense + poincaré)**
- Create segment with two named vectors: "dense" (Cosine, 1024d) + "poincare" (Poincaré, 128d)
- Insert 500 points with both vectors
- Search each named vector independently
- Verify: results differ between cosine and poincaré search (different rankings)
- Verify: payload association is correct across both vector spaces

**Test 4: Quantization recall**
- From Section 3.2 above

**Test 5: Edge cases**
- Zero vector: search should not crash, should return valid (possibly degenerate) results
- Boundary vector (||x|| = 0.99/√c): numerical stability, no NaN/Inf in results
- Single-point collection: search returns exactly 1 result
- Empty collection: search returns empty results, no crash
- Mismatched dimensions: appropriate error returned

### 4.3 Test Utilities

Helper functions for generating test data:

```rust
fn random_poincare_vectors(n: usize, dim: usize, c: f32, rng: &mut StdRng) -> Vec<Vec<f32>> {
    // Generate vectors uniformly distributed by radius in the Poincaré ball
    // Stratify: 20% near origin, 40% mid-ball, 40% near boundary
}

fn brute_force_top_k(query: &[f32], vectors: &[Vec<f32>], c: f32, k: usize) -> Vec<(usize, f32)> {
    // Compute exact Poincaré distance to all vectors, return top-k by distance
}
```

---

## 5. Klein Client-Side Pre-Filter

### 5.1 Mathematical Foundation

The Klein disk model maps Poincaré ball points to a disk where geodesics become straight chords. The conversion is:

```
klein_i = 2 * poincare_i / (1 + ||poincare||²)
```

Klein chord distance (Euclidean L2 between Klein points) approximates Poincaré distance ordering for nearby points. It's O(d) with no transcendental functions — just subtraction and multiplication.

### 5.2 Three-Stage Query Pipeline

1. **Qdrant search** (server-side): search `dense` (cosine) named vector with high limit (e.g., top-200). This is fast HNSW search.
2. **Klein pre-rank** (client-side): compute Klein chord distance between query and all 200 candidates using stored `klein_vector` payload. Keep top-50.
3. **Poincaré re-rank** (client-side): compute exact Poincaré distance for 50 survivors. Return top-10.

The expensive `acosh` is called only 50 times instead of being in every HNSW distance computation.

### 5.3 Implementation

**`hyperbolic_math.py`** — add two functions:

```python
def poincare_to_klein(vector, c=1.0):
    """Convert Poincaré ball point to Klein disk coordinates."""
    sq_norm = np.dot(vector, vector)
    return 2.0 * vector / (1.0 + c * sq_norm)

def klein_chord_distance_sq(u_klein, v_klein):
    """Squared Euclidean distance in Klein disk. No acosh needed."""
    diff = u_klein - v_klein
    return np.dot(diff, diff)
```

**`queries.py`** — add `klein_prefilter_query()`:

```python
def klein_prefilter_query(client, collection, query_poincare, curvature=5.0,
                          prefetch_limit=200, klein_top=50, final_top=10,
                          using_prefetch="dense", ef=128):
    """Three-stage search: cosine prefetch → Klein pre-rank → Poincaré re-rank."""
```

**`embed.py`** — at upsert time, compute and store `klein_vector` as a payload field (list of floats) alongside each point.

**`benchmark.py`** — add Klein pre-filter variant to the latency suite:
- Compare: direct Poincaré search vs Klein pre-filter pipeline
- Measure: latency, recall@10 (vs exact Poincaré baseline), QPS

### 5.4 Expected Performance

- Direct Poincaré: Qdrant HNSW evaluates `acosh` at every graph hop (~100-300 distance computations per query)
- Klein pipeline: 1 cosine HNSW search (fast) + 200 Klein L2 distances (trivial) + 50 Poincaré distances
- Net: fewer `acosh` calls, but additional round-trip overhead. Beneficial at scale (large collections where HNSW does more hops).

---

## 6. Geometric Filters

### 6.1 Filter Types

**InBall** — Poincaré ball neighborhood:
- Input: center vector, radius (Poincaré distance threshold)
- Keeps candidates with `poincare_distance(center, candidate) ≤ radius`
- Use case: "Find items within conceptual distance R of this query"

**InCone** — Directional sector from origin:
- Input: axis direction vector, aperture angle (radians)
- Computes angular distance between each candidate's direction (from origin) and the cone axis
- Keeps candidates within the aperture
- Use case: "Find items in the same subtree direction" — captures hierarchical relatedness without exact parent_ids

**DepthBand** — Busemann depth range (already exists, formalize):
- Input: depth_min, depth_max
- Keeps candidates with `depth_min ≤ busemann_depth ≤ depth_max`
- Already implemented as `depth_band_query()` in `queries.py`

### 6.2 Implementation

**`hyperbolic_math.py`** — add filter functions:

```python
def inball_filter(candidates, center, radius, curvature=1.0):
    """Keep candidates within Poincaré distance `radius` of `center`."""

def incone_filter(candidates, axis, aperture, tolerance=0.1):
    """Keep candidates within angular `aperture` of `axis` direction from origin."""
```

Each takes a list of candidate dicts (with "vector" key) and returns filtered list.

**`queries.py`** — add `geometric_filtered_query()`:

```python
def geometric_filtered_query(client, collection, query_vector, filters,
                              using="poincare", limit=100, final_top=10, ef=128):
    """Search Qdrant, then apply geometric filters client-side.

    Args:
        filters: list of filter specs, e.g.:
            [{"type": "inball", "center": vec, "radius": 2.0},
             {"type": "incone", "axis": vec, "aperture": 0.5}]
    """
```

Filters compose with AND logic — all must pass.

**`benchmark.py`** — add `suite_geometric_filters`:
- For each filter type, run queries and measure:
  - `filter_precision`: do all returned results actually satisfy the geometric constraint?
  - `filter_selectivity`: what fraction of initial candidates survive the filter?
  - `filter_latency_ms`: client-side filtering overhead
- Test composition: InBall + DepthBand together

### 6.3 Composability Example

A chatbot query: "Find stories about Science that are structurally close to my query":

```python
results = geometric_filtered_query(
    client, "bgc_unified", query_vector,
    filters=[
        {"type": "depth_band", "depth_min": 0.4, "depth_max": 0.9},  # story tier
        {"type": "incone", "axis": science_direction, "aperture": 0.5},  # Science subtree
        {"type": "inball", "center": query_vector, "radius": 2.0},  # nearby
    ],
    using="poincare", limit=200, final_top=10,
)
```

---

## 7. Execution Order

### Step 1: SIMD Implementation
1. Create `poincare_avx.rs`, `poincare_sse.rs`, `poincare_neon.rs`
2. Update `poincare_metric.rs` with dispatch logic
3. Update `mod.rs` with conditional module declarations
4. Run SIMD-vs-scalar validation tests
5. Run full test suite: `cargo test --lib --features hyperbolic -p segment`

### Step 2: Integration Tests
1. Create `hnsw_poincare_search_test.rs`
2. Implement Tests 1-3, 5 (basic search, curvature, named vectors, edge cases)
3. Run: `cargo test --test hnsw_poincare_search_test --features hyperbolic -p segment`

### Step 3: Quantization Validation
1. Add Test 4 (quantization recall) to the integration test file
2. Run and measure recall@10
3. Apply decision gate — fix if needed

### Step 4: Testbench Math Extensions
1. Add Klein functions to `hyperbolic_math.py`
2. Add geometric filter functions to `hyperbolic_math.py`
3. Verify with unit tests

### Step 5: Klein Pre-Filter
1. Add `klein_prefilter_query()` to `queries.py`
2. Update `embed.py` to store `klein_vector` as payload
3. Add Klein benchmark to `benchmark.py`
4. Re-embed and benchmark BGC

### Step 6: Geometric Filters
1. Add `geometric_filtered_query()` to `queries.py`
2. Add `suite_geometric_filters` to `benchmark.py`
3. Benchmark on BGC and HWV

### Step 7: Full Validation
1. Re-run all 8 Phase 4 suites + new suites on BGC
2. Re-run on HWV
3. Generate comparison report: Phase 4 baseline vs Phase 5 optimized
4. Document findings

---

## 8. Success Criteria

### Qdrant Fork (Rust)
- [ ] SIMD: AVX/SSE/NEON implementations match scalar baseline within 1e-4 tolerance
- [ ] SIMD: measurable speedup in latency suite (≥ 1.5x QPS improvement)
- [ ] Integration tests: 5 test cases pass, HNSW recall ≥ 95%
- [ ] Curvature test: c=5.0 search produces different ordering than c=1.0
- [ ] Quantization: recall@10 measured and documented; ≥ 70% with L2 mapping
- [ ] All 625+ existing tests continue to pass

### Testbench (Python)
- [ ] Klein pre-filter: recall@10 ≥ 95% vs direct Poincaré (quality preserved)
- [ ] Klein pre-filter: measurable latency reduction vs direct search
- [ ] Geometric filters: 100% filter precision (returned results satisfy constraints)
- [ ] InCone: meaningfully filters results (selectivity < 50% on random queries)
- [ ] All Phase 4 suites still pass (no regression)

---

## 9. Non-Goals (Deferred)

- Full Klein routing inside HNSW graph traversal (would require modifying Qdrant's HNSW internals)
- Anisotropic quantization (measure L2 first, only implement if recall is unacceptable)
- SONA integration (Phase NEXT — Pythia-side)
- Cross-collection knowledge transfer (Phase NEXT — needs SONA first)
- Pythia code changes (this phase is testbench-only)
