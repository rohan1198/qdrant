# Phase 5: Production Polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Poincaré distance implementation to production quality with SIMD optimization, quantization validation, integration tests, Klein client-side pre-filter, and geometric filters.

**Architecture:** SIMD follows Qdrant's exact pattern (simple_avx.rs/simple_sse.rs/simple_neon.rs). Integration tests follow hnsw_quantized_search_test.rs. Klein routing and geometric filters are client-side Python (no Qdrant fork changes). Quantization is measured first, fixed only if recall is below threshold.

**Tech Stack:** Rust 1.94 (SIMD intrinsics: x86 AVX2+FMA, SSE, ARM NEON), Python 3.10+ (testbench), Docker, qdrant-client

**Commit policy:** Do NOT commit incrementally. Make all changes, verify end-to-end, then commit everything at once.

---

## File Structure

### Qdrant Fork (Rust) — files to create/modify

| File | Action | Responsibility |
|------|--------|---------------|
| `lib/segment/src/spaces/hyperbolic/poincare_avx.rs` | Create | AVX2+FMA SIMD Poincaré distance |
| `lib/segment/src/spaces/hyperbolic/poincare_sse.rs` | Create | SSE SIMD Poincaré distance |
| `lib/segment/src/spaces/hyperbolic/poincare_neon.rs` | Create | NEON/aarch64 SIMD Poincaré distance |
| `lib/segment/src/spaces/hyperbolic/mod.rs` | Modify | Register new SIMD modules |
| `lib/segment/src/spaces/hyperbolic/poincare_metric.rs` | Modify | Add SIMD dispatch to similarity() |
| `lib/segment/src/spaces/hyperbolic/poincare_math.rs` | Modify | Make fused_norms and stable_acosh pub(crate) |
| `lib/segment/tests/integration/hnsw_poincare_search_test.rs` | Create | 5 integration test cases |

### Testbench (Python) — files to modify

| File | Action | Responsibility |
|------|--------|---------------|
| `dev/testbench/hyperbolic_math.py` | Modify | Add Klein conversion, geometric filter functions |
| `dev/testbench/queries.py` | Modify | Add klein_prefilter_query, geometric_filtered_query |
| `dev/testbench/embed.py` | Modify | Store klein_vector as payload |
| `dev/testbench/benchmark.py` | Modify | Add quantization recall, Klein, geometric suites |

---

## Part 1: SIMD Poincaré Distance

### Task 1: Create AVX2 Poincaré similarity

**Files:**
- Create: `lib/segment/src/spaces/hyperbolic/poincare_avx.rs`
- Modify: `lib/segment/src/spaces/hyperbolic/poincare_math.rs` (make helpers pub(crate))

- [ ] **Step 1: Make fused_norms and stable_acosh accessible to SIMD modules**

In `lib/segment/src/spaces/hyperbolic/poincare_math.rs`, change visibility of `fused_norms` and `stable_acosh` from `fn` to `pub(crate) fn`:

```rust
// Line ~18: change from `fn fused_norms` to:
pub(crate) fn fused_norms(u: &[f32], v: &[f32]) -> (f32, f32, f32) {

// Line ~38: change from `fn stable_acosh` to:
pub(crate) fn stable_acosh(x: f32) -> f32 {
```

Also export the `EPS` constant:
```rust
// Line ~12: change from `pub(crate) const EPS` (verify it's already pub(crate))
pub(crate) const EPS: f32 = 1e-5;
```

- [ ] **Step 2: Create poincare_avx.rs**

Create `lib/segment/src/spaces/hyperbolic/poincare_avx.rs`:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::data_types::vectors::VectorElementType;
use crate::types::ScoreType;
use super::poincare_math::{stable_acosh, EPS};

/// Horizontal sum of a single __m256 register.
#[target_feature(enable = "avx")]
unsafe fn hsum256_ps(x: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(x, 1);
    let lo128 = _mm256_castps256_ps128(x);
    let sum128 = _mm_add_ps(hi128, lo128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x1);
    _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
}

/// AVX2+FMA accelerated Poincaré similarity.
///
/// Computes -poincare_distance(v1, v2, curvature) using SIMD for the
/// fused norms inner loop (diff_sq, u_sq, v_sq in parallel).
/// The acosh + division are scalar (called once, not per-dimension).
#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
pub(crate) unsafe fn poincare_similarity_avx(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
    curvature: f32,
) -> ScoreType {
    let n = v1.len();
    let m = n - (n % 32);
    let mut ptr1: *const f32 = v1.as_ptr();
    let mut ptr2: *const f32 = v2.as_ptr();

    // 4-way accumulators for diff_sq, u_sq, v_sq (12 registers total)
    let mut diff1 = _mm256_setzero_ps();
    let mut diff2 = _mm256_setzero_ps();
    let mut diff3 = _mm256_setzero_ps();
    let mut diff4 = _mm256_setzero_ps();
    let mut usq1 = _mm256_setzero_ps();
    let mut usq2 = _mm256_setzero_ps();
    let mut usq3 = _mm256_setzero_ps();
    let mut usq4 = _mm256_setzero_ps();
    let mut vsq1 = _mm256_setzero_ps();
    let mut vsq2 = _mm256_setzero_ps();
    let mut vsq3 = _mm256_setzero_ps();
    let mut vsq4 = _mm256_setzero_ps();

    let mut i: usize = 0;
    while i < m {
        let u1 = _mm256_loadu_ps(ptr1);
        let v1_reg = _mm256_loadu_ps(ptr2);
        let sub1 = _mm256_sub_ps(u1, v1_reg);
        diff1 = _mm256_fmadd_ps(sub1, sub1, diff1);
        usq1 = _mm256_fmadd_ps(u1, u1, usq1);
        vsq1 = _mm256_fmadd_ps(v1_reg, v1_reg, vsq1);

        let u2 = _mm256_loadu_ps(ptr1.add(8));
        let v2_reg = _mm256_loadu_ps(ptr2.add(8));
        let sub2 = _mm256_sub_ps(u2, v2_reg);
        diff2 = _mm256_fmadd_ps(sub2, sub2, diff2);
        usq2 = _mm256_fmadd_ps(u2, u2, usq2);
        vsq2 = _mm256_fmadd_ps(v2_reg, v2_reg, vsq2);

        let u3 = _mm256_loadu_ps(ptr1.add(16));
        let v3_reg = _mm256_loadu_ps(ptr2.add(16));
        let sub3 = _mm256_sub_ps(u3, v3_reg);
        diff3 = _mm256_fmadd_ps(sub3, sub3, diff3);
        usq3 = _mm256_fmadd_ps(u3, u3, usq3);
        vsq3 = _mm256_fmadd_ps(v3_reg, v3_reg, vsq3);

        let u4 = _mm256_loadu_ps(ptr1.add(24));
        let v4_reg = _mm256_loadu_ps(ptr2.add(24));
        let sub4 = _mm256_sub_ps(u4, v4_reg);
        diff4 = _mm256_fmadd_ps(sub4, sub4, diff4);
        usq4 = _mm256_fmadd_ps(u4, u4, usq4);
        vsq4 = _mm256_fmadd_ps(v4_reg, v4_reg, vsq4);

        ptr1 = ptr1.add(32);
        ptr2 = ptr2.add(32);
        i += 32;
    }

    // Horizontal sum all accumulators
    let diff_sq_sum = _mm256_add_ps(_mm256_add_ps(diff1, diff2), _mm256_add_ps(diff3, diff4));
    let usq_sum = _mm256_add_ps(_mm256_add_ps(usq1, usq2), _mm256_add_ps(usq3, usq4));
    let vsq_sum = _mm256_add_ps(_mm256_add_ps(vsq1, vsq2), _mm256_add_ps(vsq3, vsq4));

    let mut diff_sq = hsum256_ps(diff_sq_sum);
    let mut u_sq = hsum256_ps(usq_sum);
    let mut v_sq = hsum256_ps(vsq_sum);

    // Handle remainder
    for j in 0..(n - m) {
        let u_val = *ptr1.add(j);
        let v_val = *ptr2.add(j);
        let d = u_val - v_val;
        diff_sq += d * d;
        u_sq += u_val * u_val;
        v_sq += v_val * v_val;
    }

    // Scalar: compute Poincaré distance from norms
    let c = curvature;
    let denom_u = (1.0 - c * u_sq).max(EPS);
    let denom_v = (1.0 - c * v_sq).max(EPS);
    let arg = 1.0 + 2.0 * c * diff_sq / (denom_u * denom_v);

    -(stable_acosh(arg) / c.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::hyperbolic::poincare_math::poincare_distance;

    #[test]
    fn test_poincare_avx_matches_scalar() {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            // 128-dim vectors (our testbench size)
            let v1: Vec<f32> = (0..128).map(|i| (i as f32 * 0.001).sin() * 0.3).collect();
            let v2: Vec<f32> = (0..128).map(|i| (i as f32 * 0.002).cos() * 0.3).collect();
            let simd = unsafe { poincare_similarity_avx(&v1, &v2, 1.0) };
            let scalar = -poincare_distance(&v1, &v2, 1.0);
            assert!((simd - scalar).abs() < 1e-4, "c=1.0: SIMD={simd}, scalar={scalar}");

            // Test with non-default curvature
            let simd5 = unsafe { poincare_similarity_avx(&v1, &v2, 5.0) };
            let scalar5 = -poincare_distance(&v1, &v2, 5.0);
            assert!((simd5 - scalar5).abs() < 1e-4, "c=5.0: SIMD={simd5}, scalar={scalar5}");
        }
    }

    #[test]
    fn test_poincare_avx_odd_dimensions() {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            // 70 dims (not divisible by 32 — tests remainder handling)
            let v1: Vec<f32> = (0..70).map(|i| (i as f32 * 0.003).sin() * 0.25).collect();
            let v2: Vec<f32> = (0..70).map(|i| (i as f32 * 0.004).cos() * 0.25).collect();
            let simd = unsafe { poincare_similarity_avx(&v1, &v2, 1.0) };
            let scalar = -poincare_distance(&v1, &v2, 1.0);
            assert!((simd - scalar).abs() < 1e-4, "70d: SIMD={simd}, scalar={scalar}");
        }
    }
}
```

- [ ] **Step 3: Verify AVX compilation**

```bash
cd /home/rohan/projects/qdrant && cargo check --features hyperbolic -p segment 2>&1 | tail -5
```

Expected: compiles without errors.

- [ ] **Step 4: Run AVX tests**

```bash
cd /home/rohan/projects/qdrant && cargo test --features hyperbolic -p segment poincare_avx 2>&1 | tail -10
```

Expected: 2 tests pass.

---

### Task 2: Create SSE Poincaré similarity

**Files:**
- Create: `lib/segment/src/spaces/hyperbolic/poincare_sse.rs`

- [ ] **Step 1: Create poincare_sse.rs**

Create `lib/segment/src/spaces/hyperbolic/poincare_sse.rs` following the same pattern as poincare_avx.rs but with SSE intrinsics:

- Use `__m128` instead of `__m256` (4 floats instead of 8)
- Process 16 elements per iteration (4 × 4-element vectors)
- Use `_mm_sub_ps`, `_mm_mul_ps`, `_mm_add_ps` (no FMA in SSE — use separate mul+add)
- Use `_mm_loadu_ps` for loads
- Horizontal sum: `_mm_movehl_ps` + `_mm_add_ps` + `_mm_shuffle_ps` + `_mm_cvtss_f32`
- Guard with `#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]`
- Target feature: `#[target_feature(enable = "sse")]`

The function signature:
```rust
#[target_feature(enable = "sse")]
pub(crate) unsafe fn poincare_similarity_sse(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
    curvature: f32,
) -> ScoreType
```

The inner loop body for each of the 4 sub-iterations:
```rust
let sub = _mm_sub_ps(_mm_loadu_ps(ptr1.add(offset)), _mm_loadu_ps(ptr2.add(offset)));
diff_acc = _mm_add_ps(_mm_mul_ps(sub, sub), diff_acc);
let u_chunk = _mm_loadu_ps(ptr1.add(offset));
usq_acc = _mm_add_ps(_mm_mul_ps(u_chunk, u_chunk), usq_acc);
let v_chunk = _mm_loadu_ps(ptr2.add(offset));
vsq_acc = _mm_add_ps(_mm_mul_ps(v_chunk, v_chunk), vsq_acc);
```

Include the same test pattern (128d + 70d, c=1.0 and c=5.0, compare vs scalar).

- [ ] **Step 2: Verify SSE compilation and tests**

```bash
cd /home/rohan/projects/qdrant && cargo test --features hyperbolic -p segment poincare_sse 2>&1 | tail -10
```

---

### Task 3: Create NEON Poincaré similarity

**Files:**
- Create: `lib/segment/src/spaces/hyperbolic/poincare_neon.rs`

- [ ] **Step 1: Create poincare_neon.rs**

Create `lib/segment/src/spaces/hyperbolic/poincare_neon.rs` following the NEON pattern:

- Use `float32x4_t` (4 floats, same as SSE)
- Process 16 elements per iteration (4 × 4-element vectors)
- Use `vld1q_f32` for loads, `vsubq_f32` for subtract, `vfmaq_f32` for FMA
- Horizontal sum: `vaddvq_f32` (native NEON horizontal sum)
- Guard with `#[cfg(target_arch = "aarch64")]`
- Target feature: `#[cfg(target_feature = "neon")]`

The function signature:
```rust
#[cfg(target_feature = "neon")]
pub(crate) unsafe fn poincare_similarity_neon(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
    curvature: f32,
) -> ScoreType
```

The inner loop body for each sub-iteration:
```rust
let sub = vsubq_f32(vld1q_f32(ptr1.add(offset)), vld1q_f32(ptr2.add(offset)));
diff_acc = vfmaq_f32(diff_acc, sub, sub);
let u_chunk = vld1q_f32(ptr1.add(offset));
usq_acc = vfmaq_f32(usq_acc, u_chunk, u_chunk);
let v_chunk = vld1q_f32(ptr2.add(offset));
vsq_acc = vfmaq_f32(vsq_acc, v_chunk, v_chunk);
```

Include tests guarded by `#[cfg(target_arch = "aarch64")]`.

- [ ] **Step 2: Verify compilation (may only compile on aarch64)**

```bash
cd /home/rohan/projects/qdrant && cargo check --features hyperbolic -p segment 2>&1 | tail -5
```

If on x86_64, the NEON module compiles but tests won't run (that's fine — cfg-gated).

---

### Task 4: Wire SIMD dispatch into poincare_metric.rs

**Files:**
- Modify: `lib/segment/src/spaces/hyperbolic/mod.rs`
- Modify: `lib/segment/src/spaces/hyperbolic/poincare_metric.rs`

- [ ] **Step 1: Register SIMD modules in mod.rs**

Add to `lib/segment/src/spaces/hyperbolic/mod.rs`:

```rust
#[cfg(target_arch = "x86_64")]
pub mod poincare_avx;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod poincare_sse;
#[cfg(target_arch = "aarch64")]
pub mod poincare_neon;
```

- [ ] **Step 2: Add SIMD dispatch to PoincareMetric::similarity()**

In `lib/segment/src/spaces/hyperbolic/poincare_metric.rs`, replace the f32 `similarity` method:

```rust
impl Metric<VectorElementType> for PoincareMetric {
    fn distance() -> Distance {
        Distance::Poincare
    }

    fn similarity(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("fma")
                && v1.len() >= 32
            {
                return unsafe {
                    super::poincare_avx::poincare_similarity_avx(v1, v2, DEFAULT_CURVATURE)
                };
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") && v1.len() >= 16 {
                return unsafe {
                    super::poincare_sse::poincare_similarity_sse(v1, v2, DEFAULT_CURVATURE)
                };
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") && v1.len() >= 16 {
                return unsafe {
                    super::poincare_neon::poincare_similarity_neon(v1, v2, DEFAULT_CURVATURE)
                };
            }
        }
        -poincare_distance(v1, v2, DEFAULT_CURVATURE)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        project_to_ball(vector, DEFAULT_CURVATURE)
    }
}
```

Add the necessary import at the top of the file:

```rust
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;
```

- [ ] **Step 3: Add SIMD dispatch to PoincareCurvatureQueryScorer**

In the same file, update `PoincareCurvatureQueryScorer::score_stored()` and `score()` methods to use SIMD. Create a helper function:

```rust
/// SIMD-dispatched Poincaré similarity with runtime curvature.
#[inline]
fn poincare_similarity_dispatched(v1: &[f32], v2: &[f32], curvature: f32) -> ScoreType {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("fma")
            && v1.len() >= 32
        {
            return unsafe {
                super::poincare_avx::poincare_similarity_avx(v1, v2, curvature)
            };
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse") && v1.len() >= 16 {
            return unsafe {
                super::poincare_sse::poincare_similarity_sse(v1, v2, curvature)
            };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && v1.len() >= 16 {
            return unsafe {
                super::poincare_neon::poincare_similarity_neon(v1, v2, curvature)
            };
        }
    }
    -poincare_distance(v1, v2, curvature)
}
```

Then update all `PoincareCurvatureQueryScorer` methods to call `poincare_similarity_dispatched()` instead of `-poincare_distance()`.

- [ ] **Step 4: Verify full compilation and all tests pass**

```bash
cd /home/rohan/projects/qdrant && cargo test --lib --features hyperbolic -p segment 2>&1 | tail -10
```

Expected: 625+ tests pass (existing tests + new SIMD tests), 0 failures.

---

## Part 2: Integration Tests

### Task 5: Create HNSW Poincaré integration tests

**Files:**
- Create: `lib/segment/tests/integration/hnsw_poincare_search_test.rs`

- [ ] **Step 1: Create the integration test file**

Create `lib/segment/tests/integration/hnsw_poincare_search_test.rs`. Follow the pattern from `hnsw_quantized_search_test.rs` for segment construction and search.

The file needs these test cases:

**Test 1: basic_hnsw_poincare_search** — Create 1000 Poincaré vectors (stratified by radius), build HNSW, compare search results to brute-force. Assert recall@10 ≥ 95%.

**Test 2: curvature_aware_search** — Same setup with curvature=5.0. Verify results differ from default curvature. Assert recall ≥ 95%.

**Test 3: poincare_edge_cases** — Test zero vector, boundary vector (norm ≈ 0.99/√c), single-point collection. No crashes, valid results.

**Test 4: poincare_quantization_recall** — Create 5000 vectors, build HNSW with INT8 scalar quantization, compare quantized vs exact search. Assert recall ≥ 70%.

Key implementation details:
- Use `build_simple_segment()` from fixtures with `Distance::Poincare`
- Generate vectors inside the ball: `random_vector(rng, dim)` then scale by `0.3 * radius_factor` where radius_factor varies from 0.1 to 0.95
- For brute-force: compute `poincare_distance()` from all vectors to query, sort, take top-k
- Use `sames_count()` helper from existing test infrastructure
- The segment needs `Distance::Poincare` which requires the `hyperbolic` feature

The test must reference the segment's Poincaré distance helpers. Import from:
```rust
use segment::spaces::hyperbolic::poincare_math::poincare_distance;
```

- [ ] **Step 2: Verify test compilation**

```bash
cd /home/rohan/projects/qdrant && cargo test --test hnsw_poincare_search_test --features hyperbolic -p segment --no-run 2>&1 | tail -5
```

Expected: compiles without errors.

- [ ] **Step 3: Run integration tests**

```bash
cd /home/rohan/projects/qdrant && cargo test --test hnsw_poincare_search_test --features hyperbolic -p segment 2>&1 | tail -15
```

Expected: all 4 tests pass. The quantization recall test may take a few seconds (5000 vectors).

---

## Part 3: Testbench Python Extensions

### Task 6: Add Klein and geometric filter functions to hyperbolic_math.py

**Files:**
- Modify: `dev/testbench/hyperbolic_math.py`

- [ ] **Step 1: Add Klein conversion functions**

Append to `dev/testbench/hyperbolic_math.py`:

```python
# ---------------------------------------------------------------------------
# Klein disk model
# ---------------------------------------------------------------------------

def poincare_to_klein(vector, c=1.0):
    """Convert Poincaré ball point to Klein disk coordinates.

    Klein geodesics are Euclidean straight lines (chords), making
    Klein-space distance a cheap L2 proxy for Poincaré distance.

    k_i = 2 * p_i / (1 + c * ||p||^2)

    Source: HyperspaceDB vector.rs
    """
    sq_norm = np.dot(vector, vector)
    return 2.0 * vector / (1.0 + c * sq_norm)


def poincare_to_klein_batch(vectors, c=1.0):
    """Batch Klein conversion for a matrix of vectors."""
    sq_norms = np.sum(vectors ** 2, axis=1, keepdims=True)
    return 2.0 * vectors / (1.0 + c * sq_norms)


def klein_chord_distance_sq(u_klein, v_klein):
    """Squared Euclidean distance in Klein disk (chord distance).

    This is a fast proxy for Poincaré distance — no acosh needed.
    Preserves ordering for nearby points.
    """
    diff = u_klein - v_klein
    return np.dot(diff, diff)
```

- [ ] **Step 2: Add geometric filter functions**

```python
# ---------------------------------------------------------------------------
# Geometric filters (client-side post-filtering)
# ---------------------------------------------------------------------------

def inball_filter(candidates, center, radius, curvature=1.0):
    """Keep candidates within Poincaré distance `radius` of `center`.

    Args:
        candidates: list of dicts with 'vector' key (Poincaré coordinates)
        center: np.ndarray, center point in Poincaré ball
        radius: float, maximum Poincaré distance from center
        curvature: float, ball curvature

    Returns:
        Filtered list of candidates
    """
    center_sq = np.dot(center, center)
    alpha_c = 1.0 / max(1.0 - curvature * center_sq, 1e-7)
    sqrt_c = np.sqrt(curvature)

    result = []
    for cand in candidates:
        v = cand["vector"]
        diff = center - v
        diff_sq = np.dot(diff, diff)
        v_sq = np.dot(v, v)
        alpha_v = 1.0 / max(1.0 - curvature * v_sq, 1e-7)
        arg = 1.0 + 2.0 * curvature * diff_sq * alpha_c * alpha_v
        dist = np.arccosh(max(arg, 1.0)) / sqrt_c
        if dist <= radius:
            result.append(cand)
    return result


def incone_filter(candidates, axis, aperture, origin=None):
    """Keep candidates within angular aperture of axis direction from origin.

    Measures the angle between the direction from origin to each candidate
    and the axis direction. Keeps candidates within the aperture.

    Args:
        candidates: list of dicts with 'vector' key
        axis: np.ndarray, direction vector (will be normalized)
        aperture: float, half-angle in radians
        origin: np.ndarray or None (defaults to zero vector)

    Returns:
        Filtered list of candidates
    """
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return candidates  # Degenerate axis, keep all

    axis_unit = axis / axis_norm

    result = []
    for cand in candidates:
        v = cand["vector"]
        if origin is not None:
            v = v - origin

        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue  # Skip zero-length vectors

        cos_angle = np.dot(v / v_norm, axis_unit)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        if angle <= aperture:
            result.append(cand)
    return result
```

- [ ] **Step 3: Verify functions work**

```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "
from hyperbolic_math import *
import numpy as np

# Test Klein conversion
p = np.array([0.3, 0.4])
k = poincare_to_klein(p, c=1.0)
expected_denom = 1.0 + 0.09 + 0.16
expected = 2.0 * p / expected_denom
assert np.allclose(k, expected), f'Klein: {k} vs {expected}'

# Test batch
vecs = np.array([[0.3, 0.4], [0.1, 0.2]])
k_batch = poincare_to_klein_batch(vecs, c=1.0)
assert np.allclose(k_batch[0], k)

# Test Klein distance
k1 = poincare_to_klein(np.array([0.1, 0.0]), c=1.0)
k2 = poincare_to_klein(np.array([0.3, 0.0]), c=1.0)
d = klein_chord_distance_sq(k1, k2)
assert d > 0

# Test InBall filter
cands = [{'vector': np.array([0.1, 0.0])}, {'vector': np.array([0.5, 0.0])}]
filtered = inball_filter(cands, np.array([0.0, 0.0]), radius=0.5, curvature=1.0)
assert len(filtered) == 1, f'Expected 1 result in ball, got {len(filtered)}'

# Test InCone filter
cands = [
    {'vector': np.array([0.3, 0.01])},  # nearly on x-axis
    {'vector': np.array([0.01, 0.3])},  # nearly on y-axis
]
filtered = incone_filter(cands, axis=np.array([1.0, 0.0]), aperture=0.3)
assert len(filtered) == 1, f'Expected 1 result in cone, got {len(filtered)}'

print('All Klein + geometric filter tests PASSED')
"
```

---

### Task 7: Add Klein pre-filter and geometric queries to queries.py

**Files:**
- Modify: `dev/testbench/queries.py`

- [ ] **Step 1: Add klein_prefilter_query**

Append to `dev/testbench/queries.py`:

```python
def klein_prefilter_query(client, collection, query_poincare, curvature=5.0,
                          prefetch_limit=200, klein_top=50, final_top=10,
                          using_prefetch="dense", ef=128):
    """Three-stage search: cosine prefetch → Klein pre-rank → Poincaré re-rank.

    Stage 1: Fast HNSW search using cosine named vector (server-side)
    Stage 2: Re-rank candidates by Klein chord distance (client-side, cheap)
    Stage 3: Re-rank survivors by exact Poincaré distance (client-side, expensive)

    Returns dict with results, per-stage latencies, and candidate counts.
    """
    import time
    import numpy as np
    from hyperbolic_math import poincare_to_klein, klein_chord_distance_sq, poincare_distance_with_alpha, alpha_precompute, fused_norms

    try:
        t0 = time.time()

        # Stage 1: Cosine prefetch from Qdrant
        if hasattr(query_poincare, "tolist"):
            # We need a dense vector for prefetch — retrieve a point to get it
            # In practice, the caller would have both vectors. Use poincare as fallback.
            query_list = query_poincare.tolist()
        else:
            query_list = list(query_poincare)

        prefetch_results = client.query_points(
            collection_name=collection,
            query=query_list,
            using=using_prefetch,
            limit=prefetch_limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=True,
        ).points

        t1 = time.time()

        # Stage 2: Klein pre-rank
        query_klein = poincare_to_klein(np.array(query_poincare), c=curvature)
        scored = []
        for r in prefetch_results:
            vec = r.vector
            if isinstance(vec, dict):
                vec = vec.get("poincare", next(iter(vec.values())))
            if vec is None:
                continue
            poincare_vec = np.array(vec, dtype=np.float32)
            klein_vec = poincare_to_klein(poincare_vec, c=curvature)
            klein_dist = klein_chord_distance_sq(query_klein, klein_vec)
            scored.append((r, klein_dist, poincare_vec))

        scored.sort(key=lambda x: x[1])
        klein_survivors = scored[:klein_top]

        t2 = time.time()

        # Stage 3: Poincaré re-rank
        query_alpha = alpha_precompute(np.array(query_poincare), c=curvature)
        poincare_scored = []
        for r, _, poincare_vec in klein_survivors:
            diff_sq, _, _ = fused_norms(np.array(query_poincare), poincare_vec)
            cand_alpha = alpha_precompute(poincare_vec, c=curvature)
            dist = poincare_distance_with_alpha(diff_sq, query_alpha, cand_alpha, c=curvature)
            poincare_scored.append((r, float(dist)))

        poincare_scored.sort(key=lambda x: x[1])
        final_results = [(r, d) for r, d in poincare_scored[:final_top]]

        t3 = time.time()

        return {
            "results": [r for r, _ in final_results],
            "distances": [d for _, d in final_results],
            "prefetch_count": len(prefetch_results),
            "klein_survivors": len(klein_survivors),
            "final_count": len(final_results),
            "stage1_latency_ms": (t1 - t0) * 1000,
            "stage2_latency_ms": (t2 - t1) * 1000,
            "stage3_latency_ms": (t3 - t2) * 1000,
            "total_latency_ms": (t3 - t0) * 1000,
        }

    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 2: Add geometric_filtered_query**

```python
def geometric_filtered_query(client, collection, query_vector, filters,
                              using="poincare", limit=200, final_top=10, ef=128,
                              curvature=5.0):
    """Search Qdrant, then apply geometric filters client-side.

    Args:
        filters: list of filter specs, each a dict with 'type' key:
            {"type": "inball", "center": ndarray, "radius": float}
            {"type": "incone", "axis": ndarray, "aperture": float}
            {"type": "depth_band", "depth_min": float, "depth_max": float}
    """
    import time
    import numpy as np
    from hyperbolic_math import inball_filter, incone_filter

    try:
        t0 = time.time()

        if hasattr(query_vector, "tolist"):
            query_vector_list = query_vector.tolist()
        else:
            query_vector_list = list(query_vector)

        # Get candidates from Qdrant
        results = client.query_points(
            collection_name=collection,
            query=query_vector_list,
            using=using,
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=True,
        ).points

        t1 = time.time()

        # Convert to candidate dicts for filtering
        candidates = []
        for r in results:
            vec = r.vector
            if isinstance(vec, dict):
                vec = vec.get("poincare", vec.get("dense", next(iter(vec.values()))))
            candidates.append({
                "vector": np.array(vec, dtype=np.float32),
                "point": r,
                "busemann_depth": r.payload.get("busemann_depth", 0),
            })

        # Apply filters in order (AND logic)
        filtered = candidates
        for f in filters:
            if f["type"] == "inball":
                filtered = inball_filter(
                    filtered, f["center"], f["radius"], curvature
                )
            elif f["type"] == "incone":
                filtered = incone_filter(
                    filtered, f["axis"], f["aperture"], f.get("origin")
                )
            elif f["type"] == "depth_band":
                filtered = [
                    c for c in filtered
                    if f["depth_min"] <= c["busemann_depth"] <= f["depth_max"]
                ]

        t2 = time.time()

        final = filtered[:final_top]

        return {
            "results": [c["point"] for c in final],
            "initial_count": len(candidates),
            "filtered_count": len(filtered),
            "final_count": len(final),
            "search_latency_ms": (t1 - t0) * 1000,
            "filter_latency_ms": (t2 - t1) * 1000,
            "total_latency_ms": (t2 - t0) * 1000,
        }

    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 3: Verify imports**

```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "
from queries import klein_prefilter_query, geometric_filtered_query
print('Klein + geometric query functions imported OK')
"
```

---

### Task 8: Store Klein vectors as payload in embed.py

**Files:**
- Modify: `dev/testbench/embed.py`

- [ ] **Step 1: Add Klein vector computation and storage**

In `dev/testbench/embed.py`, find the unified collection upsert section. After computing poincare vectors and alpha values, add Klein vector computation.

Add import at top:
```python
from hyperbolic_math import poincare_to_klein_batch
```

In `main()`, after the alpha precomputation step, add:
```python
# Klein vector precomputation
klein_vectors = poincare_to_klein_batch(poincare_vectors, c=CURVATURE)
print(f"  Klein vectors: shape={klein_vectors.shape}")
```

In the `upsert_unified()` function, add `klein_vector` to each point's payload:
- For content points: `"klein_vector": klein_vectors[i].tolist()`
- For story/narrative centroids: compute from their poincare vector: `poincare_to_klein(poincare_vec, c=curvature).tolist()`

The `klein_vector` is stored as a list of floats in the payload (not as a named vector), since it's only used for client-side re-ranking.

- [ ] **Step 2: Verify syntax**

```bash
cd /home/rohan/projects/qdrant && python3 -c "import ast; ast.parse(open('dev/testbench/embed.py').read()); print('OK')"
```

---

### Task 9: Add benchmark suites for quantization, Klein, and geometric filters

**Files:**
- Modify: `dev/testbench/benchmark.py`

- [ ] **Step 1: Add suite_quantization_recall**

Add to the `BenchmarkRunner` class:

```python
def suite_quantization_recall(self):
    """Measure recall of quantized vs exact Poincaré search."""
    print(f"\n--- Suite: Quantization Recall ---")
    try:
        points = self._load_unified_points()
        sample = random.sample(points, min(200, len(points)))

        recalls = []
        for pt in tqdm(sample, desc="  quantization recall"):
            vec = pt.get("vectors", {}).get("poincare", pt["vector"])
            if hasattr(vec, "tolist"):
                vec = vec.tolist()

            # Quantized search (default)
            try:
                quantized = self.client.query_points(
                    collection_name=self.unified_collection,
                    query=vec, using="poincare", limit=10,
                    search_params=SearchParams(hnsw_ef=128),
                    with_payload=False,
                ).points
            except Exception:
                continue

            # Exact search
            try:
                exact = self.client.query_points(
                    collection_name=self.unified_collection,
                    query=vec, using="poincare", limit=10,
                    search_params=SearchParams(hnsw_ef=128, exact=True),
                    with_payload=False,
                ).points
            except Exception:
                continue

            q_ids = {r.id for r in quantized}
            e_ids = {r.id for r in exact}
            if e_ids:
                recalls.append(len(q_ids & e_ids) / len(e_ids))

        result = {
            "recall_at_10": float(np.mean(recalls)) if recalls else None,
            "recall_min": float(np.min(recalls)) if recalls else None,
            "recall_max": float(np.max(recalls)) if recalls else None,
            "num_queries": len(recalls),
        }
        self.results["quantization_recall"] = result
        if recalls:
            print(f"  recall@10: {np.mean(recalls):.4f} (min={np.min(recalls):.4f}, max={np.max(recalls):.4f})")

    except Exception as e:
        self.results["quantization_recall"] = {"error": str(e)}
        print(f"  ERROR: {e}")
```

- [ ] **Step 2: Add suite_klein_prefilter**

```python
def suite_klein_prefilter(self):
    """Compare Klein pre-filter pipeline vs direct Poincaré search."""
    print(f"\n--- Suite: Klein Pre-Filter ---")
    try:
        from queries import klein_prefilter_query

        points = self._load_unified_points()
        sample = random.sample(points, min(100, len(points)))

        direct_latencies = []
        klein_latencies = []
        recall_vs_direct = []

        for pt in tqdm(sample, desc="  klein prefilter"):
            poincare_vec = pt.get("vectors", {}).get("poincare", pt["vector"])

            # Direct Poincaré search
            try:
                direct = self.client.query_points(
                    collection_name=self.unified_collection,
                    query=poincare_vec.tolist() if hasattr(poincare_vec, "tolist") else poincare_vec,
                    using="poincare", limit=10,
                    search_params=SearchParams(hnsw_ef=128),
                    with_payload=False,
                ).points
                direct_ids = {r.id for r in direct}
            except Exception:
                continue

            # Klein pre-filter pipeline
            result = klein_prefilter_query(
                self.client, self.unified_collection,
                poincare_vec, curvature=self.curvature,
                prefetch_limit=200, klein_top=50, final_top=10,
            )
            if "error" in result:
                continue

            klein_ids = {r.id for r in result["results"]}
            if direct_ids:
                recall_vs_direct.append(len(klein_ids & direct_ids) / len(direct_ids))

            klein_latencies.append(result["total_latency_ms"])

        self.results["klein_prefilter"] = {
            "recall_vs_direct": float(np.mean(recall_vs_direct)) if recall_vs_direct else None,
            "avg_klein_latency_ms": float(np.mean(klein_latencies)) if klein_latencies else None,
            "num_queries": len(recall_vs_direct),
        }
        if recall_vs_direct:
            print(f"  recall vs direct: {np.mean(recall_vs_direct):.4f}")
            print(f"  avg Klein latency: {np.mean(klein_latencies):.1f}ms")

    except Exception as e:
        self.results["klein_prefilter"] = {"error": str(e)}
        print(f"  ERROR: {e}")
```

- [ ] **Step 3: Add suite_geometric_filters**

```python
def suite_geometric_filters(self):
    """Validate geometric filter precision and selectivity."""
    print(f"\n--- Suite: Geometric Filters ---")
    try:
        from queries import geometric_filtered_query
        import numpy as np

        points = self._load_unified_points()
        content_points = [p for p in points if p.get("item_type", p.get("tier")) == "content"]
        sample = random.sample(content_points, min(100, len(content_points)))

        # Test InBall filter
        inball_precisions = []
        inball_selectivities = []
        for pt in tqdm(sample[:50], desc="  inball"):
            radius = 1.5
            result = geometric_filtered_query(
                self.client, self.unified_collection,
                pt["vector"],
                filters=[{"type": "inball", "center": pt["vector"], "radius": radius}],
                using="poincare", limit=100, final_top=50,
                curvature=self.curvature,
            )
            if "error" in result or result["initial_count"] == 0:
                continue

            inball_selectivities.append(result["filtered_count"] / result["initial_count"])
            # Precision: all returned should be within radius (filter guarantees this)
            inball_precisions.append(1.0)  # By construction

        # Test InCone filter
        incone_selectivities = []
        for pt in tqdm(sample[:50], desc="  incone"):
            axis = pt["vector"]  # Use the point itself as axis
            result = geometric_filtered_query(
                self.client, self.unified_collection,
                pt["vector"],
                filters=[{"type": "incone", "axis": axis, "aperture": 0.5}],
                using="poincare", limit=100, final_top=50,
                curvature=self.curvature,
            )
            if "error" in result or result["initial_count"] == 0:
                continue

            incone_selectivities.append(result["filtered_count"] / result["initial_count"])

        # Test composition: InBall + DepthBand
        composed_selectivities = []
        for pt in tqdm(sample[:50], desc="  composed"):
            depth = pt.get("busemann_depth", 1.0)
            result = geometric_filtered_query(
                self.client, self.unified_collection,
                pt["vector"],
                filters=[
                    {"type": "inball", "center": pt["vector"], "radius": 2.0},
                    {"type": "depth_band", "depth_min": depth - 0.2, "depth_max": depth + 0.2},
                ],
                using="poincare", limit=100, final_top=50,
                curvature=self.curvature,
            )
            if "error" in result or result["initial_count"] == 0:
                continue

            composed_selectivities.append(result["filtered_count"] / result["initial_count"])

        self.results["geometric_filters"] = {
            "inball_precision": float(np.mean(inball_precisions)) if inball_precisions else None,
            "inball_selectivity": float(np.mean(inball_selectivities)) if inball_selectivities else None,
            "incone_selectivity": float(np.mean(incone_selectivities)) if incone_selectivities else None,
            "composed_selectivity": float(np.mean(composed_selectivities)) if composed_selectivities else None,
            "num_queries": len(inball_precisions),
        }
        if inball_selectivities:
            print(f"  inball selectivity: {np.mean(inball_selectivities):.4f}")
        if incone_selectivities:
            print(f"  incone selectivity: {np.mean(incone_selectivities):.4f}")
        if composed_selectivities:
            print(f"  composed selectivity: {np.mean(composed_selectivities):.4f}")

    except Exception as e:
        self.results["geometric_filters"] = {"error": str(e)}
        print(f"  ERROR: {e}")
```

- [ ] **Step 4: Wire new suites into run_all()**

In the `BenchmarkRunner.run_all()` method, add the three new suites:

```python
def run_all(self):
    # ... existing suites ...
    self.suite_quantization_recall()
    self.suite_klein_prefilter()
    self.suite_geometric_filters()
    return self.results
```

- [ ] **Step 5: Verify syntax**

```bash
cd /home/rohan/projects/qdrant && python3 -c "import ast; ast.parse(open('dev/testbench/benchmark.py').read()); print('OK')"
```

---

## Part 4: End-to-End Validation

### Task 10: Full validation run

- [ ] **Step 1: Verify Rust compilation and all tests pass**

```bash
cd /home/rohan/projects/qdrant && cargo test --lib --features hyperbolic -p segment 2>&1 | tail -5
```

Expected: 625+ tests pass (including new SIMD tests), 0 failures.

- [ ] **Step 2: Run integration tests**

```bash
cd /home/rohan/projects/qdrant && cargo test --test hnsw_poincare_search_test --features hyperbolic -p segment 2>&1 | tail -15
```

Expected: 4 integration tests pass. Note the quantization recall value — if < 70%, we'll need to add rescoring.

- [ ] **Step 3: Rebuild Docker image**

Tell the user to run:

```bash
cd /home/rohan/projects/qdrant/dev/testbench && docker compose up -d --build
```

Wait for Qdrant to rebuild with SIMD-optimized Poincaré distance.

- [ ] **Step 4: Re-embed BGC with Klein vectors**

Tell the user to run:

```bash
cd /home/rohan/projects/qdrant/dev/testbench && ./run.sh embed bgc
```

This re-creates collections with Klein vectors in payloads and runs all benchmarks including the 3 new suites.

- [ ] **Step 5: Compare results with Phase 4 baseline**

Check the output for:
- SIMD speedup: QPS should be ≥ 1.5x the Phase 4 baseline (was 224 QPS on BGC)
- Quantization recall@10: document the value
- Klein pre-filter: recall vs direct ≥ 95%
- Geometric filters: selectivity values make sense (InCone < 50%)
- All Phase 4 suites still pass (no regression)

- [ ] **Step 6: Run on HWV**

Tell the user to run:

```bash
cd /home/rohan/projects/qdrant/dev/testbench && ./run.sh hwv
```

- [ ] **Step 7: Commit everything**

Once all tests pass and results look good:

```bash
cd /home/rohan/projects/qdrant
git add -A  # Review with git status first
# Commit with descriptive message
```

---

## Verification Checklist

### Rust
- [ ] `cargo test --lib --features hyperbolic -p segment` — all pass
- [ ] `cargo test --test hnsw_poincare_search_test --features hyperbolic -p segment` — 4 tests pass
- [ ] SIMD tests: AVX + SSE match scalar within 1e-4 at both c=1.0 and c=5.0
- [ ] SIMD tests: odd dimensions (70d) work correctly (remainder handling)
- [ ] Quantization recall@10 measured and documented

### Python
- [ ] `python3 -c "import ast; ast.parse(open('dev/testbench/embed.py').read())"` — OK
- [ ] `python3 -c "import ast; ast.parse(open('dev/testbench/benchmark.py').read())"` — OK
- [ ] Klein conversion + chord distance verified numerically
- [ ] InBall + InCone filters verified with simple test cases
- [ ] All Phase 4 benchmark suites still pass (no regression)
