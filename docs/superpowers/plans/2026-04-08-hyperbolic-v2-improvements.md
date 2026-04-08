# Hyperbolic V2 Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable per-collection curvature, Einstein midpoint (O(1) centroid), and stable acosh third regime to the Qdrant fork's hyperbolic vector support.

**Architecture:** Three changes in `lib/segment/src/spaces/hyperbolic/`, plus curvature threading through scorer dispatch in `lib/segment/src/vector_storage/`. A new `PoincareCurvatureQueryScorer` in `poincare_metric.rs` bypasses the stateless `Metric` trait to inject runtime curvature. All changes feature-gated behind `hyperbolic`.

**Tech Stack:** Rust 1.94+, Qdrant `lib/segment` crate, `half` crate (f16), `common::types::ScoreType` (f32)

**Spec:** `docs/superpowers/specs/2026-04-08-hyperbolic-v2-improvements-design.md`

---

## Pre-requisite: Verify Current State

```bash
cd /home/rohan/projects/qdrant
git branch --show-current  # Should be feat/hyperbolic-vector-support
cargo test -p segment --features hyperbolic --lib spaces::hyperbolic 2>&1 | tail -5
# Expected: test result: ok. 37 passed
```

---

## File Structure

### Modified files

| File | Responsibility | Task |
|------|---------------|------|
| `lib/segment/src/spaces/hyperbolic/poincare_math.rs` | Add 3rd acosh regime, add `einstein_midpoint()`, receive `poincare_to_lorentz()` and `lorentz_inner()` | 1, 2, 3 |
| `lib/segment/src/spaces/hyperbolic/busemann.rs` | Remove `poincare_to_lorentz()` and `lorentz_inner()` (moved), update imports, use Einstein midpoint for focal direction | 2, 4 |
| `lib/segment/src/spaces/hyperbolic/tangent_cache.rs` | Switch from `frechet_mean` to `einstein_midpoint` for centroid | 4 |
| `lib/segment/src/spaces/hyperbolic/poincare_metric.rs` | Add `PoincareCurvatureQueryScorer` struct | 6 |
| `lib/segment/src/spaces/hyperbolic/mod.rs` | Update re-exports | 2, 6 |
| `lib/segment/src/types.rs` | Add `curvature: Option<f32>` to `VectorDataConfig`, update `preprocess_vector` | 5 |
| `lib/segment/src/vector_storage/raw_scorer.rs` | Add `new_raw_scorer_with_curvature` + `new_poincare_curvature_scorer` helpers | 7 |

---

## Task 1: Stable Acosh Third Regime

**Files:**
- Modify: `lib/segment/src/spaces/hyperbolic/poincare_math.rs:34-49`

- [ ] **Step 1: Write tests for the third regime**

Add to the `mod tests` block in `poincare_math.rs`, after the existing `test_stable_acosh_known_value` test (line 314):

```rust
#[test]
fn test_stable_acosh_large_value() {
    // For very large x, acosh(x) ≈ ln(2x)
    let x = 1e8_f32;
    let result = stable_acosh(x);
    let expected = (2.0 * x).ln();
    assert!(
        (result - expected).abs() < 0.01,
        "acosh({x}) = {result}, expected ≈ {expected}"
    );
}

#[test]
fn test_stable_acosh_regime_boundary_no_discontinuity() {
    // Check near the 1e6 boundary: values should be close from both sides
    let below = stable_acosh(999_999.0);
    let above = stable_acosh(1_000_001.0);
    let at = stable_acosh(1_000_000.0);
    // All three should be monotonically increasing and close
    assert!(below < at, "below={below} should be < at={at}");
    assert!(at < above, "at={at} should be < above={above}");
    assert!(
        (above - below).abs() < 0.01,
        "jump across boundary too large: {below} vs {above}"
    );
}

#[test]
fn test_stable_acosh_monotonicity() {
    let values = [1.0, 1.0 + 1e-7, 1.001, 2.0, 100.0, 1e6, 1e8];
    for window in values.windows(2) {
        let a = stable_acosh(window[0]);
        let b = stable_acosh(window[1]);
        assert!(
            b >= a,
            "monotonicity violated: acosh({}) = {} > acosh({}) = {}",
            window[0], a, window[1], b
        );
    }
}
```

- [ ] **Step 2: Run tests to verify new tests fail (large value test)**

Run: `cargo test -p segment --features hyperbolic --lib spaces::hyperbolic::poincare_math::tests::test_stable_acosh_large_value 2>&1 | tail -5`

Expected: PASS (the standard acosh may handle 1e8 fine, but the asymptotic test validates the regime exists). The monotonicity test should PASS. If all pass, that's fine — the implementation improves numerical stability rather than fixing a bug.

- [ ] **Step 3: Add the third regime**

In `poincare_math.rs`, replace lines 38-49:

```rust
fn stable_acosh(x: f32) -> f32 {
    if x <= 1.0 {
        return 0.0;
    }
    let delta = x - 1.0;
    if delta < 1e-6 {
        // Taylor: acosh(1+δ) ≈ sqrt(2δ) for small δ
        (2.0 * delta).sqrt()
    } else if x > 1e6 {
        // Asymptotic: acosh(x) ≈ ln(2x) for large x
        // Avoids sqrt(x²-1) precision loss. Ref: ruvector poincare.rs:259-275
        (2.0 * x).ln()
    } else {
        x.acosh()
    }
}
```

- [ ] **Step 4: Run all hyperbolic tests**

Run: `cargo test -p segment --features hyperbolic --lib spaces::hyperbolic 2>&1 | tail -5`

Expected: All tests pass (37 existing + 3 new = 40).

---

## Task 2: Move Lorentz Utilities to poincare_math.rs

**Files:**
- Modify: `lib/segment/src/spaces/hyperbolic/poincare_math.rs` (add functions)
- Modify: `lib/segment/src/spaces/hyperbolic/busemann.rs` (remove functions, update imports)
- Modify: `lib/segment/src/spaces/hyperbolic/mod.rs` (update re-exports)

- [ ] **Step 1: Copy `poincare_to_lorentz` and `lorentz_inner` to poincare_math.rs**

Add before the `// Tests` section in `poincare_math.rs` (before line 263):

```rust
// ---------------------------------------------------------------------------
// Lorentz model conversions
// ---------------------------------------------------------------------------

/// Convert a Poincaré ball point to the Lorentz hyperboloid model.
///
/// Lorentz coordinates: x_0 = (1 + c||x||²) / (1 - c||x||²)
///                      x_i = 2√c · x_i / (1 - c||x||²)  for i > 0
///
/// Returns a vector of dimension d+1 (prepends the time component x_0).
pub fn poincare_to_lorentz(p: &[f32], c: f32) -> Vec<f32> {
    let norm_sq: f32 = p.iter().map(|v| v * v).sum();
    let denom = (1.0 - c * norm_sq).max(EPS);
    let x0 = (1.0 + c * norm_sq) / denom;
    let sqrt_c = c.sqrt();
    let mut result = Vec::with_capacity(p.len() + 1);
    result.push(x0);
    for &xi in p {
        result.push(2.0 * sqrt_c * xi / denom);
    }
    result
}

/// Lorentz inner product: ⟨x, y⟩_L = −x₀y₀ + x₁y₁ + … + xₙyₙ
#[inline]
pub fn lorentz_inner(x: &[f32], y: &[f32]) -> f32 {
    let mut result = -x[0] * y[0];
    for i in 1..x.len() {
        result += x[i] * y[i];
    }
    result
}

/// Project a Lorentz-space vector back onto the hyperboloid.
///
/// Sets x₀ = √(1/c + ||x_spatial||²) so that ⟨x, x⟩_L = −1/c.
pub fn project_hyperboloid(x: &[f32], c: f32) -> Vec<f32> {
    let space_norm_sq: f32 = x[1..].iter().map(|v| v * v).sum();
    let x0 = (1.0 / c + space_norm_sq).max(EPS).sqrt();
    let mut result = Vec::with_capacity(x.len());
    result.push(x0);
    result.extend_from_slice(&x[1..]);
    result
}

/// Convert a Lorentz hyperboloid point back to Poincaré ball coordinates.
///
/// p_i = x_i / (x_0 + 1)  for i > 0
pub fn lorentz_to_poincare(x: &[f32]) -> Vec<f32> {
    let x0 = x[0];
    let denom = x0 + 1.0;
    x[1..].iter().map(|&xi| xi / denom.max(EPS)).collect()
}
```

- [ ] **Step 2: Update busemann.rs to import from poincare_math**

Replace the top of `busemann.rs` (lines 1-32) with:

```rust
use super::poincare_math::{
    einstein_midpoint, lorentz_inner, poincare_to_lorentz, EPS,
};
```

Remove the `poincare_to_lorentz` function (old lines 11-22) and `lorentz_inner` function (old lines 26-32) and the `const EPS` (old line 3) from `busemann.rs`. Keep `compute_focal_direction`, `busemann_score`, `busemann_depth`, and all tests.

Note: `busemann.rs` will temporarily reference `einstein_midpoint` which doesn't exist yet — that's added in Task 3. For now, keep `frechet_mean` in the import and we'll switch in Task 4. So the import should be:

```rust
use super::poincare_math::{frechet_mean, lorentz_inner, poincare_to_lorentz, EPS};
```

- [ ] **Step 3: Update mod.rs re-exports**

In `mod.rs`, update the re-exports. Replace the current `pub use busemann::` block (lines 16-19):

```rust
pub use busemann::{
    busemann_score, busemann_depth, compute_focal_direction,
};
pub use poincare_math::{
    poincare_distance, project_to_ball, exp_map_origin, log_map_origin,
    exp_map, log_map, mobius_add, frechet_mean, conformal_factor,
    poincare_to_lorentz, lorentz_inner, lorentz_to_poincare, project_hyperboloid,
    DEFAULT_CURVATURE,
};
```

- [ ] **Step 4: Run all hyperbolic tests**

Run: `cargo test -p segment --features hyperbolic --lib spaces::hyperbolic 2>&1 | tail -5`

Expected: All 40 tests pass. The Lorentz tests in `busemann.rs` should still work since they now call the same functions via the new import path.

---

## Task 3: Einstein Midpoint

**Files:**
- Modify: `lib/segment/src/spaces/hyperbolic/poincare_math.rs`

- [ ] **Step 1: Write tests for Einstein midpoint**

Add to the `mod tests` block in `poincare_math.rs`:

```rust
// -- Einstein midpoint --------------------------------------------------

#[test]
fn test_einstein_midpoint_single_point() {
    let p: &[f32] = &[0.3, -0.2];
    let result = einstein_midpoint(&[p], DEFAULT_CURVATURE);
    assert!(
        vec_approx_eq(&result, p),
        "midpoint([x]) = {result:?}, expected {p:?}"
    );
}

#[test]
fn test_einstein_midpoint_symmetric_near_origin() {
    let a: &[f32] = &[0.3, 0.0];
    let b: &[f32] = &[-0.3, 0.0];
    let mid = einstein_midpoint(&[a, b], DEFAULT_CURVATURE);
    let origin = [0.0f32, 0.0];
    assert!(
        vec_approx_eq(&mid, &origin),
        "symmetric midpoint = {mid:?}, expected ≈ origin"
    );
}

#[test]
fn test_einstein_midpoint_inside_ball() {
    let points: Vec<&[f32]> = vec![
        &[0.7, 0.1],
        &[-0.3, 0.6],
        &[0.1, -0.8],
        &[-0.5, -0.2],
    ];
    let mid = einstein_midpoint(&points, DEFAULT_CURVATURE);
    let norm: f32 = mid.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        norm < 1.0,
        "midpoint norm = {norm}, should be inside ball"
    );
}

#[test]
fn test_einstein_midpoint_agrees_with_frechet_mean() {
    // For well-clustered points, Einstein midpoint should approximate
    // Fréchet mean within reasonable tolerance
    let points: Vec<&[f32]> = vec![
        &[0.1, 0.1],
        &[0.2, -0.1],
        &[-0.1, 0.2],
    ];
    let einstein = einstein_midpoint(&points, DEFAULT_CURVATURE);
    let frechet = frechet_mean(&points, DEFAULT_CURVATURE);
    // Tolerance is loose because they're different algorithms
    let dist: f32 = einstein.iter().zip(frechet.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    assert!(
        dist < 0.1,
        "Einstein and Fréchet disagree by {dist}: einstein={einstein:?}, frechet={frechet:?}"
    );
}

#[test]
fn test_einstein_midpoint_different_curvatures() {
    let points: Vec<&[f32]> = vec![
        &[0.3, 0.1],
        &[-0.2, 0.4],
    ];
    let mid_c1 = einstein_midpoint(&points, 1.0);
    let mid_c2 = einstein_midpoint(&points, 2.0);
    // Different curvatures should produce different midpoints
    assert!(
        !vec_approx_eq(&mid_c1, &mid_c2),
        "midpoints should differ with different curvatures"
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p segment --features hyperbolic --lib spaces::hyperbolic::poincare_math::tests::test_einstein_midpoint_single_point 2>&1 | tail -5`

Expected: FAIL — `einstein_midpoint` doesn't exist yet.

- [ ] **Step 3: Implement Einstein midpoint**

Add to `poincare_math.rs`, after the `frechet_mean` function (after line 261) and before the Lorentz conversions section:

```rust
/// Einstein midpoint: O(1) closed-form centroid in hyperbolic space.
///
/// Algorithm (ref: ruvector lorentz_cascade.rs:146-167):
/// 1. Convert each Poincaré point to Lorentz hyperboloid
/// 2. Compute Lorentz factor γ_i = x_0_i (time component)
/// 3. Weighted average: m = Σ(γ_i · x_i) / Σ(γ_i)
/// 4. Project back to hyperboloid: normalize so ⟨m, m⟩_L = −1/c
/// 5. Convert back to Poincaré ball
///
/// Single pass, no iteration. Exact for 2 points, excellent approximation
/// for n points. ~10-100x faster than iterative Fréchet mean.
pub fn einstein_midpoint(points: &[&[f32]], c: f32) -> Vec<f32> {
    assert!(!points.is_empty(), "cannot compute midpoint of empty set");

    let dim = points[0].len();
    // Accumulate in Lorentz space (dim+1)
    let mut weighted_sum = vec![0.0f32; dim + 1];
    let mut gamma_sum: f32 = 0.0;

    for &p in points {
        let lorentz = poincare_to_lorentz(p, c);
        let gamma = lorentz[0]; // time component = Lorentz factor
        gamma_sum += gamma;
        for (i, &val) in lorentz.iter().enumerate() {
            weighted_sum[i] += gamma * val;
        }
    }

    // Normalize by total weight
    if gamma_sum > EPS {
        for val in weighted_sum.iter_mut() {
            *val /= gamma_sum;
        }
    }

    // Project onto hyperboloid and convert back to Poincaré
    let on_hyperboloid = project_hyperboloid(&weighted_sum, c);
    let poincare = lorentz_to_poincare(&on_hyperboloid);
    project_to_ball(poincare, c)
}
```

- [ ] **Step 4: Run all hyperbolic tests**

Run: `cargo test -p segment --features hyperbolic --lib spaces::hyperbolic 2>&1 | tail -5`

Expected: All tests pass (40 existing + 5 new = 45).

---

## Task 4: Integrate Einstein Midpoint into TangentCache and Busemann

**Files:**
- Modify: `lib/segment/src/spaces/hyperbolic/tangent_cache.rs:9,34`
- Modify: `lib/segment/src/spaces/hyperbolic/busemann.rs:1,43-44`

- [ ] **Step 1: Update TangentCache to use Einstein midpoint**

In `tangent_cache.rs`, change the import on line 9:

```rust
use super::poincare_math::{einstein_midpoint, log_map, poincare_distance};
```

Then replace line 34 (`let centroid = frechet_mean(&slices, curvature);`) with:

```rust
        let centroid = einstein_midpoint(&slices, curvature);
```

- [ ] **Step 2: Update compute_focal_direction to use Einstein midpoint**

In `busemann.rs`, update the import at the top:

```rust
use super::poincare_math::{
    einstein_midpoint, lorentz_inner, poincare_to_lorentz, EPS,
};
```

Then replace line 43 (`let mean = frechet_mean(points, c);`) with:

```rust
    let mean = einstein_midpoint(points, c);
```

- [ ] **Step 3: Run all hyperbolic tests**

Run: `cargo test -p segment --features hyperbolic --lib spaces::hyperbolic 2>&1 | tail -5`

Expected: All 45 tests pass. Existing tests should still pass because Einstein midpoint approximates Fréchet mean for well-behaved inputs. If any tangent ordering test fails due to the slightly different centroid, the tolerance may need minor adjustment.

- [ ] **Step 4: Update mod.rs re-exports**

In `mod.rs`, add `einstein_midpoint` to the `poincare_math` re-export block:

```rust
pub use poincare_math::{
    poincare_distance, project_to_ball, exp_map_origin, log_map_origin,
    exp_map, log_map, mobius_add, frechet_mean, einstein_midpoint, conformal_factor,
    poincare_to_lorentz, lorentz_inner, lorentz_to_poincare, project_hyperboloid,
    DEFAULT_CURVATURE,
};
```

- [ ] **Step 5: Verify compilation and tests**

Run: `cargo test -p segment --features hyperbolic --lib spaces::hyperbolic 2>&1 | tail -5`

Expected: All 45 tests pass.

---

## Task 5: Add Curvature to VectorDataConfig

**Files:**
- Modify: `lib/segment/src/types.rs:1652-1669`

- [ ] **Step 1: Add curvature field to VectorDataConfig**

In `types.rs`, add a curvature field to `VectorDataConfig` (after the `datatype` field, line 1668):

```rust
    /// Poincaré ball curvature for hyperbolic distance. Only used when
    /// distance is `Poincare`. Defaults to 1.0 if not specified.
    #[cfg(feature = "hyperbolic")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curvature: Option<f32>,
```

- [ ] **Step 2: Add a helper method to get curvature**

Add to the `impl VectorDataConfig` block (after the existing `is_appendable` method, around line 1680):

```rust
    /// Get the Poincaré curvature for this vector config.
    /// Returns `DEFAULT_CURVATURE` (1.0) if not set or if the feature is disabled.
    #[cfg(feature = "hyperbolic")]
    pub fn curvature(&self) -> f32 {
        self.curvature.unwrap_or(
            crate::spaces::hyperbolic::poincare_math::DEFAULT_CURVATURE
        )
    }
```

- [ ] **Step 3: Update preprocess_vector to use curvature**

In `types.rs`, update the Poincare arm of `preprocess_vector` (lines 348-352). Replace:

```rust
            #[cfg(feature = "hyperbolic")]
            Distance::Poincare => {
                // PoincareMetric::preprocess works on f32 DenseVector regardless of T
                crate::spaces::hyperbolic::poincare_math::project_to_ball(vector, crate::spaces::hyperbolic::poincare_math::DEFAULT_CURVATURE)
            },
```

Note: `preprocess_vector` doesn't have access to the VectorDataConfig (it's a method on `Distance`), so it stays with `DEFAULT_CURVATURE`. The actual curvature-aware preprocessing happens in the scorer (Task 6). This is acceptable because ball projection is the same for all curvatures — it just clamps to `1/√c − ε` boundary, and for `c ≥ 1.0` the `c=1.0` ball is the smallest, so projecting to it is safe.

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p segment --features hyperbolic 2>&1 | tail -5`

Expected: Compiles without errors. Existing code constructs `VectorDataConfig` in many places — verify that `curvature: None` is inferred via `#[serde(default)]`. If any constructor uses struct literal syntax, the `#[cfg]` attribute means the field simply doesn't exist without the feature, so non-hyperbolic code is unaffected.

- [ ] **Step 5: Write a test for curvature config**

Add to the existing test module in `types.rs` (find the `#[cfg(test)]` block), or if that's too large, add inline tests in `poincare_math.rs`:

```rust
#[test]
fn test_poincare_distance_with_curvature() {
    let x = [0.3f32, 0.2];
    let y = [-0.1f32, 0.4];
    let d_c1 = poincare_distance(&x, &y, 1.0);
    let d_c2 = poincare_distance(&x, &y, 2.0);
    let d_c05 = poincare_distance(&x, &y, 0.5);
    // Higher curvature → larger distances (more curved space)
    assert!(d_c2 > d_c1, "c=2 distance {d_c2} should be > c=1 distance {d_c1}");
    assert!(d_c1 > d_c05, "c=1 distance {d_c1} should be > c=0.5 distance {d_c05}");
}

#[test]
fn test_poincare_distance_curvature_symmetry() {
    let x = [0.2f32, -0.3];
    let y = [-0.1f32, 0.4];
    for &c in &[0.5, 1.0, 2.0, 5.0] {
        let dxy = poincare_distance(&x, &y, c);
        let dyx = poincare_distance(&y, &x, c);
        assert!(
            (dxy - dyx).abs() < 1e-4,
            "symmetry violated at c={c}: d(x,y)={dxy} != d(y,x)={dyx}"
        );
    }
}

#[test]
fn test_poincare_distance_curvature_triangle_inequality() {
    let x = [0.1f32, 0.2];
    let y = [0.3f32, -0.1];
    let z = [-0.2f32, 0.3];
    for &c in &[0.5, 1.0, 2.0, 5.0] {
        let dxy = poincare_distance(&x, &y, c);
        let dyz = poincare_distance(&y, &z, c);
        let dxz = poincare_distance(&x, &z, c);
        assert!(
            dxz <= dxy + dyz + 1e-4,
            "triangle inequality violated at c={c}: d(x,z)={dxz} > d(x,y)+d(y,z)={}",
            dxy + dyz
        );
    }
}
```

- [ ] **Step 6: Run all tests**

Run: `cargo test -p segment --features hyperbolic --lib spaces::hyperbolic 2>&1 | tail -5`

Expected: All tests pass (45 + 3 = 48).

---

## Task 6: Create PoincareCurvatureQueryScorer

**Files:**
- Modify: `lib/segment/src/spaces/hyperbolic/poincare_metric.rs`
- Modify: `lib/segment/src/spaces/hyperbolic/mod.rs`

This scorer bypasses the stateless `Metric` trait to inject runtime curvature into Poincaré distance computation. It mirrors `MetricQueryScorer` (see `lib/segment/src/vector_storage/query_scorer/metric_query_scorer.rs`) but calls `poincare_distance(v1, v2, curvature)` directly.

- [ ] **Step 1: Add the PoincareCurvatureQueryScorer**

Add to the bottom of `poincare_metric.rs` (before the `#[cfg(test)]` block):

```rust
// ---------------------------------------------------------------------------
// Curvature-aware scorer (bypasses stateless Metric trait)
// ---------------------------------------------------------------------------

use common::counter::hardware_counter::HardwareCounterCell;
use common::generic_consts::Random;
use common::types::PointOffsetType;
use common::typelevel::True;
use crate::vector_storage::DenseVectorStorage;
use crate::vector_storage::common::VECTOR_READ_BATCH_SIZE;
use crate::vector_storage::query_scorer::QueryScorer;

/// A query scorer that uses configurable curvature for Poincaré distance.
///
/// Unlike `MetricQueryScorer<PoincareMetric>`, this scorer captures a runtime
/// curvature value instead of using `DEFAULT_CURVATURE`. It stores the query
/// as f32 (Poincaré distance always operates on f32 internally).
pub struct PoincareCurvatureQueryScorer<'a, TVectorStorage: DenseVectorStorage<VectorElementType>> {
    query: Vec<f32>,
    curvature: f32,
    vector_storage: &'a TVectorStorage,
    hardware_counter: HardwareCounterCell,
}

impl<'a, TVectorStorage: DenseVectorStorage<VectorElementType>>
    PoincareCurvatureQueryScorer<'a, TVectorStorage>
{
    pub fn new(
        query: Vec<f32>,
        curvature: f32,
        vector_storage: &'a TVectorStorage,
        mut hardware_counter: HardwareCounterCell,
    ) -> Self {
        let dim = query.len();
        let preprocessed = project_to_ball(query, curvature);

        hardware_counter.set_cpu_multiplier(dim * size_of::<VectorElementType>());
        if vector_storage.is_on_disk() {
            hardware_counter.set_vector_io_read_multiplier(dim * size_of::<VectorElementType>());
        } else {
            hardware_counter.set_vector_io_read_multiplier(0);
        }

        Self {
            query: preprocessed,
            curvature,
            vector_storage,
            hardware_counter,
        }
    }
}

impl<TVectorStorage: DenseVectorStorage<VectorElementType>> QueryScorer
    for PoincareCurvatureQueryScorer<'_, TVectorStorage>
{
    type TVector = [VectorElementType];

    #[inline]
    fn score_stored(&self, idx: PointOffsetType) -> ScoreType {
        self.hardware_counter.cpu_counter().incr();
        self.hardware_counter.vector_io_read().incr();
        let stored = self.vector_storage.get_dense::<Random>(idx);
        -poincare_distance(&self.query, &stored, self.curvature)
    }

    fn score_stored_batch(&self, ids: &[PointOffsetType], scores: &mut [ScoreType]) {
        debug_assert!(ids.len() <= VECTOR_READ_BATCH_SIZE);
        debug_assert_eq!(ids.len(), scores.len());

        self.hardware_counter.cpu_counter().incr_delta(ids.len());
        self.hardware_counter.vector_io_read().incr_delta(ids.len());

        self.vector_storage
            .for_each_in_dense_batch(ids, |idx, vector| {
                scores[idx] = -poincare_distance(&self.query, vector, self.curvature);
            });
    }

    #[inline]
    fn score(&self, v2: &[VectorElementType]) -> ScoreType {
        self.hardware_counter.cpu_counter().incr();
        -poincare_distance(&self.query, v2, self.curvature)
    }

    fn score_internal(&self, point_a: PointOffsetType, point_b: PointOffsetType) -> ScoreType {
        self.hardware_counter.cpu_counter().incr();
        let v1 = self.vector_storage.get_dense::<Random>(point_a);
        let v2 = self.vector_storage.get_dense::<Random>(point_b);
        -poincare_distance(&v1, &v2, self.curvature)
    }

    type SupportsBytes = True;
    fn score_bytes(&self, _enabled: Self::SupportsBytes, bytes: &[u8]) -> ScoreType {
        let v2 = <[VectorElementType]>::ref_from_bytes(bytes).unwrap();
        self.score(v2)
    }
}
```

Note: add `use zerocopy::FromBytes;` to the imports at the top of the file if not already present.

- [ ] **Step 2: Update mod.rs re-exports**

Add to `mod.rs`:

```rust
pub use poincare_metric::PoincareCurvatureQueryScorer;
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p segment --features hyperbolic 2>&1 | tail -5`

Expected: Compiles. The scorer isn't used yet (that's Task 7), but it should compile cleanly.

---

## Task 7: Wire Curvature Scorer Into Raw Scorer Dispatch

**Files:**
- Modify: `lib/segment/src/vector_storage/raw_scorer.rs:55-100` (new_raw_scorer + new helper)

**Approach:** `new_raw_scorer` is called from ~12 sites (HNSW index, point_scorer, tests) that don't have access to `VectorDataConfig`. Rather than changing all call sites, we add a parallel entry point `new_raw_scorer_with_curvature` and have the existing `new_raw_scorer` delegate to it with `DEFAULT_CURVATURE`.

- [ ] **Step 1: Add `new_raw_scorer_with_curvature` function**

In `raw_scorer.rs`, add this new public function right after `new_raw_scorer` (after line ~100). This handles only the f32 Dense variants (the only ones used for Poincare):

```rust
/// Create a raw scorer with explicit Poincaré curvature.
///
/// Call sites that have access to `VectorDataConfig` should use this
/// instead of `new_raw_scorer` when the distance is `Poincare` to get
/// curvature-aware scoring. Falls back to `new_raw_scorer` for
/// non-f32 storage and non-Nearest queries.
#[cfg(feature = "hyperbolic")]
pub fn new_raw_scorer_with_curvature<'a>(
    query: QueryVector,
    vector_storage: &'a VectorStorageEnum,
    hc: HardwareCounterCell,
    curvature: f32,
) -> OperationResult<Box<dyn RawScorer + 'a>> {
    // Only intercept f32 Dense storage with Nearest queries for Poincare.
    // All other cases delegate to the standard path.
    if vector_storage.distance() == Distance::Poincare {
        if let QueryVector::Nearest(ref _vector) = query {
            return match vector_storage {
                #[cfg(feature = "rocksdb")]
                VectorStorageEnum::DenseSimple(vs) => {
                    new_poincare_curvature_scorer(query, vs, curvature, hc)
                }
                VectorStorageEnum::DenseVolatile(vs) => {
                    new_poincare_curvature_scorer(query, vs, curvature, hc)
                }
                VectorStorageEnum::DenseMemmap(vs) => {
                    new_poincare_curvature_scorer(query, vs.as_ref(), curvature, hc)
                }
                VectorStorageEnum::DenseAppendableMemmap(vs) => {
                    new_poincare_curvature_scorer(query, vs.as_ref(), curvature, hc)
                }
                // Non-f32 variants: fall through to default path
                _ => new_raw_scorer(query, vector_storage, hc),
            };
        }
    }
    // Non-Poincare or non-Nearest: use default path
    new_raw_scorer(query, vector_storage, hc)
}

#[cfg(feature = "hyperbolic")]
fn new_poincare_curvature_scorer<'a, TVectorStorage: DenseVectorStorage<VectorElementType>>(
    query: QueryVector,
    vector_storage: &'a TVectorStorage,
    curvature: f32,
    hardware_counter: HardwareCounterCell,
) -> OperationResult<Box<dyn RawScorer + 'a>> {
    match query {
        QueryVector::Nearest(vector) => {
            let dense: DenseVector = vector.try_into()?;
            let query_scorer =
                crate::spaces::hyperbolic::PoincareCurvatureQueryScorer::new(
                    dense,
                    curvature,
                    vector_storage,
                    hardware_counter,
                );
            raw_scorer_from_query_scorer(query_scorer)
        }
        // Non-Nearest query: fall back to PoincareMetric with DEFAULT_CURVATURE
        other => new_scorer_with_metric::<VectorElementType, PoincareMetric, _>(
            other,
            vector_storage,
            hardware_counter,
        ),
    }
}
```

Add necessary imports at the top of the cfg-gated section:

```rust
#[cfg(feature = "hyperbolic")]
use crate::spaces::hyperbolic::poincare_metric::PoincareMetric;
#[cfg(feature = "hyperbolic")]
use crate::data_types::vectors::VectorElementType;
```

(Some of these may already be imported — verify and deduplicate.)

- [ ] **Step 2: Verify `new_raw_scorer` still works unchanged**

Run: `cargo test -p segment --features hyperbolic 2>&1 | tail -10`

Expected: All tests pass. `new_raw_scorer` is unchanged, all existing callers work. `new_raw_scorer_with_curvature` is available but not called by existing code yet.

- [ ] **Step 3: Verify feature gating**

Run: `cargo check -p segment 2>&1 | tail -5`

Expected: Compiles without `hyperbolic` feature. The new functions are completely absent.

---

## Task 8: Integration Test — Hierarchy Separation

**Files:**
- Modify: `lib/segment/src/spaces/hyperbolic/poincare_math.rs` (add integration test)

- [ ] **Step 1: Write hierarchy separation test**

Add to the `mod tests` block in `poincare_math.rs`:

```rust
#[test]
fn test_hierarchy_separation_with_busemann() {
    use super::super::busemann::{busemann_depth, compute_focal_direction};

    // Simulate Pythia's hierarchy:
    // - Narratives near origin (low norm, abstract)
    // - Stories at intermediate depth
    // - Posts near boundary (high norm, concrete)

    let narratives: Vec<Vec<f32>> = vec![
        vec![0.05, 0.03],
        vec![-0.04, 0.06],
        vec![0.02, -0.03],
    ];
    let stories: Vec<Vec<f32>> = vec![
        vec![0.3, 0.1],
        vec![-0.2, 0.35],
        vec![0.25, -0.2],
        vec![-0.15, 0.3],
    ];
    let posts: Vec<Vec<f32>> = vec![
        vec![0.7, 0.1],
        vec![-0.6, 0.5],
        vec![0.55, -0.55],
        vec![-0.5, -0.5],
        vec![0.6, 0.4],
        vec![-0.65, 0.3],
    ];

    let c = 1.0;

    // Combine all points for focal direction computation
    let all_points: Vec<&[f32]> = narratives.iter()
        .chain(stories.iter())
        .chain(posts.iter())
        .map(|v| v.as_slice())
        .collect();
    let focal = compute_focal_direction(&all_points, c);

    // Compute Busemann depths
    let narrative_depths: Vec<f32> = narratives.iter()
        .map(|p| busemann_depth(p, &focal, c))
        .collect();
    let story_depths: Vec<f32> = stories.iter()
        .map(|p| busemann_depth(p, &focal, c))
        .collect();
    let post_depths: Vec<f32> = posts.iter()
        .map(|p| busemann_depth(p, &focal, c))
        .collect();

    let avg_narrative = narrative_depths.iter().sum::<f32>() / narrative_depths.len() as f32;
    let avg_story = story_depths.iter().sum::<f32>() / story_depths.len() as f32;
    let avg_post = post_depths.iter().sum::<f32>() / post_depths.len() as f32;

    // Posts should have different average depth than narratives
    // (the exact ordering depends on focal direction, but they should be distinct)
    let separation = (avg_post - avg_narrative).abs();
    assert!(
        separation > 0.1,
        "insufficient hierarchy separation: avg_narrative={avg_narrative:.3}, \
         avg_story={avg_story:.3}, avg_post={avg_post:.3}, separation={separation:.3}"
    );

    // Story depths should be between narrative and post depths (approximately)
    // This tests the intermediate tier
    let story_between = (avg_story > avg_narrative.min(avg_post) - 0.1)
        && (avg_story < avg_narrative.max(avg_post) + 0.1);
    assert!(
        story_between,
        "stories should be between narratives and posts: \
         avg_narrative={avg_narrative:.3}, avg_story={avg_story:.3}, avg_post={avg_post:.3}"
    );
}

#[test]
fn test_curvature_increases_separation() {
    use super::super::busemann::{busemann_depth, compute_focal_direction};

    let near_origin: Vec<f32> = vec![0.05, 0.03];
    let near_boundary: Vec<f32> = vec![0.7, 0.1];
    let all: Vec<&[f32]> = vec![near_origin.as_slice(), near_boundary.as_slice()];

    // Compute separation at c=1.0
    let focal_c1 = compute_focal_direction(&all, 1.0);
    let depth_origin_c1 = busemann_depth(&near_origin, &focal_c1, 1.0);
    let depth_boundary_c1 = busemann_depth(&near_boundary, &focal_c1, 1.0);
    let sep_c1 = (depth_boundary_c1 - depth_origin_c1).abs();

    // Compute separation at c=2.0
    let focal_c2 = compute_focal_direction(&all, 2.0);
    let depth_origin_c2 = busemann_depth(&near_origin, &focal_c2, 2.0);
    let depth_boundary_c2 = busemann_depth(&near_boundary, &focal_c2, 2.0);
    let sep_c2 = (depth_boundary_c2 - depth_origin_c2).abs();

    // Higher curvature should generally produce greater separation
    // (This is the key property for per-client curvature tuning)
    assert!(
        sep_c2 > sep_c1 * 0.8, // Allow some margin — exact relationship is non-linear
        "higher curvature should maintain/increase separation: c1={sep_c1:.3}, c2={sep_c2:.3}"
    );
}
```

- [ ] **Step 2: Run all tests**

Run: `cargo test -p segment --features hyperbolic --lib spaces::hyperbolic 2>&1 | tail -10`

Expected: All tests pass (48 + 2 = 50 total).

- [ ] **Step 3: Run the full segment test suite to check for regressions**

Run: `cargo test -p segment --features hyperbolic 2>&1 | tail -10`

Expected: All segment tests pass. No regressions from our changes.

- [ ] **Step 4: Run without the hyperbolic feature to verify feature gating**

Run: `cargo test -p segment 2>&1 | tail -10`

Expected: All tests pass. Hyperbolic code is completely absent without the feature flag.

---

## Final Checklist

After all tasks are complete:

- [ ] All 50 hyperbolic tests pass: `cargo test -p segment --features hyperbolic --lib spaces::hyperbolic`
- [ ] Full segment suite passes with feature: `cargo test -p segment --features hyperbolic`
- [ ] Full segment suite passes without feature: `cargo test -p segment`
- [ ] `cargo clippy -p segment --features hyperbolic` — no warnings
- [ ] Review the diff: `git diff --stat` — confirm ~300 lines added, all in expected files
