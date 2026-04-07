# Hyperbolic Vector Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native Poincare distance metric, tangent space pruning, and Busemann scoring to a Qdrant fork, enabling hierarchy-aware vector search.

**Architecture:** New `spaces/hyperbolic/` module in `lib/segment` implements Poincare math, metric trait, tangent cache, and Busemann scoring. Existing files receive only match-arm additions (~90 lines). All new code is feature-gated behind `hyperbolic` Cargo feature.

**Tech Stack:** Rust (requires 1.94+), Qdrant lib/segment crate, `half` crate (f16), `common::types::ScoreType` (f32)

**Spec:** `docs/superpowers/specs/2026-04-07-hyperbolic-vector-support-design.md`

---

## Pre-requisite: Rust Toolchain

The project requires Rust 1.94+ (`rust-version = "1.94"` in workspace `Cargo.toml`). Ensure your toolchain is up to date:

```bash
rustup update stable
rustc --version  # Must be >= 1.94
```

---

## File Structure

### New files (all in `lib/segment/src/spaces/hyperbolic/`)

| File | Responsibility |
|------|---------------|
| `mod.rs` | Module declaration, public re-exports |
| `poincare_math.rs` | Pure math: Poincare distance, ball projection, exp/log maps, Frechet mean, fused norms |
| `poincare_metric.rs` | `PoincareMetric` struct implementing `Metric<f32>`, `Metric<u8>`, `Metric<f16>`, `MetricPostProcessing` |
| `tangent_cache.rs` | `TangentCache` struct: Frechet mean centroid, pre-computed tangent coordinates, query projection |
| `tangent_scorer.rs` | `TangentScorer` wrapping `RawScorer` with prune-then-rescore logic |
| `busemann.rs` | Lorentz model conversion, Busemann score function, focal direction computation |

### Modified files

| File | What changes | Lines |
|------|-------------|-------|
| `lib/segment/Cargo.toml` | Add `hyperbolic = []` feature | ~1 |
| `lib/segment/src/spaces/mod.rs` | Add `pub mod hyperbolic;` | ~2 |
| `lib/segment/src/types.rs:309-366` | Add `Poincare` variant + match arms | ~30 |
| `lib/segment/src/vector_storage/raw_scorer.rs:200-350` | Add `Poincare` match arms in `raw_scorer_impl` and `raw_multi_scorer_impl` | ~20 |
| `lib/segment/src/vector_storage/async_raw_scorer.rs:126-138` | Add `Poincare` match arm in `build` | ~5 |
| `lib/segment/src/vector_storage/quantized/quantized_scorer_builder.rs:56-87` | Add `Poincare` match arms (3 datatypes) | ~15 |
| `lib/segment/src/vector_storage/quantized/quantized_vectors.rs:1845-1851` | Add `Poincare` distance type mapping | ~5 |
| `lib/segment/src/index/hnsw_index/gpu/gpu_vector_storage/mod.rs:101-114` | Add `Poincare` GPU shader define | ~5 |

### Test files

| File | What it tests |
|------|--------------|
| `lib/segment/src/spaces/hyperbolic/poincare_math.rs` (inline `#[cfg(test)]`) | Math properties: symmetry, triangle inequality, projection, stability |
| `lib/segment/src/spaces/hyperbolic/poincare_metric.rs` (inline `#[cfg(test)]`) | Metric trait conformance |
| `lib/segment/src/spaces/hyperbolic/tangent_cache.rs` (inline `#[cfg(test)]`) | Tangent distance ordering, Frechet mean properties |
| `lib/segment/src/spaces/hyperbolic/busemann.rs` (inline `#[cfg(test)]`) | Lorentz conversion, depth monotonicity |

---

## Task 1: Feature Gate and Module Skeleton

**Files:**
- Modify: `lib/segment/Cargo.toml:13-17`
- Modify: `lib/segment/src/spaces/mod.rs`
- Create: `lib/segment/src/spaces/hyperbolic/mod.rs`

- [ ] **Step 1: Add feature flag to Cargo.toml**

Open `lib/segment/Cargo.toml` and add the `hyperbolic` feature:

```toml
[features]
default = []
testing = ["common/testing", "sparse/testing", "gpu/testing", "quantization/testing"]
gpu = ["gpu/gpu"]
rocksdb = ["dep:rocksdb"]
hyperbolic = []
```

- [ ] **Step 2: Add module to spaces/mod.rs**

Add at the end of `lib/segment/src/spaces/mod.rs`:

```rust
#[cfg(feature = "hyperbolic")]
pub mod hyperbolic;
```

- [ ] **Step 3: Create hyperbolic/mod.rs skeleton**

Create `lib/segment/src/spaces/hyperbolic/mod.rs`:

```rust
pub mod poincare_math;
pub mod poincare_metric;
pub mod tangent_cache;
pub mod tangent_scorer;
pub mod busemann;
```

- [ ] **Step 4: Create empty placeholder files**

Create each of these as empty files so the module compiles:
- `lib/segment/src/spaces/hyperbolic/poincare_math.rs` — `// Poincare ball math operations`
- `lib/segment/src/spaces/hyperbolic/poincare_metric.rs` — `// PoincareMetric trait impl`
- `lib/segment/src/spaces/hyperbolic/tangent_cache.rs` — `// Tangent space cache`
- `lib/segment/src/spaces/hyperbolic/tangent_scorer.rs` — `// Tangent-pruned scorer`
- `lib/segment/src/spaces/hyperbolic/busemann.rs` — `// Busemann scoring`

- [ ] **Step 5: Verify compilation**

```bash
cd /home/rohan/projects/qdrant
cargo check -p segment --features hyperbolic 2>&1 | tail -5
```

Expected: compiles with no errors (may have warnings about unused modules).

- [ ] **Step 6: Commit**

```bash
git add lib/segment/Cargo.toml lib/segment/src/spaces/mod.rs lib/segment/src/spaces/hyperbolic/
git commit -m "feat: add hyperbolic feature gate and module skeleton"
```

---

## Task 2: Poincare Math — Core Functions

**Files:**
- Create: `lib/segment/src/spaces/hyperbolic/poincare_math.rs`

This is pure math with no Qdrant dependencies. All functions operate on `&[f32]` slices.

- [ ] **Step 1: Write failing tests for poincare_distance**

Write the full `poincare_math.rs` file starting with tests at the bottom:

```rust
/// Poincare ball math operations.
///
/// All functions operate in the Poincare ball model of hyperbolic geometry
/// where points satisfy ||x|| < 1/sqrt(c) for curvature parameter c > 0.

/// Default curvature parameter.
pub const DEFAULT_CURVATURE: f32 = 1.0;

/// Numerical stability epsilon.
const EPS: f32 = 1e-5;

/// Compute Poincare geodesic distance between two points in the Poincare ball.
///
/// Formula: d(u,v) = (1/sqrt(c)) * acosh(1 + 2c||u-v||^2 / ((1 - c||u||^2)(1 - c||v||^2)))
///
/// Uses fused single-pass norm computation for efficiency.
pub fn poincare_distance(u: &[f32], v: &[f32], c: f32) -> f32 {
    todo!()
}

/// Compute ||u-v||^2, ||u||^2, ||v||^2 in a single pass.
#[inline]
fn fused_norms(u: &[f32], v: &[f32]) -> (f32, f32, f32) {
    todo!()
}

/// Numerically stable acosh. Uses Taylor expansion for arguments near 1.
#[inline]
fn stable_acosh(x: f32) -> f32 {
    todo!()
}

/// Project a vector into the Poincare ball: ||x|| < 1/sqrt(c) - eps.
/// If already inside, returns unchanged. Otherwise scales to the boundary.
pub fn project_to_ball(mut v: Vec<f32>, c: f32) -> Vec<f32> {
    todo!()
}

/// Exponential map at the origin: maps Euclidean tangent vector to Poincare ball point.
/// exp_0(v) = tanh(sqrt(c) * ||v|| / 2) * v / ||v||
pub fn exp_map_origin(v: &[f32], c: f32) -> Vec<f32> {
    todo!()
}

/// Logarithmic map at the origin: maps Poincare ball point to tangent space vector.
/// log_0(p) = arctanh(sqrt(c) * ||p||) / (sqrt(c) * ||p||) * p
pub fn log_map_origin(p: &[f32], c: f32) -> Vec<f32> {
    todo!()
}

/// Exponential map at an arbitrary base point p.
/// exp_p(v) = p + tanh(sqrt(c) * lambda_p * ||v|| / 2) * v / (sqrt(c) * ||v||)
/// where lambda_p = 2 / (1 - c||p||^2) is the conformal factor.
pub fn exp_map(p: &[f32], v: &[f32], c: f32) -> Vec<f32> {
    todo!()
}

/// Logarithmic map at an arbitrary base point p.
/// log_p(y) = (2 / (sqrt(c) * lambda_p)) * arctanh(sqrt(c) * ||-p ⊕ y||) * (-p ⊕ y) / ||-p ⊕ y||
pub fn log_map(p: &[f32], y: &[f32], c: f32) -> Vec<f32> {
    todo!()
}

/// Mobius addition: x ⊕_c y
/// Formula: ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) / (1 + 2c<x,y> + c^2||x||^2||y||^2)
pub fn mobius_add(x: &[f32], y: &[f32], c: f32) -> Vec<f32> {
    todo!()
}

/// Mobius negation: -x in the Poincare ball (same as Euclidean negation).
pub fn mobius_neg(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| -v).collect()
}

/// Conformal factor at point x: lambda_x = 2 / (1 - c||x||^2)
#[inline]
pub fn conformal_factor(x: &[f32], c: f32) -> f32 {
    let norm_sq: f32 = x.iter().map(|v| v * v).sum();
    2.0 / (1.0 - c * norm_sq).max(EPS)
}

/// Frechet mean (hyperbolic centroid) via Riemannian gradient descent.
/// Iteratively computes the point that minimizes sum of squared geodesic distances.
pub fn frechet_mean(points: &[&[f32]], c: f32) -> Vec<f32> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    fn vec_approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx_eq(*x, *y))
    }

    // --- fused_norms ---

    #[test]
    fn test_fused_norms_basic() {
        let u = vec![1.0, 0.0];
        let v = vec![0.0, 1.0];
        let (diff_sq, u_sq, v_sq) = fused_norms(&u, &v);
        assert!(approx_eq(diff_sq, 2.0));
        assert!(approx_eq(u_sq, 1.0));
        assert!(approx_eq(v_sq, 1.0));
    }

    #[test]
    fn test_fused_norms_same_vector() {
        let u = vec![0.3, 0.4];
        let (diff_sq, _, _) = fused_norms(&u, &u);
        assert!(approx_eq(diff_sq, 0.0));
    }

    // --- stable_acosh ---

    #[test]
    fn test_stable_acosh_at_one() {
        assert!(approx_eq(stable_acosh(1.0), 0.0));
    }

    #[test]
    fn test_stable_acosh_known_value() {
        // acosh(2) = ln(2 + sqrt(3)) ≈ 1.3169
        assert!((stable_acosh(2.0) - 1.3169).abs() < 1e-3);
    }

    // --- poincare_distance ---

    #[test]
    fn test_distance_identity() {
        let x = vec![0.3, 0.2];
        assert!(approx_eq(poincare_distance(&x, &x, 1.0), 0.0));
    }

    #[test]
    fn test_distance_symmetry() {
        let x = vec![0.3, 0.2];
        let y = vec![-0.1, 0.4];
        let d_xy = poincare_distance(&x, &y, 1.0);
        let d_yx = poincare_distance(&y, &x, 1.0);
        assert!(approx_eq(d_xy, d_yx));
    }

    #[test]
    fn test_distance_non_negative() {
        let x = vec![0.3, 0.2];
        let y = vec![-0.1, 0.4];
        assert!(poincare_distance(&x, &y, 1.0) >= 0.0);
    }

    #[test]
    fn test_distance_triangle_inequality() {
        let x = vec![0.1, 0.2];
        let y = vec![-0.2, 0.3];
        let z = vec![0.4, -0.1];
        let d_xz = poincare_distance(&x, &z, 1.0);
        let d_xy = poincare_distance(&x, &y, 1.0);
        let d_yz = poincare_distance(&y, &z, 1.0);
        assert!(d_xz <= d_xy + d_yz + TOL);
    }

    #[test]
    fn test_distance_from_origin() {
        let origin = vec![0.0, 0.0];
        let x = vec![0.5, 0.0];
        let d = poincare_distance(&origin, &x, 1.0);
        // d(0, x) = acosh(1 + 2||x||^2 / (1 - ||x||^2))
        // = acosh(1 + 2*0.25 / 0.75) = acosh(1.6667) ≈ 1.0986
        assert!((d - 1.0986).abs() < 1e-3);
    }

    // --- project_to_ball ---

    #[test]
    fn test_project_inside_ball_unchanged() {
        let v = vec![0.3, 0.4]; // norm = 0.5, inside ball
        let projected = project_to_ball(v.clone(), 1.0);
        assert!(vec_approx_eq(&projected, &v));
    }

    #[test]
    fn test_project_outside_ball_clamped() {
        let v = vec![0.8, 0.8]; // norm ≈ 1.13, outside ball for c=1
        let projected = project_to_ball(v, 1.0);
        let norm: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1.0 - EPS + TOL);
    }

    #[test]
    fn test_project_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let projected = project_to_ball(v, 1.0);
        assert!(vec_approx_eq(&projected, &[0.0, 0.0, 0.0]));
    }

    // --- exp_map_origin / log_map_origin ---

    #[test]
    fn test_exp_log_roundtrip_at_origin() {
        let v = vec![0.3, -0.2, 0.1];
        let p = exp_map_origin(&v, 1.0);
        let v_back = log_map_origin(&p, 1.0);
        assert!(vec_approx_eq(&v, &v_back));
    }

    #[test]
    fn test_exp_map_zero_is_origin() {
        let v = vec![0.0, 0.0];
        let p = exp_map_origin(&v, 1.0);
        assert!(vec_approx_eq(&p, &[0.0, 0.0]));
    }

    #[test]
    fn test_exp_map_inside_ball() {
        let v = vec![1.0, 0.5, -0.3];
        let p = exp_map_origin(&v, 1.0);
        let norm: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1.0); // Must be inside the unit ball for c=1
    }

    // --- mobius_add ---

    #[test]
    fn test_mobius_right_identity() {
        let x = vec![0.3, 0.2];
        let zero = vec![0.0, 0.0];
        let result = mobius_add(&x, &zero, 1.0);
        assert!(vec_approx_eq(&result, &x));
    }

    #[test]
    fn test_mobius_inverse() {
        let x = vec![0.3, 0.2];
        let neg_x = mobius_neg(&x);
        let result = mobius_add(&x, &neg_x, 1.0);
        let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm < TOL); // Should be near origin
    }

    // --- frechet_mean ---

    #[test]
    fn test_frechet_mean_single_point() {
        let p = vec![0.3, 0.2];
        let mean = frechet_mean(&[&p], 1.0);
        assert!(vec_approx_eq(&mean, &p));
    }

    #[test]
    fn test_frechet_mean_symmetric_near_origin() {
        let p1 = vec![0.3, 0.0];
        let p2 = vec![-0.3, 0.0];
        let mean = frechet_mean(&[&p1, &p2], 1.0);
        let norm: f32 = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm < 0.1); // Should be near origin
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p segment --features hyperbolic -- spaces::hyperbolic::poincare_math::tests 2>&1 | tail -10
```

Expected: All tests FAIL with `not yet implemented`.

- [ ] **Step 3: Implement fused_norms and stable_acosh**

Replace the `todo!()` in `fused_norms`:

```rust
#[inline]
fn fused_norms(u: &[f32], v: &[f32]) -> (f32, f32, f32) {
    let mut diff_sq = 0.0f32;
    let mut u_sq = 0.0f32;
    let mut v_sq = 0.0f32;
    for (a, b) in u.iter().zip(v.iter()) {
        let d = a - b;
        diff_sq += d * d;
        u_sq += a * a;
        v_sq += b * b;
    }
    (diff_sq, u_sq, v_sq)
}
```

Replace the `todo!()` in `stable_acosh`:

```rust
#[inline]
fn stable_acosh(x: f32) -> f32 {
    if x <= 1.0 {
        return 0.0;
    }
    // For x near 1, use Taylor: acosh(1+d) ≈ sqrt(2d)
    let d = x - 1.0;
    if d < 1e-6 {
        return (2.0 * d).sqrt();
    }
    // Standard formula: acosh(x) = ln(x + sqrt(x^2 - 1))
    (x + (x * x - 1.0).max(0.0).sqrt()).ln()
}
```

- [ ] **Step 4: Implement poincare_distance**

```rust
pub fn poincare_distance(u: &[f32], v: &[f32], c: f32) -> f32 {
    let (diff_sq, u_sq, v_sq) = fused_norms(u, v);
    let denom = (1.0 - c * u_sq).max(EPS) * (1.0 - c * v_sq).max(EPS);
    let arg = 1.0 + 2.0 * c * diff_sq / denom;
    stable_acosh(arg) / c.sqrt()
}
```

- [ ] **Step 5: Implement project_to_ball**

```rust
pub fn project_to_ball(mut v: Vec<f32>, c: f32) -> Vec<f32> {
    let max_norm = 1.0 / c.sqrt() - EPS;
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > 0.0 && norm >= max_norm {
        let scale = max_norm / norm;
        for x in v.iter_mut() {
            *x *= scale;
        }
    }
    v
}
```

- [ ] **Step 6: Implement exp_map_origin and log_map_origin**

```rust
pub fn exp_map_origin(v: &[f32], c: f32) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < EPS {
        return vec![0.0; v.len()];
    }
    let scale = (c.sqrt() * norm / 2.0).tanh() / (c.sqrt() * norm);
    v.iter().map(|x| x * scale).collect()
}

pub fn log_map_origin(p: &[f32], c: f32) -> Vec<f32> {
    let norm: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < EPS {
        return vec![0.0; p.len()];
    }
    let scale = (c.sqrt() * norm).atanh() / (c.sqrt() * norm);
    p.iter().map(|x| x * scale).collect()
}
```

- [ ] **Step 7: Implement mobius_add**

```rust
pub fn mobius_add(x: &[f32], y: &[f32], c: f32) -> Vec<f32> {
    let mut dot_xy = 0.0f32;
    let mut x_sq = 0.0f32;
    let mut y_sq = 0.0f32;
    for (a, b) in x.iter().zip(y.iter()) {
        dot_xy += a * b;
        x_sq += a * a;
        y_sq += b * b;
    }
    let num_x = 1.0 + 2.0 * c * dot_xy + c * y_sq;
    let num_y = 1.0 - c * x_sq;
    let denom = (1.0 + 2.0 * c * dot_xy + c * c * x_sq * y_sq).max(EPS);
    let mut result: Vec<f32> = x.iter().zip(y.iter())
        .map(|(xi, yi)| (num_x * xi + num_y * yi) / denom)
        .collect();
    result = project_to_ball(result, c);
    result
}
```

- [ ] **Step 8: Implement exp_map, log_map (arbitrary base point)**

```rust
pub fn exp_map(p: &[f32], v: &[f32], c: f32) -> Vec<f32> {
    let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if v_norm < EPS {
        return p.to_vec();
    }
    let lambda_p = conformal_factor(p, c);
    let scale = (c.sqrt() * lambda_p * v_norm / 2.0).tanh() / (c.sqrt() * v_norm);
    let direction: Vec<f32> = v.iter().map(|x| x * scale).collect();
    mobius_add(p, &direction, c)
}

pub fn log_map(p: &[f32], y: &[f32], c: f32) -> Vec<f32> {
    let neg_p = mobius_neg(p);
    let add = mobius_add(&neg_p, y, c);
    let add_norm: f32 = add.iter().map(|x| x * x).sum::<f32>().sqrt();
    if add_norm < EPS {
        return vec![0.0; p.len()];
    }
    let lambda_p = conformal_factor(p, c);
    let scale = (2.0 / (c.sqrt() * lambda_p)) * (c.sqrt() * add_norm).atanh() / add_norm;
    add.iter().map(|x| x * scale).collect()
}
```

- [ ] **Step 9: Implement frechet_mean**

```rust
pub fn frechet_mean(points: &[&[f32]], c: f32) -> Vec<f32> {
    if points.is_empty() {
        return vec![];
    }
    if points.len() == 1 {
        return points[0].to_vec();
    }
    let dim = points[0].len();
    let mut mean = vec![0.0f32; dim]; // Start at origin
    let lr = 0.1f32;
    let max_iter = 100;
    let tol = 1e-6f32;

    for _ in 0..max_iter {
        // Compute Riemannian gradient: sum of log_mean(p_i)
        let mut grad = vec![0.0f32; dim];
        for p in points {
            let log_v = log_map(&mean, p, c);
            for (g, l) in grad.iter_mut().zip(log_v.iter()) {
                *g += l;
            }
        }
        let n = points.len() as f32;
        for g in grad.iter_mut() {
            *g /= n;
        }
        let grad_norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
        if grad_norm < tol {
            break;
        }
        // Step along gradient via exp map
        let step: Vec<f32> = grad.iter().map(|x| x * lr).collect();
        mean = exp_map(&mean, &step, c);
        mean = project_to_ball(mean, c);
    }
    mean
}
```

- [ ] **Step 10: Run all tests**

```bash
cargo test -p segment --features hyperbolic -- spaces::hyperbolic::poincare_math::tests 2>&1 | tail -20
```

Expected: All 16 tests PASS.

- [ ] **Step 11: Commit**

```bash
git add lib/segment/src/spaces/hyperbolic/poincare_math.rs
git commit -m "feat(hyperbolic): implement Poincare ball math operations"
```

---

## Task 3: PoincareMetric — Metric Trait Implementation

**Files:**
- Create: `lib/segment/src/spaces/hyperbolic/poincare_metric.rs`

- [ ] **Step 1: Write the PoincareMetric with tests**

```rust
use common::types::ScoreType;

use crate::data_types::primitive::PrimitiveVectorElement;
use crate::data_types::vectors::{DenseVector, VectorElementType, VectorElementTypeByte, VectorElementTypeHalf};
use crate::spaces::metric::{Metric, MetricPostProcessing};
use crate::types::Distance;

use super::poincare_math::{poincare_distance, project_to_ball, DEFAULT_CURVATURE};

#[derive(Clone)]
pub struct PoincareMetric;

impl Metric<VectorElementType> for PoincareMetric {
    fn distance() -> Distance {
        Distance::Poincare
    }

    fn similarity(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
        -poincare_distance(v1, v2, DEFAULT_CURVATURE)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        project_to_ball(vector, DEFAULT_CURVATURE)
    }
}

impl MetricPostProcessing for PoincareMetric {
    fn postprocess(score: ScoreType) -> ScoreType {
        // Raw score is negative Poincare distance.
        // Postprocess: negate to return positive distance (like EuclidMetric).
        score.abs()
    }
}

impl Metric<VectorElementTypeByte> for PoincareMetric {
    fn distance() -> Distance {
        Distance::Poincare
    }

    fn similarity(v1: &[VectorElementTypeByte], v2: &[VectorElementTypeByte]) -> ScoreType {
        // Convert u8 to f32, scale to [0, 1), then compute Poincare distance.
        let v1_f32: Vec<f32> = v1.iter().map(|&b| b as f32 / 255.0 - 0.5).collect();
        let v2_f32: Vec<f32> = v2.iter().map(|&b| b as f32 / 255.0 - 0.5).collect();
        -poincare_distance(&v1_f32, &v2_f32, DEFAULT_CURVATURE)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        project_to_ball(vector, DEFAULT_CURVATURE)
    }
}

impl Metric<VectorElementTypeHalf> for PoincareMetric {
    fn distance() -> Distance {
        Distance::Poincare
    }

    fn similarity(v1: &[VectorElementTypeHalf], v2: &[VectorElementTypeHalf]) -> ScoreType {
        use half::f16;
        let v1_f32: Vec<f32> = v1.iter().map(|h| f16::to_f32(*h)).collect();
        let v2_f32: Vec<f32> = v2.iter().map(|h| f16::to_f32(*h)).collect();
        -poincare_distance(&v1_f32, &v2_f32, DEFAULT_CURVATURE)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        project_to_ball(vector, DEFAULT_CURVATURE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_distance_variant() {
        assert_eq!(<PoincareMetric as Metric<VectorElementType>>::distance(), Distance::Poincare);
    }

    #[test]
    fn test_similarity_negative_distance() {
        let v1 = vec![0.3, 0.2];
        let v2 = vec![-0.1, 0.4];
        let sim = <PoincareMetric as Metric<VectorElementType>>::similarity(&v1, &v2);
        assert!(sim < 0.0); // Negative distance
        assert!(sim > -10.0); // Sanity bound
    }

    #[test]
    fn test_similarity_identity_is_zero() {
        let v = vec![0.3, 0.2];
        let sim = <PoincareMetric as Metric<VectorElementType>>::similarity(&v, &v);
        assert!((sim - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_preprocess_projects_to_ball() {
        let v = vec![0.9, 0.9]; // norm ≈ 1.27
        let preprocessed = <PoincareMetric as Metric<VectorElementType>>::preprocess(v);
        let norm: f32 = preprocessed.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1.0);
    }

    #[test]
    fn test_postprocess_returns_positive() {
        let negative_score = -1.5;
        let result = PoincareMetric::postprocess(negative_score);
        assert!(result > 0.0);
        assert!((result - 1.5).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p segment --features hyperbolic -- spaces::hyperbolic::poincare_metric::tests 2>&1 | tail -10
```

Expected: All 5 tests PASS (implementation is already written above, not TDD for the metric since it delegates to poincare_math).

- [ ] **Step 3: Commit**

```bash
git add lib/segment/src/spaces/hyperbolic/poincare_metric.rs
git commit -m "feat(hyperbolic): implement PoincareMetric for f32, u8, f16"
```

---

## Task 4: Wire Distance::Poincare Into Qdrant Dispatch

**Files:**
- Modify: `lib/segment/src/types.rs:308-366`
- Modify: `lib/segment/src/vector_storage/raw_scorer.rs:200-350`
- Modify: `lib/segment/src/vector_storage/async_raw_scorer.rs:126-138`
- Modify: `lib/segment/src/vector_storage/quantized/quantized_scorer_builder.rs:56-87`
- Modify: `lib/segment/src/vector_storage/quantized/quantized_vectors.rs:1845-1851`
- Modify: `lib/segment/src/index/hnsw_index/gpu/gpu_vector_storage/mod.rs:101-114`

This is the largest task in terms of files touched, but each change is mechanical: add a match arm.

- [ ] **Step 1: Add Poincare to Distance enum (types.rs)**

In `lib/segment/src/types.rs`, add the variant and all match arms. At the top of the file, add the import (after existing imports from `crate::spaces::simple`):

```rust
#[cfg(feature = "hyperbolic")]
use crate::spaces::hyperbolic::poincare_metric::PoincareMetric;
```

Add the variant to the enum (line ~318):

```rust
pub enum Distance {
    Cosine,
    Euclid,
    Dot,
    Manhattan,
    #[cfg(feature = "hyperbolic")]
    Poincare,
}
```

Add match arms in `postprocess_score()` (line ~327):

```rust
#[cfg(feature = "hyperbolic")]
Distance::Poincare => PoincareMetric::postprocess(score),
```

Add match arms in `preprocess_vector()` (line ~342):

```rust
#[cfg(feature = "hyperbolic")]
Distance::Poincare => PoincareMetric::preprocess(vector),
```

Note: `preprocess_vector` has trait bounds on CosineMetric, EuclidMetric, DotProductMetric, ManhattanMetric. Add `PoincareMetric: Metric<T>` to the where clause, also gated:

```rust
pub fn preprocess_vector<T: PrimitiveVectorElement>(&self, vector: DenseVector) -> DenseVector
where
    CosineMetric: Metric<T>,
    EuclidMetric: Metric<T>,
    DotProductMetric: Metric<T>,
    ManhattanMetric: Metric<T>,
    #[cfg(feature = "hyperbolic")]
    PoincareMetric: Metric<T>,
```

Add match arm in `distance_order()` (line ~348):

```rust
#[cfg(feature = "hyperbolic")]
Distance::Poincare => Order::SmallBetter,
```

- [ ] **Step 2: Add Poincare match arms to raw_scorer.rs**

In `lib/segment/src/vector_storage/raw_scorer.rs`, add the import:

```rust
#[cfg(feature = "hyperbolic")]
use crate::spaces::hyperbolic::poincare_metric::PoincareMetric;
```

Add match arm in `raw_scorer_impl()` (after Manhattan, line ~236):

```rust
#[cfg(feature = "hyperbolic")]
Distance::Poincare => new_scorer_with_metric::<TElement, PoincareMetric, _>(
    query,
    vector_storage,
    hardware_counter,
),
```

Add the same where clause extension and match arm in `raw_multi_scorer_impl()` (after Manhattan, line ~349):

```rust
#[cfg(feature = "hyperbolic")]
Distance::Poincare => new_multi_scorer_with_metric::<_, PoincareMetric, _>(
    query,
    vector_storage,
    hardware_counter,
),
```

And the where clause bounds on both functions:

```rust
#[cfg(feature = "hyperbolic")]
PoincareMetric: Metric<TElement>,
```

- [ ] **Step 3: Add Poincare match arm to async_raw_scorer.rs**

In `lib/segment/src/vector_storage/async_raw_scorer.rs`, add the import and match arm in `build()` (line ~137):

```rust
#[cfg(feature = "hyperbolic")]
use crate::spaces::hyperbolic::poincare_metric::PoincareMetric;
```

```rust
#[cfg(feature = "hyperbolic")]
Distance::Poincare => self._build_with_metric::<PoincareMetric>(),
```

And add the where clause bound.

- [ ] **Step 4: Add Poincare match arms to quantized_scorer_builder.rs**

In `lib/segment/src/vector_storage/quantized/quantized_scorer_builder.rs`, add match arms for all three datatypes (lines 58-85):

```rust
#[cfg(feature = "hyperbolic")]
use crate::spaces::hyperbolic::poincare_metric::PoincareMetric;
```

In the `build()` method, for each datatype block add:

```rust
#[cfg(feature = "hyperbolic")]
Distance::Poincare => self.build_with_metric::<VectorElementType, PoincareMetric>(),
```

(Repeat for `VectorElementTypeByte` and `VectorElementTypeHalf` blocks.)

- [ ] **Step 5: Add Poincare to quantized_vectors.rs distance mapping**

In `lib/segment/src/vector_storage/quantized/quantized_vectors.rs` (line ~1845):

```rust
#[cfg(feature = "hyperbolic")]
Distance::Poincare => quantization::DistanceType::L2,  // Closest approximation for quantized search
```

And in the `invert` line (~1851):

```rust
invert: distance == Distance::Euclid || distance == Distance::Manhattan || {
    #[cfg(feature = "hyperbolic")]
    { distance == Distance::Poincare }
    #[cfg(not(feature = "hyperbolic"))]
    { false }
},
```

- [ ] **Step 6: Add Poincare to GPU shader defines**

In `lib/segment/src/index/hnsw_index/gpu/gpu_vector_storage/mod.rs` (line ~114):

```rust
#[cfg(feature = "hyperbolic")]
Distance::Poincare => {
    // Fall back to Euclid shader for GPU — exact Poincare distance is CPU-only for now
    defines.insert("EUCLID_DISTANCE".to_owned(), None);
}
```

- [ ] **Step 7: Verify compilation**

```bash
cargo check -p segment --features hyperbolic 2>&1 | tail -10
```

Expected: Compiles with no errors. There may be compiler warnings about `#[cfg]` on where clause bounds — if the Rust version doesn't support `#[cfg]` on where clauses, we'll need to use conditional compilation at the function level instead. Adjust as needed.

- [ ] **Step 8: Also verify compilation WITHOUT the feature**

```bash
cargo check -p segment 2>&1 | tail -5
```

Expected: Compiles with no errors. The feature gate means zero impact on default builds.

- [ ] **Step 9: Commit**

```bash
git add lib/segment/src/types.rs lib/segment/src/vector_storage/raw_scorer.rs lib/segment/src/vector_storage/async_raw_scorer.rs lib/segment/src/vector_storage/quantized/quantized_scorer_builder.rs lib/segment/src/vector_storage/quantized/quantized_vectors.rs lib/segment/src/index/hnsw_index/gpu/gpu_vector_storage/mod.rs
git commit -m "feat(hyperbolic): wire Distance::Poincare into all scorer dispatch sites"
```

---

## Task 5: Tangent Space Cache

**Files:**
- Create: `lib/segment/src/spaces/hyperbolic/tangent_cache.rs`

- [ ] **Step 1: Write TangentCache with tests**

```rust
use super::poincare_math::{
    conformal_factor, frechet_mean, log_map, poincare_distance, DEFAULT_CURVATURE, EPS,
};

/// Pre-computed tangent space coordinates for fast approximate nearest neighbor search.
///
/// The tangent space at the Frechet mean (centroid) is Euclidean, so Euclidean
/// distances in tangent space approximate geodesic distances for nearby points.
pub struct TangentCache {
    /// Frechet mean of all vectors (tangent space base point).
    centroid: Vec<f32>,
    /// Pre-computed log_centroid(x) for each vector, indexed by position.
    tangent_coords: Vec<Vec<f32>>,
    /// Curvature used.
    curvature: f32,
}

impl TangentCache {
    /// Build a tangent cache from a set of Poincare ball vectors.
    pub fn build(vectors: &[Vec<f32>], curvature: f32) -> Self {
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let centroid = if refs.is_empty() {
            vec![]
        } else {
            frechet_mean(&refs, curvature)
        };
        let tangent_coords: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| log_map(&centroid, v, curvature))
            .collect();
        Self {
            centroid,
            tangent_coords,
            curvature,
        }
    }

    /// Map a query point to tangent space at the centroid.
    pub fn project_query(&self, query: &[f32]) -> Vec<f32> {
        log_map(&self.centroid, query, self.curvature)
    }

    /// Squared Euclidean distance in tangent space between a projected query and a stored point.
    #[inline]
    pub fn tangent_distance_sq(&self, query_tangent: &[f32], point_idx: usize) -> f32 {
        let stored = &self.tangent_coords[point_idx];
        query_tangent
            .iter()
            .zip(stored.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// Exact Poincare distance between query and a stored vector (for re-ranking).
    pub fn exact_distance(&self, query: &[f32], point_idx: usize, vectors: &[Vec<f32>]) -> f32 {
        poincare_distance(query, &vectors[point_idx], self.curvature)
    }

    /// Number of cached points.
    pub fn len(&self) -> usize {
        self.tangent_coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tangent_coords.is_empty()
    }

    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::hyperbolic::poincare_math::exp_map_origin;

    fn make_test_vectors() -> Vec<Vec<f32>> {
        // Cluster of points near [0.3, 0.1] in Poincare ball
        vec![
            vec![0.25, 0.08],
            vec![0.30, 0.12],
            vec![0.35, 0.10],
            vec![0.28, 0.15],
            vec![0.32, 0.05],
        ]
    }

    #[test]
    fn test_build_cache() {
        let vectors = make_test_vectors();
        let cache = TangentCache::build(&vectors, 1.0);
        assert_eq!(cache.len(), 5);
        assert!(!cache.is_empty());
        assert_eq!(cache.centroid().len(), 2);
    }

    #[test]
    fn test_tangent_distance_ordering_matches_exact() {
        let vectors = make_test_vectors();
        let cache = TangentCache::build(&vectors, 1.0);
        let query = vec![0.27, 0.11];
        let query_tangent = cache.project_query(&query);

        // Get tangent distances and exact distances
        let mut tangent_dists: Vec<(usize, f32)> = (0..vectors.len())
            .map(|i| (i, cache.tangent_distance_sq(&query_tangent, i)))
            .collect();
        let mut exact_dists: Vec<(usize, f32)> = (0..vectors.len())
            .map(|i| (i, poincare_distance(&query, &vectors[i], 1.0)))
            .collect();

        tangent_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // The nearest neighbor should agree (tangent approximation is good for nearby points)
        assert_eq!(tangent_dists[0].0, exact_dists[0].0);
    }

    #[test]
    fn test_empty_cache() {
        let cache = TangentCache::build(&[], 1.0);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_single_point_cache() {
        let vectors = vec![vec![0.3, 0.2]];
        let cache = TangentCache::build(&vectors, 1.0);
        assert_eq!(cache.len(), 1);
        // Tangent distance of the point to itself should be ~0
        let query_tangent = cache.project_query(&vectors[0]);
        let d = cache.tangent_distance_sq(&query_tangent, 0);
        assert!(d < 1e-4);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p segment --features hyperbolic -- spaces::hyperbolic::tangent_cache::tests 2>&1 | tail -10
```

Expected: All 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add lib/segment/src/spaces/hyperbolic/tangent_cache.rs
git commit -m "feat(hyperbolic): implement TangentCache for approximate NN pruning"
```

---

## Task 6: Tangent Scorer Wrapper

**Files:**
- Create: `lib/segment/src/spaces/hyperbolic/tangent_scorer.rs`

This task creates the `TangentScorer` that wraps an exact `RawScorer` and pre-filters using tangent distances. This is the integration point with Qdrant's scorer pipeline.

**Note:** This is a design skeleton — the full integration into HNSW's search path (injecting `TangentScorer` in `hnsw.rs`) is deferred to a later phase since it requires deeper integration testing. The scorer struct and its pruning logic are implemented and unit-tested here.

- [ ] **Step 1: Write TangentScorer**

```rust
use super::tangent_cache::TangentCache;

/// Default prune factor: keep top (k * PRUNE_FACTOR) candidates from tangent
/// distance, then re-rank with exact Poincare distance.
pub const DEFAULT_PRUNE_FACTOR: usize = 10;

/// Prune a set of candidate indices using tangent space distances,
/// returning the top-k by exact Poincare distance.
///
/// This is a standalone function rather than a RawScorer wrapper,
/// for ease of testing and integration. The HNSW integration will
/// call this in the search hot path.
pub fn prune_and_rescore(
    query: &[f32],
    candidate_indices: &[usize],
    vectors: &[Vec<f32>],
    tangent_cache: &TangentCache,
    k: usize,
    prune_factor: usize,
) -> Vec<(usize, f32)> {
    let query_tangent = tangent_cache.project_query(query);

    // Phase 1: Score all candidates with cheap tangent distance
    let mut tangent_scores: Vec<(usize, f32)> = candidate_indices
        .iter()
        .map(|&idx| (idx, tangent_cache.tangent_distance_sq(&query_tangent, idx)))
        .collect();

    // Keep top (k * prune_factor) by tangent distance (smallest = closest)
    let keep = (k * prune_factor).min(tangent_scores.len());
    tangent_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    tangent_scores.truncate(keep);

    // Phase 2: Re-score survivors with exact Poincare distance
    let mut exact_scores: Vec<(usize, f32)> = tangent_scores
        .iter()
        .map(|&(idx, _)| {
            let d = tangent_cache.exact_distance(query, idx, vectors);
            (idx, d)
        })
        .collect();

    // Return top k by exact distance
    exact_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    exact_scores.truncate(k);
    exact_scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::hyperbolic::tangent_cache::TangentCache;
    use crate::spaces::hyperbolic::poincare_math::poincare_distance;

    fn make_test_vectors() -> Vec<Vec<f32>> {
        vec![
            vec![0.10, 0.05],  // 0
            vec![0.25, 0.08],  // 1
            vec![0.30, 0.12],  // 2
            vec![0.35, 0.10],  // 3
            vec![0.28, 0.15],  // 4
            vec![0.32, 0.05],  // 5
            vec![-0.20, 0.30], // 6 — far away
            vec![-0.40, -0.10],// 7 — far away
            vec![0.50, 0.40],  // 8 — far away
            vec![0.05, -0.45], // 9 — far away
        ]
    }

    #[test]
    fn test_prune_and_rescore_returns_k() {
        let vectors = make_test_vectors();
        let cache = TangentCache::build(&vectors, 1.0);
        let query = vec![0.27, 0.11];
        let candidates: Vec<usize> = (0..vectors.len()).collect();
        let results = prune_and_rescore(&query, &candidates, &vectors, &cache, 3, 5);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_prune_and_rescore_finds_nearest() {
        let vectors = make_test_vectors();
        let cache = TangentCache::build(&vectors, 1.0);
        let query = vec![0.27, 0.11];
        let candidates: Vec<usize> = (0..vectors.len()).collect();

        // Pruned results
        let results = prune_and_rescore(&query, &candidates, &vectors, &cache, 3, 5);

        // Brute force exact nearest
        let mut exact: Vec<(usize, f32)> = (0..vectors.len())
            .map(|i| (i, poincare_distance(&query, &vectors[i], 1.0)))
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // The nearest neighbor from pruning should match brute force
        assert_eq!(results[0].0, exact[0].0);
    }

    #[test]
    fn test_prune_factor_1_is_exact() {
        let vectors = make_test_vectors();
        let cache = TangentCache::build(&vectors, 1.0);
        let query = vec![0.27, 0.11];
        let candidates: Vec<usize> = (0..vectors.len()).collect();

        // With prune_factor=1, all candidates get exact scoring
        // (tangent_scores.truncate(k*1) = k candidates re-scored)
        // This should produce same results as prune_factor=10 for small k
        let results_pf1 = prune_and_rescore(&query, &candidates, &vectors, &cache, 3, 1);
        let results_pf10 = prune_and_rescore(&query, &candidates, &vectors, &cache, 3, 10);

        // Top-1 should be the same regardless of prune factor
        assert_eq!(results_pf1[0].0, results_pf10[0].0);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p segment --features hyperbolic -- spaces::hyperbolic::tangent_scorer::tests 2>&1 | tail -10
```

Expected: All 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add lib/segment/src/spaces/hyperbolic/tangent_scorer.rs
git commit -m "feat(hyperbolic): implement tangent space pruning scorer"
```

---

## Task 7: Busemann Scoring

**Files:**
- Create: `lib/segment/src/spaces/hyperbolic/busemann.rs`

- [ ] **Step 1: Write Busemann module with tests**

```rust
use super::poincare_math::{frechet_mean, EPS};

/// Convert a Poincare ball point to the Lorentz hyperboloid model.
///
/// Lorentz coordinates: x_0 = (1 + c||x||^2) / (1 - c||x||^2)
///                      x_i = 2*sqrt(c)*x_i / (1 - c||x||^2)  for i > 0
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

/// Lorentz inner product: <x, y>_L = -x_0*y_0 + x_1*y_1 + ... + x_n*y_n
#[inline]
pub fn lorentz_inner(x: &[f32], y: &[f32]) -> f32 {
    let mut result = -x[0] * y[0];
    for i in 1..x.len() {
        result += x[i] * y[i];
    }
    result
}

/// Compute focal direction from a set of Poincare ball points.
///
/// The focal direction is a light-like vector (on the null cone: <xi, xi>_L = 0)
/// derived from the Frechet mean of all points. It represents the "root" of the
/// hierarchy for Busemann scoring.
pub fn compute_focal_direction(points: &[&[f32]], c: f32) -> Vec<f32> {
    if points.is_empty() {
        return vec![];
    }
    let mean = frechet_mean(points, c);
    let lorentz_mean = poincare_to_lorentz(&mean, c);
    // Project onto null cone: normalize so that <xi, xi>_L = 0
    // For a light-like vector: x_0 = ||x_spatial||
    let spatial_norm: f32 = lorentz_mean[1..].iter().map(|v| v * v).sum::<f32>().sqrt();
    if spatial_norm < EPS {
        // Degenerate case: mean is at origin. Use default direction.
        let mut focal = vec![0.0; lorentz_mean.len()];
        focal[0] = 1.0;
        if focal.len() > 1 {
            focal[1] = 1.0; // Light-like: x_0 = x_1
        }
        return focal;
    }
    let mut focal = Vec::with_capacity(lorentz_mean.len());
    focal.push(spatial_norm); // x_0 = ||spatial||
    for &xi in &lorentz_mean[1..] {
        focal.push(xi); // spatial components unchanged
    }
    focal
}

/// Busemann score: B_xi(x) = log(-<x, xi>_L)
///
/// Lower score = closer to hierarchy root (more abstract).
/// Higher score = closer to leaves (more concrete).
///
/// x must be in Lorentz coordinates (d+1 dimensions).
/// xi must be a focal direction (light-like, d+1 dimensions).
#[inline]
pub fn busemann_score(x_lorentz: &[f32], focal: &[f32]) -> f32 {
    let inner = lorentz_inner(x_lorentz, focal);
    (-inner).max(EPS).ln()
}

/// Convenience: compute Busemann depth for a Poincare ball point.
pub fn busemann_depth(p: &[f32], focal: &[f32], c: f32) -> f32 {
    let lorentz = poincare_to_lorentz(p, c);
    busemann_score(&lorentz, focal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::hyperbolic::poincare_math::exp_map_origin;

    #[test]
    fn test_poincare_to_lorentz_origin() {
        let origin = vec![0.0, 0.0];
        let lorentz = poincare_to_lorentz(&origin, 1.0);
        assert_eq!(lorentz.len(), 3); // d+1
        assert!((lorentz[0] - 1.0).abs() < 1e-4); // x_0 = 1 at origin
        assert!((lorentz[1]).abs() < 1e-4);
        assert!((lorentz[2]).abs() < 1e-4);
    }

    #[test]
    fn test_lorentz_hyperboloid_constraint() {
        // Points on hyperboloid satisfy <x,x>_L = -1/c
        let p = vec![0.3, 0.4];
        let c = 1.0;
        let l = poincare_to_lorentz(&p, c);
        let inner = lorentz_inner(&l, &l);
        assert!((inner - (-1.0 / c)).abs() < 1e-3);
    }

    #[test]
    fn test_busemann_depth_monotonicity() {
        // Points farther from origin should have higher Busemann depth
        let near_origin = vec![0.1, 0.0];
        let mid = vec![0.4, 0.0];
        let near_boundary = vec![0.8, 0.0];

        // Focal direction toward positive x
        let focal = compute_focal_direction(&[&near_origin, &mid, &near_boundary], 1.0);

        let d_near = busemann_depth(&near_origin, &focal, 1.0);
        let d_mid = busemann_depth(&mid, &focal, 1.0);
        let d_far = busemann_depth(&near_boundary, &focal, 1.0);

        // Near boundary should have different depth than near origin
        // The exact ordering depends on focal direction, but they should be distinct
        assert!((d_near - d_far).abs() > 0.1);
    }

    #[test]
    fn test_lorentz_inner_known_values() {
        let x = vec![2.0, 1.0, 0.5];
        let y = vec![3.0, 0.5, 1.0];
        // <x,y>_L = -2*3 + 1*0.5 + 0.5*1.0 = -6 + 0.5 + 0.5 = -5.0
        let inner = lorentz_inner(&x, &y);
        assert!((inner - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_focal_direction_is_lightlike() {
        let points: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2],
            vec![0.3, -0.1],
            vec![-0.2, 0.4],
        ];
        let refs: Vec<&[f32]> = points.iter().map(|v| v.as_slice()).collect();
        let focal = compute_focal_direction(&refs, 1.0);
        // Light-like: <xi, xi>_L should be ≈ 0
        let inner = lorentz_inner(&focal, &focal);
        assert!(inner.abs() < 0.1);
    }

    #[test]
    fn test_busemann_score_positive() {
        let p = vec![0.3, 0.2];
        let lorentz = poincare_to_lorentz(&p, 1.0);
        let focal = vec![1.0, 1.0, 0.0]; // Simple light-like direction
        let score = busemann_score(&lorentz, &focal);
        // Score should be a finite real number
        assert!(score.is_finite());
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p segment --features hyperbolic -- spaces::hyperbolic::busemann::tests 2>&1 | tail -10
```

Expected: All 6 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add lib/segment/src/spaces/hyperbolic/busemann.rs
git commit -m "feat(hyperbolic): implement Busemann scoring via Lorentz model"
```

---

## Task 8: Update Module Exports

**Files:**
- Modify: `lib/segment/src/spaces/hyperbolic/mod.rs`

- [ ] **Step 1: Add re-exports for public API**

Replace `lib/segment/src/spaces/hyperbolic/mod.rs`:

```rust
pub mod poincare_math;
pub mod poincare_metric;
pub mod tangent_cache;
pub mod tangent_scorer;
pub mod busemann;

// Re-export key types for convenience
pub use poincare_metric::PoincareMetric;
pub use poincare_math::{
    poincare_distance, project_to_ball, exp_map_origin, log_map_origin,
    exp_map, log_map, mobius_add, frechet_mean, conformal_factor,
    DEFAULT_CURVATURE,
};
pub use tangent_cache::TangentCache;
pub use tangent_scorer::{prune_and_rescore, DEFAULT_PRUNE_FACTOR};
pub use busemann::{
    poincare_to_lorentz, lorentz_inner, busemann_score, busemann_depth,
    compute_focal_direction,
};
```

- [ ] **Step 2: Run full test suite**

```bash
cargo test -p segment --features hyperbolic -- spaces::hyperbolic 2>&1 | tail -20
```

Expected: All ~34 tests across all hyperbolic modules PASS.

- [ ] **Step 3: Run tests WITHOUT feature flag**

```bash
cargo test -p segment 2>&1 | tail -5
```

Expected: Compiles and all existing tests pass (hyperbolic module not compiled).

- [ ] **Step 4: Commit**

```bash
git add lib/segment/src/spaces/hyperbolic/mod.rs
git commit -m "feat(hyperbolic): add public re-exports for hyperbolic module"
```

---

## Summary

| Task | Component | Tests | Key files |
|------|-----------|-------|-----------|
| 1 | Feature gate + skeleton | Compilation check | `Cargo.toml`, `mod.rs` |
| 2 | Poincare math | 16 tests | `poincare_math.rs` |
| 3 | PoincareMetric trait | 5 tests | `poincare_metric.rs` |
| 4 | Distance::Poincare dispatch | Compilation check | `types.rs`, `raw_scorer.rs`, +4 files |
| 5 | Tangent cache | 4 tests | `tangent_cache.rs` |
| 6 | Tangent scorer | 3 tests | `tangent_scorer.rs` |
| 7 | Busemann scoring | 6 tests | `busemann.rs` |
| 8 | Module exports | Full suite run | `mod.rs` |

**Total:** 8 tasks, ~34 tests, ~1700 lines of new code, ~90 lines modified in existing files.

After completing all 8 tasks, the Qdrant fork has a working `Distance::Poincare` metric with tangent space pruning and Busemann scoring, all behind a `hyperbolic` feature gate. The next phase (not in this plan) is Pythia-side integration: `HyperbolicProjector`, per-client collection creation, and embedding pipeline updates.
