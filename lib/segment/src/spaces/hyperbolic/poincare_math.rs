/// Poincaré ball model — core math operations.
///
/// All functions operate on `&[f32]` slices and have **no** Qdrant dependencies.
/// The curvature parameter `c` is always positive; `c = 1` gives the standard
/// unit Poincaré ball.

/// Default (negative) curvature magnitude.
pub const DEFAULT_CURVATURE: f32 = 1.0;

/// Small epsilon used to avoid division-by-zero and keep points strictly inside
/// the ball.
const EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute `||u - v||^2`, `||u||^2` and `||v||^2` in a single pass over both
/// slices (avoids three separate iterations).
fn fused_norms(u: &[f32], v: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(u.len(), v.len(), "dimension mismatch");
    let mut diff_sq: f32 = 0.0;
    let mut u_sq: f32 = 0.0;
    let mut v_sq: f32 = 0.0;
    for (ui, vi) in u.iter().zip(v.iter()) {
        let d = ui - vi;
        diff_sq += d * d;
        u_sq += ui * ui;
        v_sq += vi * vi;
    }
    (diff_sq, u_sq, v_sq)
}

/// Numerically stable `acosh`.
///
/// Near `x = 1` the standard `f32::acosh` can lose precision, so we use a
/// first-order Taylor expansion: `acosh(1 + δ) ≈ sqrt(2δ)` for small δ.
fn stable_acosh(x: f32) -> f32 {
    if x <= 1.0 {
        return 0.0;
    }
    let delta = x - 1.0;
    if delta < 1e-6 {
        // Taylor: acosh(1+δ) ≈ sqrt(2δ) for small δ
        (2.0 * delta).sqrt()
    } else {
        x.acosh()
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Geodesic distance in the Poincaré ball of curvature `-c`:
///
/// ```text
/// d(u, v) = (1 / √c) · acosh(1 + 2c·‖u−v‖² / ((1 − c‖u‖²)(1 − c‖v‖²)))
/// ```
pub fn poincare_distance(u: &[f32], v: &[f32], c: f32) -> f32 {
    let (diff_sq, u_sq, v_sq) = fused_norms(u, v);

    let denom_u = (1.0 - c * u_sq).max(EPS);
    let denom_v = (1.0 - c * v_sq).max(EPS);

    let arg = 1.0 + 2.0 * c * diff_sq / (denom_u * denom_v);

    stable_acosh(arg) / c.sqrt()
}

/// Project a vector into the open Poincaré ball of radius `1/√c − ε`.
///
/// If the vector is already inside the ball it is returned unchanged.
pub fn project_to_ball(mut v: Vec<f32>, c: f32) -> Vec<f32> {
    let max_norm = 1.0 / c.sqrt() - EPS;
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm >= max_norm {
        let scale = max_norm / norm.max(EPS);
        for x in v.iter_mut() {
            *x *= scale;
        }
    }
    v
}

/// Exponential map at the **origin** of the Poincaré ball:
///
/// ```text
/// exp_0(v) = tanh(√c · ‖v‖ / 2) · v / (√c · ‖v‖)
/// ```
pub fn exp_map_origin(v: &[f32], c: f32) -> Vec<f32> {
    let sc = c.sqrt();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < EPS {
        return vec![0.0; v.len()];
    }
    let coeff = (sc * norm / 2.0).tanh() / (sc * norm);
    v.iter().map(|&x| x * coeff).collect()
}

/// Logarithmic map at the **origin** of the Poincaré ball:
///
/// ```text
/// log_0(p) = arctanh(√c · ‖p‖) / (√c · ‖p‖) · p
/// ```
pub fn log_map_origin(p: &[f32], c: f32) -> Vec<f32> {
    let sc = c.sqrt();
    let norm: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < EPS {
        return vec![0.0; p.len()];
    }
    let sc_norm = (sc * norm).min(1.0 - EPS); // clamp for atanh domain
    let coeff = sc_norm.atanh() / (sc * norm);
    p.iter().map(|&x| x * coeff).collect()
}

/// Möbius addition in the Poincaré ball:
///
/// ```text
/// x ⊕ y = ((1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 − c‖x‖²)y)
///          / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)
/// ```
pub fn mobius_add(x: &[f32], y: &[f32], c: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), y.len());
    let mut xy: f32 = 0.0;
    let mut x_sq: f32 = 0.0;
    let mut y_sq: f32 = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        xy += xi * yi;
        x_sq += xi * xi;
        y_sq += yi * yi;
    }

    let num_x = 1.0 + 2.0 * c * xy + c * y_sq;
    let num_y = 1.0 - c * x_sq;
    let denom = (1.0 + 2.0 * c * xy + c * c * x_sq * y_sq).max(EPS);

    let result: Vec<f32> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (num_x * xi + num_y * yi) / denom)
        .collect();

    project_to_ball(result, c)
}

/// Negate each element (Möbius inverse is simply `−x` in the Poincaré model).
pub fn mobius_neg(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| -v).collect()
}

/// Conformal factor (Riemannian metric scaling) at point `x`:
///
/// ```text
/// λ(x) = 2 / (1 − c‖x‖²)
/// ```
pub fn conformal_factor(x: &[f32], c: f32) -> f32 {
    let x_sq: f32 = x.iter().map(|v| v * v).sum();
    2.0 / (1.0 - c * x_sq).max(EPS)
}

/// Exponential map at an **arbitrary** base point `p`:
///
/// ```text
/// exp_p(v) = p ⊕ (tanh(√c · λ_p · ‖v‖ / 2) · v / (√c · ‖v‖))
/// ```
///
/// where `λ_p` is the conformal factor at `p`.
pub fn exp_map(p: &[f32], v: &[f32], c: f32) -> Vec<f32> {
    let sc = c.sqrt();
    let lambda = conformal_factor(p, c);
    let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if v_norm < EPS {
        return p.to_vec();
    }
    let coeff = (sc * lambda * v_norm / 2.0).tanh() / (sc * v_norm);
    let direction: Vec<f32> = v.iter().map(|&x| x * coeff).collect();
    mobius_add(p, &direction, c)
}

/// Logarithmic map at an **arbitrary** base point `p` towards `y`:
///
/// ```text
/// log_p(y) = (2 / (√c · λ_p)) · arctanh(√c · ‖−p⊕y‖) · (−p⊕y) / ‖−p⊕y‖
/// ```
pub fn log_map(p: &[f32], y: &[f32], c: f32) -> Vec<f32> {
    let sc = c.sqrt();
    let lambda = conformal_factor(p, c);

    let neg_p = mobius_neg(p);
    let add = mobius_add(&neg_p, y, c);

    let add_norm: f32 = add.iter().map(|x| x * x).sum::<f32>().sqrt();
    if add_norm < EPS {
        return vec![0.0; p.len()];
    }

    let sc_add_norm = (sc * add_norm).min(1.0 - EPS);
    let coeff = (2.0 / (sc * lambda)) * sc_add_norm.atanh() / add_norm;

    add.iter().map(|&x| x * coeff).collect()
}

/// Fréchet mean (hyperbolic centroid) of a set of points via Riemannian
/// gradient descent using exp/log maps.
///
/// Algorithm:
/// 1. Initialise the mean at the Euclidean centroid projected onto the ball.
/// 2. Repeatedly compute the tangent-space average of log-maps from the
///    current mean to every input point, then step with the exp-map.
/// 3. Stop when the tangent update norm drops below `tol` or after `max_iter`.
pub fn frechet_mean(points: &[&[f32]], c: f32) -> Vec<f32> {
    assert!(!points.is_empty(), "cannot compute mean of empty set");

    let dim = points[0].len();
    let n = points.len() as f32;

    // Initialise at the Euclidean centroid, projected inside the ball.
    let mut mean = vec![0.0f32; dim];
    for &p in points {
        for (m, &pi) in mean.iter_mut().zip(p.iter()) {
            *m += pi;
        }
    }
    for m in mean.iter_mut() {
        *m /= n;
    }
    mean = project_to_ball(mean, c);

    let lr: f32 = 0.1;
    let max_iter: usize = 100;
    let tol: f32 = 1e-6;

    for _ in 0..max_iter {
        // Riemannian gradient = average of log_mean(p_i)
        let mut grad = vec![0.0f32; dim];
        for &p in points {
            let lm = log_map(&mean, p, c);
            for (g, &l) in grad.iter_mut().zip(lm.iter()) {
                *g += l;
            }
        }
        for g in grad.iter_mut() {
            *g /= n;
        }

        let grad_norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
        if grad_norm < tol {
            break;
        }

        // Scale gradient by learning rate
        let step: Vec<f32> = grad.iter().map(|&g| g * lr).collect();

        mean = exp_map(&mean, &step, c);
        mean = project_to_ball(mean, c);
    }

    mean
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    fn vec_approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(&ai, &bi)| approx_eq(ai, bi))
    }

    // -- fused_norms ---------------------------------------------------------

    #[test]
    fn test_fused_norms_basic() {
        let u = [1.0f32, 0.0];
        let v = [0.0f32, 1.0];
        let (diff_sq, u_sq, v_sq) = fused_norms(&u, &v);
        assert!(approx_eq(diff_sq, 2.0), "diff_sq = {diff_sq}");
        assert!(approx_eq(u_sq, 1.0), "u_sq = {u_sq}");
        assert!(approx_eq(v_sq, 1.0), "v_sq = {v_sq}");
    }

    #[test]
    fn test_fused_norms_same_vector() {
        let u = [3.0f32, 4.0, 5.0];
        let (diff_sq, _u_sq, _v_sq) = fused_norms(&u, &u);
        assert!(approx_eq(diff_sq, 0.0), "diff_sq = {diff_sq}");
    }

    // -- stable_acosh --------------------------------------------------------

    #[test]
    fn test_stable_acosh_at_one() {
        assert!(approx_eq(stable_acosh(1.0), 0.0));
    }

    #[test]
    fn test_stable_acosh_known_value() {
        // acosh(2) ≈ 1.31695…
        let val = stable_acosh(2.0);
        assert!(
            (val - 1.3169).abs() < 0.001,
            "acosh(2) = {val}, expected ≈ 1.3169"
        );
    }

    // -- poincare_distance ---------------------------------------------------

    #[test]
    fn test_distance_identity() {
        let x = [0.3f32, 0.4];
        let d = poincare_distance(&x, &x, DEFAULT_CURVATURE);
        assert!(approx_eq(d, 0.0), "d(x,x) = {d}");
    }

    #[test]
    fn test_distance_symmetry() {
        let x = [0.1f32, 0.2];
        let y = [0.3f32, -0.1];
        let dxy = poincare_distance(&x, &y, DEFAULT_CURVATURE);
        let dyx = poincare_distance(&y, &x, DEFAULT_CURVATURE);
        assert!(approx_eq(dxy, dyx), "d(x,y)={dxy} != d(y,x)={dyx}");
    }

    #[test]
    fn test_distance_non_negative() {
        let x = [0.2f32, -0.3];
        let y = [-0.1f32, 0.4];
        let d = poincare_distance(&x, &y, DEFAULT_CURVATURE);
        assert!(d >= -TOL, "d = {d}");
    }

    #[test]
    fn test_distance_triangle_inequality() {
        let x = [0.1f32, 0.2];
        let y = [0.3f32, -0.1];
        let z = [-0.2f32, 0.3];
        let dxy = poincare_distance(&x, &y, DEFAULT_CURVATURE);
        let dyz = poincare_distance(&y, &z, DEFAULT_CURVATURE);
        let dxz = poincare_distance(&x, &z, DEFAULT_CURVATURE);
        assert!(
            dxz <= dxy + dyz + TOL,
            "triangle inequality violated: d(x,z)={dxz} > d(x,y)+d(y,z)={:.6}",
            dxy + dyz
        );
    }

    #[test]
    fn test_distance_from_origin() {
        // d(0, [0.5, 0]) with c=1:
        //   ||u-v||^2 = 0.25, ||u||^2 = 0, ||v||^2 = 0.25
        //   arg = 1 + 2*0.25 / (1 * 0.75) = 1 + 0.5/0.75 = 1 + 2/3 = 5/3
        //   d = acosh(5/3) ≈ 1.0986
        let origin = [0.0f32, 0.0];
        let p = [0.5f32, 0.0];
        let d = poincare_distance(&origin, &p, DEFAULT_CURVATURE);
        assert!(
            (d - 1.0986).abs() < 0.001,
            "d(0, [0.5,0]) = {d}, expected ≈ 1.0986"
        );
    }

    // -- project_to_ball -----------------------------------------------------

    #[test]
    fn test_project_inside_ball_unchanged() {
        let v = vec![0.3f32, 0.4]; // norm = 0.5
        let p = project_to_ball(v.clone(), DEFAULT_CURVATURE);
        assert!(vec_approx_eq(&p, &v), "vector changed: {p:?}");
    }

    #[test]
    fn test_project_outside_ball_clamped() {
        let v = vec![3.0f32, 4.0]; // norm = 5
        let p = project_to_ball(v, DEFAULT_CURVATURE);
        let norm: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
        let max_norm = 1.0 / DEFAULT_CURVATURE.sqrt() - EPS;
        assert!(
            norm < max_norm + TOL,
            "norm = {norm}, max = {max_norm}"
        );
    }

    #[test]
    fn test_project_zero_vector() {
        let v = vec![0.0f32, 0.0];
        let p = project_to_ball(v, DEFAULT_CURVATURE);
        assert!(vec_approx_eq(&p, &[0.0, 0.0]), "zero changed: {p:?}");
    }

    // -- exp / log at origin -------------------------------------------------

    #[test]
    fn test_exp_log_roundtrip_at_origin() {
        let v = [0.3f32, -0.2, 0.1];
        let p = exp_map_origin(&v, DEFAULT_CURVATURE);
        let recovered = log_map_origin(&p, DEFAULT_CURVATURE);
        assert!(
            vec_approx_eq(&recovered, &v),
            "roundtrip failed: v={v:?}, recovered={recovered:?}"
        );
    }

    #[test]
    fn test_exp_map_zero_is_origin() {
        let zero = [0.0f32, 0.0, 0.0];
        let result = exp_map_origin(&zero, DEFAULT_CURVATURE);
        assert!(
            vec_approx_eq(&result, &zero),
            "exp_0(0) = {result:?}"
        );
    }

    #[test]
    fn test_exp_map_inside_ball() {
        let v = [1.0f32, 2.0, 3.0];
        let p = exp_map_origin(&v, DEFAULT_CURVATURE);
        let norm: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1.0, "norm = {norm}, should be < 1");
    }

    // -- Möbius operations ---------------------------------------------------

    #[test]
    fn test_mobius_right_identity() {
        let x = [0.3f32, -0.2];
        let zero = [0.0f32, 0.0];
        let result = mobius_add(&x, &zero, DEFAULT_CURVATURE);
        assert!(
            vec_approx_eq(&result, &x),
            "x ⊕ 0 = {result:?}, expected {x:?}"
        );
    }

    #[test]
    fn test_mobius_inverse() {
        let x = [0.3f32, -0.4];
        let neg_x = mobius_neg(&x);
        let result = mobius_add(&x, &neg_x, DEFAULT_CURVATURE);
        let zero = [0.0f32, 0.0];
        assert!(
            vec_approx_eq(&result, &zero),
            "x ⊕ (−x) = {result:?}, expected ≈ 0"
        );
    }

    // -- Fréchet mean --------------------------------------------------------

    #[test]
    fn test_frechet_mean_single_point() {
        let x: &[f32] = &[0.3f32, -0.2];
        let mean = frechet_mean(&[x], DEFAULT_CURVATURE);
        assert!(
            vec_approx_eq(&mean, x),
            "mean([x]) = {mean:?}, expected {x:?}"
        );
    }

    #[test]
    fn test_frechet_mean_symmetric_near_origin() {
        let a: &[f32] = &[0.3f32, 0.0];
        let b: &[f32] = &[-0.3f32, 0.0];
        let mean = frechet_mean(&[a, b], DEFAULT_CURVATURE);
        let origin = [0.0f32, 0.0];
        assert!(
            vec_approx_eq(&mean, &origin),
            "symmetric mean = {mean:?}, expected ≈ origin"
        );
    }
}
