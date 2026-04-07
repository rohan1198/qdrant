/// Busemann scoring via the Lorentz (hyperboloid) model.
///
/// This module enables hierarchy-aware ranking — measuring how "deep" a point
/// is in a tree-like hierarchy.  Low score = near root (abstract), high score
/// = near leaves (concrete).

use super::poincare_math::{frechet_mean, EPS};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert a Poincaré ball point `p` (dimension `d`) to a point on the Lorentz
/// hyperboloid (dimension `d+1`).
///
/// Formula (curvature `-c`):
/// ```text
/// x_0   = (1 + c‖p‖²) / (1 − c‖p‖²)
/// x_i   = 2√c · p_i  / (1 − c‖p‖²)   for i ≥ 1
/// ```
///
/// The returned vector has length `d + 1`.
pub fn poincare_to_lorentz(p: &[f32], c: f32) -> Vec<f32> {
    let p_sq: f32 = p.iter().map(|x| x * x).sum();
    let denom = (1.0 - c * p_sq).max(EPS);
    let sc = c.sqrt();

    let mut result = Vec::with_capacity(p.len() + 1);
    result.push((1.0 + c * p_sq) / denom); // x_0
    for &pi in p {
        result.push(2.0 * sc * pi / denom); // x_i, i >= 1
    }
    result
}

/// Lorentz (Minkowski) inner product: `⟨x, y⟩_L = −x_0 y_0 + x_1 y_1 + … + x_n y_n`.
pub fn lorentz_inner(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len(), "dimension mismatch in lorentz_inner");
    debug_assert!(!x.is_empty(), "lorentz_inner called with empty slice");

    let mut iter = x.iter().zip(y.iter());
    // First component contributes with a minus sign
    let (x0, y0) = iter.next().unwrap();
    let mut result = -x0 * y0;
    for (xi, yi) in iter {
        result += xi * yi;
    }
    result
}

/// Compute the focal direction (a null/light-like vector on the boundary of the
/// light cone) from a set of Poincaré ball points.
///
/// Steps:
/// 1. Compute the Fréchet mean of the input points in the Poincaré ball.
/// 2. Convert that mean to a Lorentz vector.
/// 3. Project to the null cone: set `x_0 = ‖x_spatial‖` so that
///    `⟨xi, xi⟩_L = 0`.
///
/// Degenerate case (mean at or near origin): return `[1, 1, 0, …, 0]` as a
/// fallback null vector.
pub fn compute_focal_direction(points: &[&[f32]], c: f32) -> Vec<f32> {
    let mean = frechet_mean(points, c);
    let lorentz = poincare_to_lorentz(&mean, c);

    // Spatial part = lorentz[1..]
    let spatial_norm_sq: f32 = lorentz[1..].iter().map(|x| x * x).sum();
    let spatial_norm = spatial_norm_sq.sqrt();

    if spatial_norm < EPS {
        // Degenerate: mean at origin → return a canonical null vector
        let dim = lorentz.len();
        let mut focal = vec![0.0f32; dim];
        focal[0] = 1.0;
        if dim > 1 {
            focal[1] = 1.0;
        }
        return focal;
    }

    // Project to null cone: replace x_0 with ‖x_spatial‖
    let mut focal = lorentz;
    focal[0] = spatial_norm;
    focal
}

/// Busemann score of a Lorentz point `x_lorentz` with respect to a focal
/// direction `focal` (a null vector):
///
/// ```text
/// B_xi(x) = -log(-⟨x, xi⟩_L)
/// ```
///
/// Low score → near hierarchy root (abstract); high score → near leaves (concrete).
/// As a point moves deeper toward the boundary in direction xi, `-⟨x, xi⟩_L`
/// decreases toward 0, so `-log` of it increases toward +∞.
pub fn busemann_score(x_lorentz: &[f32], focal: &[f32]) -> f32 {
    let inner = lorentz_inner(x_lorentz, focal);
    // inner is negative; take -log of its absolute value so deeper points score higher.
    let abs_inner = (-inner).max(EPS);
    -abs_inner.ln()
}

/// Convenience function: convert a Poincaré ball point `p` to Lorentz
/// coordinates and compute its Busemann score relative to `focal`.
pub fn busemann_depth(p: &[f32], focal: &[f32], c: f32) -> f32 {
    let x_lorentz = poincare_to_lorentz(p, c);
    busemann_score(&x_lorentz, focal)
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

    /// Origin [0,0] → x_0 = 1, spatial ≈ 0, result dim = 3.
    #[test]
    fn test_poincare_to_lorentz_origin() {
        let p = [0.0f32, 0.0];
        let l = poincare_to_lorentz(&p, 1.0);
        assert_eq!(l.len(), 3, "expected dim 3, got {}", l.len());
        assert!(
            approx_eq(l[0], 1.0),
            "x_0 should be 1.0, got {}",
            l[0]
        );
        assert!(
            approx_eq(l[1], 0.0) && approx_eq(l[2], 0.0),
            "spatial should be 0, got {:?}",
            &l[1..]
        );
    }

    /// Points on the hyperboloid satisfy `⟨x, x⟩_L = −1/c`.
    #[test]
    fn test_lorentz_hyperboloid_constraint() {
        let c = 1.0f32;
        let points: &[&[f32]] = &[
            &[0.0, 0.0],
            &[0.3, 0.4],
            &[-0.2, 0.1],
            &[0.5, 0.0],
        ];
        for &p in points {
            let l = poincare_to_lorentz(p, c);
            let inner = lorentz_inner(&l, &l);
            assert!(
                approx_eq(inner, -1.0 / c),
                "hyperboloid constraint failed for {:?}: ⟨x,x⟩_L = {}, expected {}",
                p,
                inner,
                -1.0 / c
            );
        }
    }

    /// Points at different radii should have different Busemann depths.
    #[test]
    fn test_busemann_depth_monotonicity() {
        let c = 1.0f32;
        // All points along the positive x-axis at increasing radii.
        let p_near = [0.1f32, 0.0];
        let p_far = [0.8f32, 0.0];

        // Focal direction pointing along x-axis
        let focal_pts: &[&[f32]] = &[&[0.9f32, 0.0]];
        let focal = compute_focal_direction(focal_pts, c);

        let score_near = busemann_depth(&p_near, &focal, c);
        let score_far = busemann_depth(&p_far, &focal, c);

        assert!(
            score_far > score_near,
            "expected score_far ({}) > score_near ({})",
            score_far,
            score_near
        );
    }

    /// Known value: x=[2,1,0.5], y=[3,0.5,1] → inner = -2*3 + 1*0.5 + 0.5*1 = -5.0
    #[test]
    fn test_lorentz_inner_known_values() {
        let x = [2.0f32, 1.0, 0.5];
        let y = [3.0f32, 0.5, 1.0];
        let inner = lorentz_inner(&x, &y);
        assert!(
            approx_eq(inner, -5.0),
            "expected -5.0, got {}",
            inner
        );
    }

    /// The focal direction should satisfy `⟨xi, xi⟩_L ≈ 0` (null / light-like).
    #[test]
    fn test_compute_focal_direction_is_lightlike() {
        let c = 1.0f32;
        let points: &[&[f32]] = &[
            &[0.2f32, 0.3],
            &[0.4f32, -0.1],
            &[-0.1f32, 0.5],
        ];
        let focal = compute_focal_direction(points, c);
        let inner = lorentz_inner(&focal, &focal);
        assert!(
            inner.abs() < TOL,
            "focal direction is not light-like: ⟨xi,xi⟩_L = {}",
            inner
        );
    }

    /// Busemann score should be a finite real number.
    #[test]
    fn test_busemann_score_positive() {
        let c = 1.0f32;
        let p = [0.3f32, 0.4];
        let points: &[&[f32]] = &[&[0.5f32, 0.5]];
        let focal = compute_focal_direction(points, c);
        let score = busemann_depth(&p, &focal, c);
        assert!(
            score.is_finite(),
            "Busemann score should be finite, got {}",
            score
        );
    }
}
