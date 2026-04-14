#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::data_types::vectors::VectorElementType;
use common::types::ScoreType;

use super::poincare_math::{stable_acosh, EPS};

#[target_feature(enable = "sse")]
#[allow(clippy::missing_safety_doc)]
unsafe fn hsum128_ps_sse(v: __m128) -> f32 {
    let x64: __m128 = _mm_add_ps(v, _mm_movehl_ps(v, v));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

/// SSE-accelerated Poincaré similarity.
///
/// Computes `diff_sq = ||u - v||^2`, `u_sq = ||u||^2`, `v_sq = ||v||^2` in a
/// single pass using 12 `__m128` accumulators (4 per norm), then applies the
/// Poincaré distance formula and returns the negated distance (similarity
/// semantics, matching `PoincareMetric::similarity`).
#[target_feature(enable = "sse")]
pub(crate) unsafe fn poincare_similarity_sse(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
    curvature: f32,
) -> ScoreType {
    unsafe {
        let n = v1.len();
        let m = n - (n % 16);
        let mut ptr1: *const f32 = v1.as_ptr();
        let mut ptr2: *const f32 = v2.as_ptr();

        // 4-way accumulators for diff_sq
        let mut diff_1: __m128 = _mm_setzero_ps();
        let mut diff_2: __m128 = _mm_setzero_ps();
        let mut diff_3: __m128 = _mm_setzero_ps();
        let mut diff_4: __m128 = _mm_setzero_ps();

        // 4-way accumulators for u_sq
        let mut usq_1: __m128 = _mm_setzero_ps();
        let mut usq_2: __m128 = _mm_setzero_ps();
        let mut usq_3: __m128 = _mm_setzero_ps();
        let mut usq_4: __m128 = _mm_setzero_ps();

        // 4-way accumulators for v_sq
        let mut vsq_1: __m128 = _mm_setzero_ps();
        let mut vsq_2: __m128 = _mm_setzero_ps();
        let mut vsq_3: __m128 = _mm_setzero_ps();
        let mut vsq_4: __m128 = _mm_setzero_ps();

        let mut i: usize = 0;
        while i < m {
            // Sub-iteration 1: elements [0..4)
            let u1 = _mm_loadu_ps(ptr1);
            let v1_ = _mm_loadu_ps(ptr2);
            let sub1 = _mm_sub_ps(u1, v1_);
            diff_1 = _mm_add_ps(_mm_mul_ps(sub1, sub1), diff_1);
            usq_1 = _mm_add_ps(_mm_mul_ps(u1, u1), usq_1);
            vsq_1 = _mm_add_ps(_mm_mul_ps(v1_, v1_), vsq_1);

            // Sub-iteration 2: elements [4..8)
            let u2 = _mm_loadu_ps(ptr1.add(4));
            let v2_ = _mm_loadu_ps(ptr2.add(4));
            let sub2 = _mm_sub_ps(u2, v2_);
            diff_2 = _mm_add_ps(_mm_mul_ps(sub2, sub2), diff_2);
            usq_2 = _mm_add_ps(_mm_mul_ps(u2, u2), usq_2);
            vsq_2 = _mm_add_ps(_mm_mul_ps(v2_, v2_), vsq_2);

            // Sub-iteration 3: elements [8..12)
            let u3 = _mm_loadu_ps(ptr1.add(8));
            let v3 = _mm_loadu_ps(ptr2.add(8));
            let sub3 = _mm_sub_ps(u3, v3);
            diff_3 = _mm_add_ps(_mm_mul_ps(sub3, sub3), diff_3);
            usq_3 = _mm_add_ps(_mm_mul_ps(u3, u3), usq_3);
            vsq_3 = _mm_add_ps(_mm_mul_ps(v3, v3), vsq_3);

            // Sub-iteration 4: elements [12..16)
            let u4 = _mm_loadu_ps(ptr1.add(12));
            let v4 = _mm_loadu_ps(ptr2.add(12));
            let sub4 = _mm_sub_ps(u4, v4);
            diff_4 = _mm_add_ps(_mm_mul_ps(sub4, sub4), diff_4);
            usq_4 = _mm_add_ps(_mm_mul_ps(u4, u4), usq_4);
            vsq_4 = _mm_add_ps(_mm_mul_ps(v4, v4), vsq_4);

            ptr1 = ptr1.add(16);
            ptr2 = ptr2.add(16);
            i += 16;
        }

        // Horizontal sums
        let mut diff_sq = hsum128_ps_sse(diff_1)
            + hsum128_ps_sse(diff_2)
            + hsum128_ps_sse(diff_3)
            + hsum128_ps_sse(diff_4);
        let mut u_sq = hsum128_ps_sse(usq_1)
            + hsum128_ps_sse(usq_2)
            + hsum128_ps_sse(usq_3)
            + hsum128_ps_sse(usq_4);
        let mut v_sq = hsum128_ps_sse(vsq_1)
            + hsum128_ps_sse(vsq_2)
            + hsum128_ps_sse(vsq_3)
            + hsum128_ps_sse(vsq_4);

        // Scalar remainder
        for i in 0..n - m {
            let ui = *ptr1.add(i);
            let vi = *ptr2.add(i);
            let d = ui - vi;
            diff_sq += d * d;
            u_sq += ui * ui;
            v_sq += vi * vi;
        }

        // Poincaré distance formula
        let c = curvature;
        let denom_u = (1.0 - c * u_sq).max(EPS);
        let denom_v = (1.0 - c * v_sq).max(EPS);
        let arg = 1.0 + 2.0 * c * diff_sq / (denom_u * denom_v);

        -(stable_acosh(arg) / c.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::hyperbolic::poincare_math::poincare_distance;

    #[test]
    fn test_poincare_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse") {
            println!("sse not detected, skipping test");
            return;
        }

        // 128-dimensional vectors inside the Poincaré ball
        let dim = 128;
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.003) - 0.2).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002) + 0.05).collect();

        for &c in &[1.0, 5.0] {
            let scalar = -poincare_distance(&v1, &v2, c);
            let simd = unsafe { poincare_similarity_sse(&v1, &v2, c) };
            assert!(
                (scalar - simd).abs() < 1e-4,
                "c={c}: scalar={scalar}, simd={simd}, diff={}",
                (scalar - simd).abs()
            );
        }
    }

    #[test]
    fn test_poincare_sse_odd_dimensions() {
        if !is_x86_feature_detected!("sse") {
            println!("sse not detected, skipping test");
            return;
        }

        // 70 dimensions — not divisible by 16, exercises remainder path
        let dim = 70;
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.004) - 0.15).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.003) + 0.02).collect();

        let c = 1.0;
        let scalar = -poincare_distance(&v1, &v2, c);
        let simd = unsafe { poincare_similarity_sse(&v1, &v2, c) };
        assert!(
            (scalar - simd).abs() < 1e-4,
            "odd dim: scalar={scalar}, simd={simd}, diff={}",
            (scalar - simd).abs()
        );
    }
}
