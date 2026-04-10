#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;

#[cfg(target_feature = "neon")]
use crate::data_types::vectors::VectorElementType;
#[cfg(target_feature = "neon")]
use common::types::ScoreType;

#[cfg(target_feature = "neon")]
use super::poincare_math::{stable_acosh, EPS};

/// NEON-accelerated Poincaré similarity.
///
/// Computes `diff_sq = ||u - v||^2`, `u_sq = ||u||^2`, `v_sq = ||v||^2` in a
/// single pass using 12 `float32x4_t` accumulators (4 per norm), then applies
/// the Poincaré distance formula and returns the negated distance (similarity
/// semantics, matching `PoincareMetric::similarity`).
#[cfg(target_feature = "neon")]
pub(crate) unsafe fn poincare_similarity_neon(
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
        let mut diff_1 = vdupq_n_f32(0.);
        let mut diff_2 = vdupq_n_f32(0.);
        let mut diff_3 = vdupq_n_f32(0.);
        let mut diff_4 = vdupq_n_f32(0.);

        // 4-way accumulators for u_sq
        let mut usq_1 = vdupq_n_f32(0.);
        let mut usq_2 = vdupq_n_f32(0.);
        let mut usq_3 = vdupq_n_f32(0.);
        let mut usq_4 = vdupq_n_f32(0.);

        // 4-way accumulators for v_sq
        let mut vsq_1 = vdupq_n_f32(0.);
        let mut vsq_2 = vdupq_n_f32(0.);
        let mut vsq_3 = vdupq_n_f32(0.);
        let mut vsq_4 = vdupq_n_f32(0.);

        let mut i: usize = 0;
        while i < m {
            // Sub-iteration 1: elements [0..4)
            let u1 = vld1q_f32(ptr1);
            let v1_ = vld1q_f32(ptr2);
            let sub1 = vsubq_f32(u1, v1_);
            diff_1 = vfmaq_f32(diff_1, sub1, sub1);
            usq_1 = vfmaq_f32(usq_1, u1, u1);
            vsq_1 = vfmaq_f32(vsq_1, v1_, v1_);

            // Sub-iteration 2: elements [4..8)
            let u2 = vld1q_f32(ptr1.add(4));
            let v2_ = vld1q_f32(ptr2.add(4));
            let sub2 = vsubq_f32(u2, v2_);
            diff_2 = vfmaq_f32(diff_2, sub2, sub2);
            usq_2 = vfmaq_f32(usq_2, u2, u2);
            vsq_2 = vfmaq_f32(vsq_2, v2_, v2_);

            // Sub-iteration 3: elements [8..12)
            let u3 = vld1q_f32(ptr1.add(8));
            let v3 = vld1q_f32(ptr2.add(8));
            let sub3 = vsubq_f32(u3, v3);
            diff_3 = vfmaq_f32(diff_3, sub3, sub3);
            usq_3 = vfmaq_f32(usq_3, u3, u3);
            vsq_3 = vfmaq_f32(vsq_3, v3, v3);

            // Sub-iteration 4: elements [12..16)
            let u4 = vld1q_f32(ptr1.add(12));
            let v4 = vld1q_f32(ptr2.add(12));
            let sub4 = vsubq_f32(u4, v4);
            diff_4 = vfmaq_f32(diff_4, sub4, sub4);
            usq_4 = vfmaq_f32(usq_4, u4, u4);
            vsq_4 = vfmaq_f32(vsq_4, v4, v4);

            ptr1 = ptr1.add(16);
            ptr2 = ptr2.add(16);
            i += 16;
        }

        // Horizontal sums
        let mut diff_sq = vaddvq_f32(diff_1) + vaddvq_f32(diff_2) + vaddvq_f32(diff_3) + vaddvq_f32(diff_4);
        let mut u_sq = vaddvq_f32(usq_1) + vaddvq_f32(usq_2) + vaddvq_f32(usq_3) + vaddvq_f32(usq_4);
        let mut v_sq = vaddvq_f32(vsq_1) + vaddvq_f32(vsq_2) + vaddvq_f32(vsq_3) + vaddvq_f32(vsq_4);

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
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_poincare_neon_matches_scalar() {
        use super::*;
        use crate::spaces::hyperbolic::poincare_math::poincare_distance;

        if !std::arch::is_aarch64_feature_detected!("neon") {
            println!("neon not detected, skipping test");
            return;
        }

        // 128-dimensional vectors inside the Poincaré ball
        let dim = 128;
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.003) - 0.2).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002) + 0.05).collect();

        for &c in &[1.0, 5.0] {
            let scalar = -poincare_distance(&v1, &v2, c);
            let simd = unsafe { poincare_similarity_neon(&v1, &v2, c) };
            assert!(
                (scalar - simd).abs() < 1e-4,
                "c={c}: scalar={scalar}, simd={simd}, diff={}",
                (scalar - simd).abs()
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_poincare_neon_odd_dimensions() {
        use super::*;
        use crate::spaces::hyperbolic::poincare_math::poincare_distance;

        if !std::arch::is_aarch64_feature_detected!("neon") {
            println!("neon not detected, skipping test");
            return;
        }

        // 70 dimensions — not divisible by 16, exercises remainder path
        let dim = 70;
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.004) - 0.15).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.003) + 0.02).collect();

        let c = 1.0;
        let scalar = -poincare_distance(&v1, &v2, c);
        let simd = unsafe { poincare_similarity_neon(&v1, &v2, c) };
        assert!(
            (scalar - simd).abs() < 1e-4,
            "odd dim: scalar={scalar}, simd={simd}, diff={}",
            (scalar - simd).abs()
        );
    }
}
