#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::data_types::vectors::VectorElementType;
use common::types::ScoreType;

use super::poincare_math::{stable_acosh, EPS};

#[target_feature(enable = "avx")]
#[allow(clippy::missing_safety_doc)]
unsafe fn hsum256_ps_avx(x: __m256) -> f32 {
    let lr_sum: __m128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let hsum = _mm_hadd_ps(lr_sum, lr_sum);
    let p1 = _mm_extract_ps(hsum, 0);
    let p2 = _mm_extract_ps(hsum, 1);
    f32::from_bits(p1 as u32) + f32::from_bits(p2 as u32)
}

/// Four-way horizontal sum of `__m256` accumulators.
#[target_feature(enable = "avx")]
#[allow(clippy::missing_safety_doc)]
unsafe fn four_way_hsum(a: __m256, b: __m256, c: __m256, d: __m256) -> f32 {
    let sum1 = _mm256_add_ps(a, b);
    let sum2 = _mm256_add_ps(c, d);
    let total = _mm256_add_ps(sum1, sum2);
    unsafe { hsum256_ps_avx(total) }
}

/// AVX2+FMA accelerated Poincare similarity.
///
/// Computes `diff_sq = ||u - v||^2`, `u_sq = ||u||^2`, `v_sq = ||v||^2` in a
/// single pass using 12 `__m256` accumulators (4 per norm), then applies the
/// Poincare distance formula and returns the negated distance (similarity
/// semantics, matching `PoincareMetric::similarity`).
#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
pub(crate) unsafe fn poincare_similarity_avx(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
    curvature: f32,
) -> ScoreType {
    unsafe {
        let n = v1.len();
        let m = n - (n % 32);
        let mut ptr1: *const f32 = v1.as_ptr();
        let mut ptr2: *const f32 = v2.as_ptr();

        // 4-way accumulators for diff_sq
        let mut diff_1: __m256 = _mm256_setzero_ps();
        let mut diff_2: __m256 = _mm256_setzero_ps();
        let mut diff_3: __m256 = _mm256_setzero_ps();
        let mut diff_4: __m256 = _mm256_setzero_ps();

        // 4-way accumulators for u_sq
        let mut usq_1: __m256 = _mm256_setzero_ps();
        let mut usq_2: __m256 = _mm256_setzero_ps();
        let mut usq_3: __m256 = _mm256_setzero_ps();
        let mut usq_4: __m256 = _mm256_setzero_ps();

        // 4-way accumulators for v_sq
        let mut vsq_1: __m256 = _mm256_setzero_ps();
        let mut vsq_2: __m256 = _mm256_setzero_ps();
        let mut vsq_3: __m256 = _mm256_setzero_ps();
        let mut vsq_4: __m256 = _mm256_setzero_ps();

        let mut i: usize = 0;
        while i < m {
            // Sub-iteration 1: elements [0..8)
            let u1 = _mm256_loadu_ps(ptr1);
            let v1_ = _mm256_loadu_ps(ptr2);
            let sub1 = _mm256_sub_ps(u1, v1_);
            diff_1 = _mm256_fmadd_ps(sub1, sub1, diff_1);
            usq_1 = _mm256_fmadd_ps(u1, u1, usq_1);
            vsq_1 = _mm256_fmadd_ps(v1_, v1_, vsq_1);

            // Sub-iteration 2: elements [8..16)
            let u2 = _mm256_loadu_ps(ptr1.add(8));
            let v2_ = _mm256_loadu_ps(ptr2.add(8));
            let sub2 = _mm256_sub_ps(u2, v2_);
            diff_2 = _mm256_fmadd_ps(sub2, sub2, diff_2);
            usq_2 = _mm256_fmadd_ps(u2, u2, usq_2);
            vsq_2 = _mm256_fmadd_ps(v2_, v2_, vsq_2);

            // Sub-iteration 3: elements [16..24)
            let u3 = _mm256_loadu_ps(ptr1.add(16));
            let v3 = _mm256_loadu_ps(ptr2.add(16));
            let sub3 = _mm256_sub_ps(u3, v3);
            diff_3 = _mm256_fmadd_ps(sub3, sub3, diff_3);
            usq_3 = _mm256_fmadd_ps(u3, u3, usq_3);
            vsq_3 = _mm256_fmadd_ps(v3, v3, vsq_3);

            // Sub-iteration 4: elements [24..32)
            let u4 = _mm256_loadu_ps(ptr1.add(24));
            let v4 = _mm256_loadu_ps(ptr2.add(24));
            let sub4 = _mm256_sub_ps(u4, v4);
            diff_4 = _mm256_fmadd_ps(sub4, sub4, diff_4);
            usq_4 = _mm256_fmadd_ps(u4, u4, usq_4);
            vsq_4 = _mm256_fmadd_ps(v4, v4, vsq_4);

            ptr1 = ptr1.add(32);
            ptr2 = ptr2.add(32);
            i += 32;
        }

        // Horizontal sums
        let mut diff_sq = four_way_hsum(diff_1, diff_2, diff_3, diff_4);
        let mut u_sq = four_way_hsum(usq_1, usq_2, usq_3, usq_4);
        let mut v_sq = four_way_hsum(vsq_1, vsq_2, vsq_3, vsq_4);

        // Scalar remainder
        for i in 0..n - m {
            let ui = *ptr1.add(i);
            let vi = *ptr2.add(i);
            let d = ui - vi;
            diff_sq += d * d;
            u_sq += ui * ui;
            v_sq += vi * vi;
        }

        // Poincare distance formula
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
    fn test_poincare_avx_matches_scalar() {
        if !is_x86_feature_detected!("avx") || !is_x86_feature_detected!("fma") {
            println!("avx/fma not detected, skipping test");
            return;
        }

        // 128-dimensional vectors inside the Poincare ball
        let dim = 128;
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.003) - 0.2).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002) + 0.05).collect();

        for &c in &[1.0, 5.0] {
            let scalar = -poincare_distance(&v1, &v2, c);
            let simd = unsafe { poincare_similarity_avx(&v1, &v2, c) };
            assert!(
                (scalar - simd).abs() < 1e-4,
                "c={c}: scalar={scalar}, simd={simd}, diff={}",
                (scalar - simd).abs()
            );
        }
    }

    #[test]
    fn test_poincare_avx_odd_dimensions() {
        if !is_x86_feature_detected!("avx") || !is_x86_feature_detected!("fma") {
            println!("avx/fma not detected, skipping test");
            return;
        }

        // 70 dimensions — not divisible by 32, exercises remainder path
        let dim = 70;
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.004) - 0.15).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.003) + 0.02).collect();

        let c = 1.0;
        let scalar = -poincare_distance(&v1, &v2, c);
        let simd = unsafe { poincare_similarity_avx(&v1, &v2, c) };
        assert!(
            (scalar - simd).abs() < 1e-4,
            "odd dim: scalar={scalar}, simd={simd}, diff={}",
            (scalar - simd).abs()
        );
    }
}
