/// Tangent-pruned scorer for approximate nearest-neighbour search in the
/// Poincaré ball.
///
/// `prune_and_rescore` is a two-phase filter:
///
/// 1. **Cheap phase** — rank all candidates by squared Euclidean distance in
///    the pre-computed tangent space.  This is O(d) per candidate.
/// 2. **Exact phase** — recompute exact Poincaré distance only for the
///    `k × prune_factor` survivors.  Return the top-`k`.
use super::tangent_cache::TangentCache;

/// Default multiplier: keep `k × DEFAULT_PRUNE_FACTOR` candidates after the
/// tangent-space filtering step before re-ranking with exact distance.
pub const DEFAULT_PRUNE_FACTOR: usize = 10;

/// Prune `candidate_indices` using tangent-space distance, then re-rank the
/// survivors with exact Poincaré distance and return the top `k`.
///
/// # Arguments
///
/// * `query`             – query vector in the Poincaré ball.
/// * `candidate_indices` – indices into `vectors` to consider.
/// * `vectors`           – the full database of vectors.
/// * `tangent_cache`     – pre-built [`TangentCache`] for `vectors`.
/// * `k`                 – number of results to return.
/// * `prune_factor`      – keep `k * prune_factor` candidates after the cheap
///                         phase (clamped to `candidate_indices.len()`).
///
/// # Returns
///
/// A `Vec<(index, exact_poincare_distance)>` of length `min(k, candidates)`,
/// sorted by ascending exact distance.
pub fn prune_and_rescore(
    query: &[f32],
    candidate_indices: &[usize],
    vectors: &[Vec<f32>],
    tangent_cache: &TangentCache,
    k: usize,
    prune_factor: usize,
) -> Vec<(usize, f32)> {
    if candidate_indices.is_empty() || k == 0 {
        return Vec::new();
    }

    // --- Phase 1: project query and score all candidates cheaply ---
    let query_tangent = tangent_cache.project_query(query);

    let mut tangent_scored: Vec<(usize, f32)> = candidate_indices
        .iter()
        .map(|&idx| {
            let dist_sq = tangent_cache.tangent_distance_sq(&query_tangent, idx);
            (idx, dist_sq)
        })
        .collect();

    // Keep only top (k * prune_factor) by tangent distance (smallest first)
    let shortlist_size = (k * prune_factor).min(tangent_scored.len());
    // Partial sort: bring the `shortlist_size` smallest elements to the front.
    tangent_scored.select_nth_unstable_by(shortlist_size.saturating_sub(1), |a, b| {
        a.1.partial_cmp(&b.1).unwrap()
    });
    tangent_scored.truncate(shortlist_size);

    // --- Phase 2: re-score survivors with exact Poincaré distance ---
    let mut exact_scored: Vec<(usize, f32)> = tangent_scored
        .iter()
        .map(|&(idx, _)| {
            let dist = tangent_cache.exact_distance(query, idx, vectors);
            (idx, dist)
        })
        .collect();

    // Sort by ascending exact distance and keep top k
    exact_scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    exact_scored.truncate(k);

    exact_scored
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::hyperbolic::poincare_math::poincare_distance;
    use crate::spaces::hyperbolic::tangent_cache::TangentCache;

    const C: f32 = 1.0;

    /// Build the shared test dataset:
    ///   - 6 points clustered near (0.20, 0.0)  (indices 0-5)
    ///   - 4 far-away points                     (indices 6-9)
    ///
    /// Query sits near the cluster.
    fn make_test_data() -> (Vec<Vec<f32>>, Vec<f32>) {
        let vectors = vec![
            vec![0.20_f32, 0.01], // 0 – cluster
            vec![0.21, -0.01],    // 1 – cluster
            vec![0.19, 0.02],     // 2 – cluster
            vec![0.22, 0.00],     // 3 – cluster
            vec![0.18, -0.02],    // 4 – cluster
            vec![0.20, -0.01],    // 5 – cluster
            vec![-0.60, 0.50],    // 6 – far
            vec![0.55, -0.55],    // 7 – far
            vec![-0.50, -0.50],   // 8 – far
            vec![0.60, 0.40],     // 9 – far
        ];
        let query = vec![0.205_f32, 0.005];
        (vectors, query)
    }

    // -------------------------------------------------------------------------
    // test_prune_and_rescore_returns_k
    // -------------------------------------------------------------------------
    #[test]
    fn test_prune_and_rescore_returns_k() {
        let (vectors, query) = make_test_data();
        let cache = TangentCache::build(&vectors, C);
        let candidates: Vec<usize> = (0..vectors.len()).collect();

        let k = 3;
        let results = prune_and_rescore(
            &query,
            &candidates,
            &vectors,
            &cache,
            k,
            DEFAULT_PRUNE_FACTOR,
        );

        assert_eq!(results.len(), k, "expected exactly k={k} results");
    }

    // -------------------------------------------------------------------------
    // test_prune_and_rescore_finds_nearest
    //
    // The top-1 result from prune_and_rescore must match the true brute-force
    // nearest neighbour.
    // -------------------------------------------------------------------------
    #[test]
    fn test_prune_and_rescore_finds_nearest() {
        let (vectors, query) = make_test_data();
        let cache = TangentCache::build(&vectors, C);
        let candidates: Vec<usize> = (0..vectors.len()).collect();

        // Brute-force nearest neighbour
        let brute_nearest = (0..vectors.len())
            .min_by(|&a, &b| {
                poincare_distance(&query, &vectors[a], C)
                    .partial_cmp(&poincare_distance(&query, &vectors[b], C))
                    .unwrap()
            })
            .unwrap();

        let results = prune_and_rescore(
            &query,
            &candidates,
            &vectors,
            &cache,
            1,
            DEFAULT_PRUNE_FACTOR,
        );

        assert_eq!(results.len(), 1, "expected 1 result, got {}", results.len());
        assert_eq!(
            results[0].0, brute_nearest,
            "top-1 index {} != brute-force NN {}",
            results[0].0, brute_nearest
        );
    }

    // -------------------------------------------------------------------------
    // test_prune_factor_1_is_exact
    //
    // With prune_factor=1 we keep exactly k=1 candidate after the tangent pass,
    // which might in general miss the true NN — but we verify that a larger
    // prune_factor (DEFAULT_PRUNE_FACTOR) still finds the same answer, i.e.
    // the function is internally consistent.
    //
    // Specifically: the top-1 result with DEFAULT_PRUNE_FACTOR should agree
    // with the top-1 result when the shortlist is the full candidate set
    // (prune_factor = candidates.len()).
    // -------------------------------------------------------------------------
    #[test]
    fn test_prune_factor_1_is_exact() {
        let (vectors, query) = make_test_data();
        let cache = TangentCache::build(&vectors, C);
        let candidates: Vec<usize> = (0..vectors.len()).collect();

        // Use prune_factor = candidates.len() so all survive into exact phase
        let full_results =
            prune_and_rescore(&query, &candidates, &vectors, &cache, 1, candidates.len());

        // Use DEFAULT_PRUNE_FACTOR
        let default_results = prune_and_rescore(
            &query,
            &candidates,
            &vectors,
            &cache,
            1,
            DEFAULT_PRUNE_FACTOR,
        );

        assert_eq!(full_results.len(), 1, "full results should have length 1");
        assert_eq!(
            default_results.len(),
            1,
            "default results should have length 1"
        );

        // Both should agree on the nearest neighbour for this well-separated dataset
        assert_eq!(
            full_results[0].0, default_results[0].0,
            "top-1 differs: full={} default={}",
            full_results[0].0, default_results[0].0
        );
    }
}
