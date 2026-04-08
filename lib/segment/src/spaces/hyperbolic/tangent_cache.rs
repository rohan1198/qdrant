/// Pre-computed tangent space coordinates for fast approximate nearest-neighbour
/// search in the Poincaré ball.
///
/// The key idea: given a centroid `μ`, every database vector `x` is mapped once
/// via `log_μ(x)` into the tangent space `T_μ(B)` — a flat Euclidean space.
/// At query time we map the query the same way and use cheap Euclidean distance
/// to produce a shortlist, then re-rank the shortlist with exact Poincaré
/// distance.
use super::poincare_math::{einstein_midpoint, log_map, poincare_distance};

pub struct TangentCache {
    centroid: Vec<f32>,
    tangent_coords: Vec<Vec<f32>>,
    curvature: f32,
}

impl TangentCache {
    /// Build a cache from a slice of vectors.
    ///
    /// Steps:
    /// 1. Compute the Fréchet mean (hyperbolic centroid) of all vectors.
    /// 2. Pre-compute `log_centroid(x)` for every vector `x`.
    pub fn build(vectors: &[Vec<f32>], curvature: f32) -> Self {
        if vectors.is_empty() {
            return Self {
                centroid: Vec::new(),
                tangent_coords: Vec::new(),
                curvature,
            };
        }

        // Collect &[f32] slices for einstein_midpoint
        let slices: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let centroid = einstein_midpoint(&slices, curvature);

        // Pre-compute log-map from centroid to every point
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

    /// Project `query` into the tangent space at the centroid via `log_centroid(query)`.
    pub fn project_query(&self, query: &[f32]) -> Vec<f32> {
        log_map(&self.centroid, query, self.curvature)
    }

    /// Squared Euclidean distance between `query_tangent` and the pre-computed
    /// tangent coordinate of `point_idx`.
    ///
    /// Panics if `point_idx` is out of range.
    pub fn tangent_distance_sq(&self, query_tangent: &[f32], point_idx: usize) -> f32 {
        let pt = &self.tangent_coords[point_idx];
        query_tangent
            .iter()
            .zip(pt.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum()
    }

    /// Exact Poincaré distance from `query` to the database vector at `point_idx`.
    ///
    /// Used for re-ranking the tangent-space shortlist.
    pub fn exact_distance(&self, query: &[f32], point_idx: usize, vectors: &[Vec<f32>]) -> f32 {
        poincare_distance(query, &vectors[point_idx], self.curvature)
    }

    /// Number of vectors in the cache.
    pub fn len(&self) -> usize {
        self.tangent_coords.len()
    }

    /// Returns `true` if the cache contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.tangent_coords.is_empty()
    }

    /// The Fréchet mean used as the tangent-space base point.
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(coords: &[f32]) -> Vec<f32> {
        coords.to_vec()
    }

    const C: f32 = 1.0;

    // -------------------------------------------------------------------------
    // test_build_cache
    // -------------------------------------------------------------------------
    #[test]
    fn test_build_cache() {
        let vectors = vec![
            make_vec(&[0.1, 0.1]),
            make_vec(&[0.2, -0.1]),
            make_vec(&[-0.1, 0.2]),
            make_vec(&[0.0, 0.15]),
            make_vec(&[-0.15, -0.1]),
        ];
        let cache = TangentCache::build(&vectors, C);

        assert_eq!(cache.len(), 5, "expected len=5");
        assert!(!cache.is_empty(), "cache should not be empty");
        assert_eq!(
            cache.centroid().len(),
            2,
            "centroid dimension should match input"
        );
    }

    // -------------------------------------------------------------------------
    // test_tangent_distance_ordering_matches_exact
    //
    // Build a cache from a tight cluster plus one outlier.  A query near the
    // cluster should have its nearest neighbour agree between tangent-space and
    // exact Poincaré distance.
    // -------------------------------------------------------------------------
    #[test]
    fn test_tangent_distance_ordering_matches_exact() {
        // Six tightly-clustered points near (0.2, 0.0) and one far outlier.
        let vectors = vec![
            make_vec(&[0.20, 0.01]),  // 0 – near query
            make_vec(&[0.21, -0.01]), // 1 – near query
            make_vec(&[0.19, 0.02]),  // 2 – near query
            make_vec(&[0.22, 0.00]),  // 3 – near query
            make_vec(&[0.18, -0.02]), // 4 – near query
            make_vec(&[0.20, -0.01]), // 5 – near query
            make_vec(&[-0.6, 0.50]),  // 6 – far outlier
        ];
        let cache = TangentCache::build(&vectors, C);
        let query = &[0.205_f32, 0.005];

        // Find nearest by tangent distance
        let query_tangent = cache.project_query(query);
        let nearest_tangent = (0..vectors.len())
            .min_by(|&a, &b| {
                cache
                    .tangent_distance_sq(&query_tangent, a)
                    .partial_cmp(&cache.tangent_distance_sq(&query_tangent, b))
                    .unwrap()
            })
            .unwrap();

        // Find nearest by exact Poincaré distance
        let nearest_exact = (0..vectors.len())
            .min_by(|&a, &b| {
                cache
                    .exact_distance(query, a, &vectors)
                    .partial_cmp(&cache.exact_distance(query, b, &vectors))
                    .unwrap()
            })
            .unwrap();

        assert_eq!(
            nearest_tangent, nearest_exact,
            "tangent NN ({nearest_tangent}) != exact NN ({nearest_exact})"
        );
    }

    // -------------------------------------------------------------------------
    // test_empty_cache
    // -------------------------------------------------------------------------
    #[test]
    fn test_empty_cache() {
        let cache = TangentCache::build(&[], C);
        assert!(cache.is_empty(), "empty build should be empty");
        assert_eq!(cache.len(), 0);
    }

    // -------------------------------------------------------------------------
    // test_single_point_cache
    //
    // The tangent-space distance from a point to itself should be (close to) 0.
    // -------------------------------------------------------------------------
    #[test]
    fn test_single_point_cache() {
        let pt = make_vec(&[0.3, -0.2]);
        let vectors = vec![pt.clone()];
        let cache = TangentCache::build(&vectors, C);

        let query_tangent = cache.project_query(&pt);
        let dist_sq = cache.tangent_distance_sq(&query_tangent, 0);

        assert!(
            dist_sq < 1e-6,
            "tangent distance of a point to itself should be ≈0, got {dist_sq}"
        );
    }
}
