use common::counter::hardware_counter::HardwareCounterCell;
use common::generic_consts::Random;
use common::types::{PointOffsetType, ScoreType};
use common::typelevel::True;
use half::f16;
use zerocopy::FromBytes;

use crate::data_types::vectors::{
    DenseVector, VectorElementType, VectorElementTypeByte, VectorElementTypeHalf,
};
use crate::spaces::metric::{Metric, MetricPostProcessing};
use crate::types::Distance;
use crate::vector_storage::DenseVectorStorage;
use crate::vector_storage::common::VECTOR_READ_BATCH_SIZE;
use crate::vector_storage::query_scorer::QueryScorer;

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
        score.abs()
    }
}

impl Metric<VectorElementTypeByte> for PoincareMetric {
    fn distance() -> Distance {
        Distance::Poincare
    }

    fn similarity(v1: &[VectorElementTypeByte], v2: &[VectorElementTypeByte]) -> ScoreType {
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
        let v1_f32: Vec<f32> = v1.iter().map(|h| f16::to_f32(*h)).collect();
        let v2_f32: Vec<f32> = v2.iter().map(|h| f16::to_f32(*h)).collect();
        -poincare_distance(&v1_f32, &v2_f32, DEFAULT_CURVATURE)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        project_to_ball(vector, DEFAULT_CURVATURE)
    }
}

// ---------------------------------------------------------------------------
// Curvature-aware scorer (bypasses stateless Metric trait)
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_distance_variant() {
        assert_eq!(
            <PoincareMetric as Metric<VectorElementType>>::distance(),
            Distance::Poincare
        );
    }

    #[test]
    fn test_similarity_negative_distance() {
        let v1 = vec![0.3, 0.2];
        let v2 = vec![-0.1, 0.4];
        let sim = <PoincareMetric as Metric<VectorElementType>>::similarity(&v1, &v2);
        assert!(sim < 0.0);
        assert!(sim > -10.0);
    }

    #[test]
    fn test_similarity_identity_is_zero() {
        let v = vec![0.3, 0.2];
        let sim = <PoincareMetric as Metric<VectorElementType>>::similarity(&v, &v);
        assert!((sim - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_preprocess_projects_to_ball() {
        let v = vec![0.9, 0.9];
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
