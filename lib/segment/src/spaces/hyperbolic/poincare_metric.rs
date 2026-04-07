use common::types::ScoreType;
use half::f16;

use crate::data_types::vectors::{
    DenseVector, VectorElementType, VectorElementTypeByte, VectorElementTypeHalf,
};
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
