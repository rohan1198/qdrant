pub mod poincare_math;
pub mod poincare_metric;
pub mod tangent_cache;
pub mod tangent_scorer;
pub mod busemann;

// Re-export key types for convenience
pub use poincare_metric::PoincareMetric;
pub use poincare_metric::PoincareCurvatureQueryScorer;
pub use poincare_math::{
    poincare_distance, project_to_ball, exp_map_origin, log_map_origin,
    exp_map, log_map, mobius_add, frechet_mean, einstein_midpoint, conformal_factor,
    poincare_to_lorentz, lorentz_inner, lorentz_to_poincare, project_hyperboloid,
    DEFAULT_CURVATURE,
};
pub use tangent_cache::TangentCache;
pub use tangent_scorer::{prune_and_rescore, DEFAULT_PRUNE_FACTOR};
pub use busemann::{
    busemann_score, busemann_depth, compute_focal_direction,
};
