#![cfg(feature = "hyperbolic")]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use atomic_refcell::AtomicRefCell;
use common::budget::ResourcePermit;
use common::counter::hardware_counter::HardwareCounterCell;
use common::flags::FeatureFlags;
use common::progress_tracker::ProgressTracker;
use common::types::ScoredPointOffset;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::SegmentEntry;
use segment::index::hnsw_index::hnsw::{HNSWIndex, HnswIndexOpenArgs};
use segment::index::VectorIndex;
use segment::segment_constructor::VectorIndexBuildArgs;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::spaces::hyperbolic::poincare_math::poincare_distance;
use segment::types::{Distance, HnswConfig, HnswGlobalConfig, SearchParams};
use segment::vector_storage::quantized::quantized_vectors::{
    QuantizedVectors, QuantizedVectorsStorageType,
};
use segment::types::{QuantizationConfig, ScalarQuantizationConfig};
use tempfile::Builder;

/// Generate a random vector inside the Poincare ball with norm < max_radius.
fn random_poincare_vector(rng: &mut StdRng, dim: usize, max_radius: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-6 {
        let target_radius = rng.random_range(0.05..max_radius);
        let scale = target_radius / norm;
        for x in &mut v {
            *x *= scale;
        }
    }
    v
}

/// Count how many point IDs overlap between two result sets.
fn sames_count(a: &[Vec<ScoredPointOffset>], b: &[Vec<ScoredPointOffset>]) -> usize {
    a[0].iter()
        .map(|x| x.idx)
        .collect::<BTreeSet<_>>()
        .intersection(&b[0].iter().map(|x| x.idx).collect())
        .count()
}

#[test]
fn test_hnsw_poincare_basic_search() {
    let stopped = AtomicBool::new(false);
    let dim = 64;
    let num_vectors: u64 = 500;
    let m = 16;
    let ef_construct = 64;
    let ef = 64;
    let top = 10;
    let num_queries = 20;

    let mut rng = StdRng::seed_from_u64(42);

    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let hnsw_dir = Builder::new().prefix("hnsw_dir").tempdir().unwrap();

    let hw_counter = HardwareCounterCell::new();

    // Build segment with Poincare distance
    let mut segment = build_simple_segment(dir.path(), dim, Distance::Poincare).unwrap();

    // Insert vectors inside the Poincare ball (scaled by 0.3)
    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    for n in 0..num_vectors {
        let vector = random_poincare_vector(&mut rng, dim, 0.3);
        all_vectors.push(vector.clone());
        segment
            .upsert_point(n as u64, n.into(), only_default_vector(&vector), &hw_counter)
            .unwrap();
    }

    // Build HNSW index
    let hnsw_config = HnswConfig {
        m,
        ef_construct,
        full_scan_threshold: num_vectors as usize * 2,
        max_indexing_threads: 2,
        on_disk: Some(false),
        payload_m: None,
        inline_storage: None,
    };

    let permit_cpu_count = 2;
    let permit = Arc::new(ResourcePermit::dummy(permit_cpu_count as u32));

    let hnsw_index = HNSWIndex::build(
        HnswIndexOpenArgs {
            path: hnsw_dir.path(),
            id_tracker: segment.id_tracker.clone(),
            vector_storage: segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .clone(),
            quantized_vectors: segment.vector_data[DEFAULT_VECTOR_NAME]
                .quantized_vectors
                .clone(),
            payload_index: segment.payload_index.clone(),
            hnsw_config,
        },
        VectorIndexBuildArgs {
            permit,
            old_indices: &[],
            gpu_device: None,
            rng: &mut rng,
            stopped: &stopped,
            hnsw_global_config: &HnswGlobalConfig::default(),
            feature_flags: FeatureFlags::default(),
            progress: ProgressTracker::new_for_test(),
        },
    )
    .unwrap();

    // Run queries and compare HNSW vs brute-force
    let mut total_recall: f64 = 0.0;
    for _ in 0..num_queries {
        let query_vector = random_poincare_vector(&mut rng, dim, 0.3);

        // Brute-force: compute poincare_distance to all vectors, sort, take top-10
        let mut distances: Vec<(u64, f32)> = all_vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, poincare_distance(&query_vector, v, 1.0)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let brute_force_top: BTreeSet<u64> = distances.iter().take(top).map(|(idx, _)| *idx).collect();

        // HNSW search
        let query: QueryVector = query_vector.into();
        let hnsw_results = hnsw_index
            .search(
                &[&query],
                None,
                top,
                Some(&SearchParams {
                    hnsw_ef: Some(ef),
                    ..Default::default()
                }),
                &Default::default(),
            )
            .unwrap();

        // The HNSW results use internal point IDs which match our insert order (0..num_vectors)
        let hnsw_top: BTreeSet<u64> = hnsw_results[0]
            .iter()
            .map(|sp| sp.idx as u64)
            .collect();

        let overlap = brute_force_top.intersection(&hnsw_top).count();
        let recall = overlap as f64 / top as f64;
        total_recall += recall;
    }

    let avg_recall = total_recall / num_queries as f64;
    let avg_recall_pct = avg_recall * 100.0;
    println!(
        "Poincare HNSW basic search: avg recall = {avg_recall_pct:.1}% over {num_queries} queries"
    );
    assert!(
        avg_recall_pct >= 90.0,
        "Average recall {avg_recall_pct:.1}% is below 90% threshold"
    );
}

#[test]
fn test_hnsw_poincare_edge_cases() {
    let stopped = AtomicBool::new(false);
    let dim = 64;
    let m = 16;
    let ef_construct = 64;
    let ef = 64;
    let top = 10;

    let mut rng = StdRng::seed_from_u64(123);

    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let hnsw_dir = Builder::new().prefix("hnsw_dir").tempdir().unwrap();

    let hw_counter = HardwareCounterCell::new();

    let mut segment = build_simple_segment(dir.path(), dim, Distance::Poincare).unwrap();

    let mut op_num: u64 = 0;

    // 1. Insert a zero vector
    let zero_vector = vec![0.0f32; dim];
    segment
        .upsert_point(op_num, 0u64.into(), only_default_vector(&zero_vector), &hw_counter)
        .unwrap();
    op_num += 1;

    // 2. Insert a vector near the ball boundary (norm ~ 0.95)
    let mut boundary_vector: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    let norm: f32 = boundary_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    let target_norm = 0.95;
    let scale = target_norm / norm;
    for x in &mut boundary_vector {
        *x *= scale;
    }
    segment
        .upsert_point(op_num, 1u64.into(), only_default_vector(&boundary_vector), &hw_counter)
        .unwrap();
    op_num += 1;

    // 3. Insert a few normal vectors
    for i in 2..10u64 {
        let v = random_poincare_vector(&mut rng, dim, 0.4);
        segment
            .upsert_point(op_num, i.into(), only_default_vector(&v), &hw_counter)
            .unwrap();
        op_num += 1;
    }

    // Build HNSW index
    let hnsw_config = HnswConfig {
        m,
        ef_construct,
        full_scan_threshold: 100,
        max_indexing_threads: 2,
        on_disk: Some(false),
        payload_m: None,
        inline_storage: None,
    };

    let permit = Arc::new(ResourcePermit::dummy(2));

    let hnsw_index = HNSWIndex::build(
        HnswIndexOpenArgs {
            path: hnsw_dir.path(),
            id_tracker: segment.id_tracker.clone(),
            vector_storage: segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .clone(),
            quantized_vectors: segment.vector_data[DEFAULT_VECTOR_NAME]
                .quantized_vectors
                .clone(),
            payload_index: segment.payload_index.clone(),
            hnsw_config,
        },
        VectorIndexBuildArgs {
            permit,
            old_indices: &[],
            gpu_device: None,
            rng: &mut rng,
            stopped: &stopped,
            hnsw_global_config: &HnswGlobalConfig::default(),
            feature_flags: FeatureFlags::default(),
            progress: ProgressTracker::new_for_test(),
        },
    )
    .unwrap();

    // Search with a query near the zero vector — should not crash
    let near_zero_query: Vec<f32> = vec![0.01; dim];
    let query: QueryVector = near_zero_query.into();
    let results = hnsw_index
        .search(
            &[&query],
            None,
            top,
            Some(&SearchParams {
                hnsw_ef: Some(ef),
                ..Default::default()
            }),
            &Default::default(),
        )
        .unwrap();
    assert!(!results[0].is_empty(), "Search near zero should return results");
    for r in &results[0] {
        assert!(r.score.is_finite(), "Score should be finite, got {}", r.score);
        assert!(!r.score.is_nan(), "Score should not be NaN");
    }

    // Search with a query near the boundary — should not produce NaN/Inf
    let boundary_query = random_poincare_vector(&mut rng, dim, 0.95);
    let query: QueryVector = boundary_query.into();
    let results = hnsw_index
        .search(
            &[&query],
            None,
            top,
            Some(&SearchParams {
                hnsw_ef: Some(ef),
                ..Default::default()
            }),
            &Default::default(),
        )
        .unwrap();
    assert!(!results[0].is_empty(), "Search near boundary should return results");
    for r in &results[0] {
        assert!(r.score.is_finite(), "Score should be finite, got {}", r.score);
        assert!(!r.score.is_nan(), "Score should not be NaN");
    }

    // Single-point collection test: search in a collection with effectively 1 result requested
    let single_result = hnsw_index
        .search(
            &[&query],
            None,
            1,
            Some(&SearchParams {
                hnsw_ef: Some(ef),
                ..Default::default()
            }),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(
        single_result[0].len(),
        1,
        "Should return exactly 1 result when top=1"
    );
    assert!(
        single_result[0][0].score.is_finite(),
        "Single result score should be finite"
    );

    println!("Poincare edge case tests passed: zero vector, boundary vector, single result");
}

#[test]
fn test_poincare_quantization_recall() {
    let stopped = AtomicBool::new(false);
    let dim = 64;
    let num_vectors: u64 = 2000;
    let m = 16;
    let ef_construct = 64;
    let ef = 64;
    let top = 10;
    let num_queries = 50;

    let mut rng = StdRng::seed_from_u64(42);

    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
    let hnsw_dir = Builder::new().prefix("hnsw_dir").tempdir().unwrap();
    let quantized_data_path = Builder::new()
        .prefix("quantized_dir")
        .tempdir()
        .unwrap();

    let hw_counter = HardwareCounterCell::new();

    // Build segment with Poincare distance
    let mut segment = build_simple_segment(dir.path(), dim, Distance::Poincare).unwrap();

    // Insert vectors inside the Poincare ball
    for n in 0..num_vectors {
        let vector = random_poincare_vector(&mut rng, dim, 0.4);
        segment
            .upsert_point(n, n.into(), only_default_vector(&vector), &hw_counter)
            .unwrap();
    }

    // Create scalar quantization
    let quantization_config: QuantizationConfig = ScalarQuantizationConfig {
        r#type: Default::default(),
        quantile: None,
        always_ram: None,
    }
    .into();

    segment.vector_data.values_mut().for_each(|vector_storage| {
        let quantized_vectors = QuantizedVectors::create(
            &vector_storage.vector_storage.borrow(),
            &quantization_config,
            QuantizedVectorsStorageType::Immutable,
            quantized_data_path.path(),
            4,
            &stopped,
        )
        .unwrap();
        vector_storage.quantized_vectors = Arc::new(AtomicRefCell::new(Some(quantized_vectors)));
    });

    // Build HNSW index with quantization
    let hnsw_config = HnswConfig {
        m,
        ef_construct,
        full_scan_threshold: num_vectors as usize * 2,
        max_indexing_threads: 2,
        on_disk: Some(false),
        payload_m: None,
        inline_storage: None,
    };

    let permit = Arc::new(ResourcePermit::dummy(2));

    let hnsw_index = HNSWIndex::build(
        HnswIndexOpenArgs {
            path: hnsw_dir.path(),
            id_tracker: segment.id_tracker.clone(),
            vector_storage: segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .clone(),
            quantized_vectors: segment.vector_data[DEFAULT_VECTOR_NAME]
                .quantized_vectors
                .clone(),
            payload_index: segment.payload_index.clone(),
            hnsw_config,
        },
        VectorIndexBuildArgs {
            permit,
            old_indices: &[],
            gpu_device: None,
            rng: &mut rng,
            stopped: &stopped,
            hnsw_global_config: &HnswGlobalConfig::default(),
            feature_flags: FeatureFlags::default(),
            progress: ProgressTracker::new_for_test(),
        },
    )
    .unwrap();

    // Generate query vectors
    let query_vectors: Vec<QueryVector> = (0..num_queries)
        .map(|_| random_poincare_vector(&mut rng, dim, 0.4).into())
        .collect();

    // Compare quantized HNSW search vs exact (plain index) search
    let mut total_sames: usize = 0;
    for query in &query_vectors {
        // Exact search using plain vector index
        let exact_results = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow()
            .search(&[query], None, top, None, &Default::default())
            .unwrap();

        // HNSW quantized search
        let hnsw_results = hnsw_index
            .search(
                &[query],
                None,
                top,
                Some(&SearchParams {
                    hnsw_ef: Some(ef),
                    ..Default::default()
                }),
                &Default::default(),
            )
            .unwrap();

        total_sames += sames_count(&hnsw_results, &exact_results);
    }

    let accuracy = 100.0 * total_sames as f64 / (num_queries * top) as f64;
    println!(
        "Poincare quantization recall: sames = {total_sames}, \
         queries = {num_queries}, top = {top}, accuracy = {accuracy:.1}%"
    );
    assert!(
        accuracy >= 60.0,
        "Quantization accuracy {accuracy:.1}% is below 60% threshold"
    );
}
