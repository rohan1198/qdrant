# Hyperbolic Vector Support — Next Steps

**Status:** Phase 1 (Qdrant fork) complete. Layers A, B, C implemented. 37 tests passing.
**Date:** 2026-04-07

---

## What's Done

The Qdrant fork (`rohan1198/qdrant`, branch `feat/hyperbolic-vector-support`) has:

- `Distance::Poincare` as a first-class metric (feature-gated behind `hyperbolic`)
- Poincare ball math: geodesic distance, exp/log maps, Mobius addition, Frechet mean
- Tangent space cache + pruning scorer (5-10x speedup over naive Poincare distance)
- Busemann scoring via Lorentz model (O(d) hierarchy depth computation)
- All wired into Qdrant's scorer dispatch pipeline (raw, multi, async, quantized, GPU)

---

## Phase 2: Pythia Integration

### Step 1: HyperbolicProjector Module

**Where:** `pythia/embedding/services/hyperbolic_projector.py`

Implement the projection pipeline: 1024d Euclidean (pplx-embed-v1) → 128d Poincare ball.

```
PCA reduce (1024 → 128) → L2 normalize → exp_map_origin → Poincare point
```

Tasks:
- [ ] Implement `HyperbolicProjector` class with `project()` and `fit_pca()` methods
- [ ] Fit PCA matrix from stratified sample (~10K posts + all available stories + narratives)
- [ ] Store PCA matrix as `models/hyperbolic_pca_128d.npy`
- [ ] Add unit tests: roundtrip accuracy, output inside ball, dimension correctness

### Step 2: Per-Client Hyperbolic Collections

**Where:** `pythia/embedding/qdrant/collections.py` (or new file)

Create the unified per-client collections:

```
uk_hyperbolic, iraq_hyperbolic, syria_hyperbolic, sudan_hyperbolic, c40_hyperbolic
```

Each collection has:
- `dense` named vector: 1024d, Cosine, INT8 quantization
- `hyperbolic` named vector: 128d, Poincare (requires the fork), INT8 quantization
- `sparse` named vector: BGE-M3 sparse
- Payload indexes: `tier`, `date`, `story_ids`, `narrative_ids`, `platform`, `entity_keys`
- HNSW: `m=16`, `ef_construct=200` (no multi-tenancy needed)

Tasks:
- [ ] Define collection schema in Python (qdrant-client SDK)
- [ ] Create setup script for all client collections
- [ ] Add `tier` payload field: `"post" | "story" | "narrative"`
- [ ] Test collection creation against the fork (requires building fork Docker image or running locally)

### Step 3: Embedding Pipeline Update

**Where:** `pythia/embedding/networks/pipelines/post_pipeline.py` (and similar)

After dense + sparse encoding, add hyperbolic projection and upsert to both existing and hyperbolic collections.

Tasks:
- [ ] Add `HyperbolicProjector` to the pipeline (initialize with PCA matrix)
- [ ] After `HybridEmbeddingService.encode()`, call `projector.project(dense_vectors)`
- [ ] Build `PointStruct` with three named vectors: `dense`, `sparse`, `hyperbolic`
- [ ] Upsert to `{client}_hyperbolic` collection
- [ ] Add `tier: "post"` to payload
- [ ] Keep existing collection upserts unchanged (dual-write)

### Step 4: Stories Pipeline Update

**Where:** `pythia/stories/networks/repositories/story_repository.py` (Phase 12)

When stories are written to Qdrant, also write to the hyperbolic collection.

Tasks:
- [ ] Compute story centroid embedding (average of constituent post embeddings)
- [ ] Project centroid through `HyperbolicProjector`
- [ ] Upsert to `{client}_hyperbolic` with `tier: "story"`
- [ ] Include `story_id`, `narrative_ids`, `coherence_score`, `quality_tier` in payload

### Step 5: Narratives Pipeline Update

**Where:** `pythia/narratives/global_/services/narrative_qdrant.py` (Phase 8)

When narratives are written to Qdrant, also write to the hyperbolic collection.

Tasks:
- [ ] Compute narrative centroid embedding (from constituent story embeddings)
- [ ] Project through `HyperbolicProjector`
- [ ] Upsert to `{client}_hyperbolic` with `tier: "narrative"`
- [ ] Include `narrative_id`, `evolution_type`, `trajectory`, `engagement_total` in payload
- [ ] Store computed Busemann focal direction per entity_key (MongoDB or config)

---

## Phase 3: Validation

### Step 6: Hierarchy Separation Test

Verify that Busemann depth naturally separates the tiers.

Tasks:
- [ ] Embed ~1000 posts, ~50 stories, ~10 narratives from one client
- [ ] Project to Poincare ball
- [ ] Compute Busemann depths using stored focal direction
- [ ] Measure: are narrative depths < story depths < post depths with 90%+ separation?
- [ ] Visualize: 2D PCA of Poincare embeddings colored by tier

### Step 7: Retrieval Quality Test

Compare hyperbolic search vs current approach.

Tasks:
- [ ] For each narrative, find constituent stories via Poincare distance search
- [ ] Compare against Neo4j BELONGS_TO ground truth
- [ ] Measure recall@10 (target > 0.7)
- [ ] Compare: cosine-only vs Poincare-only vs RRF fusion

### Step 8: Performance Benchmark

Verify search latency is acceptable.

Tasks:
- [ ] Insert ~1.5M post vectors (128d) into a test collection
- [ ] Measure search latency: p50, p95, p99
- [ ] Compare: with/without tangent pruning, with/without INT8 quantization
- [ ] Target: search p95 < 50ms at ef=128
- [ ] Test quantization boundary: distance error as function of ball radius

### Step 9: Dual-Space Search Integration

Implement the PageIndex-inspired hierarchical search pattern.

Tasks:
- [ ] Build a retrieval function that does multi-prefetch:
  - Prefetch from `hyperbolic` with Busemann depth range [0.0, 0.3] (narratives)
  - Prefetch from `hyperbolic` with Busemann depth range [0.3, 0.7] (stories)
  - Prefetch from `dense` (cosine backup)
  - Fuse with RRF
- [ ] Compare retrieval quality vs current Qdrant+Neo4j+MongoDB round-trips
- [ ] Measure latency improvement

---

## Phase 4: Production Readiness

### Step 10: Curvature Tuning

Tasks:
- [ ] Test curvature values: 0.5, 1.0, 2.0, 5.0 per client
- [ ] Measure hierarchy separation at each curvature
- [ ] Select optimal per-client curvature
- [ ] Document tuning methodology

### Step 11: Fork Maintenance

Tasks:
- [ ] Set up CI for the fork (GitHub Actions)
- [ ] Track upstream Qdrant releases (~monthly)
- [ ] Rebase cadence: after each upstream release
- [ ] Tag fork releases: `v{upstream}-hyperbolic.{patch}`

### Step 12: Build & Deploy

Tasks:
- [ ] Build fork Docker image with `--features hyperbolic`
- [ ] Deploy alongside existing Qdrant instance (separate port or separate instance)
- [ ] Configure Pythia to write to both
- [ ] Gradual migration: start with one client (e.g., Sudan) for validation

---

## Future Phases (Deferred)

| Item | Revisit When |
|------|-------------|
| Layer D: Server-side projection | Multiple consumers need projection |
| Temporal decay weighting | After hierarchy validation proves value |
| Rolling window / archival collections | Data growth exceeds 3M+ per client |
| Comments and articles in hyperbolic collections | Posts + stories + narratives validated |
| Curvature canary testing (ruvector-style) | Multiple curvatures need A/B testing |
| Upstream Qdrant PR | Fork proven in production for 1+ months |
| SIMD-optimized Poincare distance | Performance profiling shows bottleneck |
| Themes tier implementation | Narrative-level validation complete |

---

## Key Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| Qdrant fork with hyperbolic feature | Done | `rohan1198/qdrant` branch `feat/hyperbolic-vector-support` |
| Rust 1.94+ | Done | Updated from 1.92 |
| PCA matrix from Pythia data | Not started | Needs stratified sample from existing embeddings |
| Fork Docker image | Not started | Needed for Pythia integration testing |
| qdrant-client Python SDK | Check | May need fork/PR for `Distance.POINCARE` support |

---

## References

- Design spec: `docs/superpowers/specs/2026-04-07-hyperbolic-vector-support-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-07-hyperbolic-vector-support.md`
- ruvector reference: `~/projects/ruvector`
- Pythia narratives (existing Poincare code): `~/projects/pythia/pythia/narratives/services/poincare.py`
- Pythia Busemann (existing): `~/projects/pythia/pythia/narratives/services/busemann.py`
