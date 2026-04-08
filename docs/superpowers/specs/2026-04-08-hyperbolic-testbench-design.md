# Hyperbolic Testbench: End-to-End Validation with Real Hierarchical Data

**Date:** 2026-04-08
**Status:** Design approved, pending implementation
**Scope:** Self-contained testbench in `dev/testbench/` that validates the Qdrant fork's hyperbolic vector support using real-world hierarchical datasets
**Depends on:** Phase 1.5 implementation (configurable curvature, Einstein midpoint, stable acosh)

---

## 1. Problem Statement

The Qdrant fork has 50 unit tests proving the math works — Poincare distance, tangent pruning, Busemann scoring, configurable curvature, Einstein midpoint. What's missing is end-to-end validation: does real-world hierarchical data, when embedded and projected into the Poincare ball, actually produce hierarchy-aware search results?

We need to answer:
1. Does Busemann depth separate hierarchy tiers in real data?
2. Does Poincare distance search retrieve hierarchy-aware results better than cosine?
3. Which curvature value works best for a given hierarchy?
4. What's the performance/latency profile compared to cosine baseline?

### 1.1 Dataset Selection

**Primary: WOS (Web of Science) Hierarchical Text Classification**
- Source: `marcelsun/wos_hierarchical_multi_label_text_classification` on HuggingFace
- ~43K academic paper abstracts with 2-level hierarchy (Domain → Area)
- 7 domains, ~134 areas
- Augmented with synthetic tier representatives: domain documents (root, ~7), area documents (mid, ~134), paper documents (leaf, ~43K)
- This creates a 3-tier lots→many→few pattern matching Pythia's posts→stories→narratives

**Secondary (future): Gene Ontology (goa_reasoning)**
- GO annotations linked to PubMed publications
- 4-6 level hierarchy (GO term DAG)
- Cross-domain validation for generalizability
- Not implemented in initial testbench; dataset loading designed to be extensible

### 1.2 Embedding Model

pplx-embed-v1 (0.6B parameters, 1024d output) from `~/projects/pythia/models/pplx-embed-v1-0.6b`. The same model used in Pythia production. Runs on CUDA (RTX 5080, 16GB VRAM).

---

## 2. Architecture

### 2.1 File Structure

```
dev/testbench/
├── Dockerfile                  # Multi-stage: build Qdrant with --features hyperbolic
├── docker-compose.yml          # Qdrant service on port 6334/6335 (REST/gRPC)
├── run.sh                      # Orchestrator: docker up → download → embed → benchmark
├── requirements.txt            # Python deps
├── download_dataset.py         # Fetch WOS from HuggingFace → data/wos/
├── embed.py                    # pplx-embed-v1 → PCA → Poincare → upsert
├── benchmark.py                # 4 benchmark suites, outputs JSON + console summary
├── data/                       # Downloaded datasets (gitignored)
└── results/                    # Benchmark output JSONs (gitignored)
```

### 2.2 Infrastructure

**Docker build from source:** Multi-stage Dockerfile compiles Qdrant from the repo root with `cargo build --release --features hyperbolic`. Second stage copies the binary into a slim runtime image. Docker build cache makes subsequent runs fast. The Dockerfile lives in `dev/testbench/` but docker-compose uses `context: ../..` (repo root) with `dockerfile: dev/testbench/Dockerfile` so the full source tree is available for compilation.

**Ports:** REST on 6334, gRPC on 6335 — avoids conflict with production Qdrant on 6333/6334.

**Orchestration via `run.sh`:**
```bash
./run.sh              # Full pipeline: build → download → embed → benchmark
./run.sh benchmark    # Re-run benchmarks only (skip download + embed)
./run.sh embed        # Re-embed and benchmark (skip download)
```

---

## 3. Pipeline

### 3.1 Dataset Download (`download_dataset.py`)

1. Download WOS via `datasets.load_dataset("marcelsun/wos_hierarchical_multi_label_text_classification")`
2. Extract text (title + abstract), domain label (L1), area label (L2)
3. Assign `tier` field: papers = "leaf"
4. Create synthetic mid-tier representatives: for each area, concatenate a sample of its papers' titles → "area document" with `tier = "mid"`
5. Create synthetic root-tier representatives: for each domain, concatenate its area descriptions → "domain document" with `tier = "root"`
6. Save to `data/wos/documents.jsonl` with fields: `id`, `text`, `tier`, `domain`, `area`, `hierarchy_path`

Result: ~7 root + ~134 mid + ~43K leaf documents.

### 3.2 Embedding & Projection (`embed.py`)

**Step 1: Dense embedding.** Load pplx-embed-v1 from `~/projects/pythia/models/pplx-embed-v1-0.6b`. Batch encode all documents on GPU (batch size 64). Output: 1024d vectors.

**Step 2: PCA reduction.** Fit PCA on stratified sample (all root + mid documents, random 10% of leaf documents). Reduce 1024d → 128d. L2 normalize. Save PCA matrix to `data/wos/pca_128d.npy`.

**Step 3: Poincare projection.** Apply `exp_map_at_origin(v * ball_scaling, c)` where `ball_scaling = 0.9`. Project at 4 curvature values: c = 0.5, 1.0, 2.0, 5.0.

**Step 4: Qdrant upsert.** Create 5 collections:

| Collection | Distance | Curvature | Quantization |
|---|---|---|---|
| `wos_c05` | Poincare | 0.5 | None (raw f32) |
| `wos_c10` | Poincare | 1.0 | None |
| `wos_c20` | Poincare | 2.0 | None |
| `wos_c50` | Poincare | 5.0 | None |
| `wos_cosine` | Cosine | N/A | None |

All collections: 128d vectors, HNSW with m=16, ef_construct=200. Payload per point: `tier`, `domain`, `area`, `hierarchy_path`.

Batch upsert: 500 points per batch.

### 3.3 Benchmarks (`benchmark.py`)

#### Benchmark 1: Hierarchy Separation

Compute Busemann depths for all points in each Poincare collection (reimplemented in Python: `poincare_to_lorentz` → `busemann_score`, same math as Rust code).

Metrics:
- **Avg depth per tier:** root, mid, leaf. Clear separation = working hierarchy.
- **Separation ratio:** `(avg_leaf_depth - avg_root_depth) / std_dev_all_depths`. Higher = cleaner.
- **Tier classification accuracy:** threshold-based classification on Busemann depth. Report accuracy and confusion matrix.

#### Benchmark 2: Retrieval Quality

Sample 500 random leaf documents. For each, search top-10 nearest neighbors via the collection's distance metric.

Metrics:
- **Same-area recall@10:** fraction of top-10 sharing the query's area label.
- **Same-domain recall@10:** fraction sharing the domain label.
- **Hierarchical precision:** weighted metric: same-area = 1.0, same-domain-different-area = 0.5, different-domain = 0.0. Average across queries.
- **Compare Poincare (all curvatures) vs cosine baseline.**

#### Benchmark 3: Curvature Comparison

Cross-compare 4 curvature values + cosine baseline. Output summary table:

```
Curvature | Tier Sep Ratio | Area Recall@10 | Domain Recall@10 | hPrecision
0.5       | ...            | ...            | ...              | ...
1.0       | ...            | ...            | ...              | ...
2.0       | ...            | ...            | ...              | ...
5.0       | ...            | ...            | ...              | ...
cosine    | N/A            | ...            | ...              | ...
```

Identifies optimal curvature for this hierarchy shape.

#### Benchmark 4: Performance & Latency

For each collection, run 1000 random searches (sequential, single-threaded).

Metrics:
- **Throughput:** queries per second.
- **Latency:** p50, p95, p99 in milliseconds.
- **Vary ef:** test at ef = 64, 128, 256.
- **Compare across curvatures and against cosine baseline.**

#### Output

All results saved to `results/benchmark_YYYYMMDD_HHMMSS.json`. Console prints formatted summary tables.

---

## 4. Key Design Decisions

**No quantization for Poincare vectors.** 128d raw f32 = 512 bytes/point. At ~43K points, total = ~22MB. Negligible. Quantization introduces boundary sensitivity artifacts (per Phase 1.5 design spec).

**Cosine baseline at 128d (not 1024d).** Apples-to-apples comparison: same PCA-reduced vectors, same dimensionality, different distance metric. This isolates the contribution of Poincare distance vs cosine distance.

**Synthetic tier representatives.** WOS has 2 hierarchy levels (domain, area), but we need 3 tiers to match Pythia. We create synthetic documents by concatenating constituent texts. These represent what a "story" or "narrative" embedding would look like — a centroid/summary of its children.

**Separate collection per curvature.** Curvature is a per-collection config (VectorDataConfig.curvature). Testing multiple curvatures requires multiple collections. This also validates the curvature config feature end-to-end.

**Python-side Busemann computation.** The testbench computes Busemann depths client-side (Python), not via Qdrant. This validates the math independently and doesn't require server-side Busemann endpoints. The Poincare-to-Lorentz conversion and Busemann score formulas are identical to the Rust implementation.

---

## 5. What This Does NOT Cover

| Item | Why |
|---|---|
| GO/goa_reasoning dataset | Secondary dataset, added later via extensible dataset loading |
| Multi-tenant collections | Single collection per test, no data_source filtering |
| Sparse vector search | Testing Poincare distance in isolation |
| RRF fusion (Poincare + cosine + sparse) | Phase 2 Pythia concern |
| Profile actor-depth | Pythia-specific, not relevant to Qdrant fork validation |
| Automated CI integration | Testbench is manual-run for now |

---

## 6. Success Criteria

The testbench validates the fork if:
1. **Busemann depth separates tiers:** avg root depth < avg mid depth < avg leaf depth (or inverse, depending on focal direction) with separation ratio > 1.0
2. **Poincare recall beats cosine:** same-area recall@10 is higher for at least one curvature than for cosine baseline
3. **Curvature matters:** different curvatures produce measurably different separation ratios and recall scores
4. **Performance is acceptable:** Poincare search p95 latency < 50ms at ef=128 for ~43K vectors
5. **No crashes or errors:** full pipeline runs end-to-end without failures

---

## 7. References

- Phase 1 design spec: `docs/superpowers/specs/2026-04-07-hyperbolic-vector-support-design.md`
- Phase 1.5 design spec: `docs/superpowers/specs/2026-04-08-hyperbolic-v2-improvements-design.md`
- WOS dataset: `https://huggingface.co/datasets/marcelsun/wos_hierarchical_multi_label_text_classification`
- GO dataset (future): `https://huggingface.co/datasets/hugging-science/goa_reasoning`
- Pythia embedding reference: `~/projects/pythia/pythia/embedding/services/dense_embedder.py`
- Pythia Poincare reference: `~/projects/pythia/pythia/narratives/services/poincare.py`
- pplx-embed-v1 model: `~/projects/pythia/models/pplx-embed-v1-0.6b`
