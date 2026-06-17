# Hyperbolic Search

Retrieval system that uses hyperbolic geometry to find hierarchically related documents. A narrative-level query retrieves its content descendants; a story-level query retrieves its posts — without any explicit parent/child links in the index.

Built on standard [Qdrant](https://qdrant.tech/). No custom database fork required.

---

## Results

### WOS (44k docs) — trained encoder vs PCA baseline

44,531 documents across 404 areas. Encoder trained with InfoNCE + Busemann-triplet loss on top of frozen pplx-embed. Evaluated on full corpus (404 areas, 404 queries), stage-1 candidate pool=1000.

**Story-level queries (394 queries, k=10, candidate pool=1000)**

| Pipeline | P@10 | R@10 | MRR | MAP | NDCG@10 |
|----------|------|------|-----|-----|---------|
| cosine (PCA) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| busemann (PCA) | 0.002 | 0.000 | 0.005 | 0.000 | 0.001 |
| busemann (trained v1) | 0.544 | 0.086 | 0.766 | 0.226 | 0.567 |
| busemann (trained v2) | 0.540 | 0.085 | **0.776** | **0.227** | 0.566 |
| **busemann (trained v3)** | **0.520** | **0.083** | 0.766 | 0.211 | **0.547** |

**Narrative-level queries (10 queries, k=10, candidate pool=1000)**

| Pipeline | P@10 | R@10 | MRR | MAP | NDCG@10 |
|----------|------|------|-----|-----|---------|
| cosine (PCA) | 0.000 | 0.000 | 0.006 | 0.013 | 0.000 |
| busemann (PCA) | 0.120 | 0.000 | 0.142 | 0.004 | 0.105 |
| busemann (trained v1) | 0.530 | 0.002 | 0.696 | 0.030 | 0.529 |
| busemann (trained v2) | 0.560 | 0.002 | **0.783** | **0.064** | 0.574 |
| **busemann (trained v3)** | **0.620** | **0.002** | 0.725 | **0.064** | **0.624** |

The PCA baselines collapse at scale because cosine ANN over PCA embeddings fails to surface descendants in a 44k-doc pool. The trained encoder fixes stage-1: InfoNCE pulls descendants together so ANN finds them, then Busemann reranks by depth.

- **v1**: base trained encoder, candidate pool 200→1000 raised narrative P@10 from 0.13 to 0.53
- **v2**: narrative positive sampling 20→200, Busemann loss weight 0.5→1.0 — best MRR (0.783)
- **v3**: harder negatives (sibling stories added to negative pool) — best P@10 (0.620) and NDCG (0.624)

---

### 20 Newsgroups (18k docs) — zero-shot transfer + in-domain training

18,331 posts across 20 newsgroups. 3-tier hierarchy: 7 narrative (top groups: alt, comp, misc, rec, sci, soc, talk) / 20 story (individual newsgroups) / 18,331 content (posts). Story and narrative query docs are synthesized from representative posts.

**Story-level queries (20 queries, k=10, candidate pool=1000)**

| Pipeline | P@10 | R@10 | MRR | MAP | NDCG@10 |
|----------|------|------|-----|-----|---------|
| cosine ANN (stage-1) | 0.850 | 0.009 | 0.975 | 0.009 | 0.873 |
| busemann (WOS v3, zero-shot) | 0.850 | 0.009 | 0.975 | 0.009 | 0.873 |
| cosine ANN (trained 20news) | 0.955 | 0.010 | 0.990 | 0.010 | 0.960 |
| **busemann (trained 20news)** | **0.955** | **0.010** | **0.990** | **0.010** | **0.960** |

**Narrative-level queries (7 queries, k=10, candidate pool=1000)**

| Pipeline | P@10 | R@10 | MRR | MAP | NDCG@10 |
|----------|------|------|-----|-----|---------|
| cosine ANN (stage-1) | 0.743 | 0.004 | 0.929 | 0.003 | 0.755 |
| busemann (WOS v3, zero-shot) | 0.800 | 0.004 | 0.786 | 0.004 | 0.796 |
| cosine ANN (trained 20news) | 0.829 | 0.005 | 0.929 | 0.005 | 0.843 |
| **busemann (trained 20news)** | **1.000** | **0.006** | **1.000** | **0.006** | **1.000** |

20 Newsgroups has very distinct topic clusters so cosine ANN already scores high at story level. At narrative level, Busemann with an in-domain head reaches P@10=1.000 — every top-10 result across all 7 narrative queries was from the correct top group. The 7 top groups (alt, comp, misc, rec, sci, soc, talk) are topically distinct enough that the trained geometry separates them cleanly.

Zero-shot transfer (WOS-trained head) already beats cosine at narrative level (+5.7 P@10). In-domain training pushes further: narrative P@10 0.800→1.000, MRR 0.786→1.000. Story level gains +10.5 P@10 (0.850→0.955). R@10 is low by construction — relevant sets are 800–3000 posts and only 10 are retrieved.

---

### WOS (Web of Science) — small corpus, PCA encoder

317 documents: 300 content papers, 10 story-level summaries, 7 narrative-level themes.
Query = a summary/theme doc. Relevant = its descendant papers. No tier filter at search time.

**Story-level queries (10 queries, k=10)**

| Pipeline | P@10 | R@10 | MRR | MAP | NDCG@10 |
|----------|------|------|-----|-----|---------|
| cosine | 0.210 | 0.070 | 0.166 | 0.163 | 0.162 |
| alpha | **0.890** | **0.297** | **1.000** | **0.714** | **0.911** |
| busemann | **0.890** | **0.297** | **1.000** | **0.714** | **0.911** |
| entailment | **0.890** | **0.297** | **1.000** | **0.714** | **0.911** |

**Narrative-level queries (7 queries, k=10)**

| Pipeline | P@10 | R@10 | MRR | MAP | NDCG@10 |
|----------|------|------|-----|-----|---------|
| cosine | 0.000 | 0.000 | 0.044 | 0.065 | 0.000 |
| alpha | 0.429 | 0.095 | 0.517 | 0.204 | 0.418 |
| **busemann** | **0.443** | **0.102** | 0.492 | **0.258** | **0.440** |

Cosine retrieves zero relevant documents at narrative level. Busemann finds 4–5 per query.

---

### Climate — structured synthetic corpus

230 documents: 200 content posts, 25 stories, 5 narratives.
5 climate narratives × 5 stories each × 8 posts each.

**Story-level queries (25 queries, k=10)**

| Pipeline | P@10 | R@10 | MRR | MAP | NDCG@10 |
|----------|------|------|-----|-----|---------|
| cosine | 0.696 | 0.870 | 0.500 | 0.642 | 0.736 |
| alpha | **0.724** | **0.905** | **1.000** | **0.882** | **0.929** |
| busemann | **0.724** | **0.905** | **1.000** | **0.882** | **0.929** |

**Narrative-level queries (5 queries, k=10)**

| Pipeline | P@10 | R@10 | MRR | MAP | NDCG@10 |
|----------|------|------|-----|-----|---------|
| cosine | 0.560 | 0.140 | 0.467 | 0.305 | 0.508 |
| alpha | 0.800 | 0.200 | **1.000** | 0.576 | 0.849 |
| **busemann** | **0.960** | **0.240** | 0.900 | **0.677** | **0.943** |

---

### What the numbers mean

- **P@10** — of the top 10 returned docs, what fraction are relevant
- **R@10** — of all relevant docs, what fraction appear in the top 10
- **MRR** — mean reciprocal rank of the first relevant doc (1.0 = always first)
- **MAP** — average precision across all recall levels
- **NDCG@10** — normalized discounted cumulative gain (rewards getting relevant docs earlier)

---

### Ablation results

**Ablation 1 — does the tier filter explain the gains?**
Adding the same tier filter to cosine (`cosine_filtered`) ties all hyperbolic methods at story level.
Conclusion: at story level, the filter does the work. At narrative level, the geometry is the difference.

**Ablation 2 — is the radius injected or learned?**
Re-ingesting with `natural_norm` projection (no tier labels, radius from PCA norm percentile):
busemann drops from 0.960 → 0.900. Alpha collapses to 0.348.
Conclusion: busemann is robust because it uses directional geometry, not just depth radius.

**Ablation 3 — does wrong query radius break things?**
Encoding narrative queries at content radius (r=0.90) instead of narrative radius (r=0.45):
busemann drops from 0.960 → 0.800. Does not collapse.
Conclusion: graceful degradation — the multi-anchor strategy absorbs the radius error.

---

## How It Works

### The core idea

Standard cosine similarity finds documents about the same topic. It has no concept of specificity or depth — a 10-word tweet and a 50-page report on climate change score equally if they use the same vocabulary.

Hyperbolic search adds a second axis: **depth in the hierarchy**. A narrative sits near the center of the ball; its content posts sit near the boundary. The geometry of the ball naturally separates them, and the Busemann scoring function retrieves "everything beneath this query" without any explicit parent/child links.

### The Poincaré ball

Think of an open ball (sphere). The boundary is "infinity" — the closer you get, the further away it becomes. Distances stretch near the boundary like a fish-eye lens.

We place documents at positions inside this ball:
- Narrative docs: radius 0.45 (near center, abstract)
- Story docs: radius 0.70 (middle)
- Content posts: radius 0.90 (near boundary, specific)

Two docs at the same radius but different directions = far apart (different topics).
Two docs on the same ray but different depths = also far apart (different hierarchy levels).

This is why flat (Euclidean) search fails for hierarchies: it treats all distances the same. Hyperbolic space stretches distances near the boundary, automatically separating depth levels.

### Why hyperbolic space fits trees

A binary tree of depth 10 has 1024 nodes. The number doubles every level — exponential growth. Flat space grows polynomially (area of a circle grows as radius²), so trees get distorted when embedded flat. Hyperbolic space grows exponentially with radius, so trees embed without distortion.

### The Busemann score

This is what makes narrative retrieval work.

From a query's position in the ball, project a ray outward to the boundary. That boundary point is the **ideal point** — the direction the query is "pointing toward."

The Busemann score of a candidate document is: how deep is this document toward my ideal point?

- A content post at radius 0.90 in the same direction as the query → high score (deep, right direction)
- A sibling narrative at radius 0.45 in a different direction → low score (wrong direction)
- A parent theme at radius 0.15 in the same direction → low score (going the wrong way, too shallow)

This is better than plain hyperbolic distance for descendant retrieval because distance cares equally about radius difference and direction difference. Busemann only cares about depth toward your direction — which is exactly what "find my descendants" requires.

### The multi-anchor trick for narratives

A narrative sits above 5 stories, each with 8 posts. Those 40 posts fan out in 5 directions.

The narrative's ideal point points toward the average of those 5 directions — roughly equidistant from all 5 clusters, pointing cleanly toward none of them. Using the narrative's own ideal point misses most descendants.

Fix: use the **story-tier docs in the candidate pool as surrogate ideal points**. Pick the top 3 stories most similar to the narrative query. Score each content post against all 3 story anchors. A post scores well if it descends from any of those stories.

This is why busemann at narrative level is 0.960 while everything else is 0.800 or below.

### The retrieval pipeline

**Trained encoder (current, best results):**
```
Query text
    ↓
pplx-embed (frozen, 0.6B) → 1024-dim dense vector
    ↓
ProjectionHead (trained, 591K params)
    Linear(1024→512) + LayerNorm + GELU + Dropout
    Linear(512→128) → L2-normalise → scale to tier radius
    ↓
Stage 1: cosine ANN on 1024-dim dense → 200 candidates
    ↓
Stage 2: Busemann score on 128-dim Poincaré → rerank
    (narrative queries use multi-anchor strategy)
    ↓
Top-k results
```

**PCA encoder (baseline, works at small scale):**
```
Query text
    ↓
Transformer (pplx-embed-v1-0.6B, 1024d)
    ↓
PCA reduction (128d) — captures topic direction
    ↓
Poincaré projection:
    direction = PCA unit vector  (what topic)
    radius    = tier label       (what depth)
    ↓
Stage 1: cosine ANN → 200 candidates
    ↓
Stage 2: Busemann score → rerank
    ↓
Top-k results
```

Three vectors stored per document in Qdrant:
- `dense` (1024d) — for stage-1 cosine ANN
- `poincare` (128d) — for exact hyperbolic reranking
- `tangent` (128d) — log-map of poincare, for tangent-space ANN pipelines

### Other pipelines

| Pipeline | Stage 1 | Stage 2 | Stage 3 | Best for |
|----------|---------|---------|---------|----------|
| `cosine` | cosine ANN | — | — | Baseline |
| `alpha` | cosine ANN | alpha proxy rerank | exact Poincaré | Story-level, fast |
| `tangent` | poincare cosine ANN | tangent distance | exact Poincaré | Peer search |
| `combined` | cosine + poincare ANN merge | — | exact Poincaré | Wide recall |
| `klein` | cosine ANN | Klein proxy | exact Poincaré | Alternative proxy |
| `busemann` | cosine ANN | Busemann score | — | Descendant retrieval |
| `entailment` | cosine ANN | Busemann filter | horoball rerank | Strict descendant |

The **alpha proxy** is a fast shortcut for hyperbolic distance: multiply the squared Euclidean gap by the stretch factor at each point. Monotonically related to true hyperbolic distance, no expensive `arccosh` needed.

The **Klein model** converts ball coordinates so geodesics become straight lines, making distances cheaper to compute as a proxy.

The **tangent space** unrolls the curved ball onto a flat plane at the origin (log map). Flat vectors can be indexed by standard HNSW.

---

## Setup

```bash
# Install dependencies
pip install qdrant-client sentence-transformers scikit-learn numpy torch

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

**Train the encoder (recommended, ~35 min one-time):**
```bash
# Step 1: embed all docs once (saves ~175MB, avoids re-encoding every epoch)
CUDA_VISIBLE_DEVICES=0 python3 scripts/precompute_embeddings.py

# Step 2: train the projection head (~8s/epoch, early stops ~epoch 3-6)
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_projection.py

# Step 3: eval trained vs PCA baselines
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_eval_trained.py \
    --head data/trained/projection_head.pt --areas 50
```

**PCA baseline evals (no training needed):**
```bash
# WOS small corpus
python3 scripts/run_eval_descendant.py

# Climate synthetic corpus
python3 scripts/run_eval_climate.py

# Ablations
python3 scripts/run_ablation_climate.py
```

GPU recommended — pplx-embed is a 0.6B param model:
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_eval_climate.py
```

---

## Project Structure

```
src/hyperbolic/
    client.py           — Qdrant wrapper (collection creation, search, upsert)
    encoder/
        base.py         — Encoder ABC
        pca_encoder.py  — PCA + Poincaré projection (baseline encoder)
        trained_encoder.py — Trained ProjectionHead encoder (best results)
    math/
        poincare.py     — All hyperbolic math (distance, Busemann, Klein, Einstein midpoint)
    pipeline/
        config.py       — PipelineConfig dataclass + preset configs
        ingest.py       — Document ingestion
    search/
        strategies.py   — All pipeline implementations (alpha, busemann, entailment, etc.)
        intent.py       — Query intent detection (precision vs recall routing)
        fusion.py       — Fusion utilities (RRF, linear alpha, Busemann-weighted)
    eval/
        harness.py      — Eval runner
        metrics.py      — P@k, R@k, MRR, MAP, NDCG
    training/
        projection_head.py  — MLP head: 1024→512→128 → Poincaré ball
        triplet_dataset.py  — Hierarchy triplet sampling (story/narrative anchors)
        losses.py           — InfoNCE + Busemann-triplet combined loss

data/
    climate/            — Synthetic climate corpus (230 docs, 3-tier hierarchy)
    wos/
        embeddings.npz  — Pre-computed pplx-embed vectors for all 44k WOS docs
        pca_matrix.npy  — PCA projection matrix (baseline)
    trained/
        projection_head.pt — Trained MLP checkpoint (591K params, epoch 3)

scripts/
    precompute_embeddings.py — Embed all WOS docs once → data/wos/embeddings.npz
    train_projection.py      — Train ProjectionHead (InfoNCE + Busemann-triplet)
    run_eval_trained.py      — Eval: cosine(PCA) vs busemann(PCA) vs busemann(trained)
    run_eval_climate.py      — Climate eval
    run_eval_descendant.py   — WOS small-corpus eval
    run_ablation_climate.py  — Three ablations (tier filter, natural norm, radius corruption)
```

---

## Deep Dive

For the full mathematics with formulas, and all ablation tables: see [BENCHMARKS.md](BENCHMARKS.md).
