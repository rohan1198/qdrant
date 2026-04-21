# Horobase — Hyperbolic Hierarchical Search

Python client library for hyperbolic vector search on top of this Qdrant fork.

Qdrant normally stores and indexes vectors with flat distances (Cosine, Euclid, Dot). This fork adds `Distance::Poincare` to the Rust core. Horobase is the Python layer that uses it: a two-stage pipeline where Qdrant handles ANN retrieval and client-side code handles the Busemann reranking that makes hierarchical retrieval work.

---

## What it does

Standard cosine similarity retrieves documents *about* the same topic. Horobase adds a second axis: **depth in a hierarchy**. A narrative query retrieves all its content descendants without any explicit parent/child links in the index.

| Pipeline | Story P@10 | Narrative P@10 | MRR |
|----------|-----------|----------------|-----|
| cosine (baseline) | 0.000 | 0.000 | 0.000 |
| busemann (PCA encoder) | 0.002 | 0.120 | 0.005 |
| **busemann (trained encoder)** | **0.520** | **0.620** | **0.766** |

Evaluated on 44k Web of Science documents, 404 areas, candidate pool=1000.

---

## How it works

Three named vectors per document in a single Qdrant collection:

```
dense    (1024d, Cosine)  — stage-1 ANN: find candidates fast
poincare (128d,  Euclid)  — stage-2 rerank: exact Busemann scoring
tangent  (128d,  Euclid)  — stage-2 rerank: tangent-space distance
```

**Stage 1:** cosine ANN on `dense` retrieves 200–2000 candidates.  
**Stage 2:** client-side Busemann scoring on `poincare` reranks to top-k.

The `Distance::Poincare` metric in this fork enables native hyperbolic distances inside Qdrant's HNSW index — stage-1 ANN can run directly in hyperbolic space when the `hyperbolic` feature flag is enabled at build time.

---

## Setup

```bash
cd python
pip install -e ".[encoder]"

# Start this fork's Qdrant (with hyperbolic feature)
cargo run --features hyperbolic --release

# Or use standard Qdrant (horobase falls back to cosine ANN for stage-1)
docker run -p 6333:6333 qdrant/qdrant
```

---

## Quick start

```python
from horobase.client import HyperbolicClient
from horobase.encoder.pca_encoder import PCAEncoder
from horobase.pipeline.config import BUSEMANN_PIPELINE
from horobase.pipeline.ingest import ingest_documents
from horobase.search.strategies import run_search

# Connect to Qdrant (fork or standard)
client = HyperbolicClient.connect("http://localhost:6333")
client.ensure_collection("my_collection")

# Encode + ingest documents
encoder = PCAEncoder()
ingest_documents(client, "my_collection", docs, encoder)

# Search
ev = encoder.encode(["my query"], tiers=["story"])
results = run_search(
    client, "my_collection",
    ev.dense[0], ev.poincare[0], ev.tangent[0],
    config=BUSEMANN_PIPELINE, limit=10,
)
```

---

## Trained encoder (best results)

Train a projection head on top of frozen pplx-embed for 5–10× better recall:

```bash
# Pre-compute embeddings once (~175MB)
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_embeddings.py

# Train (~35 min, ~8s/epoch, early stops epoch 3-6)
CUDA_VISIBLE_DEVICES=0 python scripts/train_projection.py

# Eval
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval_trained.py \
    --head data/trained/projection_head.pt
```

The projection head is a 591K-param MLP: `Linear(1024→512) + LayerNorm + GELU + Dropout + Linear(512→128)` trained with InfoNCE + Busemann-triplet loss.

---

## Architecture

```
src/hyperbolic/ (this fork: python/horobase/)
    client.py               — Qdrant wrapper (collection creation, search, upsert)
    encoder/
        pca_encoder.py      — PCA + Poincaré projection (baseline, no GPU needed)
        trained_encoder.py  — Trained ProjectionHead encoder (best results)
    math/
        poincare.py         — Poincaré distance, Busemann score, Klein, horoball
    pipeline/
        config.py           — PipelineConfig + preset configs
        ingest.py           — Document ingestion
    search/
        strategies.py       — All pipeline implementations
        intent.py           — Query intent detection (precision vs recall routing)
    training/
        projection_head.py  — MLP head: 1024→512→128 → Poincaré ball
        triplet_dataset.py  — Hierarchy triplet sampling
        losses.py           — InfoNCE + Busemann-triplet combined loss
```

---

## Integration with Distance::Poincare

When running against this fork with `--features hyperbolic`, the `poincare` named vector can use native `Distance::Poincare` for HNSW construction. This moves stage-1 ANN into hyperbolic space, improving candidate recall for deep hierarchy queries.

To create a collection with native hyperbolic indexing:

```python
from qdrant_client.http import models as rest

# Standard collection (works with any Qdrant)
client.ensure_collection("my_collection")

# When using this fork with hyperbolic feature:
# Replace Distance.EUCLID → Distance.POINCARE for the poincare named vector
# (requires qdrant-client with Poincare distance support)
```

Stage-2 Busemann reranking is always client-side regardless of the distance type
used for stage-1 indexing.
