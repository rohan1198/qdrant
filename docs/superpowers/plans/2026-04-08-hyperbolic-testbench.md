# Hyperbolic Testbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained testbench in `dev/testbench/` that embeds the WOS hierarchical dataset with pplx-embed-v1, projects to Poincaré space, upserts to a Docker-built Qdrant fork, and runs comprehensive benchmarks (hierarchy separation, retrieval quality, curvature comparison, performance).

**Architecture:** Three Python scripts orchestrated by `run.sh`: `download_dataset.py` fetches WOS from HuggingFace, `embed.py` encodes with pplx-embed-v1 + PCA→128d + exp_map→Poincaré then upserts to Qdrant, `benchmark.py` runs 4 benchmark suites. Qdrant runs in Docker, built from source with `--features hyperbolic`.

**Tech Stack:** Python 3.11+, PyTorch (CUDA), sentence-transformers, qdrant-client, scikit-learn, numpy, datasets (HuggingFace), Docker

**Spec:** `docs/superpowers/specs/2026-04-08-hyperbolic-testbench-design.md`

---

## Pre-requisite

```bash
cd /home/rohan/projects/qdrant
git branch --show-current  # Should be feat/hyperbolic-vector-support
docker --version           # Docker available
nvidia-smi                 # RTX 5080 available
ls ~/projects/pythia/models/pplx-embed-v1-0.6b/config.json  # Model exists
```

---

## File Structure

### New files (all in `dev/testbench/`)

| File | Responsibility |
|------|---------------|
| `Dockerfile` | Multi-stage build of Qdrant with `--features hyperbolic` |
| `docker-compose.yml` | Qdrant service on ports 6334/6335 |
| `requirements.txt` | Python dependencies |
| `run.sh` | Orchestrator script |
| `download_dataset.py` | Fetch WOS → `data/wos/documents.jsonl` |
| `embed.py` | pplx-embed-v1 → PCA → Poincaré → Qdrant upsert |
| `benchmark.py` | 4 benchmark suites → `results/` JSON + console output |
| `.gitignore` | Ignore `data/` and `results/` directories |

---

## Task 1: Infrastructure — Dockerfile, docker-compose, requirements, run.sh, .gitignore

**Files:**
- Create: `dev/testbench/Dockerfile`
- Create: `dev/testbench/docker-compose.yml`
- Create: `dev/testbench/requirements.txt`
- Create: `dev/testbench/run.sh`
- Create: `dev/testbench/.gitignore`

- [ ] **Step 1: Create .gitignore**

```
data/
results/
__pycache__/
*.pyc
```

- [ ] **Step 2: Create Dockerfile**

The existing Qdrant Dockerfile uses cargo-chef for caching + multi-stage build. Our testbench Dockerfile is simpler — we only need a native build (no cross-compilation), so we skip `xx` and cargo-chef:

```dockerfile
# Stage 1: Build Qdrant with hyperbolic feature
FROM rust:1.94-bookworm AS builder

RUN apt-get update && apt-get install -y \
    clang lld cmake protobuf-compiler pkg-config \
    libunwind-dev g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /qdrant
COPY . .

RUN RUSTFLAGS="-C link-arg=-fuse-ld=lld" \
    cargo build --release --features hyperbolic --bin qdrant

# Stage 2: Runtime
FROM debian:13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata libunwind8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /qdrant/target/release/qdrant /qdrant/qdrant
COPY --from=builder /qdrant/config /qdrant/config
COPY --from=builder /qdrant/tools/entrypoint.sh /qdrant/entrypoint.sh

WORKDIR /qdrant

ENV TZ=Etc/UTC RUN_MODE=production

EXPOSE 6333 6334

CMD ["./qdrant"]
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
services:
  qdrant-hyperbolic:
    build:
      context: ../..
      dockerfile: dev/testbench/Dockerfile
    ports:
      - "6334:6333"   # REST: host 6334 → container 6333
      - "6335:6334"   # gRPC: host 6335 → container 6334
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

volumes:
  qdrant_storage:
```

- [ ] **Step 4: Create requirements.txt**

```
datasets>=3.0
sentence-transformers>=3.0
torch>=2.0
qdrant-client>=1.12
scikit-learn>=1.4
numpy>=1.26
tqdm>=4.60
```

- [ ] **Step 5: Create run.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

QDRANT_URL="http://localhost:6334"

wait_for_qdrant() {
    echo "Waiting for Qdrant to be ready..."
    for i in $(seq 1 60); do
        if curl -sf "$QDRANT_URL/healthz" > /dev/null 2>&1; then
            echo "Qdrant is ready."
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Qdrant did not start within 120 seconds."
    exit 1
}

case "${1:-all}" in
    all)
        echo "=== Building Qdrant with hyperbolic feature ==="
        docker compose up -d --build
        wait_for_qdrant

        echo "=== Downloading dataset ==="
        python3 download_dataset.py

        echo "=== Embedding & uploading ==="
        python3 embed.py --qdrant-url "$QDRANT_URL"

        echo "=== Running benchmarks ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL"
        ;;
    benchmark)
        wait_for_qdrant
        echo "=== Running benchmarks ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL"
        ;;
    embed)
        wait_for_qdrant
        echo "=== Embedding & uploading ==="
        python3 embed.py --qdrant-url "$QDRANT_URL"
        echo "=== Running benchmarks ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL"
        ;;
    down)
        docker compose down -v
        ;;
    *)
        echo "Usage: ./run.sh [all|benchmark|embed|down]"
        exit 1
        ;;
esac

echo "=== Done ==="
```

- [ ] **Step 6: Make run.sh executable and verify Docker builds**

```bash
chmod +x dev/testbench/run.sh
cd dev/testbench && docker compose build 2>&1 | tail -5
```

Expected: Docker build completes (may take 10-15 minutes first time).

---

## Task 2: Dataset Download Script

**Files:**
- Create: `dev/testbench/download_dataset.py`

- [ ] **Step 1: Create download_dataset.py**

```python
"""Download and prepare the WOS hierarchical dataset for benchmarking.

Creates a 3-tier hierarchy:
- Root (~10 domain documents) — synthetic, from area descriptions
- Mid (~336 area documents) — synthetic, from paper titles in each area
- Leaf (~46K paper documents) — real paper abstracts (test+val splits)

Output: data/wos/documents.jsonl
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = Path("data/wos")
OUTPUT_FILE = OUTPUT_DIR / "documents.jsonl"

# Use test + validation splits (smaller, faster; train is 106K)
SPLITS = ["test", "validation"]

# Max papers to sample for synthetic area/domain documents
TITLES_PER_AREA = 20
AREAS_PER_DOMAIN_DESC = 10


def main():
    if OUTPUT_FILE.exists():
        print(f"Dataset already exists at {OUTPUT_FILE}, skipping download.")
        print(f"Delete {OUTPUT_FILE} to re-download.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading WOS dataset from HuggingFace...")
    ds = load_dataset(
        "marcelsun/wos_hierarchical_multi_label_text_classification",
    )

    # Collect papers grouped by (domain, area)
    papers_by_area: dict[tuple[int, int], list[dict]] = defaultdict(list)
    all_papers = []

    for split_name in SPLITS:
        split = ds[split_name]
        for sample in tqdm(split, desc=f"Processing {split_name}"):
            text = sample["token"]
            labels = sample["label"]
            domain_id = labels[0]
            area_id = labels[1]

            paper = {
                "text": text,
                "domain": domain_id,
                "area": area_id,
            }
            all_papers.append(paper)
            papers_by_area[(domain_id, area_id)].append(paper)

    print(f"Total papers: {len(all_papers)}")
    print(f"Unique domains: {len(set(p['domain'] for p in all_papers))}")
    print(f"Unique areas: {len(set(p['area'] for p in all_papers))}")

    # Build documents for all tiers
    documents = []
    doc_id = 0

    # --- Leaf tier: individual papers ---
    for paper in all_papers:
        documents.append({
            "id": f"leaf_{doc_id}",
            "text": paper["text"],
            "tier": "leaf",
            "domain": paper["domain"],
            "area": paper["area"],
            "hierarchy_path": f"{paper['domain']}/{paper['area']}",
        })
        doc_id += 1

    print(f"Leaf documents: {doc_id}")

    # --- Mid tier: area representatives ---
    areas_by_domain: dict[int, list[int]] = defaultdict(list)
    mid_count = 0

    for (domain_id, area_id), papers in papers_by_area.items():
        # Sample paper titles/abstracts to create area description
        sampled = random.sample(papers, min(TITLES_PER_AREA, len(papers)))
        # Use first 200 chars of each paper as a snippet
        snippets = [p["text"][:200] for p in sampled]
        area_text = (
            f"Research area {area_id} in domain {domain_id}. "
            f"Representative works: {' | '.join(snippets)}"
        )

        documents.append({
            "id": f"mid_{domain_id}_{area_id}",
            "text": area_text,
            "tier": "mid",
            "domain": domain_id,
            "area": area_id,
            "hierarchy_path": f"{domain_id}/{area_id}",
        })
        mid_count += 1
        areas_by_domain[domain_id].append(area_id)

    print(f"Mid (area) documents: {mid_count}")

    # --- Root tier: domain representatives ---
    root_count = 0

    for domain_id, area_ids in areas_by_domain.items():
        # Build domain description from area descriptions
        sampled_areas = random.sample(
            area_ids, min(AREAS_PER_DOMAIN_DESC, len(area_ids))
        )
        area_descs = []
        for aid in sampled_areas:
            area_papers = papers_by_area[(domain_id, aid)]
            snippets = [p["text"][:100] for p in area_papers[:5]]
            area_descs.append(f"Area {aid}: {' | '.join(snippets)}")

        domain_text = (
            f"Research domain {domain_id} spanning {len(area_ids)} areas. "
            f"Sample areas: {' '.join(area_descs)}"
        )

        documents.append({
            "id": f"root_{domain_id}",
            "text": domain_text,
            "tier": "root",
            "domain": domain_id,
            "area": -1,  # root spans all areas
            "hierarchy_path": f"{domain_id}",
        })
        root_count += 1

    print(f"Root (domain) documents: {root_count}")
    print(f"Total documents: {len(documents)}")

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the download script**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 download_dataset.py
wc -l data/wos/documents.jsonl
head -1 data/wos/documents.jsonl | python3 -m json.tool
```

Expected: ~46K lines (test+val papers + ~336 mid + ~10 root). First line is a valid JSON document with id, text, tier, domain, area, hierarchy_path fields.

---

## Task 3: Embedding & Projection Script

**Files:**
- Create: `dev/testbench/embed.py`

- [ ] **Step 1: Create embed.py**

```python
"""Embed WOS documents, project to Poincaré ball, upsert to Qdrant.

Pipeline:
1. Load pplx-embed-v1 → encode all documents → 1024d dense vectors
2. PCA 1024d → 128d (fit on stratified sample)
3. exp_map_at_origin → Poincaré ball (at multiple curvatures)
4. Create Qdrant collections and upsert
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    HnswConfigDiff,
)
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm

# --- Constants ---

MODEL_PATH = str(Path.home() / "projects/pythia/models/pplx-embed-v1-0.6b")
DATA_FILE = Path("data/wos/documents.jsonl")
PCA_FILE = Path("data/wos/pca_128d.npy")
EMBEDDINGS_FILE = Path("data/wos/embeddings_1024d.npy")
EMBEDDINGS_128D_FILE = Path("data/wos/embeddings_128d.npy")

TARGET_DIM = 128
BALL_SCALING = 0.9
EPS = 1e-5

CURVATURES = [0.5, 1.0, 2.0, 5.0]
EMBEDDING_BATCH_SIZE = 64
QDRANT_UPSERT_BATCH = 500


# --- Poincaré math (matching Pythia's poincare.py and our Rust code) ---

def exp_map_at_origin(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Map Euclidean vector to Poincaré ball via exponential map at origin."""
    norm_v = np.linalg.norm(v)
    if norm_v < EPS:
        return v.copy()
    sqrt_c = np.sqrt(c)
    coeff = np.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
    return coeff * v


def project_to_ball(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Project point to stay within the Poincaré ball."""
    norm = np.linalg.norm(x)
    if norm < EPS:
        return x
    boundary = (1.0 / np.sqrt(c)) - EPS
    limit = min(0.95, boundary)
    if norm > limit:
        return x * (limit / norm)
    return x


def project_batch(vectors: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Project a batch of vectors to the Poincaré ball via exp_map."""
    scaled = vectors * BALL_SCALING
    result = np.array([
        project_to_ball(exp_map_at_origin(v, c), c) for v in scaled
    ])
    return result


# --- Embedding ---

def load_documents() -> list[dict]:
    """Load documents from JSONL."""
    docs = []
    with open(DATA_FILE) as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"Loaded {len(docs)} documents")
    return docs


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts using pplx-embed-v1. Returns (N, 1024) array."""
    if EMBEDDINGS_FILE.exists():
        print(f"Loading cached embeddings from {EMBEDDINGS_FILE}")
        return np.load(EMBEDDINGS_FILE)

    print(f"Loading model from {MODEL_PATH}...")
    model = SentenceTransformer(
        MODEL_PATH,
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Encoding {len(texts)} texts (batch_size={EMBEDDING_BATCH_SIZE})...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    elapsed = time.time() - start
    print(f"Encoding done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/s)")

    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")
    return embeddings


def reduce_pca(
    embeddings: np.ndarray, tiers: list[str]
) -> np.ndarray:
    """PCA reduce from 1024d to 128d. Fit on stratified sample."""
    if EMBEDDINGS_128D_FILE.exists() and PCA_FILE.exists():
        print(f"Loading cached 128d embeddings from {EMBEDDINGS_128D_FILE}")
        return np.load(EMBEDDINGS_128D_FILE)

    # Stratified sample: all root + mid, 10% of leaf
    indices = []
    for i, tier in enumerate(tiers):
        if tier in ("root", "mid"):
            indices.append(i)
        elif np.random.random() < 0.1:
            indices.append(i)

    fit_sample = embeddings[indices]
    print(
        f"Fitting PCA on {len(fit_sample)} samples "
        f"(all root+mid, 10% leaf)..."
    )

    n_components = min(TARGET_DIM, fit_sample.shape[0], fit_sample.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(fit_sample)
    variance = sum(pca.explained_variance_ratio_)
    print(f"PCA variance explained: {variance:.3f}")

    reduced = pca.transform(embeddings)

    # Zero-pad if fewer components than target
    if n_components < TARGET_DIM:
        padding = np.zeros(
            (reduced.shape[0], TARGET_DIM - n_components), dtype=np.float32
        )
        reduced = np.concatenate([reduced, padding], axis=1)

    # L2 normalize
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    reduced = reduced / norms

    reduced = reduced.astype(np.float32)
    np.save(EMBEDDINGS_128D_FILE, reduced)
    np.save(PCA_FILE, pca.components_)
    print(f"Saved 128d embeddings to {EMBEDDINGS_128D_FILE}")
    return reduced


# --- Qdrant operations ---

def create_collection(
    client: QdrantClient,
    name: str,
    distance: str,
    curvature: float | None = None,
):
    """Create a Qdrant collection."""
    # Delete if exists
    try:
        client.delete_collection(name)
    except Exception:
        pass

    params = {
        "size": TARGET_DIM,
        "distance": distance,
        "hnsw_config": HnswConfigDiff(m=16, ef_construct=200),
    }

    # Add curvature for Poincare collections
    # NOTE: curvature is a custom field — if the fork's REST API doesn't
    # expose it yet, we pass it as a collection-level parameter or skip.
    # The Qdrant Python client may not have Distance.POINCARE yet either.
    # Workaround: use the REST API directly for collection creation.

    client.http.collections_api.create_collection(
        collection_name=name,
        create_collection={
            "vectors": {
                "size": TARGET_DIM,
                "distance": distance,
                "hnsw_config": {"m": 16, "ef_construct": 200},
                **({"curvature": curvature} if curvature is not None else {}),
            }
        },
    )
    print(f"Created collection '{name}' (distance={distance}, curvature={curvature})")


def upsert_vectors(
    client: QdrantClient,
    collection_name: str,
    documents: list[dict],
    vectors: np.ndarray,
):
    """Upsert documents with vectors to Qdrant."""
    print(f"Upserting {len(documents)} vectors to '{collection_name}'...")
    for batch_start in tqdm(
        range(0, len(documents), QDRANT_UPSERT_BATCH),
        desc=f"Upserting to {collection_name}",
    ):
        batch_end = min(batch_start + QDRANT_UPSERT_BATCH, len(documents))
        points = []
        for i in range(batch_start, batch_end):
            doc = documents[i]
            points.append(
                PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={
                        "tier": doc["tier"],
                        "domain": doc["domain"],
                        "area": doc["area"],
                        "hierarchy_path": doc["hierarchy_path"],
                        "doc_id": doc["id"],
                    },
                )
            )
        client.upsert(collection_name=collection_name, points=points, wait=True)


def main():
    parser = argparse.ArgumentParser(description="Embed and upload WOS dataset")
    parser.add_argument(
        "--qdrant-url", default="http://localhost:6334", help="Qdrant REST URL"
    )
    args = parser.parse_args()

    # Load documents
    documents = load_documents()
    texts = [doc["text"] for doc in documents]
    tiers = [doc["tier"] for doc in documents]

    # Step 1: Dense embedding
    embeddings_1024 = embed_texts(texts)
    print(f"Embeddings shape: {embeddings_1024.shape}")

    # Step 2: PCA reduction
    embeddings_128 = reduce_pca(embeddings_1024, tiers)
    print(f"Reduced shape: {embeddings_128.shape}")

    # Step 3 + 4: Project and upsert for each curvature + cosine baseline
    client = QdrantClient(url=args.qdrant_url, timeout=120)

    # Cosine baseline (no projection, just normalized 128d vectors)
    print("\n--- Cosine baseline ---")
    create_collection(client, "wos_cosine", "Cosine")
    upsert_vectors(client, "wos_cosine", documents, embeddings_128)

    # Poincaré collections at each curvature
    for c in CURVATURES:
        cname = f"wos_c{str(c).replace('.', '')}"
        print(f"\n--- Poincaré c={c} ({cname}) ---")

        projected = project_batch(embeddings_128, c)
        print(
            f"Projected: min_norm={np.linalg.norm(projected, axis=1).min():.4f}, "
            f"max_norm={np.linalg.norm(projected, axis=1).max():.4f}"
        )

        create_collection(client, cname, "Poincare", curvature=c)
        upsert_vectors(client, cname, documents, projected)

    print("\n=== Embedding complete ===")
    # Print collection stats
    for name in ["wos_cosine"] + [
        f"wos_c{str(c).replace('.', '')}" for c in CURVATURES
    ]:
        info = client.get_collection(name)
        print(f"  {name}: {info.points_count} points")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify embed.py syntax**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "import ast; ast.parse(open('embed.py').read()); print('Syntax OK')"
```

Expected: "Syntax OK"

**Note on Qdrant client compatibility:** The qdrant-client Python SDK may not have `Distance.POINCARE` or the `curvature` field in `VectorParams`. The `create_collection` function uses the REST API directly via `client.http.collections_api` to work around this. If that API doesn't exist in the installed client version, fall back to raw `requests.put()` to the collections endpoint. The implementer should test collection creation against the running fork and adapt the API call as needed.

---

## Task 4: Benchmark Script

**Files:**
- Create: `dev/testbench/benchmark.py`

- [ ] **Step 1: Create benchmark.py**

```python
"""Run benchmarks against embedded WOS collections in Qdrant.

Benchmarks:
1. Hierarchy Separation — Busemann depth per tier
2. Retrieval Quality — recall@10 within hierarchy
3. Curvature Comparison — cross-compare curvatures
4. Performance & Latency — throughput and p50/p95/p99
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from tqdm import tqdm

# --- Constants ---

EPS = 1e-5
CURVATURES = [0.5, 1.0, 2.0, 5.0]
RETRIEVAL_SAMPLE_SIZE = 500
LATENCY_NUM_QUERIES = 1000
EF_VALUES = [64, 128, 256]
RESULTS_DIR = Path("results")


# --- Busemann math (Python, same as Rust/Pythia) ---

def poincare_to_lorentz(p: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Convert Poincaré ball point to Lorentz hyperboloid."""
    norm_sq = float(np.sum(p ** 2))
    denom = max(1.0 - c * norm_sq, EPS)
    x0 = (1.0 + c * norm_sq) / denom
    spatial = 2.0 * np.sqrt(c) * p / denom
    return np.concatenate([[x0], spatial])


def lorentz_inner(x: np.ndarray, y: np.ndarray) -> float:
    """Lorentz inner product."""
    return float(-x[0] * y[0] + np.dot(x[1:], y[1:]))


def busemann_score(x_lorentz: np.ndarray, focal: np.ndarray) -> float:
    """Busemann function: B_xi(x) = log(-<x, xi>_L)."""
    inner = lorentz_inner(x_lorentz, focal)
    return float(np.log(max(-inner, EPS)))


def compute_focal_direction(points: list[np.ndarray], c: float = 1.0) -> np.ndarray:
    """Compute focal direction from mean of all points."""
    lorentz_points = [poincare_to_lorentz(p, c) for p in points]
    mean = np.mean(lorentz_points, axis=0)
    spatial = mean[1:]
    spatial_norm = max(np.linalg.norm(spatial), EPS)
    spatial_unit = spatial / spatial_norm
    focal = np.zeros(len(mean))
    focal[0] = 1.0
    focal[1:] = -spatial_unit
    return focal


def compute_busemann_depths(
    points: list[np.ndarray], c: float = 1.0
) -> list[float]:
    """Compute Busemann depth for each point."""
    focal = compute_focal_direction(points, c)
    depths = []
    for p in points:
        h = poincare_to_lorentz(p, c)
        depths.append(busemann_score(h, focal))
    return depths


# --- Data loading from Qdrant ---

def scroll_all(client: QdrantClient, collection: str) -> list[dict]:
    """Scroll all points from a collection."""
    points = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        batch, next_offset = result
        for pt in batch:
            points.append({
                "id": pt.id,
                "vector": np.array(pt.vector, dtype=np.float32),
                "tier": pt.payload["tier"],
                "domain": pt.payload["domain"],
                "area": pt.payload["area"],
            })
        if next_offset is None:
            break
        offset = next_offset
    return points


# --- Benchmark 1: Hierarchy Separation ---

def benchmark_hierarchy_separation(
    points: list[dict], curvature: float
) -> dict:
    """Compute Busemann depths and measure tier separation."""
    vectors = [p["vector"] for p in points]
    tiers = [p["tier"] for p in points]

    depths = compute_busemann_depths(vectors, curvature)

    # Group depths by tier
    tier_depths = {"root": [], "mid": [], "leaf": []}
    for depth, tier in zip(depths, tiers):
        tier_depths[tier].append(depth)

    avg_depths = {t: np.mean(ds) for t, ds in tier_depths.items() if ds}
    std_all = np.std(depths)

    # Separation ratio
    if "root" in avg_depths and "leaf" in avg_depths and std_all > EPS:
        sep_ratio = abs(avg_depths["leaf"] - avg_depths["root"]) / std_all
    else:
        sep_ratio = 0.0

    # Tier classification accuracy (simple threshold-based)
    # Sort tiers by avg depth, assign thresholds at midpoints
    sorted_tiers = sorted(avg_depths.items(), key=lambda x: x[1])
    correct = 0
    total = len(depths)

    if len(sorted_tiers) >= 3:
        t01 = (sorted_tiers[0][1] + sorted_tiers[1][1]) / 2
        t12 = (sorted_tiers[1][1] + sorted_tiers[2][1]) / 2
        tier_order = [s[0] for s in sorted_tiers]

        for depth, tier in zip(depths, tiers):
            if depth < t01:
                predicted = tier_order[0]
            elif depth < t12:
                predicted = tier_order[1]
            else:
                predicted = tier_order[2]
            if predicted == tier:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "avg_depths": avg_depths,
        "std_all": float(std_all),
        "separation_ratio": float(sep_ratio),
        "tier_classification_accuracy": float(accuracy),
        "tier_counts": {t: len(ds) for t, ds in tier_depths.items()},
    }


# --- Benchmark 2: Retrieval Quality ---

def benchmark_retrieval_quality(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    k: int = 10,
    num_queries: int = RETRIEVAL_SAMPLE_SIZE,
) -> dict:
    """Search for nearest neighbors and measure hierarchy-aware recall."""
    # Sample leaf documents as queries
    leaf_points = [p for p in points if p["tier"] == "leaf"]
    if len(leaf_points) > num_queries:
        query_points = random.sample(leaf_points, num_queries)
    else:
        query_points = leaf_points

    same_area_recalls = []
    same_domain_recalls = []
    h_precisions = []

    for qp in tqdm(query_points, desc=f"Retrieval {collection}"):
        results = client.query_points(
            collection_name=collection,
            query=qp["vector"].tolist(),
            limit=k + 1,  # +1 because the query itself may be in results
        )

        # Filter out the query point itself
        neighbors = [
            r for r in results.points if r.id != qp["id"]
        ][:k]

        if not neighbors:
            continue

        same_area = sum(
            1 for n in neighbors if n.payload["area"] == qp["area"]
        )
        same_domain = sum(
            1 for n in neighbors if n.payload["domain"] == qp["domain"]
        )

        same_area_recalls.append(same_area / k)
        same_domain_recalls.append(same_domain / k)

        # Hierarchical precision: same-area=1.0, same-domain=0.5, else=0.0
        h_score = 0.0
        for n in neighbors:
            if n.payload["area"] == qp["area"]:
                h_score += 1.0
            elif n.payload["domain"] == qp["domain"]:
                h_score += 0.5
        h_precisions.append(h_score / k)

    return {
        "num_queries": len(query_points),
        "same_area_recall_at_10": float(np.mean(same_area_recalls)),
        "same_domain_recall_at_10": float(np.mean(same_domain_recalls)),
        "hierarchical_precision": float(np.mean(h_precisions)),
    }


# --- Benchmark 4: Performance & Latency ---

def benchmark_latency(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    num_queries: int = LATENCY_NUM_QUERIES,
) -> dict:
    """Measure search throughput and latency."""
    leaf_points = [p for p in points if p["tier"] == "leaf"]
    query_vectors = [
        random.choice(leaf_points)["vector"].tolist()
        for _ in range(num_queries)
    ]

    results_by_ef = {}

    for ef in EF_VALUES:
        # Set search params
        latencies = []

        for qv in query_vectors:
            start = time.perf_counter()
            client.query_points(
                collection_name=collection,
                query=qv,
                limit=10,
                search_params={"hnsw_ef": ef},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        latencies_arr = np.array(latencies)
        results_by_ef[ef] = {
            "num_queries": num_queries,
            "throughput_qps": float(num_queries / (sum(latencies) / 1000)),
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
            "mean_ms": float(np.mean(latencies_arr)),
        }

    return results_by_ef


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Benchmark hyperbolic collections")
    parser.add_argument(
        "--qdrant-url", default="http://localhost:6334", help="Qdrant REST URL"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(url=args.qdrant_url, timeout=120)

    # Determine which collections exist
    collections = [c.name for c in client.get_collections().collections]
    poincare_collections = [
        c for c in collections if c.startswith("wos_c")
    ]
    cosine_collections = [c for c in collections if c == "wos_cosine"]
    all_collections = cosine_collections + sorted(poincare_collections)

    print(f"Found collections: {all_collections}")

    all_results = {}

    for collection in all_collections:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {collection}")
        print(f"{'='*60}")

        # Determine curvature from collection name
        if collection == "wos_cosine":
            curvature = 1.0  # unused for cosine
            is_poincare = False
        else:
            # Parse curvature from name: wos_c05 → 0.5, wos_c10 → 1.0
            c_str = collection.replace("wos_c", "")
            curvature = float(c_str) / 10.0
            is_poincare = True

        # Load all points
        print("Loading points...")
        points = scroll_all(client, collection)
        print(f"Loaded {len(points)} points")

        result = {"collection": collection, "curvature": curvature}

        # Benchmark 1: Hierarchy Separation (Poincaré only)
        if is_poincare:
            print("\n--- Benchmark 1: Hierarchy Separation ---")
            sep = benchmark_hierarchy_separation(points, curvature)
            result["hierarchy_separation"] = sep
            print(f"  Avg depths: {sep['avg_depths']}")
            print(f"  Separation ratio: {sep['separation_ratio']:.3f}")
            print(f"  Tier accuracy: {sep['tier_classification_accuracy']:.3f}")

        # Benchmark 2: Retrieval Quality
        print("\n--- Benchmark 2: Retrieval Quality ---")
        ret = benchmark_retrieval_quality(client, collection, points)
        result["retrieval_quality"] = ret
        print(f"  Same-area recall@10: {ret['same_area_recall_at_10']:.3f}")
        print(f"  Same-domain recall@10: {ret['same_domain_recall_at_10']:.3f}")
        print(f"  Hierarchical precision: {ret['hierarchical_precision']:.3f}")

        # Benchmark 4: Performance
        print("\n--- Benchmark 4: Performance & Latency ---")
        perf = benchmark_latency(client, collection, points)
        result["performance"] = perf
        for ef, metrics in perf.items():
            print(
                f"  ef={ef}: {metrics['throughput_qps']:.0f} QPS, "
                f"p50={metrics['p50_ms']:.1f}ms, "
                f"p95={metrics['p95_ms']:.1f}ms, "
                f"p99={metrics['p99_ms']:.1f}ms"
            )

        all_results[collection] = result

    # Benchmark 3: Curvature Comparison (cross-collection summary)
    print(f"\n{'='*60}")
    print("Benchmark 3: Curvature Comparison Summary")
    print(f"{'='*60}")

    header = f"{'Collection':<15} {'Sep Ratio':>10} {'Area R@10':>10} {'Domain R@10':>12} {'hPrecision':>11} {'p95@128':>10}"
    print(header)
    print("-" * len(header))

    for name in all_collections:
        r = all_results[name]
        sep = r.get("hierarchy_separation", {}).get("separation_ratio", "N/A")
        area_r = r["retrieval_quality"]["same_area_recall_at_10"]
        dom_r = r["retrieval_quality"]["same_domain_recall_at_10"]
        hp = r["retrieval_quality"]["hierarchical_precision"]
        p95 = r["performance"].get(128, {}).get("p95_ms", 0)

        sep_str = f"{sep:.3f}" if isinstance(sep, float) else sep
        print(
            f"{name:<15} {sep_str:>10} {area_r:>10.3f} {dom_r:>12.3f} {hp:>11.3f} {p95:>9.1f}ms"
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify benchmark.py syntax**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "import ast; ast.parse(open('benchmark.py').read()); print('Syntax OK')"
```

Expected: "Syntax OK"

---

## Task 5: End-to-End Test Run

**Files:** None (testing existing files)

This task runs the full pipeline and validates it works.

- [ ] **Step 1: Ensure Docker is running and Qdrant builds**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
docker compose up -d --build 2>&1 | tail -20
```

Expected: Qdrant container starts. First build may take 10-15 minutes.

- [ ] **Step 2: Wait for Qdrant and verify health**

```bash
# Wait for Qdrant
for i in $(seq 1 30); do
    curl -sf http://localhost:6334/healthz && break
    sleep 2
done
curl -s http://localhost:6334/collections | python3 -m json.tool
```

Expected: Empty collections list, healthy Qdrant.

- [ ] **Step 3: Run download**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 download_dataset.py
```

Expected: Creates `data/wos/documents.jsonl` with ~46K lines.

- [ ] **Step 4: Run embedding**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 embed.py --qdrant-url http://localhost:6334
```

Expected: Embeds ~46K texts (may take 5-10 min on GPU), creates 5 collections. Verify:
```bash
curl -s http://localhost:6334/collections | python3 -m json.tool
```

Should show: `wos_cosine`, `wos_c05`, `wos_c10`, `wos_c20`, `wos_c50`.

- [ ] **Step 5: Run benchmarks**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 benchmark.py --qdrant-url http://localhost:6334
```

Expected: Runs all 4 benchmarks, prints summary table, saves JSON to `results/`.

- [ ] **Step 6: Validate success criteria**

Check the output against the spec's success criteria:
1. Busemann depth separates tiers (separation ratio > 1.0)
2. Poincaré recall@10 >= cosine baseline for at least one curvature
3. Different curvatures produce different results
4. p95 latency < 50ms at ef=128
5. No crashes

- [ ] **Step 7: Fix any issues discovered during the end-to-end run**

The testbench is new code hitting a custom fork — expect API compatibility issues (qdrant-client may not support `Distance.POINCARE` or `curvature` field). The implementer should adapt the collection creation code based on what the fork's REST API actually accepts. Test with:

```bash
curl -X PUT http://localhost:6334/collections/test_poincare \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 4, "distance": "Poincare", "curvature": 1.0}}'
```

to discover the correct API format.

---

## Final Checklist

After all tasks are complete:

- [ ] `./run.sh all` executes end-to-end without errors
- [ ] `./run.sh benchmark` re-runs benchmarks without re-embedding
- [ ] Results JSON is saved to `results/`
- [ ] Summary table shows meaningful differences between curvatures
- [ ] Busemann depth separation is visible in the output
- [ ] No sensitive data (API keys, credentials) in any file
