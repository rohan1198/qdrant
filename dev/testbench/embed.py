"""Embed WOS documents and upsert into Qdrant collections.

Pipeline:
1. Load documents from data/wos/documents.jsonl
2. Embed with pplx-embed-v1 (1024d) on GPU — cached to data/wos/embeddings_1024d.npy
3. PCA 1024d → 128d, fit on stratified sample — cached to data/wos/embeddings_128d.npy
4. Project to Poincaré ball at 4 curvature values (0.5, 1.0, 2.0, 5.0)
5. Create 5 Qdrant collections and upsert

Collections:
  wos_cosine  — 128d cosine baseline (standard qdrant_client)
  wos_c05     — Poincaré c=0.5
  wos_c10     — Poincaré c=1.0
  wos_c20     — Poincaré c=2.0
  wos_c50     — Poincaré c=5.0
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import requests
from sklearn.decomposition import PCA
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "wos"
DOCUMENTS_FILE = DATA_DIR / "documents.jsonl"
EMBEDDINGS_1024_FILE = DATA_DIR / "embeddings_1024d.npy"
EMBEDDINGS_128_FILE = DATA_DIR / "embeddings_128d.npy"

MODEL_PATH = Path.home() / "projects" / "pythia" / "models" / "pplx-embed-v1-0.6b"

# ---------------------------------------------------------------------------
# Poincaré math (from Pythia's poincare.py)
# ---------------------------------------------------------------------------
EPS = 1e-5
BALL_SCALING = 0.9


def exp_map_at_origin(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Map a tangent vector at the origin to the Poincaré ball."""
    norm_v = np.linalg.norm(v)
    if norm_v < EPS:
        return v.copy()
    sqrt_c = np.sqrt(c)
    coeff = np.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
    return coeff * v


def project_to_ball(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Clip a point to strictly inside the Poincaré ball."""
    norm = np.linalg.norm(x)
    if norm < EPS:
        return x
    boundary = (1.0 / np.sqrt(c)) - EPS
    limit = min(0.95, boundary)
    if norm > limit:
        return x * (limit / norm)
    return x


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents() -> list[dict]:
    """Load all documents from JSONL file."""
    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Documents not found at {DOCUMENTS_FILE}. "
            "Run download_dataset.py first."
        )
    docs = []
    with open(DOCUMENTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    print(f"Loaded {len(docs)} documents")
    return docs


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts with pplx-embed-v1 (1024d). Results cached to disk.

    The model produces unnormalized int8-quantized embeddings with
    L2 norm ~500-1000. This is expected — PCA will handle normalization.
    """
    if EMBEDDINGS_1024_FILE.exists():
        print(f"Loading cached 1024d embeddings from {EMBEDDINGS_1024_FILE}")
        embeddings = np.load(EMBEDDINGS_1024_FILE)
        if embeddings.shape[0] == len(texts):
            print(f"Cache hit: {embeddings.shape}")
            return embeddings
        print(
            f"Cache shape mismatch ({embeddings.shape[0]} vs {len(texts)}), "
            "re-embedding."
        )

    print(f"Loading model from {MODEL_PATH} ...")
    # Import here so the script can be syntax-checked without GPU deps installed
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(MODEL_PATH),
        trust_remote_code=True,
        device="cuda",
    )

    print(f"Embedding {len(texts)} texts on GPU ...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # keep raw; PCA handles normalization
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    print(
        f"Norm stats — min: {np.linalg.norm(embeddings, axis=1).min():.1f}, "
        f"max: {np.linalg.norm(embeddings, axis=1).max():.1f}, "
        f"mean: {np.linalg.norm(embeddings, axis=1).mean():.1f}"
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_1024_FILE, embeddings)
    print(f"Saved 1024d embeddings to {EMBEDDINGS_1024_FILE}")
    return embeddings


# ---------------------------------------------------------------------------
# PCA reduction
# ---------------------------------------------------------------------------

def reduce_pca(embeddings: np.ndarray, tiers: list[str]) -> np.ndarray:
    """PCA 1024d → 128d. Fit on stratified sample: all root+mid, 10% leaf.

    Results cached to data/wos/embeddings_128d.npy.
    """
    if EMBEDDINGS_128_FILE.exists():
        print(f"Loading cached 128d embeddings from {EMBEDDINGS_128_FILE}")
        reduced = np.load(EMBEDDINGS_128_FILE)
        if reduced.shape[0] == embeddings.shape[0]:
            print(f"Cache hit: {reduced.shape}")
            return reduced
        print(
            f"Cache shape mismatch ({reduced.shape[0]} vs {embeddings.shape[0]}), "
            "re-reducing."
        )

    print("Fitting PCA on stratified sample ...")
    sample_indices = []
    leaf_indices = []

    for i, tier in enumerate(tiers):
        if tier in ("root", "mid"):
            sample_indices.append(i)
        elif tier == "leaf":
            leaf_indices.append(i)

    # Add 10% of leaf indices
    k = max(1, int(0.10 * len(leaf_indices)))
    sampled_leaf = random.sample(leaf_indices, k)
    sample_indices.extend(sampled_leaf)

    print(
        f"PCA sample: {len(sample_indices)} points "
        f"({len(sample_indices) - k} root/mid + {k} leaf)"
    )

    sample = embeddings[sample_indices]
    pca = PCA(n_components=128, random_state=42)
    pca.fit(sample)

    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance (128 components): {explained:.3f}")

    print(f"Transforming all {len(embeddings)} embeddings ...")
    reduced = pca.transform(embeddings).astype(np.float32)
    print(f"Reduced shape: {reduced.shape}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_128_FILE, reduced)
    print(f"Saved 128d embeddings to {EMBEDDINGS_128_FILE}")
    return reduced


# ---------------------------------------------------------------------------
# Poincaré projection
# ---------------------------------------------------------------------------

def project_batch(vectors: np.ndarray, c: float) -> np.ndarray:
    """Apply exp_map_at_origin + project_to_ball to every row."""
    out = np.empty_like(vectors)
    for i in range(len(vectors)):
        mapped = exp_map_at_origin(vectors[i], c=c)
        out[i] = project_to_ball(mapped, c=c)
    return out


# ---------------------------------------------------------------------------
# Qdrant collection management
# ---------------------------------------------------------------------------

def create_poincare_collection(
    qdrant_url: str,
    name: str,
    size: int,
    curvature: float,
) -> None:
    """Create a Poincaré collection via REST API.

    Tries to pass curvature inside vectors config; falls back to creating
    the collection without curvature if the fork does not accept it (will
    use DEFAULT_CURVATURE=1.0 internally).
    """
    url = f"{qdrant_url}/collections/{name}"

    # Delete existing collection if present
    del_resp = requests.delete(url)
    if del_resp.status_code not in (200, 404):
        print(
            f"  Warning: unexpected status {del_resp.status_code} "
            f"when deleting {name}: {del_resp.text[:200]}"
        )

    vectors_config: dict = {
        "size": size,
        "distance": "Poincare",
        "hnsw_config": {"m": 16, "ef_construct": 200},
        "curvature": curvature,
    }

    resp = requests.put(url, json={"vectors": vectors_config})
    if resp.status_code == 200:
        print(f"  Created collection '{name}' (Poincaré c={curvature})")
        return

    # Curvature field may not be accepted yet — retry without it
    print(
        f"  Note: PUT with curvature={curvature} returned {resp.status_code}. "
        f"Retrying without curvature field (fork will use DEFAULT_CURVATURE=1.0). "
        f"Response: {resp.text[:300]}"
    )
    vectors_config.pop("curvature")
    resp2 = requests.put(url, json={"vectors": vectors_config})
    resp2.raise_for_status()
    print(
        f"  Created collection '{name}' (Poincaré, curvature param unsupported — "
        f"using server default)"
    )


def create_cosine_collection(client, name: str, size: int) -> None:
    """Create a standard cosine collection via qdrant_client."""
    from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

    try:
        client.delete_collection(name)
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
    )
    print(f"  Created collection '{name}' (Cosine)")


# ---------------------------------------------------------------------------
# Upserting
# ---------------------------------------------------------------------------

UPSERT_BATCH_SIZE = 256


def upsert_vectors(
    client,
    collection_name: str,
    docs: list[dict],
    vectors: np.ndarray,
) -> None:
    """Batch-upsert all vectors with document metadata as payload."""
    from qdrant_client.models import PointStruct

    total = len(docs)
    n_batches = (total + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

    for batch_idx in tqdm(range(n_batches), desc=f"  Upserting {collection_name}"):
        start = batch_idx * UPSERT_BATCH_SIZE
        end = min(start + UPSERT_BATCH_SIZE, total)

        points = []
        for i in range(start, end):
            doc = docs[i]
            payload = {
                "id": doc["id"],
                "tier": doc["tier"],
                "domain": doc["domain"],
                "area": doc["area"],
                "hierarchy_path": doc["hierarchy_path"],
                # Omit full text from payload to keep storage light
            }
            points.append(
                PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)

    print(f"  Upserted {total} points into '{collection_name}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CURVATURE_COLLECTIONS = [
    ("wos_c05", 0.5),
    ("wos_c10", 1.0),
    ("wos_c20", 2.0),
    ("wos_c50", 5.0),
]
COSINE_COLLECTION = "wos_cosine"
VECTOR_DIM = 128


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed WOS documents and upsert into Qdrant."
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6334",
        help="Qdrant REST API URL (default: http://localhost:6334)",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding step (requires cached files to exist)",
    )
    parser.add_argument(
        "--skip-upsert",
        action="store_true",
        help="Prepare embeddings but do not upsert into Qdrant",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Step 1: Load documents
    # ------------------------------------------------------------------
    print("\n=== Step 1: Load documents ===")
    docs = load_documents()
    texts = [d["text"] for d in docs]
    tiers = [d["tier"] for d in docs]

    tier_counts = {}
    for t in tiers:
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"Tier breakdown: {tier_counts}")

    # ------------------------------------------------------------------
    # Step 2: Embed (1024d)
    # ------------------------------------------------------------------
    print("\n=== Step 2: Embed texts (1024d) ===")
    if args.skip_embed and EMBEDDINGS_1024_FILE.exists():
        print(f"--skip-embed: loading from {EMBEDDINGS_1024_FILE}")
        embeddings_1024 = np.load(EMBEDDINGS_1024_FILE)
    else:
        embeddings_1024 = embed_texts(texts)

    # ------------------------------------------------------------------
    # Step 3: PCA 1024d → 128d
    # ------------------------------------------------------------------
    print("\n=== Step 3: PCA reduction (1024d → 128d) ===")
    if args.skip_embed and EMBEDDINGS_128_FILE.exists():
        print(f"--skip-embed: loading from {EMBEDDINGS_128_FILE}")
        embeddings_128 = np.load(EMBEDDINGS_128_FILE)
    else:
        embeddings_128 = reduce_pca(embeddings_1024, tiers)

    if args.skip_upsert:
        print("\n--skip-upsert: stopping before Qdrant operations.")
        print("Done.")
        return

    # ------------------------------------------------------------------
    # Step 4 & 5: Create collections and upsert
    # ------------------------------------------------------------------
    from qdrant_client import QdrantClient

    qdrant_url = args.qdrant_url
    print(f"\n=== Step 4: Create Qdrant collections (url={qdrant_url}) ===")

    client = QdrantClient(url=qdrant_url)

    # Cosine baseline
    print(f"Creating cosine baseline collection ...")
    create_cosine_collection(client, COSINE_COLLECTION, size=VECTOR_DIM)

    # Poincaré collections
    for coll_name, c in CURVATURE_COLLECTIONS:
        print(f"Creating Poincaré collection c={c} ...")
        create_poincare_collection(qdrant_url, coll_name, size=VECTOR_DIM, curvature=c)

    print("\n=== Step 5: Upsert vectors ===")

    # Cosine: use PCA-reduced vectors directly
    print("Upserting cosine collection ...")
    upsert_vectors(client, COSINE_COLLECTION, docs, embeddings_128)

    # Poincaré: project before upserting
    for coll_name, c in CURVATURE_COLLECTIONS:
        print(f"Projecting to Poincaré ball (c={c}) ...")
        poincare_vecs = project_batch(embeddings_128, c=c)
        norm_stats = np.linalg.norm(poincare_vecs, axis=1)
        print(
            f"  Norm stats — min: {norm_stats.min():.4f}, "
            f"max: {norm_stats.max():.4f}, mean: {norm_stats.mean():.4f}"
        )
        upsert_vectors(client, coll_name, docs, poincare_vecs)

    print("\nAll done.")
    print("Collections created and populated:")
    print(f"  {COSINE_COLLECTION}  — 128d cosine baseline")
    for coll_name, c in CURVATURE_COLLECTIONS:
        print(f"  {coll_name:<12} — Poincaré c={c}")


if __name__ == "__main__":
    main()
