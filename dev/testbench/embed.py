"""Embed WOS documents and upsert into Qdrant collections.

Pipeline:
1. Load documents from data/wos/documents.jsonl
2. Embed with pplx-embed-v1 (1024d) on GPU — cached to data/wos/embeddings_1024d.npy
3. PCA 1024d -> 128d, fit on stratified sample — cached to data/wos/embeddings_128d.npy
4. Project to Poincare ball at c=5.0 using 4 projection strategies
5. Create 5 Qdrant collections and upsert

Collections:
  wos_cosine       — 128d cosine baseline
  wos_c50_uniform  — Poincare c=5.0, uniform scaling
  wos_c50_static   — Poincare c=5.0, static per-tier scaling
  wos_c50_einstein — Poincare c=5.0, Einstein midpoint-derived radii
  wos_c50_spread   — Poincare c=5.0, Einstein + intra-tier variance
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import requests
from sklearn.decomposition import PCA
from tqdm import tqdm

from projection import STRATEGIES
from hyperbolic_math import (
    busemann_depth_single,
    compute_focal_direction,
    einstein_midpoint,
)

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
# Constants
# ---------------------------------------------------------------------------
CURVATURE = 5.0
COSINE_COLLECTION = "wos_cosine"
VECTOR_DIM = 128

STRATEGY_COLLECTIONS = {
    "uniform": "wos_c50_uniform",
    "static": "wos_c50_static",
    "einstein": "wos_c50_einstein",
    "einstein_spread": "wos_c50_spread",
}

UNIFIED_COLLECTION = "wos_unified"

# Point ID offsets for unified collection
CONTENT_ID_OFFSET = 0        # 0 .. 45,861
STORY_ID_OFFSET = 100_000    # 100,000 .. 100,403
NARRATIVE_ID_OFFSET = 200_000  # 200,000 .. 200,009

# Cached file paths for new artifacts
FOCAL_DIRECTION_FILE = DATA_DIR / "focal_direction.npy"
BUSEMANN_DEPTHS_FILE = DATA_DIR / "busemann_depths.npy"
AREA_CENTROIDS_FILE = DATA_DIR / "area_centroids.npz"
DOMAIN_CENTROIDS_FILE = DATA_DIR / "domain_centroids.npz"

LEGACY_COLLECTIONS = [
    "wos_c50_uniform",
    "wos_c50_static",
    "wos_c50_einstein",
    "wos_c50_spread",
    # Round 1 legacy collections
    "wos_c05",
    "wos_c10",
    "wos_c20",
    "wos_c50",
]

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
        normalize_embeddings=False,
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
    """PCA 1024d -> 128d. Fit on stratified sample: all root+mid, 10% leaf.

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
# Busemann depth computation
# ---------------------------------------------------------------------------


def compute_busemann(poincare_vectors: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute Busemann depths for all vectors. Returns (depths, focal_direction).

    Results cached to disk.
    """
    if FOCAL_DIRECTION_FILE.exists() and BUSEMANN_DEPTHS_FILE.exists():
        print(f"Loading cached Busemann data ...")
        focal = np.load(FOCAL_DIRECTION_FILE)
        depths = np.load(BUSEMANN_DEPTHS_FILE)
        if depths.shape[0] == poincare_vectors.shape[0]:
            print(f"  Cache hit: {depths.shape[0]} depths, focal dim={focal.shape[0]}")
            return depths, focal

    print("Computing focal direction from all content vectors ...")
    vectors_list = [poincare_vectors[i] for i in range(len(poincare_vectors))]
    focal = compute_focal_direction(vectors_list, c=c)

    print("Computing Busemann depths ...")
    depths = np.array([
        busemann_depth_single(poincare_vectors[i], focal, c=c)
        for i in range(len(poincare_vectors))
    ], dtype=np.float32)

    print(f"  Depth stats — min: {depths.min():.4f}, max: {depths.max():.4f}, "
          f"mean: {depths.mean():.4f}, std: {depths.std():.4f}")

    np.save(FOCAL_DIRECTION_FILE, focal)
    np.save(BUSEMANN_DEPTHS_FILE, depths)
    print(f"  Saved focal direction and depths to {DATA_DIR}")
    return depths, focal


# ---------------------------------------------------------------------------
# Tier centroid generation
# ---------------------------------------------------------------------------


def generate_tier_centroids(
    docs: list[dict],
    cosine_vectors: np.ndarray,
    poincare_vectors: np.ndarray,
    busemann_depths: np.ndarray,
    focal: np.ndarray,
    c: float,
) -> tuple[list[dict], list[dict]]:
    """Generate area (story) and domain (narrative) centroids.

    Returns (area_entries, domain_entries) where each entry is a dict:
      {"id": int, "cosine": np.ndarray, "poincare": np.ndarray,
       "busemann_depth": float, "tier": str, "domain": str, "area": str,
       "hierarchy_path": str, "source_ids": list[int]}
    """
    # Group papers by area and domain
    area_groups: dict[str, list[int]] = {}
    domain_groups: dict[str, set] = {}

    for i, doc in enumerate(docs):
        area = doc["area"]
        domain = doc["domain"]
        if area not in area_groups:
            area_groups[area] = []
        area_groups[area].append(i)
        if domain not in domain_groups:
            domain_groups[domain] = set()
        domain_groups[domain].add(area)

    # Convert domain_groups values to sorted lists
    for d in domain_groups:
        domain_groups[d] = sorted(domain_groups[d])

    print(f"Generating centroids for {len(area_groups)} areas ...")

    # --- Area (Story) centroids ---
    area_entries = []
    area_name_to_entry: dict[str, dict] = {}

    for idx, (area_name, paper_indices) in enumerate(sorted(area_groups.items())):
        # Cosine centroid: L2-normalized mean
        cosine_vecs = cosine_vectors[paper_indices]
        cosine_mean = cosine_vecs.mean(axis=0)
        cosine_norm = np.linalg.norm(cosine_mean)
        if cosine_norm > 1e-8:
            cosine_mean = cosine_mean / cosine_norm
        cosine_centroid = cosine_mean.astype(np.float32)

        # Poincare centroid: Einstein midpoint
        poincare_list = [poincare_vectors[i] for i in paper_indices]
        poincare_centroid = einstein_midpoint(poincare_list, c=c)

        # Busemann depth of the centroid
        depth = busemann_depth_single(poincare_centroid, focal, c=c)

        # Find the domain for this area
        domain_name = docs[paper_indices[0]]["domain"]

        entry = {
            "id": STORY_ID_OFFSET + idx,
            "cosine": cosine_centroid,
            "poincare": poincare_centroid,
            "busemann_depth": float(depth),
            "tier": "story",
            "domain": domain_name,
            "area": area_name,
            "hierarchy_path": f"{domain_name}/{area_name}",
            "source_ids": paper_indices,
        }
        area_entries.append(entry)
        area_name_to_entry[area_name] = entry

    print(f"  Generated {len(area_entries)} area centroids")

    # --- Domain (Narrative) centroids ---
    print(f"Generating centroids for {len(domain_groups)} domains ...")
    domain_entries = []

    for idx, (domain_name, area_names) in enumerate(sorted(domain_groups.items())):
        # Cosine centroid: L2-normalized mean of area cosine centroids
        area_cosine_vecs = np.array([
            area_name_to_entry[a]["cosine"] for a in area_names
        ])
        cosine_mean = area_cosine_vecs.mean(axis=0)
        cosine_norm = np.linalg.norm(cosine_mean)
        if cosine_norm > 1e-8:
            cosine_mean = cosine_mean / cosine_norm
        cosine_centroid = cosine_mean.astype(np.float32)

        # Poincare centroid: Einstein midpoint of area poincare centroids
        area_poincare_list = [area_name_to_entry[a]["poincare"] for a in area_names]
        poincare_centroid = einstein_midpoint(area_poincare_list, c=c)

        # Busemann depth of the centroid
        depth = busemann_depth_single(poincare_centroid, focal, c=c)

        # Source IDs: all area IDs under this domain
        area_ids = [area_name_to_entry[a]["id"] for a in area_names]

        entry = {
            "id": NARRATIVE_ID_OFFSET + idx,
            "cosine": cosine_centroid,
            "poincare": poincare_centroid,
            "busemann_depth": float(depth),
            "tier": "narrative",
            "domain": domain_name,
            "area": "",
            "hierarchy_path": domain_name,
            "source_ids": area_ids,
        }
        domain_entries.append(entry)

    print(f"  Generated {len(domain_entries)} domain centroids")

    return area_entries, domain_entries


# ---------------------------------------------------------------------------
# Qdrant collection management
# ---------------------------------------------------------------------------


def create_poincare_collection(
    qdrant_url: str,
    name: str,
    size: int,
    curvature: float,
) -> None:
    """Create a Poincare collection via REST API."""
    url = f"{qdrant_url}/collections/{name}"

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
        print(f"  Created collection '{name}' (Poincare c={curvature})")
        return

    print(
        f"  Note: PUT with curvature={curvature} returned {resp.status_code}. "
        f"Retrying without curvature field (fork will use DEFAULT_CURVATURE=1.0). "
        f"Response: {resp.text[:300]}"
    )
    vectors_config.pop("curvature")
    resp2 = requests.put(url, json={"vectors": vectors_config})
    resp2.raise_for_status()
    print(
        f"  Created collection '{name}' (Poincare, curvature param unsupported — "
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
# Unified collection management
# ---------------------------------------------------------------------------


def create_unified_collection(
    qdrant_url: str,
    name: str,
    size: int,
    curvature: float,
) -> None:
    """Create a unified collection with cosine + poincare named vectors."""
    import requests

    url = f"{qdrant_url}/collections/{name}"

    # Delete if exists
    requests.delete(url)

    body = {
        "vectors": {
            "cosine": {
                "size": size,
                "distance": "Cosine",
                "hnsw_config": {"m": 16, "ef_construct": 200},
            },
            "poincare": {
                "size": size,
                "distance": "Poincare",
                "hnsw_config": {"m": 16, "ef_construct": 200},
                "curvature": curvature,
            },
        },
    }

    resp = requests.put(url, json=body)
    if resp.status_code == 200:
        print(f"  Created unified collection '{name}' (cosine + poincare c={curvature})")
        return

    # Retry without curvature if not supported
    print(f"  Note: PUT returned {resp.status_code}, retrying without curvature field ...")
    body["vectors"]["poincare"].pop("curvature")
    resp2 = requests.put(url, json=body)
    resp2.raise_for_status()
    print(f"  Created unified collection '{name}' (curvature param unsupported)")


def upsert_unified(
    client,
    collection_name: str,
    content_docs: list[dict],
    cosine_vectors: np.ndarray,
    poincare_vectors: np.ndarray,
    busemann_depths: np.ndarray,
    area_entries: list[dict],
    domain_entries: list[dict],
) -> None:
    """Upsert all three tiers into the unified collection."""
    from qdrant_client.models import PointStruct

    # --- Content tier ---
    print(f"  Upserting {len(content_docs)} content points ...")
    total = len(content_docs)
    n_batches = (total + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

    for batch_idx in tqdm(range(n_batches), desc="    content"):
        start = batch_idx * UPSERT_BATCH_SIZE
        end = min(start + UPSERT_BATCH_SIZE, total)
        points = []
        for i in range(start, end):
            doc = content_docs[i]
            points.append(PointStruct(
                id=CONTENT_ID_OFFSET + i,
                vector={
                    "cosine": cosine_vectors[i].tolist(),
                    "poincare": poincare_vectors[i].tolist(),
                },
                payload={
                    "tier": "content",
                    "busemann_depth": float(busemann_depths[i]),
                    "domain": doc["domain"],
                    "area": doc["area"],
                    "hierarchy_path": doc["hierarchy_path"],
                    "source_ids": [],
                },
            ))
        client.upsert(collection_name=collection_name, points=points)

    # --- Story tier (areas) ---
    print(f"  Upserting {len(area_entries)} story points ...")
    story_points = []
    for entry in area_entries:
        story_points.append(PointStruct(
            id=entry["id"],
            vector={
                "cosine": entry["cosine"].tolist(),
                "poincare": entry["poincare"].tolist(),
            },
            payload={
                "tier": "story",
                "busemann_depth": entry["busemann_depth"],
                "domain": entry["domain"],
                "area": entry["area"],
                "hierarchy_path": entry["hierarchy_path"],
                "source_ids": entry["source_ids"][:20],
            },
        ))
    client.upsert(collection_name=collection_name, points=story_points)

    # --- Narrative tier (domains) ---
    print(f"  Upserting {len(domain_entries)} narrative points ...")
    narrative_points = []
    for entry in domain_entries:
        narrative_points.append(PointStruct(
            id=entry["id"],
            vector={
                "cosine": entry["cosine"].tolist(),
                "poincare": entry["poincare"].tolist(),
            },
            payload={
                "tier": "narrative",
                "busemann_depth": entry["busemann_depth"],
                "domain": entry["domain"],
                "area": "",
                "hierarchy_path": entry["hierarchy_path"],
                "source_ids": entry["source_ids"],
            },
        ))
    client.upsert(collection_name=collection_name, points=narrative_points)

    total_pts = len(content_docs) + len(area_entries) + len(domain_entries)
    print(f"  Upserted {total_pts} total points into '{collection_name}'")


def create_sweep_collection(
    client,
    qdrant_url: str,
    embeddings_128: np.ndarray,
    docs: list[dict],
    tiers: list[str],
    curvature: float,
) -> str:
    """Create a temporary unified collection at a specific curvature.

    Returns the collection name.
    """
    c_str = str(curvature).replace(".", "")
    coll_name = f"wos_unified_c{c_str}"

    print(f"  Projecting with einstein_spread at c={curvature} ...")
    strategy_fn = STRATEGIES["einstein_spread"]
    result = strategy_fn(embeddings_128, tiers, c=curvature)
    poincare_vectors = result["vectors"]

    print(f"  Computing Busemann depths at c={curvature} ...")
    vectors_list = [poincare_vectors[i] for i in range(len(poincare_vectors))]
    focal = compute_focal_direction(vectors_list, c=curvature)
    depths = np.array([
        busemann_depth_single(poincare_vectors[i], focal, c=curvature)
        for i in range(len(poincare_vectors))
    ], dtype=np.float32)

    print(f"  Generating tier centroids at c={curvature} ...")
    area_entries, domain_entries = generate_tier_centroids(
        docs, embeddings_128, poincare_vectors, depths, focal, c=curvature,
    )

    print(f"  Creating collection '{coll_name}' ...")
    create_unified_collection(qdrant_url, coll_name, size=VECTOR_DIM, curvature=curvature)
    upsert_unified(
        client, coll_name, docs, embeddings_128, poincare_vectors, depths,
        area_entries, domain_entries,
    )

    # Payload indices
    client.create_payload_index(
        collection_name=coll_name,
        field_name="busemann_depth",
        field_schema="float",
    )
    client.create_payload_index(
        collection_name=coll_name,
        field_name="tier",
        field_schema="keyword",
    )

    # Print depth summary
    area_depths = [e["busemann_depth"] for e in area_entries]
    domain_depths = [e["busemann_depth"] for e in domain_entries]
    print(f"  Depths at c={curvature}:")
    print(f"    Narratives: mean={np.mean(domain_depths):.4f}")
    print(f"    Stories:    mean={np.mean(area_depths):.4f}")
    print(f"    Content:    mean={depths.mean():.4f}")

    return coll_name


def cleanup_legacy_collections(client, qdrant_url: str) -> None:
    """Delete legacy per-strategy collections to free space."""
    import requests

    existing = client.get_collections().collections
    existing_names = {c.name for c in existing}

    for name in LEGACY_COLLECTIONS:
        if name in existing_names:
            resp = requests.delete(f"{qdrant_url}/collections/{name}")
            if resp.status_code == 200:
                print(f"  Deleted legacy collection '{name}'")
            else:
                print(f"  Warning: failed to delete '{name}': {resp.status_code}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    parser.add_argument(
        "--mode",
        default="unified",
        choices=["unified", "legacy", "all"],
        help=(
            "Which collections to create. "
            "'unified' = wos_cosine + wos_unified (default). "
            "'legacy' = per-strategy collections (round 2 compat). "
            "'all' = both."
        ),
    )
    parser.add_argument(
        "--cleanup-legacy",
        action="store_true",
        help="Delete legacy per-strategy collections to free space",
    )
    parser.add_argument(
        "--curvature-sweep",
        action="store_true",
        help="Run curvature sweep: create temporary unified collections at c=1,2,5,10",
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

    tier_counts: dict[str, int] = {}
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
    # Step 3: PCA 1024d -> 128d
    # ------------------------------------------------------------------
    print("\n=== Step 3: PCA reduction (1024d -> 128d) ===")
    if args.skip_embed and EMBEDDINGS_128_FILE.exists():
        print(f"--skip-embed: loading from {EMBEDDINGS_128_FILE}")
        embeddings_128 = np.load(EMBEDDINGS_128_FILE)
    else:
        embeddings_128 = reduce_pca(embeddings_1024, tiers)

    if args.skip_upsert:
        print("\n--skip-upsert: stopping before Qdrant operations.")
        print("Done.")
        return

    from qdrant_client import QdrantClient

    qdrant_url = args.qdrant_url
    client = QdrantClient(url=qdrant_url)

    # ------------------------------------------------------------------
    # Optional: Cleanup legacy collections
    # ------------------------------------------------------------------
    if args.cleanup_legacy:
        print("\n=== Cleaning up legacy collections ===")
        cleanup_legacy_collections(client, qdrant_url)

    run_unified = args.mode in ("unified", "all")
    run_legacy = args.mode in ("legacy", "all")

    # ------------------------------------------------------------------
    # Step 4: Project to Poincare (einstein_spread)
    # ------------------------------------------------------------------
    print(f"\n=== Step 4: Project to Poincare ball (c={CURVATURE}, einstein_spread) ===")
    strategy_fn = STRATEGIES["einstein_spread"]
    result = strategy_fn(embeddings_128, tiers, c=CURVATURE)
    poincare_vectors = result["vectors"]
    metadata = result["metadata"]

    print(f"  Norm stats — min: {metadata['norm_stats']['min']:.4f}, "
          f"max: {metadata['norm_stats']['max']:.4f}, "
          f"mean: {metadata['norm_stats']['mean']:.4f}")
    if "tier_radii" in metadata:
        print(f"  Tier radii: {metadata['tier_radii']}")

    # ------------------------------------------------------------------
    # Step 5: Compute Busemann depths
    # ------------------------------------------------------------------
    print(f"\n=== Step 5: Compute Busemann depths ===")
    busemann_depths, focal = compute_busemann(poincare_vectors, c=CURVATURE)

    # ------------------------------------------------------------------
    # Step 6: Generate tier centroids
    # ------------------------------------------------------------------
    print(f"\n=== Step 6: Generate tier centroids ===")
    area_entries, domain_entries = generate_tier_centroids(
        docs, embeddings_128, poincare_vectors, busemann_depths, focal, c=CURVATURE,
    )

    # Print centroid depth stats
    area_depths = [e["busemann_depth"] for e in area_entries]
    domain_depths = [e["busemann_depth"] for e in domain_entries]
    content_depth_mean = float(busemann_depths.mean())
    print(f"  Depth summary:")
    print(f"    Narratives (domains): mean={np.mean(domain_depths):.4f}, n={len(domain_depths)}")
    print(f"    Stories (areas):      mean={np.mean(area_depths):.4f}, n={len(area_depths)}")
    print(f"    Content (papers):     mean={content_depth_mean:.4f}, n={len(docs)}")

    # ------------------------------------------------------------------
    # Step 7: Create collections and upsert
    # ------------------------------------------------------------------
    if run_unified:
        print(f"\n=== Step 7a: Unified collection ===")

        # Cosine baseline
        print("Creating cosine baseline collection ...")
        create_cosine_collection(client, COSINE_COLLECTION, size=VECTOR_DIM)
        print("Upserting cosine collection ...")
        upsert_vectors(client, COSINE_COLLECTION, docs, embeddings_128)

        # Unified collection
        print("\nCreating unified collection ...")
        create_unified_collection(qdrant_url, UNIFIED_COLLECTION, size=VECTOR_DIM, curvature=CURVATURE)
        upsert_unified(
            client, UNIFIED_COLLECTION,
            docs, embeddings_128, poincare_vectors, busemann_depths,
            area_entries, domain_entries,
        )

        # Create payload indices for server-side filtering
        print("Creating payload indices ...")
        client.create_payload_index(
            collection_name=UNIFIED_COLLECTION,
            field_name="busemann_depth",
            field_schema="float",
        )
        client.create_payload_index(
            collection_name=UNIFIED_COLLECTION,
            field_name="tier",
            field_schema="keyword",
        )
        print("  Created indices on 'busemann_depth' (float) and 'tier' (keyword)")

    if run_legacy:
        print(f"\n=== Step 7b: Legacy per-strategy collections ===")
        strategies_to_run = list(STRATEGIES.keys())

        for strategy_name in strategies_to_run:
            coll_name = STRATEGY_COLLECTIONS[strategy_name]
            print(f"\nProjecting with strategy '{strategy_name}' (c={CURVATURE}) ...")
            create_poincare_collection(qdrant_url, coll_name, size=VECTOR_DIM, curvature=CURVATURE)
            strat_fn = STRATEGIES[strategy_name]
            strat_result = strat_fn(embeddings_128, tiers, c=CURVATURE)
            upsert_vectors(client, coll_name, docs, strat_result["vectors"])

    # ------------------------------------------------------------------
    # Optional: Curvature re-sweep
    # ------------------------------------------------------------------
    if args.curvature_sweep:
        print(f"\n=== Curvature Re-sweep ===")
        sweep_curvatures = [1.0, 2.0, 5.0, 10.0]
        sweep_names = []
        for c in sweep_curvatures:
            print(f"\n--- Curvature c={c} ---")
            name = create_sweep_collection(
                client, qdrant_url, embeddings_128, docs, tiers, c,
            )
            sweep_names.append(name)
        print(f"\nSweep collections created: {sweep_names}")
        print("Run benchmark.py to compare all collections.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
