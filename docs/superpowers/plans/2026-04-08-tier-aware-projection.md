# Tier-Aware Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace uniform-radius Poincare projection with four strategies (uniform, static, einstein, einstein_spread) so the testbench can validate that tier-aware radial placement improves Busemann separation and retrieval quality.

**Architecture:** Extract shared hyperbolic math into `hyperbolic_math.py`, implement four projection strategies in `projection.py`, update `embed.py` to iterate over strategies at c=5.0, and update `benchmark.py` for dynamic collection discovery with a new band-overlap metric. All changes scoped to `dev/testbench/`.

**Tech Stack:** Python 3.12, NumPy, scikit-learn, qdrant-client, Qdrant fork with `hyperbolic` feature

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `dev/testbench/hyperbolic_math.py` | Create | Shared Poincare/Lorentz math functions |
| `dev/testbench/projection.py` | Create | Four projection strategy functions |
| `dev/testbench/embed.py` | Modify | Use projection strategies, `--strategy` CLI arg |
| `dev/testbench/benchmark.py` | Modify | Dynamic discovery, band overlap metric |

No changes to: `Dockerfile`, `docker-compose.yml`, `run.sh`, `download_dataset.py`, `requirements.txt`, Rust code.

---

### Task 1: Create `hyperbolic_math.py`

**Files:**
- Create: `dev/testbench/hyperbolic_math.py`

- [ ] **Step 1: Create the shared math module**

```python
"""Shared hyperbolic geometry math for the testbench.

Poincare ball and Lorentz hyperboloid operations used by both
embed.py (projection) and benchmark.py (Busemann scoring).
"""

import numpy as np

EPS = 1e-5


# ---------------------------------------------------------------------------
# Poincare ball operations
# ---------------------------------------------------------------------------


def exp_map_at_origin(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Map a tangent vector at the origin to the Poincare ball."""
    norm_v = np.linalg.norm(v)
    if norm_v < EPS:
        return v.copy()
    sqrt_c = np.sqrt(c)
    coeff = np.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
    return coeff * v


def project_to_ball(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Clip a point to strictly inside the Poincare ball."""
    norm = np.linalg.norm(x)
    if norm < EPS:
        return x
    boundary = (1.0 / np.sqrt(c)) - EPS
    limit = min(0.95, boundary)
    if norm > limit:
        return x * (limit / norm)
    return x


# ---------------------------------------------------------------------------
# Lorentz hyperboloid operations
# ---------------------------------------------------------------------------


def poincare_to_lorentz(p: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Convert Poincare ball point to Lorentz hyperboloid (d -> d+1 dims).

    x_0 = (1 + c||p||^2) / (1 - c||p||^2)
    x_i = 2*sqrt(c)*p_i / (1 - c||p||^2)
    """
    norm_sq = float(np.sum(p ** 2))
    denom = max(1.0 - c * norm_sq, EPS)
    x0 = (1.0 + c * norm_sq) / denom
    spatial = 2.0 * np.sqrt(c) * p / denom
    return np.concatenate([[x0], spatial])


def lorentz_to_poincare(x: np.ndarray) -> np.ndarray:
    """Convert Lorentz hyperboloid point to Poincare ball (d+1 -> d dims).

    p_i = x_i / (x_0 + 1)
    """
    x0 = x[0]
    denom = max(x0 + 1.0, EPS)
    return x[1:] / denom


def lorentz_inner(x: np.ndarray, y: np.ndarray) -> float:
    """Lorentz inner product: <x,y>_L = -x0*y0 + x1*y1 + ... + xn*yn."""
    return float(-x[0] * y[0] + np.dot(x[1:], y[1:]))


def project_hyperboloid(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Project a Lorentz-space vector onto the hyperboloid.

    Sets x0 = sqrt(1/c + ||x_spatial||^2) so that <x,x>_L = -1/c.
    """
    space_norm_sq = float(np.sum(x[1:] ** 2))
    x0 = np.sqrt(max(1.0 / c + space_norm_sq, EPS))
    return np.concatenate([[x0], x[1:]])


# ---------------------------------------------------------------------------
# Busemann scoring
# ---------------------------------------------------------------------------


def busemann_score(x_lorentz: np.ndarray, focal: np.ndarray) -> float:
    """Busemann function: B_xi(x) = log(-<x, xi>_L)."""
    inner = lorentz_inner(x_lorentz, focal)
    return float(np.log(max(-inner, EPS)))


def compute_focal_direction(
    points: list[np.ndarray], c: float = 1.0
) -> np.ndarray:
    """Compute focal direction (light-like vector) from a set of Poincare points."""
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
    points: list[dict], c: float = 1.0
) -> list[float]:
    """Return Busemann depth for each point dict (requires 'vector' key)."""
    vectors = [pt["vector"] for pt in points]
    focal = compute_focal_direction(vectors, c=c)
    lorentz_vecs = [poincare_to_lorentz(v, c=c) for v in vectors]
    return [busemann_score(lv, focal) for lv in lorentz_vecs]


# ---------------------------------------------------------------------------
# Einstein midpoint
# ---------------------------------------------------------------------------


def einstein_midpoint(
    points_poincare: list[np.ndarray], c: float = 1.0
) -> np.ndarray:
    """Closed-form centroid via Lorentz hyperboloid (Einstein midpoint).

    1. Convert each Poincare point to Lorentz coordinates.
    2. Weight by Lorentz gamma (x_0 component).
    3. Average and project back to hyperboloid.
    4. Convert back to Poincare.

    Returns a Poincare ball point.
    """
    if not points_poincare:
        raise ValueError("cannot compute midpoint of empty set")

    dim = len(points_poincare[0])
    weighted_sum = np.zeros(dim + 1, dtype=np.float64)
    gamma_sum = 0.0

    for p in points_poincare:
        lorentz = poincare_to_lorentz(p, c)
        gamma = float(lorentz[0])
        gamma_sum += gamma
        weighted_sum += gamma * lorentz

    if gamma_sum > EPS:
        weighted_sum /= gamma_sum

    on_hyperboloid = project_hyperboloid(weighted_sum.astype(np.float32), c)
    poincare = lorentz_to_poincare(on_hyperboloid)
    return project_to_ball(poincare, c)
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "import hyperbolic_math; print('OK')"`
Expected: `OK`

---

### Task 2: Create `projection.py`

**Files:**
- Create: `dev/testbench/projection.py`

- [ ] **Step 1: Create the projection strategies module**

```python
"""Projection strategies for mapping 128d PCA vectors to the Poincare ball.

Each strategy function takes (vectors, tiers, c) and returns a dict:
  {"vectors": np.ndarray, "metadata": dict}
"""

import numpy as np

from hyperbolic_math import (
    EPS,
    einstein_midpoint,
    exp_map_at_origin,
    poincare_to_lorentz,
    project_to_ball,
)

STATIC_TIER_SCALING = {"root": 0.15, "mid": 0.50, "leaf": 0.90}
UNIFORM_SCALING = 0.9
BAND_WIDTH = 0.10
TIER_ORDER = ["root", "mid", "leaf"]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit norm."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return vectors / norms


def _project_batch(vectors: np.ndarray, c: float) -> np.ndarray:
    """Apply exp_map_at_origin + project_to_ball to every row."""
    out = np.empty_like(vectors)
    for i in range(len(vectors)):
        mapped = exp_map_at_origin(vectors[i], c=c)
        out[i] = project_to_ball(mapped, c=c)
    return out


def _norm_stats(vectors: np.ndarray) -> dict:
    """Compute norm statistics for logging."""
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "min": float(norms.min()),
        "max": float(norms.max()),
        "mean": float(norms.mean()),
        "std": float(norms.std()),
    }


def _tier_indices(tiers: list[str]) -> dict[str, list[int]]:
    """Group vector indices by tier."""
    groups: dict[str, list[int]] = {"root": [], "mid": [], "leaf": []}
    for i, t in enumerate(tiers):
        if t in groups:
            groups[t].append(i)
    return groups


def project_uniform(
    vectors: np.ndarray, tiers: list[str], c: float
) -> dict:
    """Uniform scaling: L2-normalize, scale by 0.9, project."""
    normed = _l2_normalize(vectors)
    scaled = normed * UNIFORM_SCALING
    projected = _project_batch(scaled, c)
    return {
        "vectors": projected,
        "metadata": {
            "strategy": "uniform",
            "scaling": UNIFORM_SCALING,
            "norm_stats": _norm_stats(projected),
        },
    }


def project_static(
    vectors: np.ndarray, tiers: list[str], c: float
) -> dict:
    """Static per-tier scaling: L2-normalize, scale by tier, project."""
    normed = _l2_normalize(vectors)
    scaled = np.empty_like(normed)
    groups = _tier_indices(tiers)

    for tier, indices in groups.items():
        if indices:
            scale = STATIC_TIER_SCALING[tier]
            scaled[indices] = normed[indices] * scale

    projected = _project_batch(scaled, c)

    tier_norm_stats = {}
    for tier, indices in groups.items():
        if indices:
            tier_norm_stats[tier] = _norm_stats(projected[indices])

    return {
        "vectors": projected,
        "metadata": {
            "strategy": "static",
            "tier_scaling": STATIC_TIER_SCALING,
            "tier_norm_stats": tier_norm_stats,
            "norm_stats": _norm_stats(projected),
        },
    }


def _calibrate_einstein_radii(
    vectors: np.ndarray, tiers: list[str], c: float
) -> dict[str, float]:
    """Calibration pass: project uniformly, compute Einstein midpoint per tier,
    extract Poincare radius as tier center.

    Returns dict mapping tier name to derived radius.
    Enforces monotonic ordering root < mid < leaf.
    """
    normed = _l2_normalize(vectors)
    scaled = normed * UNIFORM_SCALING
    projected = _project_batch(scaled, c)

    groups = _tier_indices(tiers)
    tier_radii: dict[str, float] = {}

    for tier in TIER_ORDER:
        indices = groups[tier]
        if not indices:
            tier_radii[tier] = STATIC_TIER_SCALING[tier]
            print(f"  Warning: no vectors for tier '{tier}', using static fallback {STATIC_TIER_SCALING[tier]}")
            continue
        tier_points = [projected[i] for i in indices]
        midpoint = einstein_midpoint(tier_points, c)
        radius = float(np.linalg.norm(midpoint))
        tier_radii[tier] = radius
        print(f"  Einstein midpoint radius for '{tier}': {radius:.6f} (n={len(indices)})")

    # Enforce monotonic ordering: root < mid < leaf
    radii_list = [tier_radii[t] for t in TIER_ORDER]
    if not (radii_list[0] < radii_list[1] < radii_list[2]):
        print(f"  Warning: radii not monotonic {radii_list}, sorting and reassigning")
        radii_list.sort()
        for t, r in zip(TIER_ORDER, radii_list):
            tier_radii[t] = r

    return tier_radii


def project_einstein(
    vectors: np.ndarray, tiers: list[str], c: float
) -> dict:
    """Einstein midpoint-derived tier centers: calibrate radii from data, then project."""
    print("  Calibrating tier radii via Einstein midpoint...")
    tier_radii = _calibrate_einstein_radii(vectors, tiers, c)

    normed = _l2_normalize(vectors)
    scaled = np.empty_like(normed)
    groups = _tier_indices(tiers)

    for tier, indices in groups.items():
        if indices:
            scaled[indices] = normed[indices] * tier_radii[tier]

    projected = _project_batch(scaled, c)

    tier_norm_stats = {}
    for tier, indices in groups.items():
        if indices:
            tier_norm_stats[tier] = _norm_stats(projected[indices])

    return {
        "vectors": projected,
        "metadata": {
            "strategy": "einstein",
            "tier_radii": tier_radii,
            "tier_norm_stats": tier_norm_stats,
            "norm_stats": _norm_stats(projected),
        },
    }


def project_einstein_spread(
    vectors: np.ndarray, tiers: list[str], c: float,
    band_width: float = BAND_WIDTH,
) -> dict:
    """Einstein-derived tier centers + intra-tier variance from centroid distance."""
    print("  Calibrating tier radii via Einstein midpoint...")
    tier_radii = _calibrate_einstein_radii(vectors, tiers, c)

    normed = _l2_normalize(vectors)
    scaled = np.empty_like(normed)
    groups = _tier_indices(tiers)

    for tier, indices in groups.items():
        if not indices:
            continue
        tier_center = tier_radii[tier]
        tier_vectors = normed[indices]

        # Euclidean centroid in PCA space (pre-projection)
        centroid = tier_vectors.mean(axis=0)
        distances = np.linalg.norm(tier_vectors - centroid, axis=1)

        # Normalize distances to [0, 1]
        d_min, d_max = distances.min(), distances.max()
        if d_max - d_min > EPS:
            normalized = (distances - d_min) / (d_max - d_min)
        else:
            normalized = np.full_like(distances, 0.5)

        # Scale = tier_center + band_width * (normalized - 0.5)
        per_vector_scale = tier_center + band_width * (normalized - 0.5)

        # Clamp to [EPS, ball_boundary) for safety
        ball_boundary = (1.0 / np.sqrt(c)) - EPS
        per_vector_scale = np.clip(per_vector_scale, EPS, ball_boundary)

        scaled[indices] = tier_vectors * per_vector_scale[:, np.newaxis]

    projected = _project_batch(scaled, c)

    tier_norm_stats = {}
    for tier, indices in groups.items():
        if indices:
            tier_norm_stats[tier] = _norm_stats(projected[indices])

    return {
        "vectors": projected,
        "metadata": {
            "strategy": "einstein_spread",
            "tier_radii": tier_radii,
            "band_width": band_width,
            "tier_norm_stats": tier_norm_stats,
            "norm_stats": _norm_stats(projected),
        },
    }


STRATEGIES = {
    "uniform": project_uniform,
    "static": project_static,
    "einstein": project_einstein,
    "einstein_spread": project_einstein_spread,
}
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "from projection import STRATEGIES; print(list(STRATEGIES.keys()))"`
Expected: `['uniform', 'static', 'einstein', 'einstein_spread']`

---

### Task 3: Update `embed.py`

**Files:**
- Modify: `dev/testbench/embed.py`

- [ ] **Step 1: Replace inline math and curvature loop with strategy loop**

Replace the entire file with the updated version. Key changes:
- Remove inline `EPS`, `BALL_SCALING`, `exp_map_at_origin`, `project_to_ball`, `project_batch`
- Import from `hyperbolic_math` and `projection`
- Replace `CURVATURE_COLLECTIONS` with `STRATEGY_COLLECTIONS` at fixed c=5.0
- Add `--strategy` CLI arg (default: `all`)
- Keep everything else (load, embed, PCA, cosine collection, upsert) unchanged

```python
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
        "--strategy",
        default="all",
        help=(
            "Comma-separated projection strategies to run. "
            "Options: uniform, static, einstein, einstein_spread, all. "
            "Default: all"
        ),
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    # Parse strategies
    if args.strategy == "all":
        strategies_to_run = list(STRATEGIES.keys())
    else:
        strategies_to_run = [s.strip() for s in args.strategy.split(",")]
        for s in strategies_to_run:
            if s not in STRATEGIES:
                parser.error(
                    f"Unknown strategy '{s}'. "
                    f"Choose from: {', '.join(STRATEGIES.keys())}, all"
                )

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

    # ------------------------------------------------------------------
    # Step 4 & 5: Create collections and upsert
    # ------------------------------------------------------------------
    from qdrant_client import QdrantClient

    qdrant_url = args.qdrant_url
    print(f"\n=== Step 4: Create Qdrant collections (url={qdrant_url}) ===")

    client = QdrantClient(url=qdrant_url)

    # Cosine baseline
    print("Creating cosine baseline collection ...")
    create_cosine_collection(client, COSINE_COLLECTION, size=VECTOR_DIM)

    print("\n=== Step 5: Project and upsert (c={}, strategies={}) ===".format(
        CURVATURE, strategies_to_run
    ))

    # Cosine: use PCA-reduced vectors directly
    print("Upserting cosine collection ...")
    upsert_vectors(client, COSINE_COLLECTION, docs, embeddings_128)

    # Poincare: project with each strategy, then upsert
    for strategy_name in strategies_to_run:
        coll_name = STRATEGY_COLLECTIONS[strategy_name]
        print(f"\nProjecting with strategy '{strategy_name}' (c={CURVATURE}) ...")

        create_poincare_collection(
            qdrant_url, coll_name, size=VECTOR_DIM, curvature=CURVATURE
        )

        strategy_fn = STRATEGIES[strategy_name]
        result = strategy_fn(embeddings_128, tiers, c=CURVATURE)
        projected = result["vectors"]
        metadata = result["metadata"]

        norm_stats = metadata.get("norm_stats", {})
        print(
            f"  Norm stats — min: {norm_stats.get('min', 0):.4f}, "
            f"max: {norm_stats.get('max', 0):.4f}, "
            f"mean: {norm_stats.get('mean', 0):.4f}"
        )

        if "tier_radii" in metadata:
            print(f"  Derived tier radii: {metadata['tier_radii']}")

        if "tier_norm_stats" in metadata:
            for tier, stats in metadata["tier_norm_stats"].items():
                print(
                    f"  {tier:>5}: norm mean={stats['mean']:.4f}, "
                    f"std={stats['std']:.4f}"
                )

        upsert_vectors(client, coll_name, docs, projected)

    print("\nAll done.")
    print("Collections created and populated:")
    print(f"  {COSINE_COLLECTION:<20} — 128d cosine baseline")
    for strategy_name in strategies_to_run:
        coll_name = STRATEGY_COLLECTIONS[strategy_name]
        print(f"  {coll_name:<20} — Poincare c={CURVATURE} ({strategy_name})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "import embed; print('OK')"`
Expected: `OK`

---

### Task 4: Update `benchmark.py`

**Files:**
- Modify: `dev/testbench/benchmark.py`

- [ ] **Step 1: Replace the entire benchmark file with updated version**

Key changes:
- Import shared math from `hyperbolic_math` instead of inline definitions
- Dynamic collection discovery: scan for `wos_*`, detect cosine vs Poincare from collection config
- Add `benchmark_band_overlap` metric
- Update comparison table with `BandOvlp` column and strategy-based grouping

```python
"""Benchmark script for hyperbolic (Poincare / Busemann) vector collections in Qdrant.

Runs 4 benchmark suites:
  1. Hierarchy Separation  — Busemann depth per tier + band overlap
  2. Retrieval Quality     — recall@10 within hierarchy
  3. Curvature Comparison  — summary table (printed after 1, 2, 4)
  4. Performance & Latency — throughput and p50/p95/p99

Auto-discovers wos_* collections and adapts to whatever exists.

Usage:
  python benchmark.py [--qdrant-url http://localhost:6334]
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams

from hyperbolic_math import (
    EPS,
    compute_busemann_depths,
    compute_focal_direction,
    poincare_to_lorentz,
    busemann_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EF_VALUES = [64, 128, 256]
DEFAULT_CURVATURE = 5.0


# ---------------------------------------------------------------------------
# Collection discovery
# ---------------------------------------------------------------------------


def discover_collections(client: QdrantClient) -> list[dict]:
    """Auto-discover wos_* collections and detect their config.

    Returns a list of dicts:
      {"name": str, "is_poincare": bool, "curvature": float | None, "strategy": str | None}
    """
    existing = client.get_collections().collections
    collections = []

    for col_info in existing:
        name = col_info.name
        if not name.startswith("wos_"):
            continue

        # Determine distance type from collection config
        col_detail = client.get_collection(name)
        vectors_config = col_detail.config.params.vectors

        # vectors_config may be a VectorParams or a dict of named vectors
        if hasattr(vectors_config, "distance"):
            distance = str(vectors_config.distance)
        elif isinstance(vectors_config, dict):
            first = next(iter(vectors_config.values()))
            distance = str(first.distance) if hasattr(first, "distance") else "unknown"
        else:
            distance = "unknown"

        is_poincare = "poincare" in distance.lower()

        # Infer strategy from collection name
        strategy = None
        curvature = None
        if name == "wos_cosine":
            strategy = None
            curvature = None
        elif "_uniform" in name:
            strategy = "uniform"
            curvature = DEFAULT_CURVATURE
        elif "_static" in name:
            strategy = "static"
            curvature = DEFAULT_CURVATURE
        elif "_einstein" in name and "_spread" not in name:
            strategy = "einstein"
            curvature = DEFAULT_CURVATURE
        elif "_spread" in name:
            strategy = "einstein_spread"
            curvature = DEFAULT_CURVATURE
        elif is_poincare:
            # Legacy collections like wos_c05, wos_c50
            strategy = "uniform"
            # Try to parse curvature from name (wos_c50 -> 5.0)
            suffix = name.replace("wos_c", "")
            try:
                curvature = int(suffix) / 10.0
            except ValueError:
                curvature = DEFAULT_CURVATURE

        collections.append({
            "name": name,
            "is_poincare": is_poincare,
            "curvature": curvature,
            "strategy": strategy,
        })

    # Sort: cosine first, then by strategy name
    collections.sort(key=lambda c: (c["is_poincare"], c["strategy"] or ""))
    return collections


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def scroll_all(client: QdrantClient, collection: str) -> list[dict]:
    """Scroll all points from a collection, returning list of dicts."""
    points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        for pt in batch:
            raw_vec = pt.vector
            if isinstance(raw_vec, dict):
                raw_vec = next(iter(raw_vec.values()))
            points.append({
                "id": pt.id,
                "vector": np.array(raw_vec, dtype=np.float32),
                "tier": pt.payload.get("tier", "leaf"),
                "domain": pt.payload.get("domain", -1),
                "area": pt.payload.get("area", -1),
            })
        if next_offset is None:
            break
        offset = next_offset
    return points


# ---------------------------------------------------------------------------
# Search helper
# ---------------------------------------------------------------------------


def _search(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    limit: int,
    search_params: SearchParams | None = None,
) -> list:
    """Run nearest-neighbour search."""
    vec = query_vector.tolist()
    kwargs: dict = dict(collection_name=collection, limit=limit)
    if search_params:
        kwargs["search_params"] = search_params

    try:
        results = client.query_points(query=vec, **kwargs)
        return results.points
    except (AttributeError, TypeError):
        pass

    try:
        return client.search(query_vector=vec, **kwargs)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Benchmark 1 — Hierarchy Separation
# ---------------------------------------------------------------------------


def benchmark_hierarchy_separation(points: list[dict], curvature: float) -> dict:
    """Compute Busemann depth per tier, separation metrics, and band overlap."""
    depths = compute_busemann_depths(points, c=curvature)

    tier_depths: dict[str, list[float]] = {"root": [], "mid": [], "leaf": []}
    for pt, d in zip(points, depths):
        tier = pt["tier"]
        if tier in tier_depths:
            tier_depths[tier].append(d)

    results: dict = {}
    for tier, ds in tier_depths.items():
        if ds:
            results[f"avg_depth_{tier}"] = float(np.mean(ds))
            results[f"std_depth_{tier}"] = float(np.std(ds))
        else:
            results[f"avg_depth_{tier}"] = None
            results[f"std_depth_{tier}"] = None

    # Separation ratio = |avg_leaf - avg_root| / std_all
    all_depths = depths
    std_all = float(np.std(all_depths)) if all_depths else EPS
    avg_leaf = results["avg_depth_leaf"]
    avg_root = results["avg_depth_root"]

    if avg_leaf is not None and avg_root is not None and std_all > EPS:
        sep_ratio = abs(avg_leaf - avg_root) / std_all
    else:
        sep_ratio = None

    results["separation_ratio"] = sep_ratio
    results["num_points"] = len(points)

    # Tier classification accuracy via median threshold
    if all_depths:
        median_d = float(np.median(all_depths))
        correct = 0
        for pt, d in zip(points, depths):
            predicted = "leaf" if d >= median_d else "root"
            if pt["tier"] == predicted or (pt["tier"] == "mid" and predicted == "root"):
                correct += 1
        results["tier_classification_accuracy"] = correct / len(points)
    else:
        results["tier_classification_accuracy"] = None

    # Band overlap: fraction of vectors whose Busemann depth falls
    # in a different tier's band (using midpoint thresholds between tiers)
    results["band_overlap"] = _compute_band_overlap(tier_depths)

    return results


def _compute_band_overlap(tier_depths: dict[str, list[float]]) -> float | None:
    """Compute fraction of vectors that fall outside their tier's Busemann band.

    Bands defined by midpoint thresholds between adjacent tier means:
      threshold_root_mid = (avg_root + avg_mid) / 2
      threshold_mid_leaf = (avg_mid + avg_leaf) / 2
    A vector is "misplaced" if its depth crosses into a different tier's band.
    """
    avgs = {}
    for tier in ("root", "mid", "leaf"):
        if not tier_depths[tier]:
            return None
        avgs[tier] = float(np.mean(tier_depths[tier]))

    # Ensure the thresholds make sense (root < mid < leaf in depth)
    sorted_tiers = sorted(avgs.keys(), key=lambda t: avgs[t])

    if len(sorted_tiers) < 3:
        return None

    lo_tier, mid_tier, hi_tier = sorted_tiers
    thresh_lo_mid = (avgs[lo_tier] + avgs[mid_tier]) / 2.0
    thresh_mid_hi = (avgs[mid_tier] + avgs[hi_tier]) / 2.0

    tier_to_band = {
        lo_tier: (-np.inf, thresh_lo_mid),
        mid_tier: (thresh_lo_mid, thresh_mid_hi),
        hi_tier: (thresh_mid_hi, np.inf),
    }

    total = 0
    misplaced = 0
    for tier, ds in tier_depths.items():
        lo, hi = tier_to_band[tier]
        for d in ds:
            total += 1
            if d < lo or d >= hi:
                misplaced += 1

    return misplaced / total if total > 0 else None


# ---------------------------------------------------------------------------
# Benchmark 2 — Retrieval Quality
# ---------------------------------------------------------------------------


def benchmark_retrieval_quality(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    k: int = 10,
    num_queries: int = 500,
) -> dict:
    """Sample leaf documents and evaluate recall@10 within hierarchy."""
    leaf_points = [pt for pt in points if pt["tier"] == "leaf"]
    if not leaf_points:
        return {"error": "no leaf points found"}

    query_sample = random.sample(leaf_points, min(num_queries, len(leaf_points)))

    area_recalls: list[float] = []
    domain_recalls: list[float] = []
    h_precisions: list[float] = []

    id_to_pt = {pt["id"]: pt for pt in points}

    for query_pt in query_sample:
        try:
            results = _search(client, collection, query_pt["vector"], limit=k + 1)
        except Exception:
            continue

        query_id = query_pt["id"]
        neighbors = [r for r in results if r.id != query_id][:k]

        if not neighbors:
            continue

        same_area = 0
        same_domain = 0
        h_prec = 0.0

        for nb in neighbors:
            nb_pt = id_to_pt.get(nb.id)
            if nb_pt is None:
                continue

            if nb_pt["area"] == query_pt["area"]:
                same_area += 1
                h_prec += 1.0
            elif nb_pt["domain"] == query_pt["domain"]:
                same_domain += 1
                h_prec += 0.5

        area_recalls.append(same_area / k)
        domain_recalls.append((same_area + same_domain) / k)
        h_precisions.append(h_prec / k)

    def _safe_mean(lst: list[float]) -> float | None:
        return float(np.mean(lst)) if lst else None

    return {
        "num_queries": len(area_recalls),
        "area_recall_at_10": _safe_mean(area_recalls),
        "domain_recall_at_10": _safe_mean(domain_recalls),
        "hierarchical_precision": _safe_mean(h_precisions),
    }


# ---------------------------------------------------------------------------
# Benchmark 4 — Performance & Latency
# ---------------------------------------------------------------------------


def benchmark_latency(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    num_queries: int = 1000,
) -> dict:
    """Run sequential search at multiple ef values, measure QPS and latency."""
    leaf_points = [pt for pt in points if pt["tier"] == "leaf"]
    if not leaf_points:
        return {"error": "no leaf points found"}

    query_sample = random.sample(leaf_points, min(num_queries, len(leaf_points)))
    ef_results: dict = {}

    for ef in EF_VALUES:
        search_params = SearchParams(hnsw_ef=ef)
        latencies_ms: list[float] = []

        for query_pt in query_sample:
            t0 = time.perf_counter()
            try:
                _search(
                    client, collection, query_pt["vector"],
                    limit=10, search_params=search_params,
                )
            except Exception:
                pass
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        if latencies_ms:
            total_s = sum(latencies_ms) / 1000.0
            ef_results[f"ef_{ef}"] = {
                "qps": len(latencies_ms) / total_s if total_s > 0 else None,
                "p50_ms": float(np.percentile(latencies_ms, 50)),
                "p95_ms": float(np.percentile(latencies_ms, 95)),
                "p99_ms": float(np.percentile(latencies_ms, 99)),
                "num_queries": len(latencies_ms),
            }
        else:
            ef_results[f"ef_{ef}"] = {"error": "no results"}

    return ef_results


# ---------------------------------------------------------------------------
# Comparison table (Benchmark 3)
# ---------------------------------------------------------------------------


def print_comparison_table(
    all_results: dict, collection_configs: list[dict]
) -> None:
    """Print a side-by-side comparison table across all collections."""
    header = (
        f"{'Collection':<20} {'Strategy':>12} {'SepRatio':>10} "
        f"{'BandOvlp':>10} {'AreaRec@10':>11} {'DomRec@10':>10} "
        f"{'H-Prec':>8} {'p95@ef128':>10}"
    )
    print("\n" + "=" * len(header))
    print("Projection Strategy Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for col_cfg in collection_configs:
        col = col_cfg["name"]
        strategy = col_cfg.get("strategy") or "cosine"

        res = all_results.get(col, {})

        sep = res.get("hierarchy_separation", {})
        sep_ratio = sep.get("separation_ratio")
        sep_str = f"{sep_ratio:.3f}" if sep_ratio is not None else "N/A"

        band_ovlp = sep.get("band_overlap")
        band_str = f"{band_ovlp:.3f}" if band_ovlp is not None else "N/A"

        rq = res.get("retrieval_quality", {})
        area_rec = rq.get("area_recall_at_10")
        dom_rec = rq.get("domain_recall_at_10")
        h_prec = rq.get("hierarchical_precision")
        area_str = f"{area_rec:.3f}" if area_rec is not None else "N/A"
        dom_str = f"{dom_rec:.3f}" if dom_rec is not None else "N/A"
        h_str = f"{h_prec:.3f}" if h_prec is not None else "N/A"

        lat = res.get("latency", {})
        p95 = lat.get("ef_128", {}).get("p95_ms")
        p95_str = f"{p95:.1f}ms" if p95 is not None else "N/A"

        print(
            f"{col:<20} {strategy:>12} {sep_str:>10} "
            f"{band_str:>10} {area_str:>11} {dom_str:>10} "
            f"{h_str:>8} {p95_str:>10}"
        )

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark hyperbolic vector collections in Qdrant"
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6334",
        help="Qdrant gRPC/HTTP URL (default: http://localhost:6334)",
    )
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url)

    # Discover collections
    collection_configs = discover_collections(client)

    if not collection_configs:
        print("No wos_* collections found. Run embed.py first.")
        return

    print(f"Discovered collections: {[c['name'] for c in collection_configs]}")

    all_results: dict = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"benchmark_{timestamp}.json"

    for col_cfg in collection_configs:
        collection = col_cfg["name"]
        curvature = col_cfg["curvature"]
        strategy = col_cfg.get("strategy") or "cosine"

        print(f"\n{'='*60}")
        print(f"Collection: {collection}  (strategy={strategy}, curvature={curvature})")
        print(f"{'='*60}")

        print("  Loading points...")
        points = scroll_all(client, collection)
        print(f"  Loaded {len(points)} points")

        col_results: dict = {}

        # --- Benchmark 1: Hierarchy Separation (Poincare only) ---
        if col_cfg["is_poincare"] and curvature is not None:
            print("  [1/3] Hierarchy Separation...")
            sep_res = benchmark_hierarchy_separation(points, curvature=curvature)
            col_results["hierarchy_separation"] = sep_res
            sep_ratio = sep_res.get("separation_ratio")
            band_ovlp = sep_res.get("band_overlap")
            print(
                f"        sep_ratio={sep_ratio:.4f}" if sep_ratio is not None
                else "        sep_ratio=N/A"
            )
            print(
                f"        band_overlap={band_ovlp:.4f}" if band_ovlp is not None
                else "        band_overlap=N/A"
            )
            for tier in ("root", "mid", "leaf"):
                avg = sep_res.get(f"avg_depth_{tier}")
                if avg is not None:
                    print(f"        avg_depth_{tier}={avg:.4f}")
        else:
            print("  [1/3] Hierarchy Separation — skipped (cosine collection)")
            col_results["hierarchy_separation"] = {}

        # --- Benchmark 2: Retrieval Quality ---
        print("  [2/3] Retrieval Quality (500 queries)...")
        rq_res = benchmark_retrieval_quality(
            client, collection, points, k=10, num_queries=500
        )
        col_results["retrieval_quality"] = rq_res
        print(
            f"        area_recall@10={rq_res.get('area_recall_at_10'):.4f}"
            if rq_res.get("area_recall_at_10") is not None
            else "        area_recall@10=N/A"
        )
        print(
            f"        domain_recall@10={rq_res.get('domain_recall_at_10'):.4f}"
            if rq_res.get("domain_recall_at_10") is not None
            else "        domain_recall@10=N/A"
        )
        print(
            f"        h_precision={rq_res.get('hierarchical_precision'):.4f}"
            if rq_res.get("hierarchical_precision") is not None
            else "        h_precision=N/A"
        )

        # --- Benchmark 4: Latency ---
        print("  [3/3] Latency (1000 queries, ef=64/128/256)...")
        lat_res = benchmark_latency(client, collection, points, num_queries=1000)
        col_results["latency"] = lat_res
        for ef in EF_VALUES:
            ef_data = lat_res.get(f"ef_{ef}", {})
            qps = ef_data.get("qps")
            p95 = ef_data.get("p95_ms")
            print(
                f"        ef={ef}: qps={qps:.1f}, p95={p95:.1f}ms"
                if qps is not None and p95 is not None
                else f"        ef={ef}: N/A"
            )

        all_results[collection] = col_results

        # Save intermediate results
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # --- Comparison table ---
    print_comparison_table(all_results, collection_configs)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "import benchmark; print('OK')"`
Expected: `OK`

---

### Task 5: Smoke Test (no Qdrant needed)

**Files:**
- None (validation only)

- [ ] **Step 1: Test projection strategies produce correct shapes and distinct radii**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "
import numpy as np
from projection import STRATEGIES

np.random.seed(42)

# Simulate 100 vectors: 5 root, 15 mid, 80 leaf
n = 100
vectors = np.random.randn(n, 128).astype(np.float32)
tiers = ['root'] * 5 + ['mid'] * 15 + ['leaf'] * 80
c = 5.0

for name, fn in STRATEGIES.items():
    result = fn(vectors, tiers, c)
    vecs = result['vectors']
    meta = result['metadata']
    norms = np.linalg.norm(vecs, axis=1)

    root_mean = norms[:5].mean()
    mid_mean = norms[5:20].mean()
    leaf_mean = norms[20:].mean()

    print(f'{name:<18} shape={vecs.shape}  root={root_mean:.4f}  mid={mid_mean:.4f}  leaf={leaf_mean:.4f}')
    if 'tier_radii' in meta:
        print(f'  derived radii: {meta[\"tier_radii\"]}')
print('All strategies OK')
"
```

Expected:
- All shapes are `(100, 128)`
- `uniform`: root, mid, leaf norms are all similar (~same value)
- `static`: root < mid < leaf norms with clear separation
- `einstein`: root < mid < leaf norms with data-derived values
- `einstein_spread`: root < mid < leaf norms with slight spread within tiers

- [ ] **Step 2: Test Einstein midpoint directly**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "
import numpy as np
from hyperbolic_math import einstein_midpoint, project_to_ball, exp_map_at_origin

np.random.seed(42)
c = 5.0

# Create points at different radii
points = []
for r in [0.1, 0.1, 0.1]:  # 3 points near origin
    v = np.random.randn(128).astype(np.float32)
    v = v / np.linalg.norm(v) * r
    p = exp_map_at_origin(v, c)
    p = project_to_ball(p, c)
    points.append(p)

midpoint = einstein_midpoint(points, c)
print(f'Midpoint norm: {np.linalg.norm(midpoint):.6f}')
print(f'Input norms: {[np.linalg.norm(p):.6f for p in points]}')
assert np.linalg.norm(midpoint) < 1.0 / np.sqrt(c), 'Midpoint outside ball!'
print('Einstein midpoint OK')
"
```

Expected: Midpoint norm close to input norms (~0.1), midpoint inside ball.

- [ ] **Step 3: Test band overlap computation**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "
from benchmark import _compute_band_overlap

# Perfect separation: no overlap
perfect = {
    'root': [0.1, 0.2, 0.15],
    'mid': [0.5, 0.55, 0.52],
    'leaf': [0.9, 0.95, 0.92],
}
print(f'Perfect separation overlap: {_compute_band_overlap(perfect):.4f}')

# Total overlap: all same values
uniform = {
    'root': [0.5, 0.5, 0.5],
    'mid': [0.5, 0.5, 0.5],
    'leaf': [0.5, 0.5, 0.5],
}
print(f'Uniform overlap: {_compute_band_overlap(uniform):.4f}')
print('Band overlap OK')
"
```

Expected: Perfect separation overlap = 0.0000. Uniform overlap > 0 (likely ~0.6667 since 2/3 of tiers will be misplaced).

- [ ] **Step 4: Commit (no code changes, just validation)**

No commit needed — this is a validation-only task.

---

### Task 6: Integration Test with Qdrant

**Files:**
- None (validation only — requires running Qdrant)

- [ ] **Step 1: Ensure Qdrant is running**

Run: `curl -sf http://localhost:6334/healthz && echo "Qdrant OK" || echo "Start Qdrant first: cd dev/testbench && docker compose up -d"`

- [ ] **Step 2: Run embed.py with all strategies**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 embed.py --qdrant-url http://localhost:6334 --skip-embed --strategy all
```

Expected:
- 5 collections created (wos_cosine + 4 strategy collections)
- Each strategy prints norm stats showing distinct tier radii
- Einstein strategies print derived tier radii
- No errors during upsert

- [ ] **Step 3: Run benchmark.py**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 benchmark.py --qdrant-url http://localhost:6334
```

Expected:
- Auto-discovers all 5 collections
- Hierarchy separation runs on all 4 Poincare collections
- Band overlap metric is printed for each
- Comparison table prints with all 5 collections
- Results JSON saved to `results/`

- [ ] **Step 4: Verify results meet success criteria**

Check the comparison table output:
1. Busemann separation ratio > 10 for at least one tier-aware strategy
2. Band overlap < 0.05 for einstein and einstein_spread
3. Area recall@10 > 0.220 (20% improvement over cosine 0.184) for best strategy
4. p95 latency < 50ms at ef=128
5. Einstein-derived radii are monotonic (root < mid < leaf) without correction

- [ ] **Step 5: Commit all changes (single commit per user preference)**

```bash
git add dev/testbench/hyperbolic_math.py dev/testbench/projection.py dev/testbench/embed.py dev/testbench/benchmark.py
git commit -m "feat(testbench): tier-aware projection with 4 strategies (uniform/static/einstein/spread)

Replace uniform-radius Poincare projection with four strategies:
- uniform: L2-normalize + scale 0.9 (fixed baseline)
- static: hardcoded per-tier radii (0.15/0.50/0.90)
- einstein: Einstein midpoint-derived tier centers
- einstein_spread: Einstein + intra-tier variance (band_width=0.10)

Extract shared math into hyperbolic_math.py, strategies into projection.py.
Update benchmark.py with dynamic collection discovery and band overlap metric."
```
