# Hyperbolic Phase 3: Unified Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Rust `lorentz_to_poincare` bug, create a unified `wos_unified` collection with named vectors (cosine + poincare) and Busemann depth payloads, implement dual-space RRF search, and run a curvature re-sweep.

**Architecture:** The Rust fix is a 2-line change + call site update. The testbench gets a new `fusion.py` module for RRF, major extensions to `embed.py` for tier centroid generation and unified collection upsert, and new benchmark suites in `benchmark.py` for cross-tier retrieval and dual-space comparison. Legacy `wos_c50_*` collections are dropped to free space.

**Tech Stack:** Rust (Qdrant segment crate, `hyperbolic` feature), Python (qdrant-client, numpy, scikit-learn), Docker (custom Qdrant build)

**Spec:** `docs/superpowers/specs/2026-04-09-hyperbolic-phase3-unified-collection-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `lib/segment/src/spaces/hyperbolic/poincare_math.rs` | Modify | Fix `lorentz_to_poincare` signature + body, update `einstein_midpoint` call site, add round-trip test |
| `dev/testbench/fusion.py` | Create | RRF fusion implementation |
| `dev/testbench/embed.py` | Modify | Busemann depth computation, tier centroid generation, unified collection creation, legacy cleanup |
| `dev/testbench/hyperbolic_math.py` | Modify | Add `busemann_depth_single()` convenience function for per-point depth |
| `dev/testbench/benchmark.py` | Modify | Add cross-tier retrieval suite, dual-space comparison suite, unified collection discovery |
| `dev/testbench/run.sh` | Modify | Add `unified` run mode |

---

### Task 1: Fix Rust `lorentz_to_poincare`

**Files:**
- Modify: `lib/segment/src/spaces/hyperbolic/poincare_math.rs:353-360` (function body)
- Modify: `lib/segment/src/spaces/hyperbolic/poincare_math.rs:304` (call site in `einstein_midpoint`)

- [ ] **Step 1: Write failing round-trip test**

Add this test at the end of the `mod tests` block in `poincare_math.rs` (before the closing `}`), after line 833:

```rust
    #[test]
    fn test_lorentz_roundtrip_nonunit_curvature() {
        // Round-trip: lorentz_to_poincare(poincare_to_lorentz(p, c), c) ≈ p
        let p = vec![0.2f32, -0.3, 0.1];
        for &c in &[1.0, 2.0, 5.0, 10.0] {
            let lorentz = poincare_to_lorentz(&p, c);
            let recovered = lorentz_to_poincare(&lorentz, c);
            assert!(
                vec_approx_eq(&recovered, &p),
                "round-trip failed at c={c}: p={p:?}, recovered={recovered:?}"
            );
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/rohan/projects/qdrant
cargo test --features hyperbolic -p segment test_lorentz_roundtrip_nonunit_curvature -- --nocapture
```

Expected: Compilation error — `lorentz_to_poincare` currently takes 1 argument, test passes 2.

- [ ] **Step 3: Fix `lorentz_to_poincare` function signature and body**

Replace lines 353-360 in `poincare_math.rs`:

```rust
/// Convert a Lorentz hyperboloid point back to Poincaré ball coordinates.
///
/// p_i = x_i / (√c · (x_0 + 1))  for i > 0
///
/// The forward map `poincare_to_lorentz` scales spatial components by 2√c,
/// so the inverse must divide by √c to recover the original Poincaré point.
pub fn lorentz_to_poincare(x: &[f32], c: f32) -> Vec<f32> {
    let x0 = x[0];
    let sqrt_c = c.sqrt();
    let denom = sqrt_c * (x0 + 1.0);
    x[1..].iter().map(|&xi| xi / denom.max(EPS)).collect()
}
```

- [ ] **Step 4: Update `einstein_midpoint` call site**

Replace line 304 in `poincare_math.rs`:

Old:
```rust
    let poincare = lorentz_to_poincare(&on_hyperboloid);
```

New:
```rust
    let poincare = lorentz_to_poincare(&on_hyperboloid, c);
```

- [ ] **Step 5: Run all hyperbolic tests**

Run:
```bash
cd /home/rohan/projects/qdrant
cargo test --features hyperbolic -p segment -- --nocapture 2>&1 | tail -30
```

Expected: All tests pass, including the new `test_lorentz_roundtrip_nonunit_curvature`.

- [ ] **Step 6: Rebuild Docker image**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
docker compose up -d --build
```

This rebuilds the Qdrant binary with the fix. Takes ~5-10 minutes.

Wait for healthy:
```bash
for i in $(seq 1 60); do curl -sf http://localhost:6334/healthz > /dev/null 2>&1 && echo "Ready" && break || sleep 2; done
```

---

### Task 2: Create RRF Fusion Module

**Files:**
- Create: `dev/testbench/fusion.py`

- [ ] **Step 1: Write `fusion.py`**

```python
"""Reciprocal Rank Fusion (RRF) for dual-space search.

Combines results from two independent ranking lists into a single
fused ranking. Following ruvector's pattern (ruvector-postgres/src/hybrid/fusion.rs).

Usage:
    fused = rrf_fuse(cosine_results, poincare_results, k=60, limit=10)
"""


def rrf_fuse(
    results_a: list,
    results_b: list,
    k: int = 60,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of two result lists.

    Each result is expected to have an `.id` attribute (qdrant ScoredPoint).

    Args:
        results_a: First ranked list (e.g., cosine results).
        results_b: Second ranked list (e.g., poincare results).
        k: RRF constant (default 60, robust across scales).
        limit: Number of fused results to return.

    Returns:
        List of (point_id, rrf_score) tuples, sorted by descending score.
    """
    scores: dict[int, float] = {}

    for rank, point in enumerate(results_a):
        pid = point.id if hasattr(point, "id") else point
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

    for rank, point in enumerate(results_b):
        pid = point.id if hasattr(point, "id") else point
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:limit]
```

- [ ] **Step 2: Verify module imports**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "from fusion import rrf_fuse; print('OK')"
```

Expected: `OK`

---

### Task 3: Add `busemann_depth_single` to `hyperbolic_math.py`

**Files:**
- Modify: `dev/testbench/hyperbolic_math.py`

- [ ] **Step 1: Add convenience function**

Add after `compute_busemann_depths` (after line 117), before the Einstein midpoint section comment:

```python


def busemann_depth_single(
    vector: np.ndarray, focal: np.ndarray, c: float = 1.0
) -> float:
    """Busemann depth for a single Poincare ball vector given a focal direction."""
    lorentz = poincare_to_lorentz(vector, c)
    return busemann_score(lorentz, focal)
```

- [ ] **Step 2: Verify import**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "from hyperbolic_math import busemann_depth_single; print('OK')"
```

Expected: `OK`

---

### Task 4: Extend `embed.py` — Unified Collection Pipeline

**Files:**
- Modify: `dev/testbench/embed.py`

This is the largest task. We add: Busemann depth computation, tier centroid generation, unified collection creation/upsert, and legacy collection cleanup.

- [ ] **Step 1: Add new imports and constants**

At the top of `embed.py`, add to the imports (after line 27):

```python
from hyperbolic_math import (
    busemann_depth_single,
    compute_focal_direction,
    einstein_midpoint,
)
```

After the `STRATEGY_COLLECTIONS` dict (after line 53), add:

```python
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
```

- [ ] **Step 2: Add Busemann depth computation function**

Add after the `reduce_pca` function (after line 184):

```python
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
```

- [ ] **Step 3: Add tier centroid generation function**

Add after `compute_busemann`:

```python
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
    if AREA_CENTROIDS_FILE.exists() and DOMAIN_CENTROIDS_FILE.exists():
        print("Loading cached tier centroids ...")
        area_data = np.load(AREA_CENTROIDS_FILE, allow_pickle=True)
        domain_data = np.load(DOMAIN_CENTROIDS_FILE, allow_pickle=True)
        area_entries = list(area_data["entries"])
        domain_entries = list(domain_data["entries"])
        if area_entries and domain_entries:
            print(f"  Cache hit: {len(area_entries)} areas, {len(domain_entries)} domains")
            return area_entries, domain_entries

    # Group papers by area and domain
    area_groups: dict[str, list[int]] = {}
    domain_groups: dict[str, list[str]] = {}  # domain -> set of areas

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

    # Cache
    np.savez(AREA_CENTROIDS_FILE, entries=np.array(area_entries, dtype=object))
    np.savez(DOMAIN_CENTROIDS_FILE, entries=np.array(domain_entries, dtype=object))
    print(f"  Saved centroids to {DATA_DIR}")

    return area_entries, domain_entries
```

- [ ] **Step 4: Add unified collection creation function**

Add after `generate_tier_centroids`:

```python
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
                "source_ids": entry["source_ids"][:20],  # Truncate for payload size
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
```

- [ ] **Step 5: Add legacy cleanup function**

Add after `upsert_unified`:

```python
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
```

- [ ] **Step 6: Update `main()` function**

Replace the entire `main()` function (lines 302-449):

```python
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

    print("\nAll done.")
```

- [ ] **Step 7: Test the embedding pipeline**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 embed.py --qdrant-url http://localhost:6334 --skip-embed --cleanup-legacy
```

Expected: Creates `wos_cosine` + `wos_unified`, deletes legacy collections, prints depth summaries showing narratives < stories < content in Busemann depth.

---

### Task 5: Extend `benchmark.py` — Unified Collection Support

**Files:**
- Modify: `dev/testbench/benchmark.py`

- [ ] **Step 1: Add imports and constants**

Add to imports at top of file (after line 28):

```python
from fusion import rrf_fuse
from hyperbolic_math import busemann_depth_single, compute_focal_direction
```

After `DEFAULT_CURVATURE` (after line 37), add:

```python
UNIFIED_COLLECTION = "wos_unified"
CONTENT_ID_OFFSET = 0
STORY_ID_OFFSET = 100_000
NARRATIVE_ID_OFFSET = 200_000
```

- [ ] **Step 2: Update `discover_collections` to detect unified collection**

Add this block inside the `for col_info in existing:` loop in `discover_collections`, after the existing strategy inference (after line 99, before `collections.append`):

```python
        # Detect unified collection (named vectors)
        if name == "wos_unified":
            is_poincare = True
            strategy = "unified"
            curvature = DEFAULT_CURVATURE
```

- [ ] **Step 3: Update `scroll_all` to handle named vectors**

Replace the `scroll_all` function (lines 118-144):

```python
def scroll_all(
    client: QdrantClient,
    collection: str,
    vector_name: str | None = None,
) -> list[dict]:
    """Scroll all points from a collection, returning list of dicts.

    For unified collections, specify vector_name ('cosine' or 'poincare')
    to extract that named vector. If None, extracts the first/only vector.
    """
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
                if vector_name and vector_name in raw_vec:
                    raw_vec = raw_vec[vector_name]
                else:
                    raw_vec = next(iter(raw_vec.values()))
            points.append({
                "id": pt.id,
                "vector": np.array(raw_vec, dtype=np.float32),
                "tier": pt.payload.get("tier", "leaf"),
                "domain": pt.payload.get("domain", -1),
                "area": pt.payload.get("area", -1),
                "busemann_depth": pt.payload.get("busemann_depth"),
            })
        if next_offset is None:
            break
        offset = next_offset
    return points
```

- [ ] **Step 4: Add named-vector search helper**

Add after the `_search` function (after line 174):

```python
def _search_named(
    client: QdrantClient,
    collection: str,
    vector_name: str,
    query_vector: np.ndarray,
    limit: int,
    tier_filter: str | None = None,
) -> list:
    """Search a named vector in the unified collection."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue, NamedVector

    kwargs: dict = dict(
        collection_name=collection,
        limit=limit,
    )

    query = NamedVector(name=vector_name, vector=query_vector.tolist())

    if tier_filter:
        kwargs["query_filter"] = Filter(
            must=[FieldCondition(key="tier", match=MatchValue(value=tier_filter))]
        )

    try:
        results = client.query_points(query=query, **kwargs)
        return results.points
    except (AttributeError, TypeError):
        pass

    try:
        return client.search(query_vector=query, **kwargs)
    except Exception:
        return []
```

- [ ] **Step 5: Add Benchmark 3 — Cross-Tier Retrieval**

Add after `benchmark_retrieval_quality` (after line 341):

```python
# ---------------------------------------------------------------------------
# Benchmark 3 — Cross-Tier Retrieval
# ---------------------------------------------------------------------------


def benchmark_cross_tier(
    client: QdrantClient,
    collection: str,
    points: list[dict],
    k: int = 5,
) -> dict:
    """Evaluate cross-tier retrieval: can we find parent/child points?

    - parent_recall: query a content point, check if its parent area/domain appears
    - child_recall: query a narrative/story point, check if child points appear
    """
    # Build lookup structures
    content_points = [p for p in points if p["tier"] == "content"]
    story_points = [p for p in points if p["tier"] == "story"]
    narrative_points = [p for p in points if p["tier"] == "narrative"]

    if not story_points or not narrative_points:
        return {"error": "need story + narrative tiers in collection"}

    # Map area -> story ID, domain -> narrative ID
    area_to_story_id = {}
    for sp in story_points:
        area_to_story_id[sp["area"]] = sp["id"]

    domain_to_narrative_id = {}
    for np_ in narrative_points:
        domain_to_narrative_id[np_["domain"]] = np_["id"]

    # --- Parent recall: query content, find story/narrative in results ---
    sample_content = random.sample(content_points, min(200, len(content_points)))
    parent_story_hits = 0
    parent_narrative_hits = 0

    for qpt in sample_content:
        results = _search_named(client, collection, "poincare", qpt["vector"], limit=k + 1)
        result_ids = {r.id for r in results if r.id != qpt["id"]}

        expected_story_id = area_to_story_id.get(qpt["area"])
        expected_narrative_id = domain_to_narrative_id.get(qpt["domain"])

        if expected_story_id in result_ids:
            parent_story_hits += 1
        if expected_narrative_id in result_ids:
            parent_narrative_hits += 1

    n_content = len(sample_content)

    # --- Child recall: query narratives, find stories in results ---
    child_story_hits = 0
    child_total = 0

    for npt in narrative_points:
        results = _search_named(client, collection, "poincare", npt["vector"], limit=20)
        result_ids = {r.id for r in results if r.id != npt["id"]}

        # Count how many of this domain's stories appear
        expected_stories = [
            sp["id"] for sp in story_points if sp["domain"] == npt["domain"]
        ]
        for sid in expected_stories:
            child_total += 1
            if sid in result_ids:
                child_story_hits += 1

    return {
        "parent_story_recall": parent_story_hits / n_content if n_content > 0 else None,
        "parent_narrative_recall": parent_narrative_hits / n_content if n_content > 0 else None,
        "child_story_recall": child_story_hits / child_total if child_total > 0 else None,
        "num_content_queries": n_content,
        "num_narrative_queries": len(narrative_points),
    }
```

- [ ] **Step 6: Add Benchmark 4 — Dual-Space Comparison**

Add after `benchmark_cross_tier`:

```python
# ---------------------------------------------------------------------------
# Benchmark 4 — Dual-Space Comparison
# ---------------------------------------------------------------------------


def benchmark_dual_space(
    client: QdrantClient,
    unified_collection: str,
    cosine_collection: str,
    points_unified: list[dict],
    points_cosine: list[dict],
    k: int = 10,
    num_queries: int = 500,
) -> dict:
    """Compare cosine-only vs poincare-only vs dual-space RRF on same queries."""
    # Use content-tier points from unified collection for queries
    content_points = [p for p in points_unified if p["tier"] == "content"]
    if not content_points:
        return {"error": "no content points in unified collection"}

    query_sample = random.sample(content_points, min(num_queries, len(content_points)))

    # Build lookup from cosine collection (for baseline comparison)
    cosine_id_to_pt = {p["id"]: p for p in points_cosine}
    unified_id_to_pt = {p["id"]: p for p in points_unified}

    cosine_area_recalls = []
    poincare_area_recalls = []
    dual_area_recalls = []

    cosine_domain_recalls = []
    poincare_domain_recalls = []
    dual_domain_recalls = []

    over_fetch = k * 2  # Fetch 2x for RRF fusion

    for qpt in query_sample:
        query_id = qpt["id"]
        query_area = qpt["area"]
        query_domain = qpt["domain"]

        # --- Cosine-only (baseline collection) ---
        try:
            cos_results = _search(client, cosine_collection, qpt["vector"], limit=k + 1)
            cos_neighbors = [r for r in cos_results if r.id != query_id][:k]
        except Exception:
            continue

        # --- Poincare-only (unified, poincare named vector, content only) ---
        try:
            poincare_results = _search_named(
                client, unified_collection, "poincare", qpt["vector"],
                limit=k + 1, tier_filter="content",
            )
            poincare_neighbors = [r for r in poincare_results if r.id != query_id][:k]
        except Exception:
            poincare_neighbors = []

        # --- Dual-space RRF ---
        try:
            cos_for_rrf = _search_named(
                client, unified_collection, "cosine", qpt["vector"],
                limit=over_fetch, tier_filter="content",
            )
            poincare_for_rrf = _search_named(
                client, unified_collection, "poincare", qpt["vector"],
                limit=over_fetch, tier_filter="content",
            )
            fused = rrf_fuse(cos_for_rrf, poincare_for_rrf, k=60, limit=k)
            # Remove self
            fused = [(pid, score) for pid, score in fused if pid != query_id][:k]
        except Exception:
            fused = []

        # --- Score each ---
        def _score_results(neighbors, id_to_pt, use_tuple=False):
            same_area = 0
            same_domain = 0
            for nb in neighbors:
                if use_tuple:
                    nb_id = nb[0]
                else:
                    nb_id = nb.id
                nb_pt = id_to_pt.get(nb_id)
                if nb_pt is None:
                    continue
                if nb_pt["area"] == query_area:
                    same_area += 1
                elif nb_pt["domain"] == query_domain:
                    same_domain += 1
            return same_area / k, (same_area + same_domain) / k

        ar_cos, dr_cos = _score_results(cos_neighbors, cosine_id_to_pt)
        ar_poin, dr_poin = _score_results(poincare_neighbors, unified_id_to_pt)
        ar_dual, dr_dual = _score_results(fused, unified_id_to_pt, use_tuple=True)

        cosine_area_recalls.append(ar_cos)
        poincare_area_recalls.append(ar_poin)
        dual_area_recalls.append(ar_dual)

        cosine_domain_recalls.append(dr_cos)
        poincare_domain_recalls.append(dr_poin)
        dual_domain_recalls.append(dr_dual)

    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else None

    return {
        "num_queries": len(cosine_area_recalls),
        "cosine_only": {
            "area_recall_at_10": _safe_mean(cosine_area_recalls),
            "domain_recall_at_10": _safe_mean(cosine_domain_recalls),
        },
        "poincare_only": {
            "area_recall_at_10": _safe_mean(poincare_area_recalls),
            "domain_recall_at_10": _safe_mean(poincare_domain_recalls),
        },
        "dual_rrf": {
            "area_recall_at_10": _safe_mean(dual_area_recalls),
            "domain_recall_at_10": _safe_mean(dual_domain_recalls),
        },
    }
```

- [ ] **Step 7: Update `main()` to run all suites including unified**

Replace the `main()` function in `benchmark.py`:

```python
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark hyperbolic vector collections in Qdrant"
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6334",
        help="Qdrant gRPC/HTTP URL (default: http://localhost:6334)",
    )
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", "hierarchy", "retrieval", "cross-tier", "dual-space", "latency"],
        help="Which benchmark suite to run (default: all)",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    client = QdrantClient(url=args.qdrant_url)

    collection_configs = discover_collections(client, args.qdrant_url)
    if not collection_configs:
        print("No wos_* collections found. Run embed.py first.")
        return

    print(f"Discovered collections: {[c['name'] for c in collection_configs]}")

    all_results: dict = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"benchmark_{timestamp}.json"

    # Check for unified collection
    has_unified = any(c["name"] == UNIFIED_COLLECTION for c in collection_configs)
    has_cosine = any(c["name"] == "wos_cosine" for c in collection_configs)

    # --- Standard per-collection benchmarks (hierarchy, retrieval, latency) ---
    for col_cfg in collection_configs:
        collection = col_cfg["name"]
        curvature = col_cfg["curvature"]
        strategy = col_cfg.get("strategy") or "cosine"

        print(f"\n{'='*60}")
        print(f"Collection: {collection}  (strategy={strategy}, curvature={curvature})")
        print(f"{'='*60}")

        # For unified collection, use poincare vector for hierarchy tests
        vector_name = "poincare" if strategy == "unified" else None
        print("  Loading points...")
        points = scroll_all(client, collection, vector_name=vector_name)
        print(f"  Loaded {len(points)} points")

        col_results: dict = {}

        # --- Benchmark 1: Hierarchy Separation ---
        if args.suite in ("all", "hierarchy"):
            if col_cfg["is_poincare"] and curvature is not None:
                print("  [1] Hierarchy Separation...")
                sep_res = benchmark_hierarchy_separation(points, curvature=curvature)
                col_results["hierarchy_separation"] = sep_res
                sep_ratio = sep_res.get("separation_ratio")
                band_ovlp = sep_res.get("band_overlap")
                print(f"      sep_ratio={sep_ratio:.4f}" if sep_ratio else "      sep_ratio=N/A")
                print(f"      band_overlap={band_ovlp:.4f}" if band_ovlp else "      band_overlap=N/A")
            else:
                col_results["hierarchy_separation"] = {}

        # --- Benchmark 2: Retrieval Quality ---
        if args.suite in ("all", "retrieval"):
            print("  [2] Retrieval Quality (500 queries)...")
            rq_res = benchmark_retrieval_quality(client, collection, points, k=10, num_queries=500)
            col_results["retrieval_quality"] = rq_res
            ar = rq_res.get("area_recall_at_10")
            dr = rq_res.get("domain_recall_at_10")
            hp = rq_res.get("hierarchical_precision")
            print(f"      area_recall@10={ar:.4f}" if ar else "      area_recall@10=N/A")
            print(f"      domain_recall@10={dr:.4f}" if dr else "      domain_recall@10=N/A")
            print(f"      h_precision={hp:.4f}" if hp else "      h_precision=N/A")

        # --- Benchmark 5: Latency ---
        if args.suite in ("all", "latency"):
            print("  [5] Latency (1000 queries)...")
            lat_res = benchmark_latency(client, collection, points, num_queries=1000)
            col_results["latency"] = lat_res
            for ef in EF_VALUES:
                ef_data = lat_res.get(f"ef_{ef}", {})
                qps = ef_data.get("qps")
                p95 = ef_data.get("p95_ms")
                if qps and p95:
                    print(f"      ef={ef}: qps={qps:.1f}, p95={p95:.1f}ms")

        all_results[collection] = col_results

    # --- Unified-specific benchmarks ---
    if has_unified and has_cosine:
        # --- Benchmark 3: Cross-Tier Retrieval ---
        if args.suite in ("all", "cross-tier"):
            print(f"\n{'='*60}")
            print("Cross-Tier Retrieval (wos_unified, poincare)")
            print(f"{'='*60}")
            points_unified = scroll_all(client, UNIFIED_COLLECTION, vector_name="poincare")
            ct_res = benchmark_cross_tier(client, UNIFIED_COLLECTION, points_unified, k=5)
            all_results.setdefault(UNIFIED_COLLECTION, {})["cross_tier"] = ct_res
            print(f"  parent_story_recall:     {ct_res.get('parent_story_recall')}")
            print(f"  parent_narrative_recall:  {ct_res.get('parent_narrative_recall')}")
            print(f"  child_story_recall:       {ct_res.get('child_story_recall')}")

        # --- Benchmark 4: Dual-Space Comparison ---
        if args.suite in ("all", "dual-space"):
            print(f"\n{'='*60}")
            print("Dual-Space Comparison (cosine vs poincare vs RRF)")
            print(f"{'='*60}")
            # Load cosine vectors for unified collection queries
            points_unified_cosine = scroll_all(client, UNIFIED_COLLECTION, vector_name="cosine")
            points_cosine_baseline = scroll_all(client, "wos_cosine")
            ds_res = benchmark_dual_space(
                client, UNIFIED_COLLECTION, "wos_cosine",
                points_unified_cosine, points_cosine_baseline,
                k=10, num_queries=500,
            )
            all_results.setdefault(UNIFIED_COLLECTION, {})["dual_space"] = ds_res

            print(f"\n  {'Mode':<16} {'AreaRec@10':>12} {'DomRec@10':>12}")
            print(f"  {'-'*40}")
            for mode in ("cosine_only", "poincare_only", "dual_rrf"):
                m = ds_res.get(mode, {})
                ar = m.get("area_recall_at_10")
                dr = m.get("domain_recall_at_10")
                ar_s = f"{ar:.4f}" if ar is not None else "N/A"
                dr_s = f"{dr:.4f}" if dr is not None else "N/A"
                print(f"  {mode:<16} {ar_s:>12} {dr_s:>12}")

    # --- Comparison table ---
    print_comparison_table(all_results, collection_configs)

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
```

- [ ] **Step 8: Test benchmark discovery**

Run:
```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 benchmark.py --qdrant-url http://localhost:6334 --suite hierarchy
```

Expected: Discovers `wos_cosine` and `wos_unified`, runs hierarchy separation on the unified collection showing three distinct tiers (narrative/story/content) in Busemann depth bands.

---

### Task 6: Update `run.sh` with Unified Mode

**Files:**
- Modify: `dev/testbench/run.sh`

- [ ] **Step 1: Add `unified` run mode**

Replace the entire `run.sh`:

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

        echo "=== Embedding & uploading (unified) ==="
        python3 embed.py --qdrant-url "$QDRANT_URL" --cleanup-legacy

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
        echo "=== Embedding & uploading (unified) ==="
        python3 embed.py --qdrant-url "$QDRANT_URL" --cleanup-legacy
        echo "=== Running benchmarks ==="
        python3 benchmark.py --qdrant-url "$QDRANT_URL"
        ;;
    rebuild)
        echo "=== Rebuilding Qdrant binary ==="
        docker compose up -d --build
        wait_for_qdrant
        echo "Qdrant rebuilt and ready."
        ;;
    down)
        docker compose down -v
        ;;
    *)
        echo "Usage: ./run.sh [all|benchmark|embed|rebuild|down]"
        exit 1
        ;;
esac

echo "=== Done ==="
```

---

### Task 7: Run Full Pipeline and Capture Results

**Files:**
- Output: `dev/testbench/results/analysis_round3.md` (created by hand from benchmark output)

- [ ] **Step 1: Rebuild Qdrant with the Rust fix**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
./run.sh rebuild
```

Wait for "Qdrant rebuilt and ready."

- [ ] **Step 2: Clear Busemann/centroid caches (force recomputation)**

```bash
rm -f /home/rohan/projects/qdrant/dev/testbench/data/wos/focal_direction.npy
rm -f /home/rohan/projects/qdrant/dev/testbench/data/wos/busemann_depths.npy
rm -f /home/rohan/projects/qdrant/dev/testbench/data/wos/area_centroids.npz
rm -f /home/rohan/projects/qdrant/dev/testbench/data/wos/domain_centroids.npz
```

- [ ] **Step 3: Run embedding pipeline**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 embed.py --qdrant-url http://localhost:6334 --skip-embed --cleanup-legacy
```

Expected: Loads cached PCA embeddings, projects to Poincare, computes Busemann depths, generates tier centroids, creates wos_cosine + wos_unified, deletes legacy collections.

- [ ] **Step 4: Run full benchmarks**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 benchmark.py --qdrant-url http://localhost:6334
```

Expected: Runs all 5 suites, prints comparison table and dual-space results.

- [ ] **Step 5: Capture results in analysis doc**

After benchmarks complete, record the output in `dev/testbench/results/analysis_round3.md`. Include:
- Hierarchy separation table (sep_ratio, band_overlap per tier)
- Retrieval quality comparison (cosine vs poincare vs dual-RRF)
- Cross-tier recall results
- Latency comparison
- Comparison with Round 2 results

---

### Task 8: Curvature Re-sweep

**Files:**
- Modify: `dev/testbench/embed.py` (add sweep function)
- Modify: `dev/testbench/benchmark.py` (add sweep runner)

- [ ] **Step 1: Add curvature sweep to `embed.py`**

Add after `cleanup_legacy_collections` in `embed.py`:

```python
def create_sweep_collection(
    client,
    qdrant_url: str,
    embeddings_128: np.ndarray,
    docs: list[dict],
    tiers: list[str],
    curvature: float,
) -> str:
    """Create a temporary unified collection at a specific curvature. Returns collection name."""
    c_str = str(curvature).replace(".", "")
    coll_name = f"wos_unified_c{c_str}"

    # Project
    strategy_fn = STRATEGIES["einstein_spread"]
    result = strategy_fn(embeddings_128, tiers, c=curvature)
    poincare_vectors = result["vectors"]

    # Busemann depths
    vectors_list = [poincare_vectors[i] for i in range(len(poincare_vectors))]
    focal = compute_focal_direction(vectors_list, c=curvature)
    depths = np.array([
        busemann_depth_single(poincare_vectors[i], focal, c=curvature)
        for i in range(len(poincare_vectors))
    ], dtype=np.float32)

    # Tier centroids
    area_entries, domain_entries = generate_tier_centroids(
        docs, embeddings_128, poincare_vectors, depths, focal, c=curvature,
    )

    # Create and upsert
    create_unified_collection(qdrant_url, coll_name, size=VECTOR_DIM, curvature=curvature)
    upsert_unified(
        client, coll_name, docs, embeddings_128, poincare_vectors, depths,
        area_entries, domain_entries,
    )

    return coll_name
```

- [ ] **Step 2: Add sweep mode to `embed.py` main()**

Add a new CLI argument in `main()` (after the `--cleanup-legacy` argument):

```python
    parser.add_argument(
        "--curvature-sweep",
        action="store_true",
        help="Run curvature sweep: create temporary unified collections at c=1,2,5,10",
    )
```

Add this block at the end of `main()`, before the final print:

```python
    # ------------------------------------------------------------------
    # Optional: Curvature re-sweep
    # ------------------------------------------------------------------
    if args.curvature_sweep:
        print(f"\n=== Curvature Re-sweep ===")
        sweep_curvatures = [1.0, 2.0, 5.0, 10.0]
        for c in sweep_curvatures:
            print(f"\n--- Curvature c={c} ---")
            # Clear centroid caches to force recomputation at new curvature
            for f in [AREA_CENTROIDS_FILE, DOMAIN_CENTROIDS_FILE,
                      FOCAL_DIRECTION_FILE, BUSEMANN_DEPTHS_FILE]:
                if f.exists():
                    f.unlink()
            create_sweep_collection(client, qdrant_url, embeddings_128, docs, tiers, c)
        print(f"\nSweep collections created. Run benchmark.py to compare.")
```

- [ ] **Step 3: Run the curvature sweep**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 embed.py --qdrant-url http://localhost:6334 --skip-embed --curvature-sweep
```

Expected: Creates `wos_unified_c10`, `wos_unified_c20`, `wos_unified_c50`, `wos_unified_c100`.

- [ ] **Step 4: Benchmark sweep collections**

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 benchmark.py --qdrant-url http://localhost:6334
```

Expected: Discovers and benchmarks all sweep collections alongside `wos_unified` and `wos_cosine`.

- [ ] **Step 5: Clean up sweep collections**

After recording results, delete temporary collections:

```bash
cd /home/rohan/projects/qdrant/dev/testbench
python3 -c "
import requests
for c in ['10', '20', '50', '100']:
    resp = requests.delete(f'http://localhost:6334/collections/wos_unified_c{c}')
    print(f'wos_unified_c{c}: {resp.status_code}')
"
```

- [ ] **Step 6: Record sweep results in analysis doc**

Append curvature comparison table to `dev/testbench/results/analysis_round3.md`.

---

### Task 9: Commit All Changes

- [ ] **Step 1: Review all changes**

```bash
cd /home/rohan/projects/qdrant
git status
git diff --stat
```

- [ ] **Step 2: Run Rust tests one final time**

```bash
cargo test --features hyperbolic -p segment -- --nocapture 2>&1 | tail -20
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add lib/segment/src/spaces/hyperbolic/poincare_math.rs
git add dev/testbench/fusion.py
git add dev/testbench/embed.py
git add dev/testbench/hyperbolic_math.py
git add dev/testbench/benchmark.py
git add dev/testbench/run.sh
git add docs/superpowers/specs/2026-04-09-hyperbolic-phase3-unified-collection-design.md
git add docs/superpowers/plans/2026-04-09-hyperbolic-phase3-unified-collection.md
git add dev/testbench/results/analysis_round3.md
git commit -m "feat(hyperbolic): unified collection with named vectors, Busemann depth, dual-space RRF

- Fix lorentz_to_poincare: add sqrt(c) divisor for configurable curvature
- Create wos_unified collection with cosine + poincare named vectors
- Compute Busemann depth client-side, store in payload
- Generate tier centroids via Einstein midpoint (area=story, domain=narrative)
- Add RRF fusion module for dual-space search
- Add cross-tier and dual-space benchmark suites
- Curvature re-sweep at c={1.0, 2.0, 5.0, 10.0}
- Clean up legacy per-strategy collections"
```
