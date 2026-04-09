# Hyperbolic Phase 3: Unified Collection, Rust Fix, Busemann Payload & Dual-Space Search

**Date:** 2026-04-09
**Branch:** `feat/hyperbolic-vector-support`
**Status:** Approved

---

## Overview

This design covers four interconnected changes to the hyperbolic vector support:

1. **Rust-side `lorentz_to_poincare` bug fix** — add configurable curvature and `sqrt(c)` divisor
2. **Busemann depth in payload** — client-side computation, stored as float payload at upsert
3. **Unified collection with named vectors** — single `wos_unified` collection with `cosine` + `poincare` named vectors and tier centroids
4. **Dual-space search with RRF fusion** — query both vector spaces, fuse results client-side
5. **Curvature re-sweep** — validate c=5.0 under the new unified topology

## Context

### Where We Left Off (2026-04-08)

Phase 2 testbench validated `einstein_spread` as the winning projection strategy:
- sep_ratio 20.4, beats cosine on all retrieval metrics
- Fixed Python-side `lorentz_to_poincare` bug (missing `sqrt(c)`)
- Rust-side has the same bug, unfixed

### Pythia Production Context

Pythia's narrative pipeline (Content > Stories > Narratives) currently uses 3 databases:
- MongoDB for source of truth
- Qdrant for semantic similarity (flat cosine, no hyperbolic)
- Neo4j for relationships and hierarchies

With native Poincare support in Qdrant, the hierarchy can move into Qdrant, leaving Neo4j for actor relationships only. The endgame is per-client unified collections where all tiers coexist in a single Poincare ball.

### Design Decisions Made

- **Approach C (Unified + Cosine Baseline):** Create `wos_unified` with named vectors, keep `wos_cosine` for A/B comparison
- **Client-side Busemann depth:** Python computes depth at upsert time, stores in payload. Keeps Qdrant changes minimal for future-proofing against upstream.
- **WOS tier mapping:** Domain=Narrative (10), Area=Story (404), Paper=Content (45,862)

---

## Section 1: Rust-Side Bug Fix

### The Bug

`lorentz_to_poincare` at `lib/segment/src/spaces/hyperbolic/poincare_math.rs:353` takes no curvature parameter and is missing `sqrt(c)` in the denominator.

The forward map `poincare_to_lorentz` scales spatial components by `2*sqrt(c)`:
```
x_i = 2*sqrt(c)*p_i / (1 - c||p||^2)
```

The inverse must undo this:
```
p_i = x_i / (sqrt(c) * (x_0 + 1))
```

But the current code does:
```
p_i = x_i / (x_0 + 1)
```

### The Fix

```rust
// Before (buggy):
pub fn lorentz_to_poincare(x: &[f32]) -> Vec<f32> {
    let x0 = x[0];
    let denom = x0 + 1.0;
    x[1..].iter().map(|&xi| xi / denom.max(EPS)).collect()
}

// After (fixed):
pub fn lorentz_to_poincare(x: &[f32], c: f32) -> Vec<f32> {
    let x0 = x[0];
    let sqrt_c = c.sqrt();
    let denom = sqrt_c * (x0 + 1.0);
    x[1..].iter().map(|&xi| xi / denom.max(EPS)).collect()
}
```

### Call Sites

| Location | Current Call | Fixed Call |
|----------|-------------|-----------|
| `poincare_math.rs` — `einstein_midpoint()` | `lorentz_to_poincare(&on_hyperboloid)` | `lorentz_to_poincare(&on_hyperboloid, c)` |
| `busemann.rs` — `compute_focal_direction()` | Uses `einstein_midpoint` (indirect) | Automatically fixed |
| `busemann.rs` — `busemann_depth()` | Calls `poincare_to_lorentz` only | No change needed |

### Testing

Add a round-trip test:
```rust
// For c != 1.0, verify: lorentz_to_poincare(poincare_to_lorentz(p, c), c) ~ p
```

Existing tests must continue to pass.

### Impact

- Does NOT require re-embedding (stored vectors are pre-projected in Python)
- Affects: TangentCache construction (Einstein midpoint), Busemann focal direction (query-time)
- After fix: rebuild Qdrant binary. New collections (wos_unified) build fresh HNSW indices at upsert time.

---

## Section 2: Unified Collection Architecture

### Collection: `wos_unified`

**Named vectors:**
- `cosine`: 128d, Cosine distance (flat semantic search)
- `poincare`: 128d, Poincare distance (hierarchical search)

**Payload schema:**
```json
{
    "tier": "content" | "story" | "narrative",
    "busemann_depth": 0.347,
    "domain": "CS",
    "area": "Machine Learning",
    "hierarchy_path": "CS/Machine Learning",
    "source_ids": [12, 45, 89]
}
```

**Payload indexes:**
- `tier` — keyword, enables filtered HNSW sub-graphs
- `busemann_depth` — float, enables range filtering

### Three Tiers of Points

| Tier | WOS Mapping | Count | Cosine Vector | Poincare Vector |
|------|-------------|-------|---------------|-----------------|
| narrative | Domain | 10 | L2-normalized mean of area cosine vectors | Einstein midpoint of area Poincare vectors |
| story | Area | 404 | L2-normalized mean of paper cosine vectors | Einstein midpoint of paper Poincare vectors |
| content | Paper | 45,862 | PCA-reduced 128d embedding | `einstein_spread` projected 128d |

### Point ID Scheme

- Content (papers): IDs `0` to `45,861`
- Stories (areas): IDs `100,000` to `100,403`
- Narratives (domains): IDs `200,000` to `200,009`

### Baseline Collection: `wos_cosine`

Kept from Round 2. Flat 128d cosine, papers only. Used for A/B comparison.

---

## Section 3: Embedding Pipeline

**File:** `dev/testbench/embed.py`

### Pipeline Steps

```
Step 1: Load & Embed (existing, cached)
  documents.jsonl -> pplx-embed-v1 -> embeddings_1024d.npy

Step 2: PCA Reduce (existing, cached)
  embeddings_1024d.npy -> PCA -> embeddings_128d.npy

Step 3: Project to Poincare (existing logic)
  embeddings_128d + tier labels -> einstein_spread(c=5.0) -> poincare_128d

Step 4: Compute Busemann Depths (NEW)
  poincare_128d -> compute_focal_direction() -> busemann_depth per point
  Save focal direction to data/wos/focal_direction.npy

Step 5: Generate Tier Centroids (NEW)
  For each area (404):
    cosine_centroid = L2-normalized mean of paper cosine vectors
    poincare_centroid = einstein_midpoint(paper poincare vectors, c=5.0)
  For each domain (10):
    cosine_centroid = L2-normalized mean of area cosine centroids
    poincare_centroid = einstein_midpoint(area poincare centroids, c=5.0)

Step 6: Upsert to wos_unified (NEW)
  All three tiers with named vectors + payload

Step 7: Upsert to wos_cosine (existing, baseline)
```

### Caching

- Steps 1-3: already cached as .npy files
- Step 4: cached as `data/wos/busemann_depths.npy`
- Step 5: cached as `data/wos/area_centroids.npz`, `data/wos/domain_centroids.npz`

### Subtlety: Cosine vs Poincare Centroids

- **Cosine centroids:** Simple L2-normalized mean (Einstein midpoint in flat space)
- **Poincare centroids:** Einstein midpoint in Poincare ball (proper hyperbolic centroid, naturally closer to origin)

---

## Section 4: Benchmark Suite

**File:** `dev/testbench/benchmark.py` (extended)
**New file:** `dev/testbench/fusion.py` (RRF implementation)

### Three Search Modes

1. **Cosine-only:** Query `wos_cosine` baseline (flat, papers only)
2. **Poincare-only:** Query `wos_unified` poincare named vector, filtered by tier
3. **Dual-space RRF:** Query both named vectors in `wos_unified`, fuse with RRF (k=60)

### RRF Fusion

```python
def rrf_fuse(results_a, results_b, k=60, limit=10):
    scores = {}
    for rank, point in enumerate(results_a):
        scores[point.id] = scores.get(point.id, 0) + 1.0 / (k + rank + 1)
    for rank, point in enumerate(results_b):
        scores[point.id] = scores.get(point.id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:limit]
```

Over-fetch top-20 from each space, fuse to top-10.

### Five Benchmark Suites

| Suite | Measures | Key Metrics | Query Set |
|-------|----------|-------------|-----------|
| 1. Hierarchy Separation | Tier band distinctness in unified collection | `sep_ratio`, `band_overlap`, tier classification accuracy | All points |
| 2. Retrieval Quality | Within-tier recall across 3 search modes | `area_recall@10`, `domain_recall@10`, `h_precision@10` | 500 random papers |
| 3. Cross-Tier Retrieval | Navigate between tiers | `parent_recall@5` (paper -> area/domain), `child_recall@10` (narrative -> stories) | 50 areas + 10 domains |
| 4. Dual-Space Comparison | Does RRF beat either space alone? | Side-by-side all 3 modes on same queries | 500 random papers |
| 5. Latency | Performance per search mode | `qps`, `p50`, `p95`, `p99` | 500 random papers |

### Output

Results written to `dev/testbench/results/analysis_round3.md` with comparison tables against Round 2.

---

## Section 5: Curvature Re-sweep

### Methodology

Sweep curvatures `c in {1.0, 2.0, 5.0, 10.0}` under the unified collection topology.

For each curvature:
1. Re-project paper vectors via `einstein_spread` at that curvature
2. Compute area/domain centroids via Einstein midpoint
3. Compute Busemann depths with new focal direction
4. Create temporary `wos_unified_c{X}` collection
5. Run Suites 1-4
6. Record metrics
7. Drop temporary collection

### Key Questions

| Metric | Question |
|--------|----------|
| `sep_ratio` | Does higher curvature pull narratives further toward origin? |
| `band_overlap` | Does curvature affect tier boundary clarity? |
| `area_recall@10` | Does curvature affect within-tier retrieval? |
| `parent_recall@5` | Does curvature affect cross-tier navigation? |
| `dual_rrf_vs_cosine` | At which curvature does dual-space give biggest lift? |

### Default

Keep c=5.0 as default for `wos_unified`. Update only if another curvature wins convincingly across metrics.

---

## Future-Proofing

### Minimal Qdrant Diff

- Rust changes: 2-line fix in `lorentz_to_poincare` + call site updates
- No upsert pipeline changes (Busemann depth is client-side payload)
- No new server-side APIs (dual-space search is client-side RRF)
- All hyperbolic code behind `#[cfg(feature = "hyperbolic")]`

### Upstream Merge Safety

The hyperbolic module lives in `lib/segment/src/spaces/hyperbolic/` — a directory that does not exist in upstream Qdrant. The only touchpoints with upstream are:
- `Distance` enum (one variant added, feature-gated)
- Feature flag forwarding in Cargo.toml files

### Path to Pythia Integration (Phase 4+)

This testbench validates the architecture for Pythia's future:
- Per-client unified collections (all tiers in one Poincare ball)
- Named vectors (cosine for semantic, poincare for hierarchical)
- Busemann depth payload for tier filtering
- Client-side RRF for dual-space search
- SONA/self-learning integration (Phase 5+, validated by unified geometry)

---

## File Changes Summary

| File | Change |
|------|--------|
| `lib/segment/src/spaces/hyperbolic/poincare_math.rs` | Fix `lorentz_to_poincare` signature + body, update `einstein_midpoint` call site, add round-trip test |
| `dev/testbench/embed.py` | Add Busemann depth computation, tier centroid generation, unified collection creation |
| `dev/testbench/benchmark.py` | Add Suites 3-4 (cross-tier, dual-space), extend Suites 1-2 for unified collection |
| `dev/testbench/fusion.py` | New file: RRF fusion implementation |
| `dev/testbench/results/analysis_round3.md` | New file: Round 3 results |
