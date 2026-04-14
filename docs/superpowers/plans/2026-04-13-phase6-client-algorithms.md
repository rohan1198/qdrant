# Phase 6: Client-Side Algorithm Suite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a client-side algorithm suite (alpha precomputation, tangent space pruning, Busemann-weighted fusion, horosphere scoring) with 6 new benchmark suites, plus one gRPC proto fix.

**Architecture:** Three composable layers (pre-upload, query-time, post-processing) implemented as Python modules in `dev/testbench/`, configured via a `PipelineConfig` dataclass. One small Rust/proto server-side change for gRPC curvature. All algorithms benchmarked against Phase 5 baselines on BGC and HWV datasets.

**Tech Stack:** Python 3.10+ (numpy, qdrant-client, scipy), Rust (proto change only), protobuf

**Spec:** `docs/superpowers/specs/2026-04-13-phase6-client-algorithms-design.md`

---

## File Map

### New Files

| File | Responsibility |
|---|---|
| `dev/testbench/pipeline_config.py` | PipelineConfig dataclass + preset configs |
| `dev/testbench/tangent.py` | Tangent space centroid computation + coordinate projection |
| `dev/testbench/comparator.py` | A/B comparison framework for benchmark suites |

### Modified Files

| File | Changes |
|---|---|
| `lib/api/src/grpc/proto/collections.proto` | Add `optional float curvature = 8` to VectorParams |
| `lib/collection/src/config.rs` | Wire proto curvature field to VectorDataConfig |
| `dev/testbench/hyperbolic_math.py` | Add `log_map_origin()`, `tangent_distance_sq()` |
| `dev/testbench/projection.py` | Add `auto_curvature()` |
| `dev/testbench/collection_builder.py` | Add tangent named vector, alpha/busemann payloads, busemann index |
| `dev/testbench/queries.py` | Add `alpha_pipeline_query()`, `tangent_pipeline_query()`, `combined_pipeline_query()` |
| `dev/testbench/fusion.py` | Enhance `busemann_weighted_fuse()`, add `horosphere_score()` |
| `dev/testbench/benchmark.py` | Add suites 12-17 to BenchmarkRunner |

---

## Task 1: PipelineConfig Dataclass

**Files:**
- Create: `dev/testbench/pipeline_config.py`

- [ ] **Step 1: Create pipeline_config.py**

```python
"""Per-collection pipeline configuration.

All tunable parameters live here. Values are manually set in Phase 6,
benchmarked on BGC/HWV. Designed to be replaced by SONA-learned values
per collection in future Pythia integration.
"""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the 3-layer hyperbolic search pipeline."""

    # Layer 1: Pre-upload
    curvature: float = 1.0
    projection_strategy: str = "einstein_spread"
    tangent_centroid: str = "origin"  # "origin" | "frechet" | "einstein"

    # Layer 2: Query-time
    query_pipeline: str = "alpha"  # "alpha" | "tangent" | "combined"
    prune_factor: int = 10
    stage_sizes: tuple = (200, 50)

    # Layer 3: Post-processing
    fusion_strategy: str = "busemann"  # "busemann" | "rrf" | "linear_alpha"
    fusion_alpha: float = 0.5
    depth_weight: float = 1.0
    horo_steepness: float = 1.0
    horo_intent: str = "auto"  # "ancestors" | "descendants" | "siblings" | "auto"
    depth_band_width: float | None = None


# Preset configs for benchmarking
PHASE5_BASELINE = PipelineConfig(
    curvature=5.0,
    query_pipeline="klein",
    fusion_strategy="rrf",
    stage_sizes=(200, 50),
)

ALPHA_PIPELINE = PipelineConfig(
    curvature=1.0,
    query_pipeline="alpha",
    fusion_strategy="busemann",
    stage_sizes=(200, 50),
)

TANGENT_PIPELINE = PipelineConfig(
    curvature=1.0,
    query_pipeline="tangent",
    tangent_centroid="origin",
    fusion_strategy="busemann",
    prune_factor=10,
)

COMBINED_PIPELINE = PipelineConfig(
    curvature=1.0,
    query_pipeline="combined",
    tangent_centroid="origin",
    fusion_strategy="busemann",
    prune_factor=10,
    stage_sizes=(100, 50),
)
```

- [ ] **Step 2: Verify import works**

Run: `cd dev/testbench && python -c "from pipeline_config import PipelineConfig, ALPHA_PIPELINE; print(ALPHA_PIPELINE)"`

Expected: Prints the dataclass repr without errors.

- [ ] **Step 3: Commit**

```bash
git add dev/testbench/pipeline_config.py
git commit -m "feat(testbench): add PipelineConfig dataclass for Phase 6"
```

---

## Task 2: gRPC Proto Curvature Field

**Files:**
- Modify: `lib/api/src/grpc/proto/collections.proto:20-38`
- Modify: `lib/collection/src/config.rs:610-625`

- [ ] **Step 1: Add curvature to VectorParams proto**

In `lib/api/src/grpc/proto/collections.proto`, add field 8 to VectorParams:

```proto
message VectorParams {
  uint64 size = 1;
  Distance distance = 2;
  optional HnswConfigDiff hnsw_config = 3;
  optional QuantizationConfig quantization_config = 4;
  optional bool on_disk = 5;
  optional Datatype datatype = 6;
  optional MultiVectorConfig multivector_config = 7;
  optional float curvature = 8;  // Poincare ball curvature, default 1.0
}
```

- [ ] **Step 2: Wire curvature in config.rs**

In `lib/collection/src/config.rs`, find the line (around 618):
```rust
#[cfg(feature = "hyperbolic")]
curvature: None,
```

Replace with:
```rust
#[cfg(feature = "hyperbolic")]
curvature: curvature.or(Some(segment::spaces::hyperbolic::poincare_math::DEFAULT_CURVATURE)),
```

Note: The exact variable name for the proto curvature field depends on the generated Rust code. Check the generated struct after `cargo build --features hyperbolic` and adjust accordingly. The proto field `optional float curvature = 8` will generate a field like `curvature: Option<f32>` on the generated `VectorParams` struct.

- [ ] **Step 3: Build and verify**

Run: `cargo build --features hyperbolic 2>&1 | tail -20`

Expected: Build succeeds with no errors related to curvature.

- [ ] **Step 4: Commit**

```bash
git add lib/api/src/grpc/proto/collections.proto lib/collection/src/config.rs
git commit -m "feat(grpc): add curvature field to VectorParams proto"
```

---

## Task 3: Hyperbolic Math Additions

**Files:**
- Modify: `dev/testbench/hyperbolic_math.py` (add after line 372)

- [ ] **Step 1: Add log_map_origin()**

Append to `dev/testbench/hyperbolic_math.py`:

```python
def log_map_origin(p: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Logarithmic map at the origin (fast path, no Mobius ops).

    Formula: log_0(p) = (2/sqrt(c)) * arctanh(sqrt(c) * ||p||) * p / ||p||

    ~3x cheaper than general log_map because the origin has conformal
    factor lambda=2 and Mobius addition simplifies to identity.
    """
    sc = np.sqrt(c)
    norm = np.linalg.norm(p)
    if norm < EPS:
        return np.zeros_like(p)
    sc_norm = min(sc * norm, 1.0 - EPS)  # clamp for arctanh domain
    coeff = (2.0 / sc) * np.arctanh(sc_norm) / norm
    return p * coeff


def log_map(base: np.ndarray, target: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Logarithmic map at arbitrary base point.

    Formula: log_p(y) = (2/(sqrt(c)*lambda_p)) * arctanh(sqrt(c)*||-p +_c y||)
                        * (-p +_c y) / ||-p +_c y||
    """
    sc = np.sqrt(c)

    # Mobius negation + addition: -p +_c y
    neg_p = -base
    # Mobius addition formula
    neg_p_sq = np.dot(neg_p, neg_p)
    y_sq = np.dot(target, target)
    neg_p_dot_y = np.dot(neg_p, target)

    num_left = (1.0 + 2.0 * c * neg_p_dot_y + c * y_sq)
    num_right = (1.0 - c * neg_p_sq)
    denom = 1.0 + 2.0 * c * neg_p_dot_y + c**2 * neg_p_sq * y_sq
    denom = max(denom, EPS)

    add = (num_left * neg_p + num_right * target) / denom

    add_norm = np.linalg.norm(add)
    if add_norm < EPS:
        return np.zeros_like(base)

    # Conformal factor at base
    base_sq = np.dot(base, base)
    lambda_p = 2.0 / max(1.0 - c * base_sq, EPS)

    sc_add_norm = min(sc * add_norm, 1.0 - EPS)
    coeff = (2.0 / (sc * lambda_p)) * np.arctanh(sc_add_norm) / add_norm

    return add * coeff


def frechet_mean(points: list[np.ndarray], c: float = 1.0,
                 max_iter: int = 100, lr: float = 0.1, tol: float = 1e-6) -> np.ndarray:
    """Iterative Frechet mean on the Poincare ball via Riemannian gradient descent."""
    dim = len(points[0])
    n = len(points)

    # Initialize at projected Euclidean centroid
    mean = np.mean(points, axis=0)
    mean = project_to_ball(mean, c)

    for _ in range(max_iter):
        grad = np.zeros(dim)
        for p in points:
            lm = log_map(mean, p, c)
            grad += lm
        grad /= n

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        step = grad * lr
        # exp_map at mean: mean +_c tanh(sqrt(c)*lambda_mean*||step||/2) * step/(sqrt(c)*||step||)
        mean = _exp_map_at(mean, step, c)
        mean = project_to_ball(mean, c)

    return mean


def _exp_map_at(base: np.ndarray, v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Exponential map at arbitrary base point (tangent vector v -> manifold)."""
    sc = np.sqrt(c)
    v_norm = np.linalg.norm(v)
    if v_norm < EPS:
        return base.copy()

    base_sq = np.dot(base, base)
    lambda_p = 2.0 / max(1.0 - c * base_sq, EPS)

    tanh_arg = sc * lambda_p * v_norm / 2.0
    direction = v / v_norm
    second = np.tanh(tanh_arg) * direction / sc

    # Mobius addition: base +_c second
    b_sq = np.dot(base, base)
    s_sq = np.dot(second, second)
    b_dot_s = np.dot(base, second)

    num_left = (1.0 + 2.0 * c * b_dot_s + c * s_sq)
    num_right = (1.0 - c * b_sq)
    denom = 1.0 + 2.0 * c * b_dot_s + c**2 * b_sq * s_sq
    denom = max(denom, EPS)

    return (num_left * base + num_right * second) / denom


def tangent_distance_sq(u: np.ndarray, v: np.ndarray) -> float:
    """Euclidean L2 squared distance between tangent space vectors."""
    diff = u - v
    return float(np.dot(diff, diff))
```

- [ ] **Step 2: Verify functions work**

Run: `cd dev/testbench && python -c "
import numpy as np
from hyperbolic_math import log_map_origin, log_map, frechet_mean, tangent_distance_sq, project_to_ball

# log_map_origin roundtrip: should recover direction
p = project_to_ball(np.array([0.3, 0.4, 0.2]), 1.0)
t = log_map_origin(p, 1.0)
print(f'log_map_origin: norm={np.linalg.norm(t):.4f} (should be > 0)')

# log_map at origin should match log_map_origin
t2 = log_map(np.zeros(3), p, 1.0)
print(f'log_map vs log_map_origin diff: {np.linalg.norm(t - t2):.8f} (should be ~0)')

# frechet_mean of single point = that point
pts = [np.array([0.1, 0.2, 0.1])]
m = frechet_mean(pts, 1.0)
print(f'frechet_mean single diff: {np.linalg.norm(m - pts[0]):.8f} (should be ~0)')

# tangent_distance_sq
print(f'tangent_dist_sq: {tangent_distance_sq(np.array([1.0, 0.0]), np.array([0.0, 1.0])):.1f} (should be 2.0)')
"`

Expected: All assertions printed, no errors.

- [ ] **Step 3: Commit**

```bash
git add dev/testbench/hyperbolic_math.py
git commit -m "feat(testbench): add log_map_origin, log_map, frechet_mean, tangent_distance_sq"
```

---

## Task 4: Tangent Space Module

**Files:**
- Create: `dev/testbench/tangent.py`

- [ ] **Step 1: Create tangent.py**

```python
"""Tangent space computation for approximate hyperbolic nearest-neighbor search.

Three centroid strategies:
  - origin: zero vector, cheapest (log_map_origin, no Mobius ops)
  - frechet: iterative Frechet mean, most accurate, expensive
  - einstein: closed-form Einstein midpoint, nearly as accurate as frechet

Tangent coordinates are uploaded as a "tangent" named vector with Euclidean
distance. Qdrant's native HNSW indexes them — no fork changes needed.
"""

import numpy as np
from hyperbolic_math import (
    log_map_origin,
    log_map,
    frechet_mean,
    einstein_midpoint,
    project_to_ball,
)


def compute_centroid(
    points: list[np.ndarray],
    strategy: str,
    curvature: float = 1.0,
) -> np.ndarray:
    """Compute the tangent space centroid using the given strategy.

    Args:
        points: list of Poincare ball vectors
        strategy: "origin" | "frechet" | "einstein"
        curvature: Poincare ball curvature

    Returns:
        centroid vector (same dimensionality as input points)
    """
    if strategy == "origin":
        return np.zeros(len(points[0]))
    elif strategy == "frechet":
        return frechet_mean(points, curvature)
    elif strategy == "einstein":
        return einstein_midpoint(points, curvature)
    else:
        raise ValueError(f"Unknown centroid strategy: {strategy}")


def compute_tangent_coords(
    points: list[np.ndarray],
    centroid: np.ndarray,
    curvature: float = 1.0,
) -> np.ndarray:
    """Project all points to tangent space at the given centroid.

    If centroid is the origin (all zeros), uses the fast log_map_origin path.

    Returns:
        np.ndarray of shape (len(points), dim) — tangent coordinates
    """
    is_origin = np.linalg.norm(centroid) < 1e-9

    if is_origin:
        coords = [log_map_origin(p, curvature) for p in points]
    else:
        coords = [log_map(centroid, p, curvature) for p in points]

    return np.array(coords, dtype=np.float32)


def tangent_query(
    query_poincare: np.ndarray,
    centroid: np.ndarray,
    curvature: float = 1.0,
) -> np.ndarray:
    """Project a single query point to tangent space.

    Returns:
        tangent space vector (same dimensionality as query)
    """
    is_origin = np.linalg.norm(centroid) < 1e-9

    if is_origin:
        return log_map_origin(query_poincare, curvature).astype(np.float32)
    else:
        return log_map(centroid, query_poincare, curvature).astype(np.float32)
```

- [ ] **Step 2: Verify import and basic usage**

Run: `cd dev/testbench && python -c "
import numpy as np
from tangent import compute_centroid, compute_tangent_coords, tangent_query
from hyperbolic_math import project_to_ball

pts = [project_to_ball(np.random.randn(8) * 0.3, 1.0) for _ in range(20)]

for strategy in ['origin', 'frechet', 'einstein']:
    c = compute_centroid(pts, strategy, 1.0)
    coords = compute_tangent_coords(pts, c, 1.0)
    q = tangent_query(pts[0], c, 1.0)
    print(f'{strategy}: centroid_norm={np.linalg.norm(c):.4f}, coords_shape={coords.shape}, query_shape={q.shape}')
"`

Expected: Three lines printed, coords_shape=(20, 8) for each, no errors.

- [ ] **Step 3: Commit**

```bash
git add dev/testbench/tangent.py
git commit -m "feat(testbench): add tangent space module with 3 centroid strategies"
```

---

## Task 5: Auto-Curvature in Projection

**Files:**
- Modify: `dev/testbench/projection.py` (append after line 240)

- [ ] **Step 1: Add auto_curvature function**

Append to `dev/testbench/projection.py`:

```python


def auto_curvature(delta: float) -> float:
    """Select Poincare ball curvature from Gromov delta.

    Thresholds are initial estimates — Suite 15 (Curvature Sweep) will
    empirically determine optimal breakpoints on BGC and HWV.

    Lower delta = more tree-like = higher curvature (tighter ball).
    """
    if delta < 0.10:
        return 2.0
    if delta < 0.20:
        return 1.0
    if delta < 0.35:
        return 0.5
    return 0.25
```

Also add to the `STRATEGIES` dict at line 235 — no change needed there, `auto_curvature` is a standalone utility, not a projection strategy.

- [ ] **Step 2: Verify**

Run: `cd dev/testbench && python -c "
from projection import auto_curvature
print(auto_curvature(0.05))   # 2.0
print(auto_curvature(0.057))  # 2.0 (BGC delta)
print(auto_curvature(0.15))   # 1.0
print(auto_curvature(0.30))   # 0.5
print(auto_curvature(0.50))   # 0.25
"`

Expected: `2.0`, `2.0`, `1.0`, `0.5`, `0.25`

- [ ] **Step 3: Commit**

```bash
git add dev/testbench/projection.py
git commit -m "feat(testbench): add auto_curvature() for Gromov delta-guided curvature selection"
```

---

## Task 6: Collection Builder — Tangent Vector + Payload Fields

**Files:**
- Modify: `dev/testbench/collection_builder.py`

- [ ] **Step 1: Update create_unified_collection() to support tangent vector**

Find the `create_unified_collection` function (line 142). Add a `tangent_dim` parameter and include the tangent vector config in the REST body. Replace the function signature and body:

Change the function signature from:
```python
def create_unified_collection(qdrant_url: str, dataset: str, dense_dim: int = 1024, poincare_dim: int = 128, curvature: float = 5.0) -> str:
```
to:
```python
def create_unified_collection(qdrant_url: str, dataset: str, dense_dim: int = 1024, poincare_dim: int = 128, curvature: float = 5.0, tangent_dim: int | None = None) -> str:
```

In the vectors_config dict construction inside the function, after the `"poincare"` entry, add:

```python
if tangent_dim is not None:
    vectors_config["tangent"] = {
        "size": tangent_dim,
        "distance": "Euclid",
    }
```

- [ ] **Step 2: Update _create_payload_indices() to index busemann_depth**

In `_create_payload_indices` (line 68), add a busemann_depth index request after the existing indices:

```python
# Busemann depth range index for server-side depth-band filtering
requests.put(
    f"{qdrant_url}/collections/{name}/index",
    json={"field_name": "busemann_depth", "field_schema": "float"},
    timeout=30,
)
```

- [ ] **Step 3: Update build_points_from_data() to accept tangent vectors**

Change the `build_points_from_data` function signature (line 377) from:
```python
def build_points_from_data(documents: list[dict], cosine_vectors: np.ndarray, poincare_vectors: np.ndarray | None, busemann_depths: np.ndarray | None, alpha_values: np.ndarray | None, tier_map: dict[int, str] | None = None, synthetic_timestamps: list[str] | None = None) -> list[dict]:
```
to:
```python
def build_points_from_data(documents: list[dict], cosine_vectors: np.ndarray, poincare_vectors: np.ndarray | None, busemann_depths: np.ndarray | None, alpha_values: np.ndarray | None, tangent_vectors: np.ndarray | None = None, tier_map: dict[int, str] | None = None, synthetic_timestamps: list[str] | None = None) -> list[dict]:
```

In the point construction loop inside `build_points_from_data`, where named vectors are assembled, add the tangent vector:

```python
if tangent_vectors is not None:
    vectors["tangent"] = tangent_vectors[i].tolist()
```

- [ ] **Step 4: Verify the builder can be imported without errors**

Run: `cd dev/testbench && python -c "from collection_builder import create_unified_collection, build_points_from_data; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add dev/testbench/collection_builder.py
git commit -m "feat(testbench): add tangent vector + busemann_depth index to collection builder"
```

---

## Task 7: Query Functions — Alpha, Tangent, Combined Pipelines

**Files:**
- Modify: `dev/testbench/queries.py` (append after line 518)

- [ ] **Step 1: Add alpha_pipeline_query()**

Append to `dev/testbench/queries.py`, first adding the new imports at the top of the file:

Add to imports (after existing imports at line 5):
```python
import numpy as np
from hyperbolic_math import alpha_precompute, fused_norms, poincare_distance_with_alpha
from tangent import tangent_query
```

Then append:

```python
def alpha_pipeline_query(
    client,
    collection,
    query_poincare,
    curvature=1.0,
    prefetch_limit=200,
    alpha_top=50,
    final_top=10,
    using_prefetch="dense",
    ef=128,
):
    """3-stage alpha pipeline: cosine prefetch -> alpha re-rank -> exact Poincare.

    Replaces Klein pipeline. Alpha proxy gives exact ordering (monotonic with
    Poincare distance), no model conversion needed.
    """
    t_start = time.time()

    # Stage 1: Cosine prefetch (server-side)
    t1 = time.time()
    prefetch_results = client.query_points(
        collection_name=collection,
        query=query_poincare.tolist(),
        using=using_prefetch,
        with_payload=True,
        with_vectors=["poincare"],
        limit=prefetch_limit,
        search_params=SearchParams(ef=ef),
    ).points
    t1_end = time.time()

    if not prefetch_results:
        return {
            "results": [], "distances": [],
            "prefetch_count": 0, "alpha_survivors": 0, "final_count": 0,
            "stage1_latency_ms": 0, "stage2_latency_ms": 0,
            "stage3_latency_ms": 0, "total_latency_ms": 0,
        }

    # Stage 2: Alpha re-rank (client-side, no acosh)
    t2 = time.time()
    query_alpha = alpha_precompute(query_poincare, curvature)

    scored = []
    for pt in prefetch_results:
        cand_vec = np.array(pt.vector["poincare"])
        cand_alpha = pt.payload.get("alpha")
        if cand_alpha is None:
            cand_alpha = alpha_precompute(cand_vec, curvature)
        diff = query_poincare - cand_vec
        diff_sq = float(np.dot(diff, diff))
        proxy = diff_sq * query_alpha * cand_alpha
        scored.append((pt, proxy))

    scored.sort(key=lambda x: x[1])
    survivors = scored[:alpha_top]
    t2_end = time.time()

    # Stage 3: Exact Poincare (client-side, reuse proxy)
    t3 = time.time()
    exact = []
    for pt, proxy in survivors:
        sqrt_c = np.sqrt(curvature)
        arg = 1.0 + 2.0 * curvature * proxy
        dist = float(np.arccosh(max(arg, 1.0))) / sqrt_c
        exact.append((pt, dist))

    exact.sort(key=lambda x: x[1])
    final = exact[:final_top]
    t3_end = time.time()

    return {
        "results": [pt for pt, _ in final],
        "distances": [d for _, d in final],
        "prefetch_count": len(prefetch_results),
        "alpha_survivors": len(survivors),
        "final_count": len(final),
        "stage1_latency_ms": (t1_end - t1) * 1000,
        "stage2_latency_ms": (t2_end - t2) * 1000,
        "stage3_latency_ms": (t3_end - t3) * 1000,
        "total_latency_ms": (t3_end - t_start) * 1000,
    }
```

- [ ] **Step 2: Add tangent_pipeline_query()**

Append to `dev/testbench/queries.py`:

```python
def tangent_pipeline_query(
    client,
    collection,
    query_poincare,
    centroid,
    curvature=1.0,
    prune_factor=10,
    final_top=10,
    ef=128,
):
    """2-stage tangent pipeline: tangent HNSW (server) -> exact Poincare (client).

    Uses Qdrant's native Euclidean HNSW on the "tangent" named vector for
    approximate hyperbolic nearest-neighbor search.
    """
    t_start = time.time()

    # Pre-query: project query to tangent space
    q_tangent = tangent_query(query_poincare, centroid, curvature)

    # Stage 1: Tangent HNSW search (server-side Euclidean)
    t1 = time.time()
    tangent_limit = final_top * prune_factor
    tangent_results = client.query_points(
        collection_name=collection,
        query=q_tangent.tolist(),
        using="tangent",
        with_payload=True,
        with_vectors=["poincare"],
        limit=tangent_limit,
        search_params=SearchParams(ef=ef),
    ).points
    t1_end = time.time()

    if not tangent_results:
        return {
            "results": [], "distances": [],
            "tangent_candidates": 0, "final_count": 0,
            "stage1_latency_ms": 0, "stage2_latency_ms": 0,
            "total_latency_ms": 0,
        }

    # Stage 2: Exact Poincare re-rank (client-side)
    t2 = time.time()
    exact = []
    for pt in tangent_results:
        cand_vec = np.array(pt.vector["poincare"])
        diff_sq, u_sq, v_sq = fused_norms(query_poincare, cand_vec)
        denom_u = max(1.0 - curvature * u_sq, 1e-7)
        denom_v = max(1.0 - curvature * v_sq, 1e-7)
        arg = 1.0 + 2.0 * curvature * diff_sq / (denom_u * denom_v)
        dist = float(np.arccosh(max(arg, 1.0))) / np.sqrt(curvature)
        exact.append((pt, dist))

    exact.sort(key=lambda x: x[1])
    final = exact[:final_top]
    t2_end = time.time()

    return {
        "results": [pt for pt, _ in final],
        "distances": [d for _, d in final],
        "tangent_candidates": len(tangent_results),
        "final_count": len(final),
        "stage1_latency_ms": (t1_end - t1) * 1000,
        "stage2_latency_ms": (t2_end - t2) * 1000,
        "total_latency_ms": (t2_end - t_start) * 1000,
    }
```

- [ ] **Step 3: Add combined_pipeline_query()**

Append to `dev/testbench/queries.py`:

```python
def combined_pipeline_query(
    client,
    collection,
    query_poincare,
    centroid,
    curvature=1.0,
    cosine_limit=100,
    tangent_limit=100,
    alpha_top=50,
    final_top=10,
    ef=128,
):
    """Combined pipeline: cosine + tangent prefetch -> alpha re-rank -> exact Poincare.

    Merges two independent retrieval signals (semantic similarity from cosine,
    hierarchical proximity from tangent) before alpha re-ranking.
    """
    t_start = time.time()

    # Stage 1a: Cosine prefetch (server)
    cosine_results = client.query_points(
        collection_name=collection,
        query=query_poincare.tolist(),
        using="dense",
        with_payload=True,
        with_vectors=["poincare"],
        limit=cosine_limit,
        search_params=SearchParams(ef=ef),
    ).points

    # Stage 1b: Tangent prefetch (server)
    q_tangent = tangent_query(query_poincare, centroid, curvature)
    tangent_results = client.query_points(
        collection_name=collection,
        query=q_tangent.tolist(),
        using="tangent",
        with_payload=True,
        with_vectors=["poincare"],
        limit=tangent_limit,
        search_params=SearchParams(ef=ef),
    ).points
    t1_end = time.time()

    # Merge + deduplicate by point ID
    seen_ids = set()
    merged = []
    for pt in cosine_results + tangent_results:
        if pt.id not in seen_ids:
            seen_ids.add(pt.id)
            merged.append(pt)

    # Stage 2: Alpha re-rank (client, no acosh)
    t2 = time.time()
    query_alpha = alpha_precompute(query_poincare, curvature)
    scored = []
    for pt in merged:
        cand_vec = np.array(pt.vector["poincare"])
        cand_alpha = pt.payload.get("alpha")
        if cand_alpha is None:
            cand_alpha = alpha_precompute(cand_vec, curvature)
        diff = query_poincare - cand_vec
        diff_sq = float(np.dot(diff, diff))
        proxy = diff_sq * query_alpha * cand_alpha
        scored.append((pt, proxy))

    scored.sort(key=lambda x: x[1])
    survivors = scored[:alpha_top]
    t2_end = time.time()

    # Stage 3: Exact Poincare
    t3 = time.time()
    exact = []
    for pt, proxy in survivors:
        sqrt_c = np.sqrt(curvature)
        arg = 1.0 + 2.0 * curvature * proxy
        dist = float(np.arccosh(max(arg, 1.0))) / sqrt_c
        exact.append((pt, dist))

    exact.sort(key=lambda x: x[1])
    final = exact[:final_top]
    t3_end = time.time()

    return {
        "results": [pt for pt, _ in final],
        "distances": [d for _, d in final],
        "cosine_count": len(cosine_results),
        "tangent_count": len(tangent_results),
        "merged_count": len(merged),
        "alpha_survivors": len(survivors),
        "final_count": len(final),
        "stage1_latency_ms": (t1_end - t_start) * 1000,
        "stage2_latency_ms": (t2_end - t2) * 1000,
        "stage3_latency_ms": (t3_end - t3) * 1000,
        "total_latency_ms": (t3_end - t_start) * 1000,
    }
```

- [ ] **Step 4: Verify imports**

Run: `cd dev/testbench && python -c "from queries import alpha_pipeline_query, tangent_pipeline_query, combined_pipeline_query; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add dev/testbench/queries.py
git commit -m "feat(testbench): add alpha, tangent, and combined pipeline queries"
```

---

## Task 8: Enhanced Fusion + Horosphere Scoring

**Files:**
- Modify: `dev/testbench/fusion.py` (append after line 167)

- [ ] **Step 1: Add import and enhance busemann_weighted_fuse**

Add at the top of `dev/testbench/fusion.py`:

```python
import math
```

Then append to end of file:

```python
def busemann_depth_proximity_fuse(
    results_a: list,
    results_b: list,
    query_depth: float,
    id_to_depth: dict[int, float],
    alpha: float = 0.5,
    depth_weight: float = 1.0,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Busemann fusion with depth proximity weighting.

    Enhanced version: results at the same hierarchy depth as the query are
    weighted higher. Uses exponential decay based on Busemann depth distance.

    Args:
        results_a: first result set (e.g., cosine)
        results_b: second result set (e.g., Poincare)
        query_depth: Busemann depth of the query point
        id_to_depth: mapping from point ID to Busemann depth
        alpha: blend weight (0=all_b, 1=all_a)
        depth_weight: steepness of depth proximity decay
        limit: max results to return
    """
    ids_a, scores_a = _extract_scores(results_a)
    ids_b, scores_b = _extract_scores(results_b)

    norm_a = _min_max_normalize(scores_a)
    norm_b = _min_max_normalize(scores_b)

    score_map_a = dict(zip(ids_a, norm_a))
    score_map_b = dict(zip(ids_b, norm_b))

    all_ids = set(ids_a) | set(ids_b)
    fused = []
    for pid in all_ids:
        sa = score_map_a.get(pid, 0.0)
        sb = score_map_b.get(pid, 0.0)
        cand_depth = id_to_depth.get(pid, query_depth)
        depth_proximity = math.exp(-abs(cand_depth - query_depth) * depth_weight)
        score = alpha * sa + (1.0 - alpha) * sb * depth_proximity
        fused.append((pid, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:limit]


def horosphere_score(
    candidates: list[tuple[int, float]],
    query_depth: float,
    id_to_depth: dict[int, float],
    intent: str = "auto",
    steepness: float = 1.0,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Re-weight candidates by horosphere (hierarchy depth) preference.

    Horospheres are level sets of the Busemann function — hyperbolic "floors"
    at a given depth.

    Args:
        candidates: list of (point_id, fused_score) tuples
        query_depth: Busemann depth of the query point
        id_to_depth: mapping from point ID to Busemann depth
        intent: "ancestors" (prefer shallower), "descendants" (prefer deeper),
                "siblings" (prefer same depth), "auto" (defaults to siblings)
        steepness: how aggressively to penalize wrong-depth results
        limit: max results to return
    """
    rescored = []
    for pid, score in candidates:
        cand_depth = id_to_depth.get(pid, query_depth)
        depth_delta = cand_depth - query_depth

        if intent == "ancestors":
            # Penalize candidates deeper than query
            horo_weight = math.exp(-max(depth_delta, 0.0) * steepness)
        elif intent == "descendants":
            # Penalize candidates shallower than query
            horo_weight = math.exp(min(depth_delta, 0.0) * steepness)
        elif intent in ("siblings", "auto"):
            # Penalize any depth difference
            horo_weight = math.exp(-abs(depth_delta) * steepness)
        else:
            horo_weight = 1.0

        rescored.append((pid, score * horo_weight))

    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:limit]
```

- [ ] **Step 2: Verify**

Run: `cd dev/testbench && python -c "
from fusion import busemann_depth_proximity_fuse, horosphere_score
# Smoke test with mock data
results_a = [type('P', (), {'id': i, 'score': 1.0 - i*0.1})() for i in range(5)]
results_b = [type('P', (), {'id': i, 'score': 0.5 + i*0.05})() for i in range(5)]
id_to_depth = {i: float(i) for i in range(5)}

fused = busemann_depth_proximity_fuse(results_a, results_b, 2.0, id_to_depth, alpha=0.5, depth_weight=1.0, limit=5)
print(f'Fused: {len(fused)} results, top={fused[0]}')

horo = horosphere_score(fused, 2.0, id_to_depth, intent='siblings', steepness=1.0, limit=3)
print(f'Horosphere: {len(horo)} results, top={horo[0]}')
"`

Expected: Prints fused and horosphere results without errors.

- [ ] **Step 3: Commit**

```bash
git add dev/testbench/fusion.py
git commit -m "feat(testbench): add busemann_depth_proximity_fuse and horosphere_score"
```

---

## Task 9: A/B Comparator Framework

**Files:**
- Create: `dev/testbench/comparator.py`

- [ ] **Step 1: Create comparator.py**

```python
"""A/B comparison framework for benchmarking pipeline configurations.

Runs the same queries through different PipelineConfigs and produces
side-by-side comparison tables.
"""

import time
import numpy as np
from scipy import stats
from pipeline_config import PipelineConfig


def compute_recall_at_k(retrieved_ids: list[int], ground_truth_ids: list[int], k: int = 10) -> float:
    """Compute recall@k: fraction of ground truth found in top-k retrieved."""
    retrieved_set = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids[:k])
    if not gt_set:
        return 0.0
    return len(retrieved_set & gt_set) / len(gt_set)


def compute_rank_correlation(ordering_a: list[int], ordering_b: list[int]) -> float:
    """Spearman rank correlation between two orderings of the same IDs.

    Returns correlation coefficient in [-1, 1]. Higher = more similar ordering.
    """
    common = set(ordering_a) & set(ordering_b)
    if len(common) < 3:
        return 0.0

    ranks_a = {pid: rank for rank, pid in enumerate(ordering_a) if pid in common}
    ranks_b = {pid: rank for rank, pid in enumerate(ordering_b) if pid in common}

    ids = sorted(common)
    ra = [ranks_a[pid] for pid in ids]
    rb = [ranks_b[pid] for pid in ids]

    corr, _ = stats.spearmanr(ra, rb)
    return float(corr) if not np.isnan(corr) else 0.0


def compare_report(results: dict[str, list[dict]], metric_keys: list[str] = None) -> str:
    """Generate a side-by-side comparison report.

    Args:
        results: {config_name: [per_query_metrics_dict, ...]}
        metric_keys: which metrics to include (default: all numeric keys)

    Returns:
        Formatted comparison table as string
    """
    if not results:
        return "No results to compare."

    config_names = list(results.keys())
    sample = results[config_names[0]][0]

    if metric_keys is None:
        metric_keys = [k for k, v in sample.items() if isinstance(v, (int, float))]

    lines = []
    header = f"{'Metric':<30}" + "".join(f"{name:<20}" for name in config_names)
    lines.append(header)
    lines.append("-" * len(header))

    for key in metric_keys:
        row = f"{key:<30}"
        values = []
        for name in config_names:
            vals = [q[key] for q in results[name] if key in q]
            if vals:
                avg = np.mean(vals)
                values.append(avg)
                row += f"{avg:<20.4f}"
            else:
                values.append(None)
                row += f"{'N/A':<20}"

        # Mark best value (highest for recall/correlation, lowest for latency)
        if values and all(v is not None for v in values):
            is_latency = "latency" in key.lower() or "time" in key.lower()
            best_idx = int(np.argmin(values)) if is_latency else int(np.argmax(values))
            row += f"  <- {config_names[best_idx]}"

        lines.append(row)

    return "\n".join(lines)
```

- [ ] **Step 2: Verify**

Run: `cd dev/testbench && python -c "
from comparator import compute_recall_at_k, compute_rank_correlation, compare_report

# recall@k
print(compute_recall_at_k([1,2,3,4,5], [1,3,5,7,9], k=5))  # 3/5 = 0.6

# rank correlation (identical ordering)
print(compute_rank_correlation([1,2,3,4,5], [1,2,3,4,5]))  # 1.0

# compare_report
results = {
    'alpha': [{'recall': 0.9, 'latency_ms': 12.0}, {'recall': 0.8, 'latency_ms': 15.0}],
    'klein': [{'recall': 0.7, 'latency_ms': 18.0}, {'recall': 0.75, 'latency_ms': 20.0}],
}
print(compare_report(results))
"`

Expected: 0.6, 1.0, then a formatted table.

- [ ] **Step 3: Commit**

```bash
git add dev/testbench/comparator.py
git commit -m "feat(testbench): add A/B comparator framework"
```

---

## Task 10: Benchmark Suites 12-13 (Alpha vs Klein, Tangent Centroid)

**Files:**
- Modify: `dev/testbench/benchmark.py`

- [ ] **Step 1: Add new imports to benchmark.py**

Add to the import block (after line 49):

```python
from scipy import stats as scipy_stats
from queries import (
    alpha_pipeline_query,
    tangent_pipeline_query,
    combined_pipeline_query,
    klein_prefilter_query,
)
from tangent import compute_centroid, compute_tangent_coords, tangent_query
from fusion import busemann_depth_proximity_fuse, horosphere_score
from comparator import compute_recall_at_k, compute_rank_correlation, compare_report
from pipeline_config import PipelineConfig, PHASE5_BASELINE, ALPHA_PIPELINE, TANGENT_PIPELINE
from projection import auto_curvature
```

- [ ] **Step 2: Add suite_alpha_vs_klein (Suite 12)**

Add to the `BenchmarkRunner` class (after the last suite method, around line 2070):

```python
    def suite_alpha_vs_klein(self, num_queries: int = 50) -> dict:
        """Suite 12: Alpha pipeline vs Klein pipeline."""
        print("\n=== Suite 12: Alpha vs Klein ===")
        points = self._load_unified_points("poincare")
        if not points:
            return {"error": "no points"}

        all_vecs = np.array([p["vector"] for p in points])
        all_ids = [p["id"] for p in points]

        # Brute-force ground truth
        rng = np.random.default_rng(42)
        query_indices = rng.choice(len(points), min(num_queries, len(points)), replace=False)

        alpha_recalls, klein_recalls = [], []
        alpha_latencies, klein_latencies = [], []
        rank_correlations = []

        for qi in tqdm(query_indices, desc="Alpha vs Klein"):
            query_vec = all_vecs[qi]

            # Ground truth: brute-force Poincare distance
            dists = []
            for i, v in enumerate(all_vecs):
                if i == qi:
                    continue
                diff = query_vec - v
                diff_sq = float(np.dot(diff, diff))
                u_sq = float(np.dot(query_vec, query_vec))
                v_sq = float(np.dot(v, v))
                denom = max((1 - self.curvature * u_sq) * (1 - self.curvature * v_sq), 1e-9)
                arg = 1.0 + 2.0 * self.curvature * diff_sq / denom
                d = float(np.arccosh(max(arg, 1.0))) / np.sqrt(self.curvature)
                dists.append((all_ids[i], d))
            dists.sort(key=lambda x: x[1])
            gt_ids = [pid for pid, _ in dists[:10]]

            # Alpha pipeline
            try:
                alpha_res = alpha_pipeline_query(
                    self.client, self.unified_name, query_vec,
                    curvature=self.curvature, prefetch_limit=200, alpha_top=50, final_top=10,
                )
                alpha_ids = [r.id for r in alpha_res["results"]]
                alpha_recalls.append(compute_recall_at_k(alpha_ids, gt_ids, 10))
                alpha_latencies.append(alpha_res["stage2_latency_ms"])

                # Rank correlation vs exact
                rank_correlations.append(compute_rank_correlation(alpha_ids, gt_ids))
            except Exception as e:
                print(f"  Alpha query failed: {e}")

            # Klein pipeline
            try:
                klein_res = klein_prefilter_query(
                    self.client, self.unified_name, query_vec,
                    curvature=self.curvature, prefetch_limit=200, klein_top=50, final_top=10,
                )
                klein_ids = [r.id for r in klein_res["results"]]
                klein_recalls.append(compute_recall_at_k(klein_ids, gt_ids, 10))
                klein_latencies.append(klein_res["stage2_latency_ms"])
            except Exception as e:
                print(f"  Klein query failed: {e}")

        result = {
            "suite": "alpha_vs_klein",
            "num_queries": len(query_indices),
            "alpha_recall_mean": float(np.mean(alpha_recalls)) if alpha_recalls else 0,
            "klein_recall_mean": float(np.mean(klein_recalls)) if klein_recalls else 0,
            "alpha_stage2_latency_ms": float(np.mean(alpha_latencies)) if alpha_latencies else 0,
            "klein_stage2_latency_ms": float(np.mean(klein_latencies)) if klein_latencies else 0,
            "rank_correlation_mean": float(np.mean(rank_correlations)) if rank_correlations else 0,
        }
        print(f"  Alpha recall: {result['alpha_recall_mean']:.3f}, Klein recall: {result['klein_recall_mean']:.3f}")
        print(f"  Alpha latency: {result['alpha_stage2_latency_ms']:.1f}ms, Klein latency: {result['klein_stage2_latency_ms']:.1f}ms")
        print(f"  Rank correlation: {result['rank_correlation_mean']:.3f}")
        return result
```

- [ ] **Step 3: Add suite_tangent_centroid (Suite 13)**

Add to the `BenchmarkRunner` class:

```python
    def suite_tangent_centroid(self, num_queries: int = 50) -> dict:
        """Suite 13: Tangent centroid comparison (origin vs frechet vs einstein)."""
        print("\n=== Suite 13: Tangent Centroid Comparison ===")
        points = self._load_unified_points("poincare")
        if not points:
            return {"error": "no points"}

        all_vecs = [np.array(p["vector"]) for p in points]
        all_ids = [p["id"] for p in points]

        rng = np.random.default_rng(42)
        query_indices = rng.choice(len(points), min(num_queries, len(points)), replace=False)

        # Brute-force ground truth (same as suite 12)
        def brute_force_gt(qi):
            query_vec = all_vecs[qi]
            dists = []
            for i, v in enumerate(all_vecs):
                if i == qi:
                    continue
                diff = query_vec - v
                diff_sq = float(np.dot(diff, diff))
                u_sq = float(np.dot(query_vec, query_vec))
                v_sq = float(np.dot(v, v))
                denom = max((1 - self.curvature * u_sq) * (1 - self.curvature * v_sq), 1e-9)
                arg = 1.0 + 2.0 * self.curvature * diff_sq / denom
                d = float(np.arccosh(max(arg, 1.0))) / np.sqrt(self.curvature)
                dists.append((all_ids[i], d))
            dists.sort(key=lambda x: x[1])
            return [pid for pid, _ in dists[:10]]

        results_by_strategy = {}
        for strategy in ["origin", "frechet", "einstein"]:
            print(f"  Testing centroid: {strategy}")
            t_cent = time.time()
            centroid = compute_centroid(all_vecs, strategy, self.curvature)
            centroid_time = time.time() - t_cent

            recalls = []
            correlations = []
            for qi in tqdm(query_indices, desc=f"  {strategy}"):
                gt_ids = brute_force_gt(qi)
                try:
                    res = tangent_pipeline_query(
                        self.client, self.unified_name, all_vecs[qi], centroid,
                        curvature=self.curvature, prune_factor=10, final_top=10,
                    )
                    ret_ids = [r.id for r in res["results"]]
                    recalls.append(compute_recall_at_k(ret_ids, gt_ids, 10))
                    correlations.append(compute_rank_correlation(ret_ids, gt_ids))
                except Exception as e:
                    print(f"    Query failed: {e}")

            results_by_strategy[strategy] = {
                "recall_mean": float(np.mean(recalls)) if recalls else 0,
                "rank_correlation": float(np.mean(correlations)) if correlations else 0,
                "centroid_time_s": centroid_time,
            }
            print(f"    Recall: {results_by_strategy[strategy]['recall_mean']:.3f}, "
                  f"Correlation: {results_by_strategy[strategy]['rank_correlation']:.3f}, "
                  f"Centroid time: {centroid_time:.2f}s")

        return {"suite": "tangent_centroid", "strategies": results_by_strategy}
```

- [ ] **Step 4: Register suites in run_all and --suites dispatch**

In the `run_all` method (line 1142), add after the existing suite calls:

```python
        results["alpha_vs_klein"] = self.suite_alpha_vs_klein()
        results["tangent_centroid"] = self.suite_tangent_centroid()
```

In the `main()` function, update the `--suites` choices list to include the new suite names: `"alpha_vs_klein"`, `"tangent_centroid"`.

- [ ] **Step 5: Commit**

```bash
git add dev/testbench/benchmark.py
git commit -m "feat(testbench): add suites 12-13 (alpha vs klein, tangent centroid)"
```

---

## Task 11: Benchmark Suites 14-15 (Prune Factor Sweep, Curvature Sweep)

**Files:**
- Modify: `dev/testbench/benchmark.py`

- [ ] **Step 1: Add suite_prune_factor_sweep (Suite 14)**

Add to the `BenchmarkRunner` class:

```python
    def suite_prune_factor_sweep(self, num_queries: int = 50) -> dict:
        """Suite 14: Tangent prune factor sweep (5, 10, 20)."""
        print("\n=== Suite 14: Prune Factor Sweep ===")
        points = self._load_unified_points("poincare")
        if not points:
            return {"error": "no points"}

        all_vecs = [np.array(p["vector"]) for p in points]
        all_ids = [p["id"] for p in points]

        # Use origin centroid (cheapest, likely winner from suite 13)
        centroid = np.zeros(len(all_vecs[0]))

        rng = np.random.default_rng(42)
        query_indices = rng.choice(len(points), min(num_queries, len(points)), replace=False)

        def brute_force_gt(qi):
            query_vec = all_vecs[qi]
            dists = []
            for i, v in enumerate(all_vecs):
                if i == qi:
                    continue
                diff = query_vec - v
                diff_sq = float(np.dot(diff, diff))
                u_sq = float(np.dot(query_vec, query_vec))
                v_sq = float(np.dot(v, v))
                denom = max((1 - self.curvature * u_sq) * (1 - self.curvature * v_sq), 1e-9)
                arg = 1.0 + 2.0 * self.curvature * diff_sq / denom
                d = float(np.arccosh(max(arg, 1.0))) / np.sqrt(self.curvature)
                dists.append((all_ids[i], d))
            dists.sort(key=lambda x: x[1])
            return [pid for pid, _ in dists[:10]]

        results_by_factor = {}
        for pf in [5, 10, 20]:
            recalls = []
            exact_count = pf * 10  # prune_factor * final_top
            for qi in tqdm(query_indices, desc=f"  pf={pf}"):
                gt_ids = brute_force_gt(qi)
                try:
                    res = tangent_pipeline_query(
                        self.client, self.unified_name, all_vecs[qi], centroid,
                        curvature=self.curvature, prune_factor=pf, final_top=10,
                    )
                    ret_ids = [r.id for r in res["results"]]
                    recalls.append(compute_recall_at_k(ret_ids, gt_ids, 10))
                except Exception as e:
                    print(f"    Query failed: {e}")

            results_by_factor[str(pf)] = {
                "recall_mean": float(np.mean(recalls)) if recalls else 0,
                "exact_dist_computations": exact_count,
            }
            print(f"  pf={pf}: recall={results_by_factor[str(pf)]['recall_mean']:.3f}, "
                  f"exact_comps={exact_count}")

        return {"suite": "prune_factor_sweep", "factors": results_by_factor}
```

- [ ] **Step 2: Add suite_curvature_sweep (Suite 15)**

Add to the `BenchmarkRunner` class:

```python
    def suite_curvature_sweep(self, num_queries: int = 50) -> dict:
        """Suite 15: Curvature sweep across c = 0.25, 0.5, 1.0, 2.0, 5.0."""
        print("\n=== Suite 15: Curvature Sweep ===")
        # This suite requires re-projecting data at different curvatures.
        # It operates on pre-loaded PCA vectors and measures drill-down recall.
        points = self._load_unified_points("poincare")
        if not points:
            return {"error": "no points"}

        all_vecs = [np.array(p["vector"]) for p in points]
        all_ids = [p["id"] for p in points]
        tiers = [p.get("tier", "unknown") for p in points]

        # Gromov delta (curvature-independent, measured on raw vectors)
        delta, rec = gromov_delta(all_vecs, num_samples=1000)
        print(f"  Gromov delta: {delta:.4f} (recommendation: {rec})")
        print(f"  auto_curvature suggests: c={auto_curvature(delta)}")

        results_by_c = {}
        for c in [0.25, 0.5, 1.0, 2.0, 5.0]:
            # Compute drill-down recall at this curvature via brute-force Poincare
            # Re-project is expensive, so we approximate: compute distances with different c
            # on the SAME vectors (curvature only affects the distance formula, not projection)
            rng = np.random.default_rng(42)
            query_indices = rng.choice(len(points), min(num_queries, len(points)), replace=False)

            # Filter narrative-tier queries for drill-down
            narrative_indices = [i for i in query_indices if tiers[i] in ("narratives", "root")]
            if not narrative_indices:
                narrative_indices = query_indices[:10]

            drill_recalls = []
            for qi in narrative_indices:
                query_vec = all_vecs[qi]
                # Find children in ground truth
                query_tier = tiers[qi]
                child_tiers = {"narratives": ["stories", "mid"], "root": ["mid", "stories"]}.get(query_tier, [])

                dists = []
                for i, v in enumerate(all_vecs):
                    if i == qi:
                        continue
                    diff = query_vec - v
                    diff_sq = float(np.dot(diff, diff))
                    u_sq = float(np.dot(query_vec, query_vec))
                    v_sq = float(np.dot(v, v))
                    denom = max((1 - c * u_sq) * (1 - c * v_sq), 1e-9)
                    arg = 1.0 + 2.0 * c * diff_sq / denom
                    d = float(np.arccosh(max(arg, 1.0))) / np.sqrt(c)
                    dists.append((i, d))
                dists.sort(key=lambda x: x[1])
                top10_ids = [idx for idx, _ in dists[:10]]
                child_count = sum(1 for idx in top10_ids if tiers[idx] in child_tiers)
                drill_recalls.append(child_count / max(len(child_tiers), 1))

            results_by_c[str(c)] = {
                "drill_down_recall": float(np.mean(drill_recalls)) if drill_recalls else 0,
                "num_queries": len(narrative_indices),
            }
            print(f"  c={c}: drill_recall={results_by_c[str(c)]['drill_down_recall']:.3f}")

        return {
            "suite": "curvature_sweep",
            "gromov_delta": delta,
            "auto_curvature": auto_curvature(delta),
            "curvatures": results_by_c,
        }
```

- [ ] **Step 3: Register suites in run_all and --suites dispatch**

Add to `run_all`:
```python
        results["prune_factor_sweep"] = self.suite_prune_factor_sweep()
        results["curvature_sweep"] = self.suite_curvature_sweep()
```

Add `"prune_factor_sweep"` and `"curvature_sweep"` to the `--suites` choices list.

- [ ] **Step 4: Commit**

```bash
git add dev/testbench/benchmark.py
git commit -m "feat(testbench): add suites 14-15 (prune factor sweep, curvature sweep)"
```

---

## Task 12: Benchmark Suites 16-17 (Fusion Comparison, End-to-End)

**Files:**
- Modify: `dev/testbench/benchmark.py`

- [ ] **Step 1: Add suite_fusion_comparison (Suite 16)**

Add to the `BenchmarkRunner` class:

```python
    def suite_fusion_comparison(self, num_queries: int = 50) -> dict:
        """Suite 16: Fusion strategy comparison (RRF vs linear alpha vs Busemann vs horosphere)."""
        print("\n=== Suite 16: Fusion Comparison ===")
        points = self._load_unified_points("poincare")
        if not points:
            return {"error": "no points"}

        all_vecs = [np.array(p["vector"]) for p in points]
        all_ids = [p["id"] for p in points]
        id_to_depth = {}
        for p in points:
            depth = p.get("busemann_depth", 0.0)
            id_to_depth[p["id"]] = depth

        rng = np.random.default_rng(42)
        query_indices = rng.choice(len(points), min(num_queries, len(points)), replace=False)

        strategy_results = {s: [] for s in ["rrf", "linear_alpha", "busemann", "horo_siblings", "horo_ancestors", "horo_descendants"]}

        for qi in tqdm(query_indices, desc="Fusion comparison"):
            query_vec = all_vecs[qi]
            query_depth = id_to_depth.get(all_ids[qi], 0.0)

            # Get cosine results and Poincare results from server
            try:
                cosine_res = self.client.query_points(
                    collection_name=self.unified_name,
                    query=query_vec.tolist(), using="dense",
                    with_payload=True, limit=50,
                ).points
                poincare_res = self.client.query_points(
                    collection_name=self.unified_name,
                    query=query_vec.tolist(), using="poincare",
                    with_payload=True, limit=50,
                ).points
            except Exception as e:
                print(f"  Query failed: {e}")
                continue

            if not cosine_res or not poincare_res:
                continue

            # Apply each fusion strategy
            rrf_result = rrf_fuse(cosine_res, poincare_res, limit=10)
            linear_result = linear_alpha_fuse(cosine_res, poincare_res, alpha=0.5, limit=10)
            busemann_result = busemann_depth_proximity_fuse(
                cosine_res, poincare_res, query_depth, id_to_depth, alpha=0.5, depth_weight=1.0, limit=10,
            )

            # Horosphere variants (applied on top of busemann fusion)
            horo_sib = horosphere_score(busemann_result, query_depth, id_to_depth, intent="siblings", limit=10)
            horo_anc = horosphere_score(busemann_result, query_depth, id_to_depth, intent="ancestors", limit=10)
            horo_desc = horosphere_score(busemann_result, query_depth, id_to_depth, intent="descendants", limit=10)

            strategy_results["rrf"].append({"ids": [pid for pid, _ in rrf_result]})
            strategy_results["linear_alpha"].append({"ids": [pid for pid, _ in linear_result]})
            strategy_results["busemann"].append({"ids": [pid for pid, _ in busemann_result]})
            strategy_results["horo_siblings"].append({"ids": [pid for pid, _ in horo_sib]})
            strategy_results["horo_ancestors"].append({"ids": [pid for pid, _ in horo_anc]})
            strategy_results["horo_descendants"].append({"ids": [pid for pid, _ in horo_desc]})

        return {"suite": "fusion_comparison", "strategies": {
            name: {"num_queries": len(res)} for name, res in strategy_results.items()
        }}
```

- [ ] **Step 2: Add suite_end_to_end (Suite 17)**

Add to the `BenchmarkRunner` class:

```python
    def suite_end_to_end(self, num_queries: int = 50) -> dict:
        """Suite 17: End-to-end pipeline comparison (Phase 6 best vs Phase 5 baseline)."""
        print("\n=== Suite 17: End-to-End Pipeline ===")
        points = self._load_unified_points("poincare")
        if not points:
            return {"error": "no points"}

        all_vecs = [np.array(p["vector"]) for p in points]
        all_ids = [p["id"] for p in points]

        rng = np.random.default_rng(42)
        query_indices = rng.choice(len(points), min(num_queries, len(points)), replace=False)

        # Brute-force ground truth
        def brute_force_gt(qi):
            query_vec = all_vecs[qi]
            dists = []
            for i, v in enumerate(all_vecs):
                if i == qi:
                    continue
                diff = query_vec - v
                diff_sq = float(np.dot(diff, diff))
                u_sq = float(np.dot(query_vec, query_vec))
                v_sq = float(np.dot(v, v))
                denom = max((1 - self.curvature * u_sq) * (1 - self.curvature * v_sq), 1e-9)
                arg = 1.0 + 2.0 * self.curvature * diff_sq / denom
                d = float(np.arccosh(max(arg, 1.0))) / np.sqrt(self.curvature)
                dists.append((all_ids[i], d))
            dists.sort(key=lambda x: x[1])
            return [pid for pid, _ in dists[:10]]

        baseline_recalls, baseline_latencies = [], []
        alpha_recalls, alpha_latencies = [], []

        for qi in tqdm(query_indices, desc="End-to-end"):
            gt_ids = brute_force_gt(qi)
            query_vec = all_vecs[qi]

            # Phase 5 baseline: Klein pipeline
            try:
                klein_res = klein_prefilter_query(
                    self.client, self.unified_name, query_vec,
                    curvature=self.curvature, prefetch_limit=200, klein_top=50, final_top=10,
                )
                klein_ids = [r.id for r in klein_res["results"]]
                baseline_recalls.append(compute_recall_at_k(klein_ids, gt_ids, 10))
                baseline_latencies.append(klein_res["total_latency_ms"])
            except Exception as e:
                print(f"  Klein failed: {e}")

            # Phase 6: Alpha pipeline
            try:
                alpha_res = alpha_pipeline_query(
                    self.client, self.unified_name, query_vec,
                    curvature=self.curvature, prefetch_limit=200, alpha_top=50, final_top=10,
                )
                alpha_ids = [r.id for r in alpha_res["results"]]
                alpha_recalls.append(compute_recall_at_k(alpha_ids, gt_ids, 10))
                alpha_latencies.append(alpha_res["total_latency_ms"])
            except Exception as e:
                print(f"  Alpha failed: {e}")

        result = {
            "suite": "end_to_end",
            "num_queries": len(query_indices),
            "baseline_klein_recall": float(np.mean(baseline_recalls)) if baseline_recalls else 0,
            "phase6_alpha_recall": float(np.mean(alpha_recalls)) if alpha_recalls else 0,
            "baseline_klein_latency_ms": float(np.mean(baseline_latencies)) if baseline_latencies else 0,
            "phase6_alpha_latency_ms": float(np.mean(alpha_latencies)) if alpha_latencies else 0,
        }
        print(f"  Baseline (Klein): recall={result['baseline_klein_recall']:.3f}, latency={result['baseline_klein_latency_ms']:.1f}ms")
        print(f"  Phase 6 (Alpha):  recall={result['phase6_alpha_recall']:.3f}, latency={result['phase6_alpha_latency_ms']:.1f}ms")
        return result
```

- [ ] **Step 3: Register suites and add linear_alpha_fuse import**

Add to imports at top:
```python
from fusion import linear_alpha_fuse
```

Add to `run_all`:
```python
        results["fusion_comparison"] = self.suite_fusion_comparison()
        results["end_to_end"] = self.suite_end_to_end()
```

Add `"fusion_comparison"` and `"end_to_end"` to the `--suites` choices list.

- [ ] **Step 4: Commit**

```bash
git add dev/testbench/benchmark.py
git commit -m "feat(testbench): add suites 16-17 (fusion comparison, end-to-end pipeline)"
```

---

## Task 13: Integration Verification

**Files:** None (verification only)

- [ ] **Step 1: Verify all imports resolve**

Run: `cd dev/testbench && python -c "
from pipeline_config import PipelineConfig, PHASE5_BASELINE, ALPHA_PIPELINE, TANGENT_PIPELINE, COMBINED_PIPELINE
from tangent import compute_centroid, compute_tangent_coords, tangent_query
from comparator import compute_recall_at_k, compute_rank_correlation, compare_report
from projection import auto_curvature
from queries import alpha_pipeline_query, tangent_pipeline_query, combined_pipeline_query
from fusion import busemann_depth_proximity_fuse, horosphere_score
from hyperbolic_math import log_map_origin, log_map, frechet_mean, tangent_distance_sq
print('All imports OK')
"`

Expected: `All imports OK`

- [ ] **Step 2: Verify benchmark suite list**

Run: `cd dev/testbench && python -c "
from benchmark import BenchmarkRunner
suites = [m for m in dir(BenchmarkRunner) if m.startswith('suite_')]
print(f'{len(suites)} suites:')
for s in sorted(suites):
    print(f'  {s}')
" 2>/dev/null`

Expected: 17 suite methods listed (11 existing + 6 new).

- [ ] **Step 3: Verify Rust builds with proto change**

Run: `cargo build --features hyperbolic 2>&1 | tail -5`

Expected: Build succeeds.

- [ ] **Step 4: Run existing tests to verify no regressions**

Run: `cargo test --features hyperbolic -p segment -- poincare 2>&1 | tail -10`

Expected: All existing Poincare tests pass.

- [ ] **Step 5: Final commit (all changes together)**

If any files were missed in earlier commits:
```bash
git status
git add -A dev/testbench/ lib/api/src/grpc/proto/collections.proto lib/collection/src/config.rs
git commit -m "feat(hyperbolic): Phase 6 — client-side algorithm suite + 6 benchmark suites"
```
