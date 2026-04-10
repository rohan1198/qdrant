# Phase 4: Unified Architecture Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix critical bugs in the Qdrant Poincaré implementation, overhaul the testbench with unified collection architecture and chatbot query benchmarks, and add HWV WikiVitals as a 6-level validation dataset.

**Architecture:** Qdrant fork provides a minimal `Distance::Poincare` primitive with configurable curvature (~550 lines). Everything application-specific (Busemann scoring, Einstein midpoint aggregation, chatbot queries, hierarchy encoding) lives in the Python testbench and will later carry to Pythia. Two datasets (BGC 4-level, HWV 6-level) validate the unified collection pattern against separate per-tier collections.

**Tech Stack:** Rust 1.94 (Qdrant fork), Python 3.10+ (testbench), Docker, qdrant-client, numpy, scikit-learn, sentence-transformers, tqdm

**Commit policy:** Do NOT commit incrementally. Make all changes, verify end-to-end, then commit everything at once.

---

## File Structure

### Qdrant Fork (Rust) — files to modify

| File | Action | Responsibility |
|------|--------|---------------|
| `lib/api/src/grpc/proto/collections.proto` | Modify | Add `Poincare = 5` to Distance enum |
| `lib/api/src/grpc/conversions.rs` | Modify | Add Poincare ↔ Distance::Poincare conversion |
| `lib/segment/src/types.rs` | Modify | Fix `preprocess_vector` to accept curvature |
| `lib/segment/src/index/hnsw_index/point_scorer.rs` | Modify | Wire curvature through FilteredScorer |
| `lib/segment/src/vector_storage/raw_scorer.rs` | Modify | Import adjustment (already has curvature scorer) |
| `lib/segment/src/spaces/hyperbolic/mod.rs` | Modify | Remove busemann/tangent re-exports |
| `lib/segment/src/spaces/hyperbolic/busemann.rs` | Delete | Moves to Python |
| `lib/segment/src/spaces/hyperbolic/tangent_cache.rs` | Delete | Deferred optimization |
| `lib/segment/src/spaces/hyperbolic/tangent_scorer.rs` | Delete | Deferred optimization |

### Testbench (Python) — files to create/modify

| File | Action | Responsibility |
|------|--------|---------------|
| `dev/testbench/hyperbolic_math.py` | Modify | Add Gromov delta, alpha precompute, fused norms |
| `dev/testbench/collections.py` | Create | Unified/separate/cosine collection creation + population |
| `dev/testbench/queries.py` | Create | 5 chatbot query implementations |
| `dev/testbench/benchmark.py` | Rewrite | 8-suite benchmark engine |
| `dev/testbench/embed.py` | Modify | HWV support, aggregate embeddings, Gromov analysis |
| `dev/testbench/run.sh` | Modify | Add hwv subcommands |
| `dev/datasets/hwv/prepare.py` | Create | HWV download + parse + JSONL output |

---

## Part 1: Qdrant Fork Fixes

### Task 1: Add Poincaré to gRPC proto and conversions

**Files:**
- Modify: `lib/api/src/grpc/proto/collections.proto:137-143`
- Modify: `lib/api/src/grpc/conversions.rs:2603-2628`

- [ ] **Step 1: Add Poincare to the proto Distance enum**

In `lib/api/src/grpc/proto/collections.proto`, add `Poincare = 5` to the Distance enum:

```protobuf
enum Distance {
  UnknownDistance = 0;
  Cosine = 1;
  Euclid = 2;
  Dot = 3;
  Manhattan = 4;
  Poincare = 5;
}
```

- [ ] **Step 2: Add Poincare to the TryFrom conversion (gRPC → segment)**

In `lib/api/src/grpc/conversions.rs`, find the `TryFrom<Distance> for segment::types::Distance` impl (~line 2603) and add the Poincare arm:

```rust
impl TryFrom<Distance> for segment::types::Distance {
    type Error = Status;

    fn try_from(value: Distance) -> Result<Self, Self::Error> {
        Ok(match value {
            Distance::UnknownDistance => {
                return Err(Status::invalid_argument(
                    "Malformed distance parameter: UnknownDistance",
                ));
            }
            Distance::Cosine => segment::types::Distance::Cosine,
            Distance::Euclid => segment::types::Distance::Euclid,
            Distance::Dot => segment::types::Distance::Dot,
            Distance::Manhattan => segment::types::Distance::Manhattan,
            #[cfg(feature = "hyperbolic")]
            Distance::Poincare => segment::types::Distance::Poincare,
            #[cfg(not(feature = "hyperbolic"))]
            Distance::Poincare => {
                return Err(Status::invalid_argument(
                    "Poincare distance requires the 'hyperbolic' feature flag",
                ));
            }
        })
    }
}
```

- [ ] **Step 3: Add Poincare to the reverse conversion (segment → gRPC)**

Search `conversions.rs` for the `From<segment::types::Distance> for Distance` impl (or similar — may be named `to_grpc_dist` or `impl From`). Add the Poincare arm:

```rust
#[cfg(feature = "hyperbolic")]
segment::types::Distance::Poincare => Distance::Poincare,
```

If no reverse conversion exists, verify by grepping for `Distance::Cosine =>` in conversions.rs to find all match blocks that dispatch on Distance. Add Poincare to each.

- [ ] **Step 4: Verify proto compilation**

```bash
cd /home/rohan/projects/qdrant && cargo check --features hyperbolic -p api 2>&1 | head -20
```

Expected: compiles without errors. If proto auto-generation is needed, the build script handles it.

---

### Task 2: Wire curvature through scorer chain

**Files:**
- Modify: `lib/segment/src/index/hnsw_index/point_scorer.rs:114-139`

This is the critical fix. `FilteredScorer::new` currently calls `new_raw_scorer()` which ignores curvature. We need it to call `new_raw_scorer_with_curvature()` when curvature is available.

- [ ] **Step 1: Add curvature parameter to FilteredScorer::new**

In `lib/segment/src/index/hnsw_index/point_scorer.rs`, modify `FilteredScorer::new` to accept curvature and use it:

```rust
impl<'a> FilteredScorer<'a> {
    pub fn new(
        query: QueryVector,
        vectors: &'a VectorStorageEnum,
        quantized_vectors: Option<&'a QuantizedVectors>,
        filter_context: Option<BoxCow<'a, dyn FilterContext + 'a>>,
        point_deleted: &'a BitSlice,
        hardware_counter: HardwareCounterCell,
        curvature: Option<f32>,
    ) -> OperationResult<Self> {
        let raw_scorer = match quantized_vectors {
            Some(quantized_vectors) => quantized_vectors.raw_scorer(query, hardware_counter)?,
            None => {
                #[cfg(feature = "hyperbolic")]
                if let Some(c) = curvature {
                    new_raw_scorer_with_curvature(query, vectors, hardware_counter, c)?
                } else {
                    new_raw_scorer(query, vectors, hardware_counter)?
                }
                #[cfg(not(feature = "hyperbolic"))]
                {
                    let _ = curvature;
                    new_raw_scorer(query, vectors, hardware_counter)?
                }
            }
        };
        Ok(FilteredScorer {
            raw_scorer,
            filters: ScorerFilters {
                filter_context,
                point_deleted,
                vec_deleted: vectors.deleted_vector_bitslice(),
            },
            scores_buffer: Vec::new(),
        })
    }
```

- [ ] **Step 2: Add import for new_raw_scorer_with_curvature**

At the top of `point_scorer.rs`, ensure the import exists:

```rust
#[cfg(feature = "hyperbolic")]
use crate::vector_storage::raw_scorer::new_raw_scorer_with_curvature;
```

- [ ] **Step 3: Update new_internal similarly**

In the same file, `new_internal` (~line 141) also calls `new_raw_scorer`. Apply the same pattern — add `curvature: Option<f32>` parameter and use `new_raw_scorer_with_curvature` when appropriate.

- [ ] **Step 4: Fix all callers of FilteredScorer::new**

Search for all call sites of `FilteredScorer::new(` across the codebase:

```bash
cd /home/rohan/projects/qdrant && grep -rn "FilteredScorer::new(" lib/ --include="*.rs" | grep -v "test" | head -20
```

For each caller:
- If the caller has access to `VectorDataConfig` (e.g., in HNSW index search), extract curvature with:
  ```rust
  #[cfg(feature = "hyperbolic")]
  let curvature = if vector_config.distance == Distance::Poincare {
      Some(vector_config.curvature())
  } else {
      None
  };
  #[cfg(not(feature = "hyperbolic"))]
  let curvature = None;
  ```
  Then pass `curvature` as the last argument.

- If the caller does NOT have access to VectorDataConfig, pass `None` (preserves existing behavior with DEFAULT_CURVATURE).

- [ ] **Step 5: Verify compilation**

```bash
cd /home/rohan/projects/qdrant && cargo check --features hyperbolic -p segment 2>&1 | head -30
```

Expected: compiles without errors. Fix any remaining callers that weren't updated.

---

### Task 3: Fix preprocess_vector to use curvature

**Files:**
- Modify: `lib/segment/src/types.rs:336-354`

- [ ] **Step 1: Add curvature-aware preprocess method**

The existing `preprocess_vector` is a method on the `Distance` enum and has no access to curvature. Rather than changing its signature (which would break many callers), add a separate method:

```rust
impl Distance {
    /// Preprocess vector with curvature for Poincaré distance.
    /// For non-Poincaré distances, this is identical to `preprocess_vector`.
    #[cfg(feature = "hyperbolic")]
    pub fn preprocess_vector_with_curvature<T: PrimitiveVectorElement>(
        &self,
        vector: DenseVector,
        curvature: f32,
    ) -> DenseVector
    where
        CosineMetric: Metric<T>,
        EuclidMetric: Metric<T>,
        DotProductMetric: Metric<T>,
        ManhattanMetric: Metric<T>,
    {
        match self {
            Distance::Poincare => {
                crate::spaces::hyperbolic::poincare_math::project_to_ball(vector, curvature)
            }
            other => other.preprocess_vector::<T>(vector),
        }
    }
}
```

- [ ] **Step 2: Update vector ingestion to use curvature-aware preprocessing**

Search for calls to `preprocess_vector` that handle Poincaré vectors:

```bash
cd /home/rohan/projects/qdrant && grep -rn "preprocess_vector\|preprocess_dense_vector" lib/segment/src/data_types/ --include="*.rs"
```

In `lib/segment/src/data_types/named_vectors.rs`, the `preprocess()` function calls `config.distance.preprocess_vector()`. Where `config: &VectorDataConfig` is available, update the Poincaré path to use `preprocess_vector_with_curvature(vector, config.curvature())`.

- [ ] **Step 3: Verify compilation**

```bash
cd /home/rohan/projects/qdrant && cargo check --features hyperbolic -p segment 2>&1 | head -20
```

---

### Task 4: Remove Busemann and tangent modules

**Files:**
- Delete: `lib/segment/src/spaces/hyperbolic/busemann.rs`
- Delete: `lib/segment/src/spaces/hyperbolic/tangent_cache.rs`
- Delete: `lib/segment/src/spaces/hyperbolic/tangent_scorer.rs`
- Modify: `lib/segment/src/spaces/hyperbolic/mod.rs`

- [ ] **Step 1: Delete the three module files**

```bash
rm lib/segment/src/spaces/hyperbolic/busemann.rs
rm lib/segment/src/spaces/hyperbolic/tangent_cache.rs
rm lib/segment/src/spaces/hyperbolic/tangent_scorer.rs
```

- [ ] **Step 2: Update mod.rs to remove deleted modules**

Replace the contents of `lib/segment/src/spaces/hyperbolic/mod.rs` with:

```rust
pub mod poincare_math;
pub mod poincare_metric;

// Re-export key types for convenience
pub use poincare_metric::PoincareMetric;
pub use poincare_metric::PoincareCurvatureQueryScorer;
pub use poincare_math::{
    poincare_distance, project_to_ball, exp_map_origin, log_map_origin,
    exp_map, log_map, mobius_add, frechet_mean, einstein_midpoint, conformal_factor,
    poincare_to_lorentz, lorentz_inner, lorentz_to_poincare, project_hyperboloid,
    DEFAULT_CURVATURE,
};
```

- [ ] **Step 3: Remove any imports of deleted modules**

```bash
cd /home/rohan/projects/qdrant && grep -rn "busemann\|tangent_cache\|tangent_scorer\|TangentCache\|prune_and_rescore\|DEFAULT_PRUNE_FACTOR\|busemann_score\|busemann_depth\|compute_focal_direction" lib/ --include="*.rs" | grep -v "target/"
```

Remove or comment out any imports or uses of the deleted types. These should only be in the `mod.rs` we just updated, but verify.

- [ ] **Step 4: Verify tests pass**

```bash
cd /home/rohan/projects/qdrant && cargo test --lib --features hyperbolic -p segment 2>&1 | tail -5
```

Expected: all tests pass. The test count may drop slightly (busemann.rs had 6 tests, tangent modules had tests too) but no failures.

- [ ] **Step 5: Verify full compilation**

```bash
cd /home/rohan/projects/qdrant && cargo check --features hyperbolic 2>&1 | tail -10
```

Expected: clean compilation across the full workspace.

- [ ] **Step 6: Rebuild Docker image**

Tell the user to run:

```bash
cd /home/rohan/projects/qdrant/dev/testbench && docker compose up -d --build
```

This rebuilds the Qdrant binary with `--features hyperbolic` and starts the container.

---

## Part 2: Testbench Math Extensions

### Task 5: Extend hyperbolic_math.py

**Files:**
- Modify: `dev/testbench/hyperbolic_math.py`

- [ ] **Step 1: Add Gromov delta-hyperbolicity analysis**

Append to `dev/testbench/hyperbolic_math.py`:

```python
def gromov_delta(vectors, num_samples=1000):
    """Compute Gromov delta-hyperbolicity via 4-point condition.

    Lower delta = more tree-like = better for hyperbolic geometry.
    Returns (delta, recommendation).

    Source: HyperspaceDB gromov.rs
    """
    n = len(vectors)
    if n < 4:
        return 0.0, "insufficient_data"

    max_delta = 0.0
    rng = np.random.default_rng(42)

    for _ in range(num_samples):
        idx = rng.choice(n, 4, replace=False)
        x, y, u, v = vectors[idx[0]], vectors[idx[1]], vectors[idx[2]], vectors[idx[3]]

        d_xy = np.linalg.norm(x - y)
        d_uv = np.linalg.norm(u - v)
        d_xu = np.linalg.norm(x - u)
        d_yv = np.linalg.norm(y - v)
        d_xv = np.linalg.norm(x - v)
        d_yu = np.linalg.norm(y - u)

        sums = sorted([d_xy + d_uv, d_xu + d_yv, d_xv + d_yu], reverse=True)
        delta = (sums[0] - sums[1]) / 2.0
        max_delta = max(max_delta, delta)

    if max_delta < 0.15:
        rec = "lorentz"
    elif max_delta < 0.30:
        rec = "poincare"
    elif max_delta < 0.50:
        rec = "cosine"
    else:
        rec = "l2"

    return max_delta, rec
```

- [ ] **Step 2: Add alpha precomputation**

```python
def alpha_precompute(vector, c=1.0):
    """Precompute conformal factor 1/(1 - c*||x||^2) for fast Poincaré distance.

    Store this as a payload alongside each vector. Reduces distance
    computation cost by ~30% since the norm doesn't need recomputing.

    Source: HyperspaceDB vector.rs
    """
    sq_norm = np.dot(vector, vector)
    denom = max(1.0 - c * sq_norm, 1e-7)
    return 1.0 / denom


def alpha_precompute_batch(vectors, c=1.0):
    """Batch alpha precomputation for a matrix of vectors."""
    sq_norms = np.sum(vectors ** 2, axis=1)
    denoms = np.maximum(1.0 - c * sq_norms, 1e-7)
    return 1.0 / denoms
```

- [ ] **Step 3: Add fused norms (Python version)**

```python
def fused_norms(u, v):
    """Compute ||u-v||^2, ||u||^2, ||v||^2 in a single pass.

    Avoids three separate np.dot calls. Most useful in loops;
    for batch operations use vectorized numpy instead.

    Source: RuVector poincare.rs
    """
    diff = u - v
    return np.dot(diff, diff), np.dot(u, u), np.dot(v, v)


def poincare_distance_with_alpha(diff_sq, alpha_u, alpha_v, c=1.0):
    """Poincaré distance from precomputed norms and alphas.

    d(u,v) = (1/sqrt(c)) * acosh(1 + 2*c * diff_sq * alpha_u * alpha_v)

    Use with fused_norms() and alpha_precompute() for maximum speed.
    """
    sqrt_c = np.sqrt(c)
    arg = 1.0 + 2.0 * c * diff_sq * alpha_u * alpha_v
    return (1.0 / sqrt_c) * np.arccosh(np.clip(arg, 1.0, None))
```

- [ ] **Step 4: Verify existing tests still work**

```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "
from hyperbolic_math import *
import numpy as np

# Test gromov delta on random vectors (should be high delta = not tree-like)
vecs = np.random.randn(100, 32).astype(np.float32)
delta, rec = gromov_delta(vecs, num_samples=500)
print(f'Random vectors: delta={delta:.4f}, rec={rec}')
assert delta > 0.1, 'Random vectors should not be tree-like'

# Test alpha precompute
v = np.array([0.3, 0.4, 0.0])
alpha = alpha_precompute(v, c=1.0)
expected = 1.0 / (1.0 - 0.09 - 0.16)
assert abs(alpha - expected) < 1e-5, f'alpha={alpha}, expected={expected}'

# Test fused norms
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])
d_sq, u_sq, v_sq = fused_norms(u, v)
assert abs(d_sq - 27.0) < 1e-5
assert abs(u_sq - 14.0) < 1e-5
assert abs(v_sq - 77.0) < 1e-5

# Test poincare_distance_with_alpha matches regular distance
p1 = np.array([0.1, 0.2])
p2 = np.array([0.3, 0.4])
d_sq, _, _ = fused_norms(p1, p2)
a1 = alpha_precompute(p1, c=1.0)
a2 = alpha_precompute(p2, c=1.0)
d_fast = poincare_distance_with_alpha(d_sq, a1, a2, c=1.0)

from hyperbolic_math import exp_map_at_origin, project_to_ball
# Compare with existing poincare distance
p1_ball = project_to_ball(p1, c=1.0)
p2_ball = project_to_ball(p2, c=1.0)
# Manual distance calc
norm_sq_1 = np.dot(p1_ball, p1_ball)
norm_sq_2 = np.dot(p2_ball, p2_ball)
diff_sq_manual = np.sum((p1_ball - p2_ball)**2)
arg_manual = 1.0 + 2.0 * diff_sq_manual / max((1.0 - norm_sq_1) * (1.0 - norm_sq_2), 1e-7)
d_manual = np.arccosh(max(arg_manual, 1.0))
print(f'd_fast={d_fast:.6f}, d_manual={d_manual:.6f}')

print('All hyperbolic_math tests passed!')
"
```

Expected: "All hyperbolic_math tests passed!"

---

## Part 3: HWV Dataset Preparation

### Task 6: Create HWV WikiVitals preparation script

**Files:**
- Create: `dev/datasets/hwv/prepare.py`

- [ ] **Step 1: Create the HWV directory**

```bash
mkdir -p /home/rohan/projects/qdrant/dev/datasets/hwv
```

- [ ] **Step 2: Write the preparation script**

Create `dev/datasets/hwv/prepare.py`:

```python
#!/usr/bin/env python3
"""Download and prepare HWV WikiVitals dataset for the testbench.

HWV (Hierarchical WikiVitals) is a 6-level hierarchy with 1,186 categories
and 10,013 Wikipedia article abstracts. Variable-depth paths (2-6 levels),
extreme class imbalance, single-path leaf (SPL) policy.

Source: https://github.com/RomanPlaud/revisitingHTC (MIT license)
Paper: Plaud et al., CoNLL 2024

Output: data/documents.jsonl with fields:
  id, text, tier, depth, domain, area, hierarchy_path, all_paths,
  parent_ids, child_ids, label_path
"""

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_URL = "https://github.com/RomanPlaud/revisitingHTC.git"
CLONE_DIR = Path(__file__).parent / "_hwv_repo"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "documents.jsonl"
HIERARCHY_FILE = OUTPUT_DIR / "hierarchy.json"

# Tier mapping: HWV levels → Pythia tiers
TIER_MAP = {
    0: "theme",
    1: "narrative",
    2: "story",
}
# Levels 3+ are all "content" at varying depths


def clone_repo():
    """Clone the HWV repository if not present."""
    if CLONE_DIR.exists():
        print(f"Repository already cloned at {CLONE_DIR}")
        return
    print(f"Cloning {REPO_URL}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)],
        check=True,
    )


def load_hierarchy(repo_dir):
    """Load the HWV hierarchy tree.

    Returns:
        nodes: dict mapping node_id → {name, depth, parent, children}
        label_to_node: dict mapping label string → node_id
    """
    # HWV stores hierarchy in various formats — check what's available
    data_dir = repo_dir / "data"
    if not data_dir.exists():
        data_dir = repo_dir

    nodes = {}
    label_to_node = {}
    parent_map = defaultdict(list)

    # Try to load from the dataset files
    # HWV typically has train/dev/test splits with label paths
    for split in ["train", "dev", "test"]:
        split_file = data_dir / f"hwv_{split}.json"
        if not split_file.exists():
            split_file = data_dir / f"{split}.json"
        if not split_file.exists():
            # Try jsonl
            split_file = data_dir / f"hwv_{split}.jsonl"
        if not split_file.exists():
            split_file = data_dir / f"{split}.jsonl"
        if not split_file.exists():
            continue

        with open(split_file) as f:
            if split_file.suffix == ".jsonl":
                items = [json.loads(line) for line in f if line.strip()]
            else:
                items = json.load(f)
                if isinstance(items, dict):
                    items = items.get("data", items.get("examples", [items]))

        for item in items:
            # Extract label path — format varies by dataset
            label_path = None
            if "labels" in item:
                label_path = item["labels"]
            elif "label_path" in item:
                label_path = item["label_path"]
            elif "categories" in item:
                label_path = item["categories"]

            if label_path is None:
                continue

            # Ensure it's a list of labels forming a path
            if isinstance(label_path, str):
                label_path = label_path.split("/")
            elif isinstance(label_path, list) and len(label_path) > 0:
                if isinstance(label_path[0], list):
                    label_path = label_path[0]  # Take first path

            # Build tree from paths
            for depth, label in enumerate(label_path):
                label_str = str(label).strip()
                if label_str not in label_to_node:
                    node_id = f"node_{len(nodes)}"
                    label_to_node[label_str] = node_id
                    nodes[node_id] = {
                        "name": label_str,
                        "depth": depth,
                        "parent": None,
                        "children": [],
                    }

                if depth > 0:
                    parent_label = str(label_path[depth - 1]).strip()
                    parent_id = label_to_node.get(parent_label)
                    child_id = label_to_node[label_str]
                    if parent_id and child_id not in nodes[parent_id]["children"]:
                        nodes[parent_id]["children"].append(child_id)
                    nodes[child_id]["parent"] = parent_id

    return nodes, label_to_node


def load_documents(repo_dir, label_to_node):
    """Load document texts and their label assignments."""
    data_dir = repo_dir / "data"
    if not data_dir.exists():
        data_dir = repo_dir

    documents = []

    for split in ["train", "dev", "test"]:
        for pattern in [f"hwv_{split}", split]:
            for ext in [".json", ".jsonl"]:
                split_file = data_dir / f"{pattern}{ext}"
                if split_file.exists():
                    break
            if split_file.exists():
                break

        if not split_file.exists():
            continue

        with open(split_file) as f:
            if split_file.suffix == ".jsonl":
                items = [json.loads(line) for line in f if line.strip()]
            else:
                items = json.load(f)
                if isinstance(items, dict):
                    items = items.get("data", items.get("examples", [items]))

        for item in items:
            text = item.get("text", item.get("abstract", item.get("content", "")))
            title = item.get("title", "")
            if title and text:
                text = f"{title}. {text}"
            elif title:
                text = title

            label_path = item.get("labels", item.get("label_path", item.get("categories", [])))
            if isinstance(label_path, str):
                label_path = label_path.split("/")
            elif isinstance(label_path, list) and len(label_path) > 0 and isinstance(label_path[0], list):
                label_path = label_path[0]

            label_path = [str(l).strip() for l in label_path]
            depth = len(label_path)

            # Map to tiers
            tier = TIER_MAP.get(depth - 1, "content") if depth > 0 else "content"

            domain = label_path[0] if len(label_path) > 0 else "unknown"
            area = label_path[1] if len(label_path) > 1 else domain

            # Build parent/child IDs from label path
            parent_ids = []
            if len(label_path) > 1:
                parent_label = label_path[-2]
                parent_node_id = label_to_node.get(parent_label)
                if parent_node_id:
                    parent_ids.append(parent_node_id)

            doc_id = f"hwv_doc_{len(documents)}"
            hierarchy_path = "/".join(label_path)

            documents.append({
                "id": doc_id,
                "text": text,
                "tier": "content",  # Leaf documents are always content
                "depth": depth,
                "domain": domain,
                "area": area,
                "hierarchy_path": hierarchy_path,
                "all_paths": [hierarchy_path],
                "parent_ids": parent_ids,
                "child_ids": [],
                "label_path": label_path,
                "split": split,
            })

    return documents


def generate_internal_nodes(nodes, documents, label_to_node):
    """Generate synthetic document entries for internal hierarchy nodes.

    Each internal node gets a synthetic text = concatenation of its name
    and all child names, to be replaced by Einstein midpoint embedding later.
    """
    internal_docs = []

    # Reverse map: node_id → list of document IDs that are its leaves
    node_to_doc_ids = defaultdict(list)
    for doc in documents:
        for label in doc["label_path"]:
            label_str = str(label).strip()
            node_id = label_to_node.get(label_str)
            if node_id:
                node_to_doc_ids[node_id].append(doc["id"])

    for node_id, node in nodes.items():
        depth = node["depth"]
        tier = TIER_MAP.get(depth, "content")
        domain = node["name"] if depth == 0 else ""
        area = node["name"] if depth == 1 else ""

        # Walk up to find domain/area
        current = node
        path_labels = [current["name"]]
        while current["parent"]:
            current = nodes[current["parent"]]
            path_labels.insert(0, current["name"])

        domain = path_labels[0] if len(path_labels) > 0 else "unknown"
        area = path_labels[1] if len(path_labels) > 1 else domain

        child_ids = node["children"]
        parent_ids = [node["parent"]] if node["parent"] else []
        hierarchy_path = "/".join(path_labels)

        # Synthetic text for embedding
        child_names = [nodes[c]["name"] for c in child_ids if c in nodes]
        text = f"{node['name']}: {', '.join(child_names)}" if child_names else node["name"]

        internal_docs.append({
            "id": node_id,
            "text": text,
            "tier": tier,
            "depth": depth,
            "domain": domain,
            "area": area,
            "hierarchy_path": hierarchy_path,
            "all_paths": [hierarchy_path],
            "parent_ids": parent_ids,
            "child_ids": child_ids,
            "label_path": path_labels,
            "source_doc_ids": node_to_doc_ids.get(node_id, []),
        })

    return internal_docs


def main():
    if OUTPUT_FILE.exists():
        print(f"Output already exists: {OUTPUT_FILE}")
        print("Delete it to re-generate.")
        return

    clone_repo()

    print("Loading hierarchy...")
    nodes, label_to_node = load_hierarchy(CLONE_DIR)
    print(f"  {len(nodes)} hierarchy nodes")

    # Depth distribution
    depth_counts = defaultdict(int)
    for n in nodes.values():
        depth_counts[n["depth"]] += 1
    for d in sorted(depth_counts):
        print(f"  Depth {d}: {depth_counts[d]} nodes")

    print("Loading documents...")
    documents = load_documents(CLONE_DIR, label_to_node)
    print(f"  {len(documents)} documents loaded")

    print("Generating internal node entries...")
    internal_docs = generate_internal_nodes(nodes, documents, label_to_node)
    print(f"  {len(internal_docs)} internal nodes")

    # Combine
    all_docs = documents + internal_docs

    # Print tier stats
    tier_counts = defaultdict(int)
    for doc in all_docs:
        tier_counts[doc["tier"]] += 1
    print(f"\nTier breakdown:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for doc in all_docs:
            f.write(json.dumps(doc) + "\n")
    print(f"\nSaved {len(all_docs)} entries to {OUTPUT_FILE}")

    # Save hierarchy
    with open(HIERARCHY_FILE, "w") as f:
        json.dump({"nodes": nodes, "label_to_node": label_to_node}, f, indent=2)
    print(f"Saved hierarchy to {HIERARCHY_FILE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test the preparation script**

Tell the user to run:

```bash
cd /home/rohan/projects/qdrant/dev/datasets/hwv && python3 prepare.py
```

Expected: Downloads repo, parses hierarchy, outputs `data/documents.jsonl` and `data/hierarchy.json` with ~10K documents + ~1.2K internal nodes.

**Note:** The HWV repo format may differ from what's expected. If the script fails, inspect the cloned repo structure:
```bash
ls -la dev/datasets/hwv/_hwv_repo/
find dev/datasets/hwv/_hwv_repo/ -name "*.json" -o -name "*.jsonl" -o -name "*.csv" -o -name "*.tsv" | head -20
```
Then adapt the `load_hierarchy` and `load_documents` functions to match the actual format.

---

## Part 4: Collection Architecture

### Task 7: Create collections.py

**Files:**
- Create: `dev/testbench/collections.py`

- [ ] **Step 1: Write the collection creation module**

Create `dev/testbench/collections.py`:

```python
#!/usr/bin/env python3
"""Unified, separate, and cosine collection creation for the testbench.

Creates 3 collection variants per dataset:
1. {dataset}_cosine       — cosine-only baseline, all tiers
2. {dataset}_unified      — named vectors (dense + poincare + sparse), all tiers
3. {dataset}_narratives / _stories / _content — separate per-tier

Each variant gets the same data with proper payload indices.
"""

import json
import time
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, TextIndexParams, TokenizerType,
    Filter, FieldCondition, MatchValue, Range,
)


# ID offset scheme for unified collections
CONTENT_ID_OFFSET = 0
STORY_ID_OFFSET = 100_000
NARRATIVE_ID_OFFSET = 200_000
THEME_ID_OFFSET = 300_000

TIER_OFFSETS = {
    "content": CONTENT_ID_OFFSET,
    "story": STORY_ID_OFFSET,
    "narrative": NARRATIVE_ID_OFFSET,
    "theme": THEME_ID_OFFSET,
}


def _rest_create_collection(qdrant_url, name, vectors_config):
    """Create collection via REST API (supports Poincare distance)."""
    url = f"{qdrant_url}/collections/{name}"
    requests.delete(url)
    time.sleep(0.5)
    body = {
        "vectors": vectors_config,
        "optimizers_config": {"indexing_threshold": 10000},
    }
    resp = requests.put(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _create_payload_indices(qdrant_url, name):
    """Create payload indices for filtering."""
    indices = {
        "item_type": "keyword",
        "tier": "integer",
        "busemann_depth": "float",
        "domain": "keyword",
        "area": "keyword",
        "parent_ids": "keyword",
        "child_ids": "keyword",
        "created_at": "datetime",
        "point_id": "keyword",
    }
    for field, schema_type in indices.items():
        url = f"{qdrant_url}/collections/{name}/index"
        body = {"field_name": field, "field_schema": schema_type}
        resp = requests.put(url, json=body, timeout=30)
        if resp.status_code not in (200, 409):  # 409 = already exists
            print(f"  Warning: index {field} on {name}: {resp.status_code}")


def create_cosine_collection(qdrant_url, dataset, dim=1024):
    """Create cosine-only baseline collection."""
    name = f"{dataset}_cosine"
    print(f"Creating {name}...")
    vectors_config = {
        "size": dim,
        "distance": "Cosine",
    }
    _rest_create_collection(qdrant_url, name, vectors_config)
    _create_payload_indices(qdrant_url, name)
    return name


def create_unified_collection(qdrant_url, dataset, dense_dim=1024,
                               poincare_dim=128, curvature=5.0):
    """Create unified collection with named vectors."""
    name = f"{dataset}_unified"
    print(f"Creating {name}...")
    vectors_config = {
        "dense": {"size": dense_dim, "distance": "Cosine"},
        "poincare": {
            "size": poincare_dim,
            "distance": "Poincare",
            "hnsw_config": {"m": 16, "ef_construct": 200},
        },
    }
    # Add curvature if supported
    vectors_config["poincare"]["curvature"] = curvature
    _rest_create_collection(qdrant_url, name, vectors_config)
    _create_payload_indices(qdrant_url, name)
    return name


def create_separate_collections(qdrant_url, dataset, dense_dim=1024,
                                 poincare_dim=128, curvature=5.0):
    """Create separate per-tier collections (current Pythia pattern)."""
    names = {}
    for tier in ["narratives", "stories", "content"]:
        name = f"{dataset}_{tier}"
        print(f"Creating {name}...")
        vectors_config = {
            "dense": {"size": dense_dim, "distance": "Cosine"},
            "poincare": {
                "size": poincare_dim,
                "distance": "Poincare",
                "curvature": curvature,
                "hnsw_config": {"m": 16, "ef_construct": 200},
            },
        }
        _rest_create_collection(qdrant_url, name, vectors_config)
        _create_payload_indices(qdrant_url, name)
        names[tier] = name
    return names


def populate_collection(client, collection_name, points, batch_size=256):
    """Upsert points into a collection in batches.

    Each point is a dict with keys:
      id, dense_vector, poincare_vector (optional), payload
    """
    total = len(points)
    for i in tqdm(range(0, total, batch_size), desc=f"  Upserting to {collection_name}"):
        batch = points[i:i + batch_size]
        structs = []
        for pt in batch:
            vector = pt.get("dense_vector")
            if "poincare_vector" in pt:
                # Named vectors
                vector = {
                    "dense": pt["dense_vector"].tolist() if hasattr(pt["dense_vector"], "tolist") else pt["dense_vector"],
                    "poincare": pt["poincare_vector"].tolist() if hasattr(pt["poincare_vector"], "tolist") else pt["poincare_vector"],
                }
            else:
                vector = pt["dense_vector"].tolist() if hasattr(pt["dense_vector"], "tolist") else pt["dense_vector"]

            structs.append(PointStruct(
                id=pt["id"],
                vector=vector,
                payload=pt["payload"],
            ))
        client.upsert(collection_name=collection_name, points=structs)


def build_points_from_data(documents, cosine_vectors, poincare_vectors,
                            busemann_depths, alpha_values,
                            tier_map=None, synthetic_timestamps=None):
    """Convert raw data arrays into point dicts for upserting.

    Args:
        documents: list of doc dicts (with id, tier, domain, area, parent_ids, etc.)
        cosine_vectors: np.ndarray (N, 1024)
        poincare_vectors: np.ndarray (N, 128)
        busemann_depths: np.ndarray (N,)
        alpha_values: np.ndarray (N,)
        tier_map: dict mapping tier name → int (default: content=2, story=1, narrative=0)
        synthetic_timestamps: list of datetime strings (optional)

    Returns:
        list of point dicts ready for populate_collection()
    """
    if tier_map is None:
        tier_map = {"narrative": 0, "theme": 0, "story": 1, "content": 2}

    points = []
    for i, doc in enumerate(documents):
        tier_name = doc.get("tier", "content")
        tier_int = tier_map.get(tier_name, 2)
        offset = TIER_OFFSETS.get(tier_name, CONTENT_ID_OFFSET)
        point_id = offset + i

        payload = {
            "item_type": tier_name,
            "tier": tier_int,
            "busemann_depth": float(busemann_depths[i]),
            "alpha": float(alpha_values[i]),
            "domain": doc.get("domain", ""),
            "area": doc.get("area", ""),
            "parent_ids": doc.get("parent_ids", []),
            "child_ids": doc.get("child_ids", []),
            "point_id": doc.get("id", f"pt_{i}"),
            "hierarchy_path": doc.get("hierarchy_path", ""),
        }

        if synthetic_timestamps and i < len(synthetic_timestamps):
            payload["created_at"] = synthetic_timestamps[i]

        points.append({
            "id": point_id,
            "dense_vector": cosine_vectors[i],
            "poincare_vector": poincare_vectors[i],
            "payload": payload,
        })

    return points
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "import collections as c; print('collections.py imported OK')"
```

Wait — `collections` conflicts with the Python stdlib. Rename to `collection_builder.py` instead.

Rename the file to `dev/testbench/collection_builder.py` and update the import check:

```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "import collection_builder; print('collection_builder.py imported OK')"
```

---

## Part 5: Chatbot Query Implementations

### Task 8: Create queries.py

**Files:**
- Create: `dev/testbench/queries.py`

- [ ] **Step 1: Write the chatbot query module**

Create `dev/testbench/queries.py`:

```python
#!/usr/bin/env python3
"""Chatbot query implementations for the testbench.

Five query patterns that mirror real Pythia/Lens chatbot usage:
1. drill_down  — vertical traversal (narrative → stories → content)
2. lateral     — horizontal search (same-tier related items)
3. cross_branch — structural similarity across domains
4. temporal    — time-windowed search within tier
5. depth_band  — Busemann depth-filtered search
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny, Range,
    SearchParams, NamedVector,
)


def drill_down_query(client, collection, start_point_id, using="poincare",
                     limit=10, ef=128):
    """Multi-hop drill-down: find children at each tier below the start point.

    1. Retrieve start point to get its vector and child_ids
    2. Search for stories (tier=1) with parent_ids containing start_point_id
    3. For each story found, search for content (tier=2) with matching parent_ids

    Returns dict with hop1_results, hop2_results, and timing.
    """
    import time

    t0 = time.time()

    # Step 1: Retrieve start point
    start_points = client.retrieve(
        collection_name=collection,
        ids=[start_point_id],
        with_vectors=True,
        with_payload=True,
    )
    if not start_points:
        return {"error": f"Point {start_point_id} not found"}

    start = start_points[0]
    query_vector = (
        start.vector.get(using, start.vector)
        if isinstance(start.vector, dict)
        else start.vector
    )
    start_pid = start.payload.get("point_id", str(start_point_id))

    # Step 2: Search for children (stories)
    hop1_filter = Filter(must=[
        FieldCondition(key="parent_ids", match=MatchValue(value=start_pid)),
    ])

    if isinstance(start.vector, dict) and using in start.vector:
        hop1_results = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=using,
            query_filter=hop1_filter,
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=True,
        ).points
    else:
        hop1_results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=hop1_filter,
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
            with_vectors=True,
        )

    t1 = time.time()

    # Step 3: For each story, search for its children (content)
    hop2_results = []
    for story in hop1_results[:5]:  # Limit second hop to top 5
        story_pid = story.payload.get("point_id", str(story.id))
        hop2_filter = Filter(must=[
            FieldCondition(key="parent_ids", match=MatchValue(value=story_pid)),
        ])
        story_vector = (
            story.vector.get(using, story.vector)
            if isinstance(story.vector, dict)
            else story.vector
        )

        try:
            if isinstance(story.vector, dict) and using in story.vector:
                h2 = client.query_points(
                    collection_name=collection,
                    query=story_vector,
                    using=using,
                    query_filter=hop2_filter,
                    limit=limit,
                    search_params=SearchParams(hnsw_ef=ef),
                    with_payload=True,
                ).points
            else:
                h2 = client.search(
                    collection_name=collection,
                    query_vector=story_vector,
                    query_filter=hop2_filter,
                    limit=limit,
                    search_params=SearchParams(hnsw_ef=ef),
                    with_payload=True,
                )
            hop2_results.extend(h2)
        except Exception as e:
            hop2_results.append({"error": str(e)})

    t2 = time.time()

    return {
        "hop1_results": hop1_results,
        "hop2_results": hop2_results,
        "hop1_latency_ms": (t1 - t0) * 1000,
        "hop2_latency_ms": (t2 - t1) * 1000,
        "total_latency_ms": (t2 - t0) * 1000,
    }


def lateral_query(client, collection, query_point_id, using="dense",
                  limit=10, ef=128):
    """Same-tier search: find related items at the query's tier level.

    Filters by tier == query's tier, then searches using the specified vector.
    """
    import time

    t0 = time.time()

    # Retrieve query point
    pts = client.retrieve(
        collection_name=collection,
        ids=[query_point_id],
        with_vectors=True,
        with_payload=True,
    )
    if not pts:
        return {"error": f"Point {query_point_id} not found"}

    pt = pts[0]
    tier = pt.payload.get("tier", 2)
    query_vector = (
        pt.vector.get(using, pt.vector)
        if isinstance(pt.vector, dict)
        else pt.vector
    )

    tier_filter = Filter(must=[
        FieldCondition(key="tier", match=MatchValue(value=tier)),
    ])

    if isinstance(pt.vector, dict) and using in pt.vector:
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=using,
            query_filter=tier_filter,
            limit=limit + 1,  # +1 to exclude self
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
        ).points
    else:
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=tier_filter,
            limit=limit + 1,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
        )

    # Remove self from results
    results = [r for r in results if r.id != query_point_id][:limit]

    t1 = time.time()

    return {
        "results": results,
        "query_tier": tier,
        "query_domain": pt.payload.get("domain", ""),
        "query_area": pt.payload.get("area", ""),
        "latency_ms": (t1 - t0) * 1000,
    }


def cross_branch_query(client, collection, narrative_point_id, using="poincare",
                       limit=10, ef=128):
    """Cross-domain structural similarity: find narratives in OTHER domains
    that have similar hierarchical structure.

    Filters: item_type=narrative AND domain != source_domain
    """
    import time

    t0 = time.time()

    pts = client.retrieve(
        collection_name=collection,
        ids=[narrative_point_id],
        with_vectors=True,
        with_payload=True,
    )
    if not pts:
        return {"error": f"Point {narrative_point_id} not found"}

    pt = pts[0]
    source_domain = pt.payload.get("domain", "")
    query_vector = (
        pt.vector.get(using, pt.vector)
        if isinstance(pt.vector, dict)
        else pt.vector
    )

    # Filter: narrative tier, different domain
    cross_filter = Filter(
        must=[
            FieldCondition(key="item_type", match=MatchValue(value="narrative")),
        ],
        must_not=[
            FieldCondition(key="domain", match=MatchValue(value=source_domain)),
        ],
    )

    if isinstance(pt.vector, dict) and using in pt.vector:
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=using,
            query_filter=cross_filter,
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
        ).points
    else:
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=cross_filter,
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
        )

    t1 = time.time()

    return {
        "results": results,
        "source_domain": source_domain,
        "latency_ms": (t1 - t0) * 1000,
    }


def temporal_query(client, collection, query_vector, date_gte, date_lt,
                   tier=None, using="dense", limit=10, ef=128):
    """Time-windowed search within an optional tier filter.

    Args:
        query_vector: the search vector (list or np.ndarray)
        date_gte: ISO datetime string for range start
        date_lt: ISO datetime string for range end
        tier: optional int tier filter
    """
    import time

    t0 = time.time()

    must_conditions = [
        FieldCondition(key="created_at", range=Range(gte=date_gte, lt=date_lt)),
    ]
    if tier is not None:
        must_conditions.append(
            FieldCondition(key="tier", match=MatchValue(value=tier))
        )

    time_filter = Filter(must=must_conditions)

    if using in ("dense", "poincare"):
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=using,
            query_filter=time_filter,
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
        ).points
    else:
        results = client.search(
            collection_name=collection,
            query_vector=query_vector if not hasattr(query_vector, "tolist") else query_vector.tolist(),
            query_filter=time_filter,
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
        )

    t1 = time.time()

    return {
        "results": results,
        "latency_ms": (t1 - t0) * 1000,
    }


def depth_band_query(client, collection, query_vector, depth_min, depth_max,
                     using="poincare", limit=10, ef=128):
    """Search within a Busemann depth band.

    Finds points whose busemann_depth falls within [depth_min, depth_max],
    ranked by vector similarity.
    """
    import time

    t0 = time.time()

    depth_filter = Filter(must=[
        FieldCondition(
            key="busemann_depth",
            range=Range(gte=depth_min, lte=depth_max),
        ),
    ])

    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    if using in ("dense", "poincare"):
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=using,
            query_filter=depth_filter,
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
        ).points
    else:
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=depth_filter,
            limit=limit,
            search_params=SearchParams(hnsw_ef=ef),
            with_payload=True,
        )

    t1 = time.time()

    return {
        "results": results,
        "latency_ms": (t1 - t0) * 1000,
    }
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd /home/rohan/projects/qdrant/dev/testbench && python3 -c "import queries; print('queries.py imported OK')"
```

---

## Part 6: Benchmark Engine and Embed Pipeline

### Task 9: Extend embed.py with HWV support and aggregate embeddings

**Files:**
- Modify: `dev/testbench/embed.py`

- [ ] **Step 1: Add HWV to DATASET_CONFIGS**

In `embed.py`, add the HWV entry to `DATASET_CONFIGS` (~line 49):

```python
"hwv": {
    "source_dir": Path(__file__).parent.parent / "datasets" / "hwv" / "data",
    "doc_file": "documents.jsonl",
    "tier_map": {"theme": "root", "narrative": "root", "story": "mid", "content": "leaf"},
    "has_abstract": False,
},
```

- [ ] **Step 2: Add Gromov analysis function**

Add after the Busemann computation function:

```python
def run_gromov_analysis(poincare_vectors, dataset_name):
    """Run Gromov delta-hyperbolicity analysis on the dataset."""
    from hyperbolic_math import gromov_delta

    print(f"\n=== Gromov Delta Analysis ({dataset_name}) ===")
    delta, recommendation = gromov_delta(poincare_vectors, num_samples=2000)
    print(f"  Delta: {delta:.4f}")
    print(f"  Recommendation: {recommendation}")
    print(f"  Tree-like: {'yes' if delta < 0.3 else 'no'}")
    return {"delta": delta, "recommendation": recommendation}
```

- [ ] **Step 3: Add alpha precomputation to the pipeline**

After the Busemann computation step in `main()`, add:

```python
from hyperbolic_math import alpha_precompute_batch

alpha_values = alpha_precompute_batch(poincare_vectors, c=CURVATURE)
print(f"  Alpha values: min={alpha_values.min():.4f}, max={alpha_values.max():.4f}")
```

- [ ] **Step 4: Add synthetic timestamp generation**

Add a helper function:

```python
def generate_synthetic_timestamps(documents, window_days=30):
    """Generate synthetic created_at timestamps for temporal benchmarks.

    Content: random dates within window
    Stories: median of children's dates
    Narratives: earliest of children's dates
    """
    import datetime
    import random

    base_date = datetime.datetime(2026, 3, 1, tzinfo=datetime.timezone.utc)
    timestamps = {}

    # Content gets random dates
    for i, doc in enumerate(documents):
        if doc.get("tier") in ("content", "leaf"):
            dt = base_date + datetime.timedelta(
                days=random.uniform(0, window_days),
                hours=random.uniform(0, 24),
            )
            timestamps[doc["id"]] = dt.isoformat()

    # Stories get median of children
    for doc in documents:
        if doc.get("tier") in ("story", "mid"):
            child_ids = doc.get("child_ids", [])
            child_times = [timestamps[cid] for cid in child_ids if cid in timestamps]
            if child_times:
                child_times.sort()
                timestamps[doc["id"]] = child_times[len(child_times) // 2]
            else:
                dt = base_date + datetime.timedelta(days=random.uniform(0, window_days))
                timestamps[doc["id"]] = dt.isoformat()

    # Narratives get earliest of children
    for doc in documents:
        if doc.get("tier") in ("narrative", "root", "theme"):
            child_ids = doc.get("child_ids", [])
            child_times = [timestamps[cid] for cid in child_ids if cid in timestamps]
            if child_times:
                timestamps[doc["id"]] = min(child_times)
            else:
                dt = base_date + datetime.timedelta(days=random.uniform(0, window_days))
                timestamps[doc["id"]] = dt.isoformat()

    return [timestamps.get(doc["id"], base_date.isoformat()) for doc in documents]
```

- [ ] **Step 5: Wire everything into the unified collection upsert**

In the unified collection creation section of `main()`, update to use `collection_builder.py` and pass alpha values and timestamps as payloads.

---

### Task 10: Rewrite benchmark.py with 8 suites

**Files:**
- Modify: `dev/testbench/benchmark.py`

This is the largest task. The benchmark engine needs to be restructured as a class with 8 suite methods.

- [ ] **Step 1: Refactor benchmark.py into BenchmarkRunner class**

The existing benchmark.py has the functions for hierarchy separation, retrieval quality, cross-tier, dual-space, and latency. Restructure as:

```python
class BenchmarkRunner:
    def __init__(self, client, qdrant_url, dataset, curvature=5.0):
        self.client = client
        self.qdrant_url = qdrant_url
        self.dataset = dataset
        self.curvature = curvature
        self.results = {}

    def run_all(self):
        """Run all 8 benchmark suites."""
        self.suite_gromov_delta()
        self.suite_hierarchy_separation()
        self.suite_drill_down()
        self.suite_lateral()
        self.suite_cross_branch()
        self.suite_depth_band()
        self.suite_unified_vs_separate()
        self.suite_latency()
        return self.results
```

- [ ] **Step 2: Implement Suite 1 — Gromov Delta**

```python
def suite_gromov_delta(self):
    """Suite 1: Analyze how tree-like the dataset is."""
    from hyperbolic_math import gromov_delta

    collection = f"{self.dataset}_unified"
    points = scroll_all(self.client, collection, "poincare")
    vectors = np.array([p["vector"] for p in points])

    delta, rec = gromov_delta(vectors, num_samples=2000)
    self.results["gromov_delta"] = {
        "delta": delta,
        "recommendation": rec,
        "num_points": len(vectors),
    }
    print(f"  Gromov delta: {delta:.4f} → {rec}")
```

- [ ] **Step 3: Implement Suite 2 — Hierarchy Separation (upgraded)**

Keep existing `_compute_hierarchy_separation` logic, add Einstein vs Fréchet comparison and alpha speedup measurement. Add to `self.results["hierarchy_separation"]`.

- [ ] **Step 4: Implement Suite 3 — Drill-Down Queries**

```python
def suite_drill_down(self):
    """Suite 3: Vertical traversal — narrative → stories → content."""
    from queries import drill_down_query

    collection = f"{self.dataset}_unified"
    points = scroll_all(self.client, collection, "poincare")

    # Select narrative-tier points
    narratives = [p for p in points if p.get("tier") == 0
                  or p.get("item_type") == "narrative"]
    if not narratives:
        print("  No narrative points found, skipping drill-down")
        return

    sample = narratives[:min(100, len(narratives))]
    results_by_method = {}

    for method in ["poincare", "dense"]:
        hop1_recalls = []
        hop2_recalls = []
        precisions = []
        latencies = []

        for narr in tqdm(sample, desc=f"  drill-down ({method})"):
            result = drill_down_query(
                self.client, collection, narr["id"], using=method
            )
            if "error" in result:
                continue

            # Measure hop1 recall: what fraction of true children were found?
            true_children = set(narr.get("child_ids", []))
            found_pids = {r.payload.get("point_id", "") for r in result["hop1_results"]}
            if true_children:
                recall = len(true_children & found_pids) / len(true_children)
                hop1_recalls.append(recall)

            # Precision: what fraction of results are true descendants?
            if result["hop1_results"]:
                correct = sum(1 for r in result["hop1_results"]
                            if r.payload.get("point_id", "") in true_children)
                precisions.append(correct / len(result["hop1_results"]))

            latencies.append(result["total_latency_ms"])

        results_by_method[method] = {
            "hop1_recall_mean": np.mean(hop1_recalls) if hop1_recalls else 0,
            "precision_mean": np.mean(precisions) if precisions else 0,
            "latency_p50_ms": np.median(latencies) if latencies else 0,
            "latency_p95_ms": np.percentile(latencies, 95) if latencies else 0,
            "num_queries": len(sample),
        }

    self.results["drill_down"] = results_by_method
```

- [ ] **Step 5: Implement Suite 4 — Lateral Exploration**

```python
def suite_lateral(self):
    """Suite 4: Same-tier search."""
    from queries import lateral_query

    collection = f"{self.dataset}_unified"
    points = scroll_all(self.client, collection, "dense")

    sample = random.sample(points, min(200, len(points)))
    results_by_method = {}

    for method in ["dense", "poincare"]:
        area_recalls = []
        domain_recalls = []
        diversities = []

        for pt in tqdm(sample, desc=f"  lateral ({method})"):
            result = lateral_query(self.client, collection, pt["id"], using=method)
            if "error" in result:
                continue

            query_area = result["query_area"]
            query_domain = result["query_domain"]

            # Area recall: fraction in same area
            if result["results"]:
                same_area = sum(1 for r in result["results"]
                              if r.payload.get("area") == query_area)
                area_recalls.append(same_area / len(result["results"]))

                same_domain = sum(1 for r in result["results"]
                                if r.payload.get("domain") == query_domain)
                domain_recalls.append(same_domain / len(result["results"]))

                # Diversity: unique areas in results
                unique_areas = len(set(r.payload.get("area", "") for r in result["results"]))
                diversities.append(unique_areas)

        results_by_method[method] = {
            "area_recall_mean": np.mean(area_recalls) if area_recalls else 0,
            "domain_recall_mean": np.mean(domain_recalls) if domain_recalls else 0,
            "diversity_mean": np.mean(diversities) if diversities else 0,
            "num_queries": len(sample),
        }

    self.results["lateral"] = results_by_method
```

- [ ] **Step 6: Implement Suites 5-8**

Follow the same pattern for:
- **Suite 5 (cross_branch):** Use `cross_branch_query`, measure structural similarity
- **Suite 6 (depth_band):** Use `depth_band_query`, measure filter precision and recall
- **Suite 7 (unified_vs_separate):** Run suites 3-6 on both unified and separate collections, compute deltas
- **Suite 8 (latency):** Keep existing latency benchmark logic, extend to all collection variants

- [ ] **Step 7: Update main() to use BenchmarkRunner**

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default="http://localhost:6334")
    parser.add_argument("--dataset", default="bgc", choices=["bgc", "hwv", "wos", "eurlex"])
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url, timeout=60)

    runner = BenchmarkRunner(client, args.qdrant_url, args.dataset)
    results = runner.run_all()

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = results_dir / f"benchmark_{args.dataset}_{timestamp}.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output}")
```

---

## Part 7: Integration and End-to-End Test

### Task 11: Update run.sh and run end-to-end

**Files:**
- Modify: `dev/testbench/run.sh`

- [ ] **Step 1: Add HWV subcommands to run.sh**

Update the case statement in `run.sh`:

```bash
case "${1:-all}" in
    all)
        docker compose up -d --build
        wait_for_qdrant
        python3 embed.py --qdrant-url "$QDRANT_URL" --dataset "${2:-bgc}"
        python3 benchmark.py --qdrant-url "$QDRANT_URL" --dataset "${2:-bgc}"
        ;;
    hwv)
        wait_for_qdrant
        python3 ../datasets/hwv/prepare.py
        python3 embed.py --qdrant-url "$QDRANT_URL" --dataset hwv
        python3 benchmark.py --qdrant-url "$QDRANT_URL" --dataset hwv
        ;;
    benchmark)
        wait_for_qdrant
        python3 benchmark.py --qdrant-url "$QDRANT_URL" --dataset "${2:-bgc}"
        ;;
    embed)
        wait_for_qdrant
        python3 embed.py --qdrant-url "$QDRANT_URL" --dataset "${2:-bgc}"
        ;;
    rebuild)
        docker compose up -d --build
        wait_for_qdrant
        ;;
    down)
        docker compose down -v
        ;;
    *)
        echo "Usage: ./run.sh [all|hwv|benchmark|embed|rebuild|down] [dataset]"
        exit 1
        ;;
esac
```

- [ ] **Step 2: End-to-end verification — BGC**

Tell the user to run:

```bash
cd /home/rohan/projects/qdrant/dev/testbench
./run.sh rebuild        # Rebuild Qdrant with fixed fork
./run.sh all bgc        # Full pipeline: embed + benchmark
```

Expected: No errors. Results JSON written to `results/benchmark_bgc_*.json` with all 8 suites populated.

- [ ] **Step 3: End-to-end verification — HWV**

Tell the user to run:

```bash
cd /home/rohan/projects/qdrant/dev/testbench
./run.sh hwv            # Prepare + embed + benchmark HWV
```

Expected: HWV data downloaded, embedded, collections created, benchmarks run.

- [ ] **Step 4: Review results**

Check the output JSON for success criteria:
- Gromov delta < 0.3 for BGC
- Drill-down recall@10 > 80% on BGC
- Unified vs separate recall delta < 5%
- All 8 suites have populated results (no empty dicts)

- [ ] **Step 5: Commit everything**

Once all tests pass and results look good:

```bash
cd /home/rohan/projects/qdrant
git add -A
git status  # Review all changes
# Then commit with a descriptive message
```

---

## Verification Checklist

- [ ] Qdrant fork: `cargo test --lib --features hyperbolic -p segment` — all tests pass
- [ ] Qdrant fork: `cargo check --features hyperbolic` — full workspace compiles
- [ ] Docker: Qdrant container starts and serves health check
- [ ] gRPC: Poincare distance in proto (verify with `grep Poincare lib/api/src/grpc/proto/collections.proto`)
- [ ] Curvature: Non-default curvature flows to scorer (verify with a c=5.0 collection)
- [ ] Busemann/tangent: Deleted from Rust, functions exist in Python
- [ ] hyperbolic_math.py: gromov_delta, alpha_precompute, fused_norms all work
- [ ] HWV: prepare.py produces documents.jsonl with hierarchy
- [ ] BGC: Unified collection created with named vectors + payloads
- [ ] Queries: All 5 query functions execute without errors
- [ ] Benchmark: All 8 suites produce results
- [ ] Results: JSON output with non-empty metrics for every suite
