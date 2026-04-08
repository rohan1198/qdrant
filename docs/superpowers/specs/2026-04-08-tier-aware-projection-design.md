# Tier-Aware Projection for Hyperbolic Testbench

**Date:** 2026-04-08
**Status:** Draft
**Scope:** `dev/testbench/` only (no Rust-side changes)

## Problem

All vectors in the Poincare ball land at the same radius due to uniform L2 normalization
(`BALL_SCALING=0.9`) before projection. Busemann separation (~6.2) comes from directional
differences only, not depth. The full power of hyperbolic space — encoding hierarchy via
radial distance from the origin — is not leveraged.

**First run results (2026-04-08):**

| Collection | Curvature | SepRatio | AreaRec@10 | DomRec@10 | H-Prec | p95@ef128 |
|---|---|---|---|---|---|---|
| wos_cosine | cosine | N/A | 0.184 | 0.583 | 0.384 | 2.4ms |
| wos_c50 | 5.0 | 6.367 | 0.196 | 0.578 | 0.387 | 4.1ms |

Poincare c=5.0 only marginally beats cosine because all vectors occupy a thin shell rather
than utilizing the full ball depth.

## Context

### Pythia Hierarchy

Pythia has a 4-tier hierarchy (Content > Stories > Narratives > Themes), but Themes don't
have embeddings yet. For this testbench we target **3 tiers** mapped to the WOS dataset:

| WOS Tier | Count | Pythia Equivalent | Ball Position |
|---|---|---|---|
| root (domains) | 10 | Narratives | Near origin (abstract) |
| mid (areas) | 336 | Stories | Intermediate |
| leaf (papers) | 45,448 | Content/Posts | Near boundary (concrete) |

### Key Insight: Tier is Always Known

In Pythia's cascading pipelines (embedding -> stories -> narratives), the tier label is always
known at embed time. Each pipeline stage produces a distinct tier. This means we can inject
tier as an explicit scaling parameter during projection — no inference needed.

### Inspiration from ruvector

The ruvector library (`~/projects/ruvector`) provides direct design precedent:

1. **ruvector-dag** (`hierarchical_lorentz.rs`) maps known depth to Poincare radius via
   `tanh(depth * 0.5)` — explicit depth-aware radial placement.
2. **Einstein midpoint** weights by Lorentz gamma factor `gamma = 1/sqrt(1 + c * ||x||^2)`,
   which naturally encodes depth. Points near the boundary get lower aggregation weight.
3. **Conformal factor** `lambda = 2/(1 - c*||x||^2)` grows with norm — the natural depth
   measure that appears throughout exp/log maps.

## Design

### Projection Strategies

Four strategies, each a function `(vectors_128d, tiers, curvature) -> projected_vectors`:

**Important preprocessing step:** All strategies first L2-normalize the 128d PCA vectors to
unit norm. Without this, varying PCA norms (which are large enough to saturate the exp_map's
`tanh`) swamp any tier-based scaling. Note: the existing code defines `BALL_SCALING=0.9` but
never applies it — `project_batch` passes raw PCA vectors directly to `exp_map_at_origin`,
where `tanh` saturates and `project_to_ball` clips everything to 0.95. This is the root cause
of the uniform-radius problem.

#### 1. `uniform` (existing baseline)

L2-normalize, then scale all vectors by 0.9, then exp_map + project_to_ball. No tier
awareness. This fixes the "BALL_SCALING defined but unused" bug while preserving the
uniform-scaling approach for fair comparison.

#### 2. `static` (hardcoded tier radii)

Per-tier scaling applied before exp_map:

```
STATIC_TIER_SCALING = {"root": 0.15, "mid": 0.50, "leaf": 0.90}
```

Each vector is multiplied by its tier's scale factor, then exp_map, then project_to_ball.
Simple and deterministic.

#### 3. `einstein` (data-driven tier centers)

Two-pass approach using Einstein midpoint to derive tier radii from the data:

**Pass 1 — Calibrate:**
1. Project all vectors uniformly (scale=0.9) into the Poincare ball.
2. Group by tier. Compute Einstein midpoint per tier in Lorentz space:
   - Convert each point to Lorentz: `x_0 = (1 + c*||p||^2)/(1 - c*||p||^2)`, spatial = `2*sqrt(c)*p/(1 - c*||p||^2)`
   - Weighted average: `m = sum(gamma_i * x_i) / sum(gamma_i)` where `gamma_i = x_i[0]`
   - Project to hyperboloid: enforce `<m,m>_L = -1/c`
3. Convert each midpoint back to Poincare and extract radius: `tier_radius = ||midpoint_poincare||`
4. Enforce monotonic ordering: if `root < mid < leaf` is violated, sort and reassign with a warning.

**Pass 2 — Project:**
Go back to raw 128d PCA vectors and project fresh using derived tier radii as the per-tier
scaling factor before exp_map.

#### 4. `einstein_spread` (data-driven + intra-tier variance)

Same calibration as `einstein`, plus intra-tier spread:

1. For each tier, compute the Euclidean centroid of its vectors in 128d PCA space (pre-projection).
2. For each vector, compute its L2 distance from its tier's centroid.
3. Normalize distances to [0, 1] within each tier.
4. Apply: `scale = tier_center + BAND_WIDTH * (normalized_distance - 0.5)`
   - `BAND_WIDTH = 0.10` (each band spans +/-0.05 around center)
   - Vectors near centroid land at band center; outliers land at band edges.

### Collection Matrix

Fixed curvature c=5.0 (winner from round 1), vary projection strategy:

| Collection | Distance | Curvature | Projection | Purpose |
|---|---|---|---|---|
| `wos_cosine` | Cosine | -- | -- | Flat baseline |
| `wos_c50_uniform` | Poincare | 5.0 | uniform | Existing approach baseline |
| `wos_c50_static` | Poincare | 5.0 | static | Hardcoded radii |
| `wos_c50_einstein` | Poincare | 5.0 | einstein | Data-driven radii |
| `wos_c50_spread` | Poincare | 5.0 | einstein_spread | Data-driven + variance |

5 collections total. The old c=0.5/1.0/2.0 collections are dropped (already shown inferior).

### Benchmark Changes

**Collection discovery:** Replace hardcoded `COLLECTIONS` list with dynamic discovery. Scan
for `wos_*` collections, auto-detect distance type and curvature from collection config.

**New metric — Band Overlap:**
```
band_overlap = fraction of vectors whose Busemann depth falls
               in a different tier's band (using median thresholds
               between adjacent tiers)
```

Complements the existing `separation_ratio`. Expected values:
- `uniform`: high overlap (all on one shell)
- `static`/`einstein`/`einstein_spread`: near-zero overlap (clean radial bands)

**Comparison table:** Adapts to print whatever collections exist, grouped by projection
strategy. Same columns (SepRatio, AreaRec@10, DomRec@10, H-Prec, p95) plus `BandOvlp`.

**No changes** to retrieval quality or latency benchmarks.

### File Structure

**New: `dev/testbench/hyperbolic_math.py`**

Shared math extracted from both scripts plus new implementations:
- From embed.py: `exp_map_at_origin`, `project_to_ball`
- From benchmark.py: `poincare_to_lorentz`, `lorentz_inner`, `busemann_score`,
  `compute_focal_direction`
- New Python implementations (ported from Rust `poincare_math.rs`): `lorentz_to_poincare`,
  `project_hyperboloid`
- New: `einstein_midpoint(points_poincare, c) -> midpoint_poincare`

**New: `dev/testbench/projection.py`**

The four projection strategy functions:
- `project_uniform(vectors, tiers, c) -> {"vectors": ndarray, "metadata": dict}`
- `project_static(vectors, tiers, c) -> {"vectors": ndarray, "metadata": dict}`
- `project_einstein(vectors, tiers, c) -> {"vectors": ndarray, "metadata": dict}`
- `project_einstein_spread(vectors, tiers, c, band_width=0.10) -> {"vectors": ndarray, "metadata": dict}`

Each returns projected vectors plus metadata (derived radii, band widths, per-tier norm
stats) for logging and analysis.

**Modified: `dev/testbench/embed.py`**

- Import from `hyperbolic_math` and `projection`
- Remove inline Poincare math (now in `hyperbolic_math.py`)
- Add `--strategy` CLI arg:
  - `--strategy all` (default): run all 4 strategies
  - `--strategy uniform,einstein`: run specific subset
- Collection creation loop iterates over strategies at c=5.0 instead of curvatures
- Cosine baseline collection unchanged

**Modified: `dev/testbench/benchmark.py`**

- Import shared math from `hyperbolic_math`
- Remove inline Lorentz/Busemann math (now in `hyperbolic_math.py`)
- Dynamic collection discovery instead of hardcoded list
- Add band overlap metric to hierarchy separation benchmark
- Adapt comparison table for strategy-based columns

**Unchanged:** `Dockerfile`, `docker-compose.yml`, `run.sh`, `download_dataset.py`,
`requirements.txt`, all Rust code.

## Success Criteria

1. **Busemann separation ratio > 10** for at least one tier-aware strategy (up from ~6.4 with uniform)
2. **Band overlap < 0.05** for `einstein` and `einstein_spread` (clean tier separation)
3. **Area recall@10 improvement** over cosine baseline (0.184) by at least 20% for best strategy
4. **Latency remains < 50ms** p95 at ef=128 (no regression from projection changes)
5. **Einstein-derived radii** produce sensible monotonic ordering without manual correction

## Non-Goals

- Rust-side changes to Qdrant fork
- Pythia integration (Phase 3)
- Curvature sweeps (can revisit if results are surprising)
- Band width hyperparameter sweep (use 0.10, only sweep if results are ambiguous)
- Themes tier (not yet available in Pythia)
