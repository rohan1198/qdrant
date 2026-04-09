# SciHTC Reconstruction + 6-Level Deep Testbench

**Date:** 2026-04-09
**Branch:** `feat/hyperbolic-vector-support`
**Status:** Approved
**Sub-project:** A (of A/B split — B covers hierarchical metrics + workflow benchmarks)
**Prerequisite:** Phase 3.5 complete (unified collection architecture validated on WOS)

---

## Overview

Reconstruct the SciHTC dataset (186K ACM papers, 6-level ACM CCS hierarchy, 1,233 categories) via Semantic Scholar API and integrate it into the testbench. This tests whether our hyperbolic approach scales to deeper hierarchies and validates the Euclidean vs Hyperbolic advantage at production depth.

SciHTC's poly-hierarchical nature (papers belong to multiple branches) mirrors Pythia's reality where a single post can support multiple stories and narratives.

## Design Decisions

- **Approach B (Dataset Module + Multi-Label Payload + Adaptive Tiers)**
- **Discovery-first pipeline** — inspect downloaded data before processing, don't assume formats
- **Multi-path labels preserved** — all hierarchy paths stored in payload, evaluation matches against ANY valid path
- **Longest path determines Poincaré position** — deepest path sets Busemann depth, other paths stored for evaluation
- **Adaptive tier assignment** — auto-detects hierarchy depth, maps levels to tiers dynamically
- **Same benchmark suites as WOS** — no new metrics in this sub-project (deferred to Sub-project B)

---

## Section 1: Dataset Reconstruction Pipeline

### Directory Structure

```
dev/
├── datasets/
│   └── scihtc/
│       ├── download.py      # External I/O: Google Drive + Semantic Scholar API
│       ├── prepare.py       # Transform raw data into testbench JSONL format
│       ├── README.md        # Dataset card: source, license, reconstruction steps
│       └── data/            # Output directory (gitignored)
│           ├── raw_papers.jsonl
│           ├── raw_labels.jsonl
│           ├── documents.jsonl
│           ├── hierarchy.json
│           └── manifest.json
└── testbench/               # Existing (modified for --dataset flag)
```

### `download.py` — Discovery-First External I/O

1. Download SciHTC public data from Google Drive (paper IDs + labels for train/test/dev splits)
2. **Inspect downloaded files** — log format (CSV? TSV? JSON?), column names, sample rows, delimiter. Don't assume structure.
3. **Discover paper ID format** — DOIs? ACM internal IDs? Numeric? Log samples and decide Semantic Scholar lookup strategy.
4. **Discover label format** — how are hierarchy paths encoded? Tilde-separated? Slash-separated? Multi-label delimiter?
5. Batch-query Semantic Scholar API for titles + abstracts, with fallback:
   - Primary: DOI lookup
   - Fallback: title-based search if DOI fails
   - Title-only entries for papers without abstracts
6. Save raw outputs:
   - `data/raw_papers.jsonl` — id, title, abstract (nullable), keywords
   - `data/raw_labels.jsonl` — id, all label paths as discovered
   - `data/manifest.json` — format discoveries, coverage stats, API version, timestamp

**Error handling:** If format is unexpected, print detailed diagnostics and exit rather than silently producing garbage. Coverage warnings at <90%, hard fail at <80%.

**Rate limiting:** Conservative approach (5 req/s batch endpoint). Full 186K papers in ~75 seconds of API time.

### `prepare.py` — Transform to Testbench Format

1. Load raw papers + labels from whatever `download.py` produced
2. **Discover hierarchy structure** — build tree from all observed label paths, log:
   - Actual depth (expected ~6 levels)
   - Nodes per level
   - Multi-parent nodes (poly-hierarchy)
   - Branching factor
3. Compute hierarchy depth per paper (max depth across all paths)
4. Select primary path (deepest) for Poincaré embedding position
5. Assign tiers adaptively based on discovered depth
6. Output `data/documents.jsonl` — standard schema compatible with WOS:
   - `id`, `text`, `tier`, `hierarchy_path` (primary), `domain` (L1), `area` (L2)
   - Plus multi-label extensions: `all_paths`, `depth`, `keywords`
7. Output `data/hierarchy.json` — discovered tree structure with node counts

### Schema (documents.jsonl)

```json
{
    "id": "<paper_id>",
    "text": "<title>. <abstract or empty>",
    "tier": "<dynamically assigned>",
    "depth": 5,
    "domain": "<L1 category>",
    "area": "<L2 category>",
    "hierarchy_path": "<primary full path>",
    "all_paths": ["<path1>", "<path2>"],
    "keywords": ["<kw1>", "<kw2>"],
    "has_abstract": true
}
```

The `tier`, `domain`, `area` fields maintain compatibility with WOS. The `all_paths`, `depth`, `keywords`, `has_abstract` fields are extensions for richer evaluation.

---

## Section 2: Testbench Integration

### `embed.py` — Dataset Flag

New `--dataset` argument:
- `--dataset wos` (default) — current behavior, loads from `data/wos/documents.jsonl`
- `--dataset scihtc` — loads from `../datasets/scihtc/data/documents.jsonl`

The flag controls:
- Document source path
- Embedding cache directory (`data/wos/` vs `data/scihtc/`)
- Collection name prefix (`wos_*` vs `scihtc_*`)

Everything else (PCA, projection, Busemann, centroids, upsert) stays the same.

### Adaptive Tier Assignment

Instead of hardcoded tier names, derive from data:
- Read documents.jsonl, find max hierarchy depth
- WOS (depth 3): narrative/story/content (current)
- SciHTC (depth 6): theme/narrative/sub_narrative/story/sub_story/content
- Tier centroids generated at each level via Einstein midpoint
- Deepest level = content (leaves), shallowest = theme (root)

### Multi-Label Payload

For SciHTC, the unified collection payload includes:
- `tier` — based on primary (deepest) path
- `busemann_depth` — computed from Poincaré position
- `hierarchy_path` — primary (deepest) path
- `all_paths` — list of ALL valid hierarchy paths
- `domain`, `area` — L1 and L2 from primary path

For WOS, `all_paths` is a single-element list (backwards compatible).

### Multi-Label Evaluation

Benchmark evaluation treats a retrieval result as correct if the result's `domain`/`area` matches ANY of the query's valid paths. This is controlled by checking `all_paths` intersection rather than exact `area` match only.

### What Doesn't Change

`fusion.py`, `hyperbolic_math.py`, `projection.py`, `run.sh`, Qdrant collection schema (named vectors + payload indices). The core math is dataset-independent.

### `benchmark.py` — Collection Discovery

Update `discover_collections` to detect `scihtc_*` collections alongside `wos_*`. The discovery function checks for any collection matching `{prefix}_cosine` or `{prefix}_unified`. The `--dataset` flag (or auto-detection from collection names) determines which prefix to use.

---

## Section 3: Output and Success Criteria

### Benchmark Output

Same 5 suites as WOS, on `scihtc_unified` collection:

1. **Hierarchy Separation** — sep_ratio_content, cross_tier_sep, tier_accuracy across 6 tiers
2. **Multi-Mode Retrieval** — unfiltered, tier_filtered, depth_range, cosine_tier_filtered
3. **Cross-Tier Retrieval** — parent/child recall across 6 levels
4. **Fusion Strategy Comparison** — alpha sweep, Busemann-weighted, depth-band
5. **Latency** — 186K points + 6-level centroids performance

### Euclidean vs Hyperbolic Comparison

Side-by-side table for BOTH datasets:

```
Dataset   Depth   Metric           Cosine    Poincaré    Delta
WOS       3       area_recall@10   0.209     0.222       +6.2%
SciHTC    6       area_recall@10   ???       ???         ???
```

Key hypothesis: the hyperbolic advantage grows with hierarchy depth.

### Success Criteria

1. **Reconstruction coverage** — recover 80%+ of SciHTC papers with text from Semantic Scholar
2. **Tier separation** — Busemann depth separates 6 tiers with tier_accuracy > 90%
3. **Hyperbolic beats Euclidean** — depth-range Poincaré outperforms cosine baseline on SciHTC
4. **Advantage scales** — hyperbolic delta on SciHTC (6 levels) > delta on WOS (3 levels)

---

## File Changes Summary

| File | Action | Responsibility |
|------|--------|---------------|
| `dev/datasets/scihtc/download.py` | Create | Google Drive download + Semantic Scholar API reconstruction |
| `dev/datasets/scihtc/prepare.py` | Create | Transform raw data to testbench JSONL with hierarchy discovery |
| `dev/datasets/scihtc/README.md` | Create | Dataset card |
| `dev/testbench/embed.py` | Modify | Add --dataset flag, adaptive tier assignment, multi-label payload |
| `dev/testbench/benchmark.py` | Modify | Multi-label evaluation (all_paths matching), adaptive tier names |
