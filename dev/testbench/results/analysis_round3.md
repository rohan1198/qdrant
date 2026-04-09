# Round 3 Analysis: Unified Collection with Named Vectors

**Date:** 2026-04-09
**Branch:** `feat/hyperbolic-vector-support`
**Dataset:** WOS (45,862 docs, 10 domains, 337 areas)

## Changes from Round 2

1. Fixed Rust `lorentz_to_poincare` — added `sqrt(c)` divisor for configurable curvature
2. Created unified collection `wos_unified` with named vectors (cosine 128d + poincare 128d)
3. Generated tier centroids: 10 domain (narrative) + 337 area (story) via Einstein midpoint
4. Added Busemann depth to payload (client-side computation)
5. Implemented RRF fusion for dual-space search
6. Deleted legacy per-strategy collections

## Collection Summary

| Collection | Points | Vectors | Distance |
|-----------|--------|---------|----------|
| wos_cosine | 45,862 | 128d | Cosine |
| wos_unified | 46,209 | 128d cosine + 128d poincare | Cosine + Poincare |

Unified collection tiers:
- Content (papers): 45,862 points (IDs 0-45861)
- Story (areas): 337 points (IDs 100000+)
- Narrative (domains): 10 points (IDs 200000+)

## Busemann Depth Distribution

| Tier | Mean Depth | Count |
|------|-----------|-------|
| Narrative (domains) | 0.4602 | 10 |
| Story (areas) | 0.7371 | 337 |
| Content (papers) | 1.2302 | 45,862 |

Correct ordering: narratives near origin (abstract), content at edge (concrete).

## Benchmark Results

### 1. Hierarchy Separation (wos_unified, poincare)

| Metric | Round 3 (Unified) | Round 2 (einstein_spread) |
|--------|-------------------|--------------------------|
| sep_ratio | 9.523 | 20.435 |
| band_overlap | 0.010 | 0.001 |

Sep ratio dropped from 20.4 to 9.5. Likely cause: the 347 story/narrative centroids
(at lower Busemann depths) compress the overall depth distribution, reducing the
ratio |avg_leaf - avg_root| / std_all.

### 2. Retrieval Quality

| Metric | Cosine Baseline | Unified (Poincare) | Round 2 (einstein_spread) |
|--------|----------------|---------------------|--------------------------|
| area_recall@10 | 0.209 | 0.178 | 0.216 |
| domain_recall@10 | 0.565 | 0.504 | 0.583 |
| h_precision | 0.387 | 0.341 | 0.399 |

Poincare retrieval dropped in the unified collection compared to both cosine baseline
and Round 2's separate einstein_spread collection.

Possible causes:
1. HNSW graph includes 347 non-content nodes — may distort graph neighborhoods
2. Tier centroids at low radius create "attractors" that pull graph traversal paths
3. No tier filter applied during retrieval benchmark — story/narrative nodes can
   appear in results and displace relevant content neighbors

### 3. Cross-Tier Retrieval (NEW)

| Metric | Value |
|--------|-------|
| parent_story_recall@5 | 0.425 |
| parent_narrative_recall@5 | 0.180 |
| child_story_recall | 0.294 |

First-ever cross-tier navigation results. 42.5% of paper queries find their parent
area centroid in top-5 results — a strong signal that the hyperbolic geometry
encodes hierarchy correctly.

### 4. Dual-Space Comparison (NEW)

| Mode | AreaRec@10 | DomRec@10 |
|------|-----------|-----------|
| cosine_only | 0.2204 | 0.6002 |
| poincare_only | 0.2188 | 0.5992 |
| dual_rrf (k=60) | 0.2002 | 0.5428 |

RRF underperforms both individual spaces. The two vector spaces produce highly
correlated rankings for this dataset, so fusion dilutes rather than diversifies.
Possible improvements:
- Lower k constant (e.g., k=10 or k=20) for more aggressive re-ranking
- Weight the spaces differently (alpha-blended linear fusion instead of RRF)
- Use Busemann depth as a third signal in the fusion

### 5. Latency

| Collection | p95 @ef128 |
|-----------|-----------|
| wos_cosine | 2.4ms |
| wos_unified (poincare) | 2.9ms |
| Round 2 (einstein_spread) | 6.1ms |

Unified collection is faster than Round 2's separate poincare collection (2.9ms vs 6.1ms).

## Key Insights

1. **Hierarchy encoding works** — Busemann depth clearly separates tiers, cross-tier
   retrieval finds parent/child nodes with meaningful recall.

2. **Flat retrieval suffers in unified collection** — mixing tiers in one HNSW graph
   hurts content-to-content retrieval quality. The Round 2 separate collection
   (content only) performed better for flat recall.

3. **RRF fusion needs tuning** — the naive RRF with k=60 and equal weighting doesn't
   help. The spaces are too correlated for rank-based fusion to add value. Consider:
   - Score-based fusion with alpha weighting
   - Busemann depth as a fusion signal
   - Tier-filtered search (restrict to content tier) before fusion

4. **Curvature re-sweep needed** — the unified topology with centroids may have
   a different optimal curvature than c=5.0.

## Open Questions

- Would tier-filtered HNSW (payload_m on "tier" field) improve content retrieval?
- Would a different curvature better balance hierarchy separation and retrieval quality?
- Should RRF be replaced with learned/weighted fusion?
- Is the sep_ratio drop just a measurement artifact from including centroid points?
