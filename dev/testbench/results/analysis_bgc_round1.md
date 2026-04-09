# BGC Round 1 Analysis: 4-Level Hierarchy Validation

**Date:** 2026-04-09
**Dataset:** BlurbGenreCollection (91,894 book blurbs, 4-level genre hierarchy)

## Hierarchy Metrics (Excellent)

| Metric | BGC (4 levels) | WOS (3 levels) |
|--------|---------------|----------------|
| sep_ratio_content | **13.51** | 11.42 |
| cross_tier_sep | **2.42** | 1.97 |
| band_overlap | **0.003** | 0.010 |
| tier_accuracy | **99.72%** | 98.99% |
| child_story_recall | **90.9%** | 29.4% |

## Multi-Mode Retrieval

| Mode | BGC AreaRec@10 | BGC DomRec@10 |
|------|---------------|---------------|
| cosine baseline | 0.584 | 0.808 |
| tier_filtered (Poincare) | 0.585 | 0.816 |
| depth_range (Poincare) | 0.586 | 0.813 |
| cosine_tier_filtered | 0.589 | 0.818 |

Poincare matches cosine for flat retrieval on BGC. No significant advantage or disadvantage.

## Key Finding: The Value Is Hierarchy, Not Flat Retrieval

The hyperbolic advantage depends on query type:
- Flat retrieval (similar books): Cosine and Poincare are tied
- Hierarchy drill-down (narrative -> stories): 90.9% child recall — cosine cannot do this
- Cross-tier navigation: unique capability of hyperbolic geometry

BGC genres are semantically very distinct (Fiction vs Nonfiction vs Children's).
Cosine already separates them well, leaving little room for hyperbolic improvement.
WOS academic sub-areas are semantically subtle, which is where hyperbolic helps more.

## Busemann Depth Distribution

| Tier | BGC | WOS |
|------|-----|-----|
| Narrative | 0.354 | 0.460 |
| Story | 0.697 | 0.737 |
| Content | 1.243 | 1.230 |
