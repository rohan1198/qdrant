# EurLex Round 1 Analysis: Multi-label Legal Domain

**Date:** 2026-04-09
**Dataset:** Multi-EurLex (17K EU legal documents, 3-level EUROVOC hierarchy, 556 L3 labels)

## Key Finding: Domain Semantics Trumps Hierarchy Depth

EurLex shows poor hierarchy separation (sep_ratio 1.45, tier_accuracy 94.5%)
because legal documents use similar language regardless of topic level.
Einstein midpoint radii are nearly identical across tiers (0.411/0.415/0.416).

BUT Poincare still slightly beats cosine on flat retrieval (0.477 vs 0.472 area,
0.705 vs 0.667 domain = +5.7% domain recall). The exponential volume of
hyperbolic space helps with the many-label problem (556 L3 categories).

## Cross-Dataset Comparison

| Dataset | Domain | Labels | Cosine Area | Poincare Area | Sep Ratio | Child Recall |
|---------|--------|--------|-------------|---------------|-----------|-------------|
| WOS | Academic CS | 141 | 0.210 | 0.203 | 11.42 | 29.4% |
| BGC | Book genres | 146 | 0.584 | 0.586 | 13.51 | 90.9% |
| EurLex | EU legal | 556 | 0.472 | 0.477 | 1.45 | 0.8% |

## Implication for Pythia

Pythia's hierarchy (Content > Stories > Narratives) has natural semantic
distinctness — posts are short/specific, narratives are abstract/persistent.
This is more like BGC (distinct genres) than EurLex (homogeneous legal text).
Hyperbolic geometry should work well for Pythia.
