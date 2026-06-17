"""
Retrieval evaluation metrics — pure functions, no Qdrant dependency.

All functions take:
  retrieved_ids : list of doc IDs in rank order (rank 1 first)
  relevant_ids  : set/list of ground-truth relevant doc IDs

Conventions
-----------
- k=0 means "use all retrieved results"
- Scores are in [0, 1] unless noted
- Empty retrieved list → 0.0 for all metrics
- Empty relevant set  → 0.0 for all metrics (undefined, treated as zero)
"""
from __future__ import annotations

import math
from typing import Union


def _to_set(ids) -> set:
    return set(ids) if not isinstance(ids, set) else ids


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def precision_at_k(
    retrieved_ids: list,
    relevant_ids: Union[list, set],
    k: int = 10,
) -> float:
    """Fraction of top-k retrieved that are relevant."""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    rel = _to_set(relevant_ids)
    top_k = retrieved_ids[:k] if k else retrieved_ids
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r in rel) / len(top_k)


def recall_at_k(
    retrieved_ids: list,
    relevant_ids: Union[list, set],
    k: int = 10,
) -> float:
    """Fraction of all relevant docs that appear in top-k."""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    rel = _to_set(relevant_ids)
    top_k = retrieved_ids[:k] if k else retrieved_ids
    return sum(1 for r in top_k if r in rel) / len(rel)


def reciprocal_rank(
    retrieved_ids: list,
    relevant_ids: Union[list, set],
) -> float:
    """1 / rank of the first relevant result. 0 if none found."""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    rel = _to_set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def average_precision(
    retrieved_ids: list,
    relevant_ids: Union[list, set],
) -> float:
    """
    Average Precision (AP) — area under the precision-recall curve.
    Used to compute MAP across queries.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    rel = _to_set(relevant_ids)
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in rel:
            hits += 1
            total += hits / rank
    return total / len(rel) if rel else 0.0


def ndcg_at_k(
    retrieved_ids: list,
    relevant_ids: Union[list, set],
    k: int = 10,
) -> float:
    """
    Normalised Discounted Cumulative Gain at k.
    Binary relevance (1 if relevant, 0 otherwise).
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    rel = _to_set(relevant_ids)
    top_k = retrieved_ids[:k] if k else retrieved_ids

    def dcg(ids: list) -> float:
        return sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(ids, start=1)
            if doc_id in rel
        )

    actual = dcg(top_k)
    # Ideal: put all relevant docs first
    n_ideal = min(len(rel), len(top_k))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_ideal + 1))
    return actual / ideal if ideal > 0 else 0.0


def f1_at_k(
    retrieved_ids: list,
    relevant_ids: Union[list, set],
    k: int = 10,
) -> float:
    """Harmonic mean of P@k and R@k."""
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregate over a list of queries
# ---------------------------------------------------------------------------

def mean_metrics(
    results: list[dict],
    k: int = 10,
) -> dict[str, float]:
    """
    Compute mean metrics from a list of per-query result dicts.

    Each result dict must have:
      retrieved_ids : list of doc IDs in rank order
      relevant_ids  : list/set of ground-truth relevant IDs

    Returns dict with keys: P@k, R@k, F1@k, MRR, MAP, NDCG@k
    """
    if not results:
        return {f"P@{k}": 0.0, f"R@{k}": 0.0, f"F1@{k}": 0.0,
                "MRR": 0.0, "MAP": 0.0, f"NDCG@{k}": 0.0}

    p, r, f1, rr, ap, ndcg = [], [], [], [], [], []
    for res in results:
        rid = res["retrieved_ids"]
        rel = res["relevant_ids"]
        p.append(precision_at_k(rid, rel, k))
        r.append(recall_at_k(rid, rel, k))
        f1.append(f1_at_k(rid, rel, k))
        rr.append(reciprocal_rank(rid, rel))
        ap.append(average_precision(rid, rel))
        ndcg.append(ndcg_at_k(rid, rel, k))

    def mean(xs):
        return sum(xs) / len(xs)

    return {
        f"P@{k}":    round(mean(p), 4),
        f"R@{k}":    round(mean(r), 4),
        f"F1@{k}":   round(mean(f1), 4),
        "MRR":       round(mean(rr), 4),
        "MAP":       round(mean(ap), 4),
        f"NDCG@{k}": round(mean(ndcg), 4),
    }
