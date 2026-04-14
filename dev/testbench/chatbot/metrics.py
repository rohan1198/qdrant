"""
Hierarchy-awareness metrics for chatbot retrieval evaluation.

All functions are pure, with no external dependencies beyond stdlib math and
collections.Counter. All return 0 or 0.0 for empty inputs.
"""

import math
from collections import Counter


def ancestor_precision(result_ids: list, ancestor_ids: set) -> float:
    """Fraction of results that are actual hierarchy ancestors of the query."""
    if not result_ids:
        return 0.0
    return sum(1 for r in result_ids if r in ancestor_ids) / len(result_ids)


def ancestor_hit_rate(result_ids: list, ancestor_ids: set) -> float:
    """Binary: did we find ANY ancestor in the results? 1.0 = yes, 0.0 = no."""
    if not result_ids or not ancestor_ids:
        return 0.0
    return 1.0 if any(r in ancestor_ids for r in result_ids) else 0.0


def subtree_coherence(result_ids: list, subtree_ids: set) -> float:
    """Fraction of results belonging to the query's subtree."""
    if not result_ids:
        return 0.0
    return sum(1 for r in result_ids if r in subtree_ids) / len(result_ids)


def depth_coverage(result_tiers: list, all_tiers: set) -> float:
    """Fraction of hierarchy levels represented in results."""
    if not all_tiers:
        return 0.0
    represented = set(result_tiers) & all_tiers
    return len(represented) / len(all_tiers)


def depth_distribution_entropy(result_tiers: list) -> float:
    """Shannon entropy of tier distribution (H = -sum(p_tier * log(p_tier)))."""
    if not result_tiers:
        return 0.0
    counts = Counter(result_tiers)
    n = len(result_tiers)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def sibling_recall(result_ids: list, sibling_ids: set, k: int = 10) -> float:
    """Fraction of top-k results that are true siblings."""
    top_k = result_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in sibling_ids)
    return hits / len(top_k)


def intruder_rate(result_parent_ids: list, query_parent_id: str) -> float:
    """Fraction of results from a different parent branch."""
    if not result_parent_ids:
        return 0.0
    intruders = sum(1 for p in result_parent_ids if p != query_parent_id)
    return intruders / len(result_parent_ids)


def depth_precision(result_tiers: list, target_tier: str) -> float:
    """Fraction of results at the target tier."""
    if not result_tiers:
        return 0.0
    return sum(1 for t in result_tiers if t == target_tier) / len(result_tiers)


def branch_diversity(result_branches: list) -> int:
    """Number of distinct narrative branches in results."""
    if not result_branches:
        return 0
    return len(set(result_branches))


def branch_coverage(result_branches: list, all_branches: set) -> float:
    """Fraction of relevant branches represented in results."""
    if not all_branches:
        return 0.0
    represented = set(result_branches) & all_branches
    return len(represented) / len(all_branches)


def redundancy_ratio(result_branches: list) -> float:
    """
    Fraction of results sharing a branch with another result.

    Lower is better — a value of 0.0 means every result is from a unique branch.
    """
    if not result_branches:
        return 0.0
    counts = Counter(result_branches)
    redundant = sum(c for c in counts.values() if c > 1)
    return redundant / len(result_branches)
