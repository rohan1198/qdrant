"""
Search strategies — all 10 patterns from the testbench, adapted to use
HyperbolicClient and the encoder ABC.

Every strategy returns List[ScoredPoint] and accepts a PipelineConfig,
so the calling code never needs to know which strategy is active.

Design: strategies are pure functions — no state, no class hierarchy.
"""
from __future__ import annotations

import logging
from typing import Optional, Any

import numpy as np
from qdrant_client.http import models as rest

from hyperbolic.client import (
    HyperbolicClient, DENSE_VECTOR, POINCARE_VECTOR, TANGENT_VECTOR
)
from hyperbolic.math.poincare import (
    poincare_dist_batch, poincare_to_klein_batch, alpha_precompute_batch,
    busemann_score_batch, busemann_score_multi_xi, horoball_score_batch,
)
from hyperbolic.pipeline.config import PipelineConfig
from hyperbolic.search.intent import QueryIntentDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector extraction helpers — eliminates copy-paste across all re-rankers
# ---------------------------------------------------------------------------

def _get_poincare_vec(p: rest.ScoredPoint) -> list:
    """Extract the Poincaré vector from a ScoredPoint regardless of format."""
    return p.vector[POINCARE_VECTOR] if isinstance(p.vector, dict) else p.vector


def _get_tangent_vec(p: rest.ScoredPoint) -> list:
    """Extract the tangent vector from a ScoredPoint, returning [] if absent."""
    if isinstance(p.vector, dict):
        return p.vector.get(TANGENT_VECTOR, [])
    return []


def _poincare_vecs(candidates: list[rest.ScoredPoint]) -> np.ndarray:
    """Stack Poincaré vectors for a list of candidates into (N, D) float32."""
    return np.array([_get_poincare_vec(p) for p in candidates], dtype=np.float32)


# ---------------------------------------------------------------------------
# Narrative-level anchor selection — shared by busemann and entailment
# ---------------------------------------------------------------------------

def _select_narrative_anchors(
    query_poincare: np.ndarray,
    candidates: list[rest.ScoredPoint],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[rest.ScoredPoint]]:
    """For narrative-level queries: split candidates into story anchors and content docs.

    Returns ``(selected_story_vecs, content_vecs, content_candidates)``.
    ``selected_story_vecs`` is ``None`` when no story-tier docs appear in the pool.
    """
    story_pts = [p for p in candidates if (p.payload or {}).get("tier") == "story"]
    content_candidates = [p for p in candidates if (p.payload or {}).get("tier") == "content"]

    if not story_pts or not content_candidates:
        return None, None, content_candidates

    story_vecs = _poincare_vecs(story_pts)
    q_unit = query_poincare / (np.linalg.norm(query_poincare) + 1e-8)
    story_cosines = story_vecs @ q_unit.astype(np.float64)
    n_anchors = min(max(1, len(story_pts) // 2), 3)   # top half, max 3
    top_idx = np.argsort(-story_cosines)[:n_anchors]
    selected_story_vecs = story_vecs[top_idx]

    content_vecs = _poincare_vecs(content_candidates)
    return selected_story_vecs, content_vecs, content_candidates


# ---------------------------------------------------------------------------
# Exact client-side Poincaré re-ranking
# ---------------------------------------------------------------------------

def _exact_poincare_rerank(
    query_vec: np.ndarray,
    candidates: list[rest.ScoredPoint],
    c: float,
    limit: int,
) -> list[rest.ScoredPoint]:
    """Re-rank candidates by exact Poincaré distance."""
    vecs = _poincare_vecs(candidates)
    dists = poincare_dist_batch(query_vec, vecs, c)
    order = np.argsort(dists)[:limit]
    return [
        rest.ScoredPoint(
            id=candidates[i].id, version=candidates[i].version,
            score=-float(dists[i]),   # negative distance = higher score
            payload=candidates[i].payload, vector=candidates[i].vector,
        )
        for i in order
    ]


# ---------------------------------------------------------------------------
# Busemann re-ranking (replaces exact Poincaré re-rank)
# ---------------------------------------------------------------------------

def _busemann_rerank(
    query_vec: np.ndarray,
    candidates: list[rest.ScoredPoint],
    c: float,
    limit: int,
    radius_sigma: Optional[float] = None,
) -> list[rest.ScoredPoint]:
    """Re-rank candidates by Busemann score relative to query's ideal boundary point."""
    vecs = _poincare_vecs(candidates)
    scores = busemann_score_batch(query_vec, vecs, c, radius_sigma=radius_sigma)
    order = np.argsort(-scores)[:limit]
    return [
        rest.ScoredPoint(
            id=candidates[i].id, version=candidates[i].version,
            score=float(scores[i]),
            payload=candidates[i].payload, vector=candidates[i].vector,
        )
        for i in order
    ]


def _entailment_rerank(
    query_vec: np.ndarray,
    candidates: list[rest.ScoredPoint],
    c: float,
    horo_weight: float,
    limit: int,
    radius_sigma: Optional[float] = None,
) -> list[rest.ScoredPoint]:
    """Re-rank by combined Busemann (with radius penalty) + horoball proximity score."""
    vecs = _poincare_vecs(candidates)
    bus = busemann_score_batch(query_vec, vecs, c, radius_sigma=radius_sigma)
    horo = horoball_score_batch(query_vec, vecs, c)
    scores = bus + horo_weight * horo
    order = np.argsort(-scores)[:limit]
    return [
        rest.ScoredPoint(
            id=candidates[i].id, version=candidates[i].version,
            score=float(scores[i]),
            payload=candidates[i].payload, vector=candidates[i].vector,
        )
        for i in order
    ]


# ---------------------------------------------------------------------------
# Alpha pipeline (alpha proxy → exact Poincaré)
# ---------------------------------------------------------------------------

def alpha_pipeline(
    client: HyperbolicClient,
    collection: str,
    query_dense: np.ndarray,
    query_poincare: np.ndarray,
    *,
    config: PipelineConfig,
    limit: int = 10,
    query_filter: Optional[rest.Filter] = None,
) -> list[rest.ScoredPoint]:
    """Stage 1: cosine ANN → Stage 2: alpha proxy re-rank → Stage 3: exact Poincaré."""
    prefetch_limit, alpha_top = config.stage_sizes

    # Stage 1
    candidates = client.search(
        collection, query_dense,
        using=DENSE_VECTOR, limit=prefetch_limit, ef=config.ef,
        query_filter=query_filter, with_vectors=True,
    )
    if not candidates:
        return []

    # Stage 2: alpha proxy
    vecs = np.array([_get_poincare_vec(p) or [] for p in candidates], dtype=np.float32)
    if vecs.ndim < 2 or vecs.shape[1] == 0:
        return candidates[:limit]

    query_alpha = 1.0 / (1.0 - config.curvature * float(np.dot(query_poincare, query_poincare)) + 1e-5)
    cand_alphas = alpha_precompute_batch(vecs, config.curvature)
    diff_sq = np.sum((vecs - query_poincare)**2, axis=-1)
    proxy = diff_sq * query_alpha * cand_alphas
    top_idx = np.argsort(proxy)[:alpha_top]
    stage2 = [candidates[i] for i in top_idx]

    # Stage 3: exact Poincaré
    return _exact_poincare_rerank(query_poincare, stage2, config.curvature, limit)


# ---------------------------------------------------------------------------
# Tangent pipeline (tangent HNSW → exact Poincaré)
# ---------------------------------------------------------------------------

def tangent_pipeline(
    client: HyperbolicClient,
    collection: str,
    query_tangent: np.ndarray,
    query_poincare: np.ndarray,
    *,
    config: PipelineConfig,
    limit: int = 10,
    query_filter: Optional[rest.Filter] = None,
) -> list[rest.ScoredPoint]:
    """Stage 1: dense cosine ANN → Stage 2: client-side tangent proximity → Stage 3: exact Poincaré."""
    prefetch = limit * config.prune_factor

    candidates = client.search(
        collection, query_poincare,
        using=POINCARE_VECTOR, limit=prefetch, ef=config.ef,
        query_filter=query_filter, with_vectors=True,
    )
    if not candidates:
        return []

    tang_vecs = np.array([_get_tangent_vec(p) or [] for p in candidates], dtype=np.float32)
    if tang_vecs.ndim == 2 and tang_vecs.shape[1] > 0:
        tang_dists = np.sum((tang_vecs - query_tangent) ** 2, axis=-1)
        top_idx = np.argsort(tang_dists)[:prefetch]
        candidates = [candidates[i] for i in top_idx]

    return _exact_poincare_rerank(query_poincare, candidates, config.curvature, limit)


# ---------------------------------------------------------------------------
# Combined pipeline (cosine + tangent → alpha → exact Poincaré)
# ---------------------------------------------------------------------------

def combined_pipeline(
    client: HyperbolicClient,
    collection: str,
    query_dense: np.ndarray,
    query_poincare: np.ndarray,
    query_tangent: np.ndarray,
    *,
    config: PipelineConfig,
    limit: int = 10,
    query_filter: Optional[rest.Filter] = None,
) -> list[rest.ScoredPoint]:
    """Stage 1: dense + poincaré cosine ANN → merge → Stage 2: exact Poincaré."""
    cosine_limit, poincare_limit = config.stage_sizes

    cosine_candidates = client.search(
        collection, query_dense,
        using=DENSE_VECTOR, limit=cosine_limit, ef=config.ef,
        query_filter=query_filter, with_vectors=True,
    )
    poincare_candidates = client.search(
        collection, query_poincare,
        using=POINCARE_VECTOR, limit=poincare_limit, ef=config.ef,
        query_filter=query_filter, with_vectors=True,
    )

    seen: dict[Any, rest.ScoredPoint] = {}
    for p in cosine_candidates + poincare_candidates:
        if p.id not in seen:
            seen[p.id] = p
    merged = list(seen.values())

    if not merged:
        return []

    return _exact_poincare_rerank(query_poincare, merged, config.curvature, limit)


# ---------------------------------------------------------------------------
# Klein pipeline (fast proxy)
# ---------------------------------------------------------------------------

def klein_pipeline(
    client: HyperbolicClient,
    collection: str,
    query_dense: np.ndarray,
    query_poincare: np.ndarray,
    *,
    config: PipelineConfig,
    limit: int = 10,
    query_filter: Optional[rest.Filter] = None,
) -> list[rest.ScoredPoint]:
    """Stage 1: cosine → Stage 2: Klein proxy → Stage 3: exact Poincaré."""
    prefetch_limit, klein_top = config.stage_sizes

    candidates = client.search(
        collection, query_dense,
        using=DENSE_VECTOR, limit=prefetch_limit, ef=config.ef,
        query_filter=query_filter, with_vectors=True,
    )
    if not candidates:
        return []

    vecs = _poincare_vecs(candidates)
    if vecs.shape[-1] == 0:
        return candidates[:limit]

    vecs_klein = poincare_to_klein_batch(vecs, config.curvature)
    q_klein = 2.0 * query_poincare / (1.0 + config.curvature * float(np.dot(query_poincare, query_poincare)))
    klein_dists = np.sum((vecs_klein - q_klein)**2, axis=-1)
    top_idx = np.argsort(klein_dists)[:klein_top]
    stage2 = [candidates[i] for i in top_idx]

    return _exact_poincare_rerank(query_poincare, stage2, config.curvature, limit)


# ---------------------------------------------------------------------------
# Busemann pipeline (cosine ANN → Busemann re-rank)
# No tier filter required — the geometry suppresses ancestor docs naturally.
# ---------------------------------------------------------------------------

def busemann_pipeline(
    client: HyperbolicClient,
    collection: str,
    query_dense: np.ndarray,
    query_poincare: np.ndarray,
    *,
    config: PipelineConfig,
    limit: int = 10,
    query_filter: Optional[rest.Filter] = None,
) -> list[rest.ScoredPoint]:
    """Stage 1: cosine ANN → Stage 2: Busemann score re-rank.

    Unlike Poincaré distance, the Busemann score naturally suppresses ancestor
    documents (low radius, same direction) without an explicit tier filter:
    a theme at r=0.15 in the query direction scores ~0.3, while a content peer
    at r=0.90 scores ~3.0.  No hard filter needed.
    """
    prefetch_limit = config.stage_sizes[0]
    candidates = client.search(
        collection, query_dense,
        using=DENSE_VECTOR, limit=prefetch_limit, ef=config.ef,
        query_filter=query_filter, with_vectors=True,
    )
    if not candidates:
        return []

    query_norm = float(np.linalg.norm(query_poincare))

    if query_norm < 0.60:
        # Narrative-level descendant retrieval.
        # The narrative's ξ is the centroid of 10+ diverse descendants — it sits
        # ~60° off each one.  Use story-tier docs in the candidate pool as M ideal
        # points instead: each story's ξ_k points into one content sub-cluster,
        # so content descendants near ANY story anchor score well.
        story_vecs, content_vecs, content_candidates = _select_narrative_anchors(
            query_poincare, candidates
        )
        if story_vecs is not None:
            scores = busemann_score_multi_xi(story_vecs, content_vecs, config.curvature)
            order = np.argsort(-scores)[:limit]
            return [
                rest.ScoredPoint(
                    id=content_candidates[i].id, version=content_candidates[i].version,
                    score=float(scores[i]),
                    payload=content_candidates[i].payload,
                    vector=content_candidates[i].vector,
                )
                for i in order
            ]
        # Fallback: no story anchors in pool — use query's own ξ, no sigma
        return _busemann_rerank(query_poincare, candidates, config.curvature, limit,
                                radius_sigma=None)

    # Peer mode (r≥0.80) or story-level (0.60≤r<0.80): single-ξ Busemann.
    # Disable sigma in story-level descendant mode (target content docs are deeper).
    sigma = getattr(config, "busemann_sigma", 0.10) if query_norm >= 0.80 else None
    return _busemann_rerank(
        query_poincare, candidates, config.curvature, limit,
        radius_sigma=sigma,
    )


# ---------------------------------------------------------------------------
# Entailment pipeline (cosine ANN → Busemann pre-filter → entailment re-rank)
# ---------------------------------------------------------------------------

def entailment_pipeline(
    client: HyperbolicClient,
    collection: str,
    query_dense: np.ndarray,
    query_poincare: np.ndarray,
    *,
    config: PipelineConfig,
    limit: int = 10,
    query_filter: Optional[rest.Filter] = None,
) -> list[rest.ScoredPoint]:
    """Stage 1: cosine ANN → Stage 2: Busemann filter → Stage 3: entailment re-rank.

    The entailment cone at the query point selects documents that are
    "more specific" instances of the query concept — geometrically, those
    that lie inside the hyperbolic cone opening from the query toward
    the ball boundary in the query's direction.
    """
    prefetch_limit, entailment_top = config.stage_sizes
    cone_weight = getattr(config, "cone_weight", 2.0)

    candidates = client.search(
        collection, query_dense,
        using=DENSE_VECTOR, limit=prefetch_limit, ef=config.ef,
        query_filter=query_filter, with_vectors=True,
    )
    if not candidates:
        return []

    query_norm = float(np.linalg.norm(query_poincare))

    if query_norm < 0.60:
        # Narrative-level descendant retrieval: same multi-ξ strategy as busemann_pipeline.
        story_vecs, content_vecs, content_candidates = _select_narrative_anchors(
            query_poincare, candidates
        )
        if story_vecs is not None:
            # Stage 2: multi-ξ Busemann pre-filter → top entailment_top content candidates
            multi_scores = busemann_score_multi_xi(story_vecs, content_vecs, config.curvature)
            top_idx = np.argsort(-multi_scores)[:entailment_top]
            stage2 = [content_candidates[i] for i in top_idx]

            # Stage 3: horoball re-rank using the narrative's own ξ for depth alignment
            return _entailment_rerank(
                query_poincare, stage2, config.curvature,
                horo_weight=cone_weight, limit=limit,
                radius_sigma=None,
            )
        # Fallback: no story anchors in pool
        return _busemann_rerank(query_poincare, candidates, config.curvature, limit,
                                radius_sigma=None)

    # Peer mode (r≥0.80) or story-level (0.60≤r<0.80): single-ξ Busemann.
    sigma = getattr(config, "busemann_sigma", 0.10) if query_norm >= 0.80 else None

    # Stage 2: Busemann pre-filter to top directional candidates
    stage2 = _busemann_rerank(query_poincare, candidates, config.curvature, entailment_top,
                               radius_sigma=sigma)

    # Stage 3: horoball proximity re-rank (combined Busemann + horoball)
    return _entailment_rerank(
        query_poincare, stage2, config.curvature,
        horo_weight=cone_weight, limit=limit,
        radius_sigma=sigma,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _tier_filter(tier: str, base_filter: Optional[rest.Filter] = None) -> rest.Filter:
    """
    Build a Qdrant filter for a tier label or sentinel.

    Sentinels:
      "content"          → tier == "content"
      "content_or_story" → tier in ["content", "story"]
    """
    if tier == "content_or_story":
        tier_cond = rest.FieldCondition(
            key="tier",
            match=rest.MatchAny(any=["content", "story"]),
        )
    else:
        tier_cond = rest.FieldCondition(
            key="tier",
            match=rest.MatchValue(value=tier),
        )
    if base_filter is None:
        return rest.Filter(must=[tier_cond])
    must = list(base_filter.must or []) + [tier_cond]
    return rest.Filter(must=must)


def _descendant_filter(
    query_poincare: np.ndarray,
    base_filter: Optional[rest.Filter] = None,
) -> Optional[rest.Filter]:
    """
    Return a tier filter that restricts stage-1 ANN to tiers *deeper* than the query.

    Thresholds match the explicit_radial projection defaults:
      r < 0.60  → narrative tier  → retrieve story + content
      0.60 ≤ r < 0.80 → story tier → retrieve content only
      r ≥ 0.80  → content tier (peer) → no filter
    """
    norm = float(np.linalg.norm(query_poincare))
    if norm >= 0.80:
        return base_filter
    elif norm >= 0.60:
        return _tier_filter("content", base_filter)
    else:
        return _tier_filter("content_or_story", base_filter)


def run_search(
    client: HyperbolicClient,
    collection: str,
    query_dense: np.ndarray,
    query_poincare: np.ndarray,
    query_tangent: np.ndarray,
    *,
    config: PipelineConfig,
    limit: int = 10,
    query_filter: Optional[rest.Filter] = None,
    query_tier: Optional[str] = None,
    query_text: Optional[str] = None,
) -> list[rest.ScoredPoint]:
    """Dispatch to the right pipeline based on config.query_pipeline and config.intent."""
    intent_mode = getattr(config, "intent", "auto")

    INTENT_MODES = {"auto", "precision", "recall", "alpha", "text"}
    if intent_mode in INTENT_MODES and query_tier is None:
        detector = QueryIntentDetector(curvature=config.curvature)
        poincare_vec = np.asarray(query_poincare, dtype=np.float32).ravel()
        routing = detector.routing(query_text or "", poincare_vec, mode=intent_mode)
        query_tier = routing["tier_filter"]
        logger.debug(
            "Intent routing: mode=%s intent=%s α=%.3f text_score=%.2f → tier_filter=%s",
            intent_mode, routing["intent"], routing["alpha"], routing["text_score"], query_tier,
        )

    p = config.query_pipeline

    FILTERLESS_PIPELINES = {"busemann", "entailment"}
    if query_tier and p not in FILTERLESS_PIPELINES:
        effective_filter = _tier_filter(query_tier, query_filter)
    else:
        effective_filter = query_filter

    if p == "alpha":
        return alpha_pipeline(client, collection, query_dense, query_poincare,
                              config=config, limit=limit, query_filter=effective_filter)
    if p == "tangent":
        return tangent_pipeline(client, collection, query_tangent, query_poincare,
                                config=config, limit=limit, query_filter=effective_filter)
    if p == "combined":
        return combined_pipeline(client, collection, query_dense, query_poincare, query_tangent,
                                 config=config, limit=limit, query_filter=effective_filter)
    if p == "klein":
        return klein_pipeline(client, collection, query_dense, query_poincare,
                              config=config, limit=limit, query_filter=effective_filter)
    if p == "busemann":
        busemann_filter = _descendant_filter(query_poincare, query_filter)
        return busemann_pipeline(client, collection, query_dense, query_poincare,
                                 config=config, limit=limit, query_filter=busemann_filter)
    if p == "entailment":
        entailment_filter = _descendant_filter(query_poincare, query_filter)
        return entailment_pipeline(client, collection, query_dense, query_poincare,
                                   config=config, limit=limit, query_filter=entailment_filter)
    if p == "cosine_filtered":
        return client.search(collection, query_dense, using=DENSE_VECTOR, limit=limit, ef=config.ef,
                             query_filter=effective_filter)
    # Fallback / baseline: plain cosine
    return client.search(collection, query_dense, using=DENSE_VECTOR, limit=limit, ef=config.ef,
                         query_filter=query_filter)
