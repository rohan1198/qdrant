"""
EvalHarness — ingest a synthetic corpus and evaluate retrieval pipelines.

Usage::

    corpus, queries = load_corpus_and_queries("data/synthetic")
    harness = EvalHarness.build("http://localhost:6333", corpus)
    report = harness.compare_pipelines(queries, k=10)
    print_report(report)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from hyperbolic.client import HyperbolicClient
from hyperbolic.encoder.pca_encoder import PCAEncoder
from hyperbolic.pipeline.ingest import Ingestor, Document
from hyperbolic.pipeline.config import (
    PipelineConfig,
    ALPHA_PIPELINE, TANGENT_PIPELINE, COMBINED_PIPELINE, KLEIN_PIPELINE,
    DENSE_COSINE_PIPELINE, DENSE_COSINE_FILTERED_PIPELINE,
    BUSEMANN_PIPELINE, ENTAILMENT_PIPELINE,
)
from hyperbolic.search.strategies import run_search
from hyperbolic.eval.metrics import mean_metrics

logger = logging.getLogger(__name__)

COLLECTION = "eval_synthetic"


def _build_tier_filter(use_broad: bool) -> Optional[str]:
    """
    Return the tier label to pass as query_tier to run_search.
    strict → "content" only (themes/narratives/stories excluded)
    broad  → None means we need a multi-value filter; handled in strategies._tier_filter
    We abuse query_tier=None to mean "content+story" via a separate path below.
    Actually: return a sentinel so strategies knows what to do.
    """
    # For strict eval: content-only filter keeps ancestor docs out
    # For broad eval: content+story filter — narratives/themes still excluded
    return "content_or_story" if use_broad else "content"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_corpus_and_queries(data_dir: str = "data/synthetic"):
    """Load corpus docs and eval queries from JSON files."""
    import json
    from pathlib import Path
    d = Path(data_dir)
    with open(d / "corpus.json") as f:
        raw_docs = json.load(f)
    with open(d / "queries.json") as f:
        queries = json.load(f)
    return raw_docs, queries


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@dataclass
class EvalHarness:
    client: HyperbolicClient
    encoder: PCAEncoder
    collection: str = COLLECTION

    @classmethod
    def build(
        cls,
        qdrant_url: str = "http://localhost:6333",
        raw_docs: Optional[list[dict]] = None,
        data_dir: str = "data/synthetic",
        curvature: float = 1.0,
        recreate: bool = False,
    ) -> "EvalHarness":
        """
        Connect to Qdrant, fit PCA on corpus texts, ingest all documents.

        If the collection already exists and recreate=False, skips ingestion.
        """
        if raw_docs is None:
            raw_docs, _ = load_corpus_and_queries(data_dir)

        client = HyperbolicClient.connect(qdrant_url)

        texts = [d["text"] for d in raw_docs]
        tiers = [d["tier"] for d in raw_docs]

        from pathlib import Path
        pca_cache = Path(data_dir) / "pca_matrix.npy"
        encoder = PCAEncoder(curvature=curvature, pca_path=str(pca_cache))
        if encoder._pca_matrix is None or recreate:
            logger.info("Fitting PCA on %d documents...", len(texts))
            encoder.fit_pca(texts, save_path=str(pca_cache))
        else:
            logger.info("Loaded cached PCA matrix from %s", pca_cache)

        harness = cls(client=client, encoder=encoder, collection=COLLECTION)

        if recreate or not client.raw.collection_exists(COLLECTION):
            client.ensure_collection(COLLECTION, curvature=curvature, recreate=recreate)
            docs = [
                Document(
                    id=d["id"],
                    text=d["text"],
                    tier=d["tier"],
                    domain=d["domain"],
                    area=d["area"],
                    parent_ids=[d["parent_id"]] if d.get("parent_id") is not None else [],
                    child_ids=d.get("child_ids", []),
                )
                for d in raw_docs
            ]
            ingestor = Ingestor(client, encoder, COLLECTION)
            ingestor.ingest(docs, curvature=curvature)
            logger.info("Ingested %d documents into %r", len(docs), COLLECTION)
        else:
            logger.info("Collection %r exists — skipping ingestion", COLLECTION)

        return harness

    # ------------------------------------------------------------------

    def run_query(
        self,
        query_text: str,
        relevant_ids: list[int],
        config: PipelineConfig,
        k: int = 10,
        encoded_query=None,
        use_broad: bool = False,
    ) -> dict:
        """Run a single query and return retrieved IDs + metrics.

        Tier filter policy
        ------------------
        Busemann and entailment pipelines receive no tier filter — they suppress
        ancestor docs geometrically, so a hard filter would be redundant.

        All other pipelines use a tier filter controlled by use_broad:
          use_broad=False → content only (strict relevance)
          use_broad=True  → content + story (broad relevance)

        Intent routing (config.intent in INTENT_MODES) additionally lets the
        detector choose the tier filter based on query signals, but never
        overrides config.query_pipeline.
        """
        ev = encoded_query if encoded_query is not None else self.encoder.encode_query(query_text)

        FILTERLESS = {"busemann", "entailment"}
        INTENT_MODES = {"auto", "precision", "recall", "alpha", "text"}
        intent_mode = getattr(config, "intent", None)

        if config.query_pipeline in FILTERLESS:
            # Geometric pipelines: no tier filter, let run_search handle it
            hits = run_search(
                self.client, self.collection,
                query_dense=ev.dense[0],
                query_poincare=ev.poincare[0],
                query_tangent=ev.tangent[0],
                config=config,
                limit=max(k, len(relevant_ids)),
                query_text=query_text,
            )
        elif intent_mode in INTENT_MODES:
            # Intent routing active — detector picks tier, pipeline stays from config.
            # When use_broad=True, override the tier to content_or_story so stories
            # are included; passing query_tier also bypasses intent routing (which
            # would otherwise always pick "content").
            hits = run_search(
                self.client, self.collection,
                query_dense=ev.dense[0],
                query_poincare=ev.poincare[0],
                query_tangent=ev.tangent[0],
                config=config,
                limit=max(k, len(relevant_ids)),
                query_text=query_text,
                query_tier=_build_tier_filter(use_broad) if use_broad else None,
            )
        else:
            # Legacy: caller controls tier via use_broad
            tier_filter = _build_tier_filter(use_broad)
            hits = run_search(
                self.client, self.collection,
                query_dense=ev.dense[0],
                query_poincare=ev.poincare[0],
                query_tangent=ev.tangent[0],
                config=config,
                limit=max(k, len(relevant_ids)),
                query_tier=tier_filter,
            )
        retrieved_ids = [h.id for h in hits]
        return {
            "query_text": query_text,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
        }

    def evaluate_pipeline(
        self,
        queries: list[dict],
        config: PipelineConfig,
        k: int = 10,
        use_broad: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate a single pipeline across all queries.

        use_broad=True uses broad_relevant_ids (stories + content),
        use_broad=False uses strict relevant_ids (content only).
        """
        results = []
        rel_key = "broad_relevant_ids" if use_broad else "relevant_ids"
        for q in queries:
            res = self.run_query(
                q["query_text"],
                q[rel_key],
                config=config,
                k=k,
                use_broad=use_broad,
            )
            results.append(res)
        return mean_metrics(results, k=k)

    def _encode_queries_cached(
        self,
        queries: list[dict],
        cache_path: str,
        recreate: bool = False,
    ) -> list:
        """
        Encode all query texts once and cache to disk.
        Returns list of EncodedVectors, one per query.
        Subsequent runs load from cache — no model load needed.
        """
        import numpy as np
        from pathlib import Path
        from hyperbolic.encoder.base import EncodedVectors

        cache = Path(cache_path)
        if cache.exists() and not recreate:
            logger.info("Loading cached query vectors from %s", cache)
            data = np.load(cache)
            return [
                EncodedVectors(
                    dense=data["dense"][i:i+1],
                    poincare=data["poincare"][i:i+1],
                    tangent=data["tangent"][i:i+1],
                )
                for i in range(len(queries))
            ]

        logger.info("Encoding %d queries (will cache to %s)...", len(queries), cache)
        encoded = [self.encoder.encode_query(q["query_text"]) for q in queries]
        np.savez(
            cache,
            dense=np.concatenate([e.dense for e in encoded], axis=0),
            poincare=np.concatenate([e.poincare for e in encoded], axis=0),
            tangent=np.concatenate([e.tangent for e in encoded], axis=0),
        )
        return encoded

    def compare_pipelines(
        self,
        queries: list[dict],
        k: int = 10,
        use_broad: bool = False,
        query_cache: str = "data/synthetic/query_vectors.npz",
        recreate: bool = False,
    ) -> dict[str, dict[str, float]]:
        """
        Run all pipelines + cosine baseline and return comparison dict.

        Returns: {pipeline_name: {metric: value}}
        """
        # Encode all queries once, reuse across all pipeline evaluations
        encoded_queries = self._encode_queries_cached(queries, query_cache, recreate=recreate)

        pipelines = {
            "cosine":          DENSE_COSINE_PIPELINE,
            "cosine_filtered": DENSE_COSINE_FILTERED_PIPELINE,
            "alpha":           ALPHA_PIPELINE,
            "tangent":     TANGENT_PIPELINE,
            "combined":    COMBINED_PIPELINE,
            "klein":       KLEIN_PIPELINE,
            "busemann":    BUSEMANN_PIPELINE,
            "entailment":  ENTAILMENT_PIPELINE,
        }
        rel_key = "broad_relevant_ids" if use_broad else "relevant_ids"
        report = {}
        for name, cfg in pipelines.items():
            logger.info("Evaluating pipeline: %s", name)
            results = []
            for q, ev in zip(queries, encoded_queries):
                res = self.run_query(
                    q["query_text"], q[rel_key], config=cfg, k=k,
                    encoded_query=ev, use_broad=use_broad,
                )
                results.append(res)
            report[name] = mean_metrics(results, k=k)
        return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: dict[str, dict[str, float]], k: int = 10):
    """Print a formatted comparison table."""
    pipelines = list(report.keys())
    if not pipelines:
        print("No results.")
        return

    metrics = list(report[pipelines[0]].keys())
    col_w = 10
    name_w = 10

    header = f"{'Pipeline':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print()
    print(header)
    print("─" * len(header))
    for pipeline, scores in report.items():
        row = f"{pipeline:<{name_w}}" + "".join(f"{scores[m]:>{col_w}.4f}" for m in metrics)
        print(row)
    print()

    # Highlight best per metric
    print("Best per metric:")
    for m in metrics:
        best = max(report.items(), key=lambda x: x[1][m])
        print(f"  {m}: {best[0]} ({best[1][m]:.4f})")
    print()
