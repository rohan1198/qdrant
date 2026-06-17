#!/usr/bin/env python3
"""
run_eval_climate.py — Descendant retrieval eval on the climate corpus.

This mirrors run_eval_descendant.py but uses the pre-generated climate
corpus in data/climate/ rather than the WOS dataset.

Query tiers:
  story (r=0.70)     → relevant = 8 content posts for that story
  narrative (r=0.45) → relevant = all 40 content posts in that narrative

Usage:
    python3 scripts/run_eval_climate.py
    python3 scripts/run_eval_climate.py --regen        # re-ingest collection
    python3 scripts/run_eval_climate.py --k 20         # deeper retrieval
    python3 scripts/run_eval_climate.py --story-only   # only story-level queries
    python3 scripts/run_eval_climate.py --narrative-only
"""
import sys
import argparse
import logging
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from data.climate.generator import generate_corpus, generate_descendant_queries
from hyperbolic.eval.harness import EvalHarness, print_report
from hyperbolic.pipeline.config import (
    DENSE_COSINE_PIPELINE,
    ALPHA_PIPELINE,
    TANGENT_PIPELINE,
    COMBINED_PIPELINE,
    KLEIN_PIPELINE,
    BUSEMANN_PIPELINE,
    ENTAILMENT_PIPELINE,
)
from hyperbolic.eval.metrics import mean_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CORPUS_JSON = root / "data" / "climate" / "corpus.json"
COLLECTION = "eval_climate"


def main():
    parser = argparse.ArgumentParser(
        description="Descendant retrieval eval on climate corpus (query=story/narrative, relevant=content posts)"
    )
    parser.add_argument("--k",              type=int, default=10,           help="Retrieval depth")
    parser.add_argument("--url",            default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--regen",          action="store_true",             help="Re-ingest collection")
    parser.add_argument("--story-only",     action="store_true",             help="Only run story-level queries")
    parser.add_argument("--narrative-only", action="store_true",             help="Only run narrative-level queries")
    args = parser.parse_args()

    print("=" * 60)
    print("CLIMATE CORPUS — DESCENDANT RETRIEVAL EVAL")
    print("Query = story or narrative doc")
    print("Relevant = content posts that are descendants")
    print("No tier filter — geometry must find them among all docs")
    print("=" * 60)
    print()

    # ── Load or generate corpus ──────────────────────────────────────────
    import json

    if CORPUS_JSON.exists() and not args.regen:
        print(f"Loading climate corpus from {CORPUS_JSON}...")
        with open(CORPUS_JSON) as f:
            raw_docs = json.load(f)
        print(f"Loaded {len(raw_docs)} documents")
    else:
        print("Generating climate corpus...")
        from data.climate.generator import save as save_climate
        save_climate(str(root / "data" / "climate"))
        with open(CORPUS_JSON) as f:
            raw_docs = json.load(f)
        print(f"Generated {len(raw_docs)} documents")

    # Convert to the format expected by generate_descendant_queries
    # (which takes ClimateDoc objects — so we regenerate from source)
    print()
    print("Building corpus objects...")
    docs = generate_corpus()
    print(f"  {sum(1 for d in docs if d.tier == 'narrative')} narratives, "
          f"{sum(1 for d in docs if d.tier == 'story')} stories, "
          f"{sum(1 for d in docs if d.tier == 'content')} posts")

    # ── Build Qdrant collection ──────────────────────────────────────────
    pca_cache = root / "data" / "climate" / "pca_matrix.npy"
    print()
    print("Building eval harness...")

    from hyperbolic.client import HyperbolicClient
    from hyperbolic.encoder.pca_encoder import PCAEncoder
    from hyperbolic.pipeline.ingest import Ingestor, Document

    client = HyperbolicClient.connect(args.url)
    encoder = PCAEncoder(curvature=1.0, pca_path=str(pca_cache))

    if encoder._pca_matrix is None or args.regen:
        print(f"  Fitting PCA on {len(raw_docs)} documents...")
        encoder.fit_pca([d["text"] for d in raw_docs], save_path=str(pca_cache))
    else:
        print(f"  Loaded cached PCA from {pca_cache}")

    harness = EvalHarness(client=client, encoder=encoder, collection=COLLECTION)

    if args.regen or not client.raw.collection_exists(COLLECTION):
        client.ensure_collection(COLLECTION, curvature=1.0, recreate=args.regen)
        ingest_docs = [
            Document(
                id=d["id"],
                text=d["text"],
                tier=d["tier"],
                domain=d["domain"],
                area=d["area"],
                parent_ids=d.get("parent_ids", []),
                child_ids=d.get("child_ids", []),
            )
            for d in raw_docs
        ]
        ingestor = Ingestor(client, encoder, COLLECTION)
        ingestor.ingest(ingest_docs, curvature=1.0)
        print(f"  Ingested {len(ingest_docs)} documents into '{COLLECTION}'")
    else:
        print(f"  Collection '{COLLECTION}' exists — skipping ingestion (use --regen to re-ingest)")

    print("Ready.\n")

    # ── Generate descendant queries ──────────────────────────────────────
    print("Generating descendant queries...")
    all_desc_queries = generate_descendant_queries(docs)
    print()

    # Filter by tier if requested
    if args.story_only:
        desc_queries = [q for q in all_desc_queries if q["query_tier"] == "story"]
        print(f"Running story-level queries only ({len(desc_queries)} queries)")
    elif args.narrative_only:
        desc_queries = [q for q in all_desc_queries if q["query_tier"] == "narrative"]
        print(f"Running narrative-level queries only ({len(desc_queries)} queries)")
    else:
        desc_queries = all_desc_queries

    if not desc_queries:
        print("No descendant queries generated.")
        sys.exit(1)

    # ── Evaluate by tier ─────────────────────────────────────────────────
    tier_groups: dict[str, list[dict]] = {}
    if not args.story_only and not args.narrative_only:
        tier_groups["story"] = [q for q in desc_queries if q["query_tier"] == "story"]
        tier_groups["narrative"] = [q for q in desc_queries if q["query_tier"] == "narrative"]
    tier_groups["all"] = desc_queries

    pipelines = {
        "cosine":     DENSE_COSINE_PIPELINE,
        "alpha":      ALPHA_PIPELINE,
        "tangent":    TANGENT_PIPELINE,
        "combined":   COMBINED_PIPELINE,
        "klein":      KLEIN_PIPELINE,
        "busemann":   BUSEMANN_PIPELINE,
        "entailment": ENTAILMENT_PIPELINE,
    }

    for tier_label, queries in tier_groups.items():
        if not queries:
            continue

        print(f"\n{'=' * 60}")
        print(f"TIER: {tier_label.upper()}  ({len(queries)} queries)")
        print(f"{'=' * 60}")

        print("Encoding queries at natural tier radii...")
        encoded_queries = []
        for q in queries:
            query_tier = q["query_tier"]
            ev = encoder.encode([q["query_text"]], tiers=[query_tier])
            encoded_queries.append(ev)
        print(f"  {len(encoded_queries)} queries encoded "
              f"(story→r=0.70, narrative→r=0.45)")
        print()

        report = {}
        for name, cfg in pipelines.items():
            logger.info("Evaluating pipeline: %s  tier=%s", name, tier_label)
            results = []
            for q, ev in zip(queries, encoded_queries):
                res = harness.run_query(
                    query_text=q["query_text"],
                    relevant_ids=q["relevant_ids"],
                    config=cfg,
                    k=args.k,
                    encoded_query=ev,
                )
                results.append(res)
            report[name] = mean_metrics(results, k=args.k)

        print_report(report, k=args.k)


if __name__ == "__main__":
    main()
