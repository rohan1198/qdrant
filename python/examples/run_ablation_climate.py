#!/usr/bin/env python3
"""
run_ablation_climate.py — Three critical ablations on the climate corpus.

Ablation 1: cosine_filtered
  Adds cosine + tier-filter to the table. Isolates whether the geometry
  or the tier filter explains hyperbolic gains.

Ablation 2: natural_norm projection
  Re-ingests corpus with natural_norm (PCA norm percentile → radius, no
  tier labels). If busemann still wins, the geometry is real signal.
  If it collapses, the manually-assigned radius was doing the work.

Ablation 3: radius corruption
  Encodes narrative queries at r=0.90 (content radius) instead of r=0.45.
  If busemann still wins, the branching logic is robust to query radius.
  If it collapses, the pipeline was just reading the tier label back.

Usage:
    python3 scripts/run_ablation_climate.py
    python3 scripts/run_ablation_climate.py --ablation 1   # only ablation 1
    python3 scripts/run_ablation_climate.py --ablation 2
    python3 scripts/run_ablation_climate.py --ablation 3
    python3 scripts/run_ablation_climate.py --url http://localhost:6333
"""
import sys
import json
import logging
import argparse
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from data.climate.generator import generate_corpus, generate_descendant_queries
from hyperbolic.client import HyperbolicClient
from hyperbolic.encoder.pca_encoder import PCAEncoder, ProjectionStrategy
from hyperbolic.pipeline.ingest import Ingestor, Document
from hyperbolic.eval.harness import EvalHarness, print_report
from hyperbolic.eval.metrics import mean_metrics
from hyperbolic.search.strategies import run_search
from hyperbolic.pipeline.config import (
    DENSE_COSINE_PIPELINE,
    DENSE_COSINE_FILTERED_PIPELINE,
    ALPHA_PIPELINE,
    BUSEMANN_PIPELINE,
    ENTAILMENT_PIPELINE,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CORPUS_JSON   = root / "data" / "climate" / "corpus.json"
PCA_EXPLICIT  = root / "data" / "climate" / "pca_matrix.npy"          # explicit_radial (default)
PCA_NATURAL   = root / "data" / "climate" / "pca_matrix_natural.npy"  # natural_norm
COLL_EXPLICIT = "eval_climate"
COLL_NATURAL  = "eval_climate_natural"


# ── helpers ────────────────────────────────────────────────────────────────

def load_corpus():
    with open(CORPUS_JSON) as f:
        return json.load(f)

def make_encoder(strategy: ProjectionStrategy, pca_path: str) -> PCAEncoder:
    return PCAEncoder(curvature=1.0, pca_path=pca_path, strategy=strategy)

def ensure_ingested(client, encoder, collection, raw_docs, recreate=False):
    if recreate or not client.raw.collection_exists(collection):
        client.ensure_collection(collection, curvature=1.0, recreate=recreate)
        docs = [
            Document(
                id=d["id"], text=d["text"], tier=d["tier"],
                domain=d["domain"], area=d["area"],
                parent_ids=d.get("parent_ids", []),
                child_ids=d.get("child_ids", []),
            )
            for d in raw_docs
        ]
        Ingestor(client, encoder, collection).ingest(docs, curvature=1.0)
        print(f"  Ingested {len(docs)} docs → '{collection}'")
    else:
        print(f"  Collection '{collection}' exists (use --regen to rebuild)")

def run_pipeline_set(harness, queries, encoded_queries, pipelines, k):
    report = {}
    for name, cfg in pipelines.items():
        results = []
        for q, ev in zip(queries, encoded_queries):
            res = harness.run_query(
                query_text=q["query_text"],
                relevant_ids=q["relevant_ids"],
                config=cfg, k=k, encoded_query=ev,
            )
            results.append(res)
        report[name] = mean_metrics(results, k=k)
    return report

def encode_queries(encoder, queries, tier_override=None):
    """Encode queries at their natural tier (or override tier for ablation 3)."""
    out = []
    for q in queries:
        t = tier_override if tier_override else q["query_tier"]
        out.append(encoder.encode([q["query_text"]], tiers=[t]))
    return out

def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ── ablation 1: cosine_filtered ─────────────────────────────────────────────

def ablation1(client, encoder, desc_queries, k):
    section("ABLATION 1 — cosine_filtered  (does tier filter explain hyperbolic gains?)")
    print("Compares:  cosine  |  cosine_filtered  |  alpha  |  busemann")
    print("Hypothesis: if cosine_filtered ≈ busemann, the filter does the work.\n")

    harness = EvalHarness(client=client, encoder=encoder, collection=COLL_EXPLICIT)
    pipelines = {
        "cosine":          DENSE_COSINE_PIPELINE,
        "cosine_filtered": DENSE_COSINE_FILTERED_PIPELINE,
        "alpha":           ALPHA_PIPELINE,
        "busemann":        BUSEMANN_PIPELINE,
        "entailment":      ENTAILMENT_PIPELINE,
    }

    for tier_label in ("story", "narrative"):
        queries = [q for q in desc_queries if q["query_tier"] == tier_label]
        if not queries:
            continue
        print(f"── {tier_label.upper()} tier ({len(queries)} queries) ──")
        encoded = encode_queries(encoder, queries)
        report = run_pipeline_set(harness, queries, encoded, pipelines, k)
        print_report(report, k=k)


# ── ablation 2: natural_norm projection ────────────────────────────────────

def ablation2(client, desc_queries, k, raw_docs, regen=False):
    section("ABLATION 2 — natural_norm projection  (is the radius doing the work?)")
    print("Re-ingests corpus with PCA-norm-based radius (no tier labels injected).")
    print("Hypothesis: if busemann collapses, the manual radius was the signal.\n")

    encoder_nat = make_encoder(ProjectionStrategy.NATURAL_NORM, str(PCA_NATURAL))
    if encoder_nat._pca_matrix is None or regen:
        print(f"Fitting natural_norm PCA on {len(raw_docs)} docs...")
        encoder_nat.fit_pca([d["text"] for d in raw_docs], save_path=str(PCA_NATURAL))
    else:
        print(f"Loaded natural_norm PCA from {PCA_NATURAL}")

    ensure_ingested(client, encoder_nat, COLL_NATURAL, raw_docs, recreate=regen)

    harness = EvalHarness(client=client, encoder=encoder_nat, collection=COLL_NATURAL)
    pipelines = {
        "cosine":   DENSE_COSINE_PIPELINE,
        "alpha":    ALPHA_PIPELINE,
        "busemann": BUSEMANN_PIPELINE,
    }

    for tier_label in ("story", "narrative"):
        queries = [q for q in desc_queries if q["query_tier"] == tier_label]
        if not queries:
            continue
        print(f"── {tier_label.upper()} tier ({len(queries)} queries, natural_norm) ──")
        # For natural_norm we don't have tier labels — encode at "content" tier
        # to simulate no prior knowledge of depth
        encoded = encode_queries(encoder_nat, queries, tier_override="content")
        report = run_pipeline_set(harness, queries, encoded, pipelines, k)
        print_report(report, k=k)


# ── ablation 3: radius corruption ──────────────────────────────────────────

def ablation3(client, encoder, desc_queries, k):
    section("ABLATION 3 — radius corruption  (does query radius drive the branch?)")
    print("Encodes narrative queries at r=0.90 (content radius) instead of r=0.45.")
    print("Hypothesis: if busemann collapses, the pipeline was reading the tier label.\n")

    harness = EvalHarness(client=client, encoder=encoder, collection=COLL_EXPLICIT)
    pipelines = {
        "busemann":   BUSEMANN_PIPELINE,
        "alpha":      ALPHA_PIPELINE,
        "entailment": ENTAILMENT_PIPELINE,
    }

    narrative_queries = [q for q in desc_queries if q["query_tier"] == "narrative"]
    if not narrative_queries:
        print("No narrative queries found.")
        return

    print(f"Narrative queries: {len(narrative_queries)}\n")

    # Correct encoding (r=0.45)
    print("── CORRECT encoding  (narrative → r=0.45) ──")
    encoded_correct = encode_queries(encoder, narrative_queries)
    report_correct = run_pipeline_set(harness, narrative_queries, encoded_correct, pipelines, k)
    print_report(report_correct, k=k)

    # Corrupted: encode at r=0.90 (pretend they're content posts)
    print("── CORRUPTED encoding  (narrative encoded as content → r=0.90) ──")
    encoded_corrupt = encode_queries(encoder, narrative_queries, tier_override="content")
    report_corrupt = run_pipeline_set(harness, narrative_queries, encoded_corrupt, pipelines, k)
    print_report(report_corrupt, k=k)

    # Also try story radius (r=0.70)
    print("── CORRUPTED encoding  (narrative encoded as story → r=0.70) ──")
    encoded_story = encode_queries(encoder, narrative_queries, tier_override="story")
    report_story = run_pipeline_set(harness, narrative_queries, encoded_story, pipelines, k)
    print_report(report_story, k=k)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Climate corpus ablation suite")
    parser.add_argument("--k",       type=int, default=10)
    parser.add_argument("--url",     default="http://localhost:6333")
    parser.add_argument("--regen",   action="store_true", help="Re-ingest collections")
    parser.add_argument("--ablation",type=int, choices=[1, 2, 3], default=None,
                        help="Run only one ablation (default: all three)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  CLIMATE RETRIEVAL ABLATION SUITE")
    print("  Ablation 1: cosine_filtered baseline")
    print("  Ablation 2: natural_norm projection (no tier labels)")
    print("  Ablation 3: narrative radius corruption")
    print("=" * 70)

    raw_docs = load_corpus()
    docs     = generate_corpus()
    desc_queries = generate_descendant_queries(docs)

    n_story = sum(1 for q in desc_queries if q["query_tier"] == "story")
    n_narr  = sum(1 for q in desc_queries if q["query_tier"] == "narrative")
    print(f"\nCorpus: {len(raw_docs)} docs | Queries: {n_story} story + {n_narr} narrative\n")

    client  = HyperbolicClient.connect(args.url)
    encoder = make_encoder(ProjectionStrategy.EXPLICIT_RADIAL, str(PCA_EXPLICIT))

    if encoder._pca_matrix is None or args.regen:
        print(f"Fitting explicit_radial PCA on {len(raw_docs)} docs...")
        encoder.fit_pca([d["text"] for d in raw_docs], save_path=str(PCA_EXPLICIT))
    else:
        print(f"Loaded explicit_radial PCA from {PCA_EXPLICIT}")

    ensure_ingested(client, encoder, COLL_EXPLICIT, raw_docs, recreate=args.regen)

    run_all = args.ablation is None

    if run_all or args.ablation == 1:
        ablation1(client, encoder, desc_queries, args.k)

    if run_all or args.ablation == 2:
        ablation2(client, desc_queries, args.k, raw_docs, regen=args.regen)

    if run_all or args.ablation == 3:
        ablation3(client, encoder, desc_queries, args.k)

    print("\n✓ Ablation suite complete.\n")


if __name__ == "__main__":
    main()
