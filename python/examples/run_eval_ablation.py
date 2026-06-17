#!/usr/bin/env python3
"""
run_eval_ablation.py — Compare projection strategies to isolate the depth signal.

Three strategies, same corpus, same queries, same 7 pipelines:

  explicit_radial  tier label → radius  (current system, upper bound)
  natural_norm     PCA norm  → radius   (data-driven, no tier labels at encode time)
  uniform          all docs at r=0.90   (no depth at all, lower bound)

If busemann beats cosine under natural_norm but not under uniform, the model's
embeddings genuinely encode hierarchical depth. If busemann only beats cosine
under explicit_radial, we were injecting the signal via tier labels.

Usage:
    python scripts/run_eval_ablation.py [--broad] [--k 10]
"""
import sys
import argparse
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from data.synthetic.generator import save as generate_corpus
from hyperbolic.eval.harness import load_corpus_and_queries, EvalHarness, print_report
from hyperbolic.client import HyperbolicClient
from hyperbolic.encoder.pca_encoder import PCAEncoder, ProjectionStrategy
from hyperbolic.pipeline.ingest import Ingestor, Document


STRATEGIES = [
    ("explicit_radial", ProjectionStrategy.EXPLICIT_RADIAL,
     "tier label → radius (injected depth)"),
    ("natural_norm",    ProjectionStrategy.NATURAL_NORM,
     "PCA norm → radius  (data-driven depth)"),
    ("uniform",         ProjectionStrategy.UNIFORM,
     "all docs r=0.90   (no depth signal)"),
]


def build_collection(client, raw_docs, encoder, collection, regen):
    if not regen and client.raw.collection_exists(collection):
        print(f"  Reusing existing collection '{collection}'")
        return
    client.ensure_collection(collection, curvature=1.0, recreate=True)
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
    print(f"  Ingested {len(docs)} docs into '{collection}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",     type=int, default=10)
    parser.add_argument("--url",   default="http://localhost:6333")
    parser.add_argument("--broad", action="store_true")
    parser.add_argument("--regen", action="store_true")
    args = parser.parse_args()

    data_dir = Path("data/synthetic")
    if args.regen or not (data_dir / "corpus.json").exists():
        generate_corpus(str(data_dir))

    raw_docs, queries = load_corpus_and_queries(str(data_dir))
    print(f"Corpus: {len(raw_docs)} docs  Queries: {len(queries)}")
    print(f"Mode: {'broad' if args.broad else 'strict'}  k={args.k}\n")

    client = HyperbolicClient.connect(args.url)

    # Fit PCA once and reuse across all strategies
    pca_cache = data_dir / "pca_matrix.npy"
    base_encoder = PCAEncoder(curvature=1.0, pca_path=str(pca_cache))
    if base_encoder._pca_matrix is None or args.regen:
        texts = [d["text"] for d in raw_docs]
        print("Fitting PCA (shared across strategies)...")
        base_encoder.fit_pca(texts, save_path=str(pca_cache))

    all_results = {}

    for strategy_name, strategy_enum, description in STRATEGIES:
        print(f"{'='*60}")
        print(f"Strategy: {strategy_name}  ({description})")
        print(f"{'='*60}")

        collection = f"eval_ablation_{strategy_name}"

        # Build encoder with this strategy, reuse the fitted PCA
        encoder = PCAEncoder(
            curvature=1.0,
            pca_path=str(pca_cache),
            strategy=strategy_enum,
        )

        print(f"  Building collection...")
        build_collection(client, raw_docs, encoder, collection, args.regen)

        # Verify actual radii stored
        records = client.scroll_all(collection, with_vectors=True)
        import numpy as np
        tier_norms = {}
        for r in records:
            tier = r.payload.get("tier", "?")
            if "poincare" in (r.vector or {}):
                norm = float(np.linalg.norm(r.vector["poincare"]))
                tier_norms.setdefault(tier, []).append(norm)
        print("  Poincaré radii by tier:")
        for tier, norms in sorted(tier_norms.items()):
            norms = np.array(norms)
            print(f"    {tier:12s}: mean={norms.mean():.4f}  std={norms.std():.4f}  "
                  f"[{norms.min():.3f}, {norms.max():.3f}]")

        harness = EvalHarness(client=client, encoder=encoder, collection=collection)

        query_cache = data_dir / f"query_vectors_{strategy_name}.npz"
        rel_key = "broad_relevant_ids" if args.broad else "relevant_ids"
        encoded_queries = harness._encode_queries_cached(
            queries, str(query_cache), recreate=args.regen
        )

        print(f"  Evaluating 7 pipelines × {len(queries)} queries...")
        results = {}
        for name, cfg in [
            ("cosine",     __import__("hyperbolic.pipeline.config", fromlist=["DENSE_COSINE_PIPELINE"]).DENSE_COSINE_PIPELINE),
            ("alpha",      __import__("hyperbolic.pipeline.config", fromlist=["ALPHA_PIPELINE"]).ALPHA_PIPELINE),
            ("tangent",    __import__("hyperbolic.pipeline.config", fromlist=["TANGENT_PIPELINE"]).TANGENT_PIPELINE),
            ("combined",   __import__("hyperbolic.pipeline.config", fromlist=["COMBINED_PIPELINE"]).COMBINED_PIPELINE),
            ("klein",      __import__("hyperbolic.pipeline.config", fromlist=["KLEIN_PIPELINE"]).KLEIN_PIPELINE),
            ("busemann",   __import__("hyperbolic.pipeline.config", fromlist=["BUSEMANN_PIPELINE"]).BUSEMANN_PIPELINE),
            ("entailment", __import__("hyperbolic.pipeline.config", fromlist=["ENTAILMENT_PIPELINE"]).ENTAILMENT_PIPELINE),
        ]:
            from hyperbolic.eval.metrics import mean_metrics
            rows = []
            for q, ev in zip(queries, encoded_queries):
                res = harness.run_query(
                    q["query_text"], q[rel_key],
                    config=cfg, k=args.k,
                    encoded_query=ev, use_broad=args.broad,
                )
                rows.append(res)
            results[name] = mean_metrics(rows, k=args.k)

        print_report(results, k=args.k)
        all_results[strategy_name] = results

    # Summary table: P@10 and NDCG@10 for busemann vs cosine across strategies
    print("\n" + "="*60)
    print("ABLATION SUMMARY  (busemann vs cosine)")
    print("="*60)
    print(f"{'Strategy':<18} {'cosine P@10':>12} {'busemann P@10':>14} {'delta':>8} {'busemann NDCG':>14}")
    print("-"*68)
    for strategy_name, results in all_results.items():
        cp = results["cosine"]["P@10"]
        bp = results["busemann"]["P@10"]
        bn = results["busemann"]["NDCG@10"]
        print(f"{strategy_name:<18} {cp:>12.4f} {bp:>14.4f} {bp-cp:>+8.4f} {bn:>14.4f}")
    print()


if __name__ == "__main__":
    main()
