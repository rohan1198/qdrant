#!/usr/bin/env python3
"""
run_eval_wos.py — Evaluate all pipelines on the WOS hierarchical dataset.

Downloads WOS from the testbench cache (qdrant/dev/testbench/data/wos/documents.jsonl),
samples N_AREAS areas, converts to our corpus format, and runs the same 7-pipeline
eval as run_eval.py.

Usage:
    python scripts/run_eval_wos.py [--k 10] [--url http://localhost:6333] [--broad]
    python scripts/run_eval_wos.py --areas 20 --leaves 50  # larger sample
"""
import sys
import argparse
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from data.wos.loader import load_wos
from hyperbolic.eval.harness import EvalHarness, print_report


WOS_JSONL = root / "qdrant/dev/testbench/data/wos/documents.jsonl"
COLLECTION = "eval_wos"


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipelines on WOS dataset")
    parser.add_argument("--k",      type=int, default=10,                   help="Retrieval depth")
    parser.add_argument("--url",    default="http://localhost:6333",         help="Qdrant URL")
    parser.add_argument("--broad",  action="store_true",                     help="Use broad relevance")
    parser.add_argument("--regen",  action="store_true",                     help="Re-ingest collection")
    parser.add_argument("--areas",  type=int, default=10,                    help="Number of areas to sample")
    parser.add_argument("--leaves", type=int, default=30,                    help="Max leaves per area")
    parser.add_argument("--queries",type=int, default=2,                     help="Queries per area")
    parser.add_argument("--wos",    default=str(WOS_JSONL),                  help="Path to documents.jsonl")
    args = parser.parse_args()

    if not Path(args.wos).exists():
        print(f"WOS dataset not found at: {args.wos}")
        print("Run: cd qdrant/dev/testbench && python3 download_dataset.py")
        sys.exit(1)

    print("Loading WOS dataset...")
    raw_docs, queries = load_wos(
        args.wos,
        n_areas=args.areas,
        max_leaves=args.leaves,
        n_queries=args.queries,
    )
    print(f"Loaded {len(raw_docs)} documents, {len(queries)} queries")
    print(f"Retrieval depth k={args.k}, relevance={'broad' if args.broad else 'strict (content only)'}")
    print()

    # Fit PCA + ingest
    pca_cache = Path("data/wos/pca_matrix.npy")
    print("Building eval harness (fitting PCA + ingesting)...")

    from hyperbolic.client import HyperbolicClient
    from hyperbolic.encoder.pca_encoder import PCAEncoder
    from hyperbolic.pipeline.ingest import Ingestor, Document

    client = HyperbolicClient.connect(args.url)

    texts = [d["text"] for d in raw_docs]
    encoder = PCAEncoder(curvature=1.0, pca_path=str(pca_cache))
    if encoder._pca_matrix is None or args.regen:
        print(f"  Fitting PCA on {len(texts)} documents...")
        encoder.fit_pca(texts, save_path=str(pca_cache))
    else:
        print(f"  Loaded cached PCA from {pca_cache}")

    harness = EvalHarness(client=client, encoder=encoder, collection=COLLECTION)

    if args.regen or not client.raw.collection_exists(COLLECTION):
        client.ensure_collection(COLLECTION, curvature=1.0, recreate=args.regen)
        docs = [
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
        ingestor.ingest(docs, curvature=1.0)
        print(f"  Ingested {len(docs)} documents into '{COLLECTION}'")
    else:
        print(f"  Collection '{COLLECTION}' exists — skipping ingestion (use --regen to re-ingest)")

    print("Ready.\n")

    n_pipelines = 7
    print(f"Evaluating {n_pipelines} pipelines × {len(queries)} queries...")
    report = harness.compare_pipelines(
        queries,
        k=args.k,
        use_broad=args.broad,
        query_cache="data/wos/query_vectors.npz",
        recreate=args.regen,
    )
    print_report(report, k=args.k)


if __name__ == "__main__":
    main()
