#!/usr/bin/env python3
"""
run_eval_20news.py — Evaluate trained projection head on 20 Newsgroups.

Pipeline:
  1. Load 20news JSONL (build_20news.py) + precomputed embeddings
  2. Ingest into Qdrant collection `eval_20news_trained`
  3. Evaluate story queries (20) + narrative queries (7)

Story eval:     query = synthetic story doc for newsgroup X
                relevant = all content posts in newsgroup X

Narrative eval: query = synthetic narrative doc for top group Y
                relevant = all content posts in all newsgroups under Y

Usage:
    # Build data first (once):
    python3 scripts/build_20news.py
    CUDA_VISIBLE_DEVICES=1 python3 scripts/precompute_20news_embeddings.py

    # Eval:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/run_eval_20news.py \\
        --head data/trained/projection_head_v3.pt

    # Force re-ingest:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/run_eval_20news.py \\
        --head data/trained/projection_head_v3.pt --regen
"""
import sys, json, argparse, logging
from pathlib import Path
from collections import defaultdict

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

JSONL_PATH = root / "data" / "20news" / "documents.jsonl"
EMBED_PATH = root / "data" / "20news" / "embeddings.npz"
COLLECTION = "eval_20news_trained"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--head",  required=True, help="Trained projection head .pt")
    p.add_argument("--emb",   default=str(EMBED_PATH))
    p.add_argument("--jsonl", default=str(JSONL_PATH))
    p.add_argument("--k",     type=int, default=10)
    p.add_argument("--pool",  type=int, default=1000)
    p.add_argument("--url",   default="http://localhost:6333")
    p.add_argument("--regen", action="store_true")
    return p.parse_args()


def load_embeddings(path: str):
    """Return (id_to_idx dict, embeddings array) — avoids per-entry Python objects."""
    data = np.load(path, allow_pickle=True)
    ids  = data["ids"]
    embs = data["embeddings"]          # shape [N, 1024], stays as contiguous array
    id_to_idx = {str(ids[i]): i for i in range(len(ids))}
    return id_to_idx, embs


def busemann_scores(head, emb_np: np.ndarray, query_poincare: np.ndarray, device: str) -> np.ndarray:
    t = torch.from_numpy(emb_np).to(device)
    tiers = ["content"] * len(emb_np)
    with torch.no_grad():
        poincare, _ = head(t, tiers=tiers)
    q = torch.from_numpy(query_poincare).to(device)
    q_norm   = q / q.norm().clamp(min=1e-8)
    norm_sq  = (poincare * poincare).sum(-1)
    depth    = torch.log((1.0 - norm_sq).clamp(min=1e-8))
    diff     = poincare - q_norm.unsqueeze(0)
    dist_sq  = (diff * diff).sum(-1).clamp(min=1e-8)
    return (depth - torch.log(dist_sq)).cpu().numpy()


def mean_metrics(results, k):
    from hyperbolic.eval.metrics import (
        precision_at_k, recall_at_k, reciprocal_rank, average_precision, ndcg_at_k,
    )
    ps  = [precision_at_k(r["retrieved"], r["relevant"], k) for r in results]
    rs  = [recall_at_k(r["retrieved"], r["relevant"], k)    for r in results]
    ms  = [reciprocal_rank(r["retrieved"], r["relevant"])    for r in results]
    aps = [average_precision(r["retrieved"], r["relevant"])  for r in results]
    ns  = [ndcg_at_k(r["retrieved"], r["relevant"], k)       for r in results]
    return {
        f"P@{k}": np.mean(ps), f"R@{k}": np.mean(rs),
        "MRR": np.mean(ms), "MAP": np.mean(aps), f"NDCG@{k}": np.mean(ns),
    }


def print_table(label, m, k):
    print(f"  {label}")
    print(f"  P@{k}={m[f'P@{k}']:.3f}  R@{k}={m[f'R@{k}']:.3f}  "
          f"MRR={m['MRR']:.3f}  MAP={m['MAP']:.3f}  NDCG@{k}={m[f'NDCG@{k}']:.3f}")


def ingest(client, head, id_to_idx, embs, collection, raw_docs, device, batch_size=512):
    """Ingest using precomputed dense embeddings — no pplx-embed needed.

    dense  = precomputed pplx-embed vector (from embeddings.npz)
    poincare/tangent = projection head applied to dense emb
    """
    from qdrant_client.http import models as rest
    from hyperbolic.client import DENSE_VECTOR, POINCARE_VECTOR, TANGENT_VECTOR
    from hyperbolic.math.poincare import log_map_origin
    import torch

    client.ensure_collection(collection, recreate=True)
    total = len(raw_docs)

    for start in range(0, total, batch_size):
        chunk = raw_docs[start: start + batch_size]
        tiers = [d["tier"] for d in chunk]

        dense_np = np.stack([
            embs[id_to_idx[d["id"]]] if d["id"] in id_to_idx else np.zeros(1024, dtype=np.float32)
            for d in chunk
        ])
        t = torch.from_numpy(dense_np).to(device)
        with torch.no_grad():
            poincare_t, _ = head(t, tiers=tiers)
        poincare_np = poincare_t.cpu().numpy()
        tangent_np  = np.stack([log_map_origin(p, c=1.0) for p in poincare_np])

        points = [
            rest.PointStruct(
                id=d["_int_id"],
                vector={
                    DENSE_VECTOR:    dense_np[i].tolist(),
                    POINCARE_VECTOR: poincare_np[i].tolist(),
                    TANGENT_VECTOR:  tangent_np[i].tolist(),
                },
                payload={"tier": d["tier"], "domain": d["domain"], "area": d["area"], "id": d["id"]},
            )
            for i, d in enumerate(chunk)
        ]
        client.upsert(collection, points)
        logger.info("  Ingested %d / %d", min(start + batch_size, total), total)

    logger.info("Done — ingested %d docs into '%s'", total, collection)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from hyperbolic.client import HyperbolicClient
    from hyperbolic.encoder.trained_encoder import TrainedEncoder
    from hyperbolic.training.projection_head import ProjectionHead
    from hyperbolic.pipeline.config import PipelineConfig
    from hyperbolic.search.strategies import run_search

    if not Path(args.jsonl).exists():
        print(f"Missing {args.jsonl} — run build_20news.py first"); sys.exit(1)
    if not Path(args.emb).exists():
        print(f"Missing {args.emb} — run precompute_20news_embeddings.py first"); sys.exit(1)

    # ── Load corpus ──────────────────────────────────────────────────────
    raw_docs = [json.loads(l) for l in open(args.jsonl)]
    for i, d in enumerate(raw_docs):
        d["_int_id"] = i

    id_to_int    = {d["id"]: i for i, d in enumerate(raw_docs)}
    int_to_strid = {i: d["id"] for i, d in enumerate(raw_docs)}

    content_by_ng:     dict[int, list[int]] = defaultdict(list)
    content_by_domain: dict[int, list[int]] = defaultdict(list)
    stories, narratives = [], []

    for d in raw_docs:
        if d["tier"] == "content":
            content_by_ng[d["area"]].append(id_to_int[d["id"]])
            content_by_domain[d["domain"]].append(id_to_int[d["id"]])
        elif d["tier"] == "story":
            stories.append(d)
        elif d["tier"] == "narrative":
            narratives.append(d)

    story_queries = [
        {"text": s["text"], "tier": "story", "relevant": content_by_ng.get(s["area"], [])}
        for s in stories if content_by_ng.get(s["area"])
    ]
    narr_queries = [
        {"text": n["text"], "tier": "narrative", "relevant": content_by_domain.get(n["domain"], [])}
        for n in narratives if content_by_domain.get(n["domain"])
    ]

    print(f"Corpus: {len(raw_docs)} docs  |  "
          f"Story queries: {len(story_queries)}  |  Narrative queries: {len(narr_queries)}")

    # ── Load precomputed embeddings + projection head (no pplx-embed yet) ──
    id_to_idx, embs = load_embeddings(args.emb)
    head    = ProjectionHead.load(args.head, device=device)
    head.eval()
    client  = HyperbolicClient.connect(args.url)
    print(f"Loaded head: {args.head}")

    # ── Ingest using precomputed embeddings (projection head only) ────────
    if args.regen or not client.raw.collection_exists(COLLECTION):
        logger.info("Ingesting %d docs (projection head only, no pplx-embed)…", len(raw_docs))
        ingest(client, head, id_to_idx, embs, COLLECTION, raw_docs, device)
    else:
        logger.info("Collection '%s' exists — using cached (--regen to rebuild)", COLLECTION)

    # ── Load encoder for query encoding (pplx-embed) ─────────────────────
    enc = TrainedEncoder(args.head, device=device)

    # ── Eval config ───────────────────────────────────────────────────────
    eval_cfg = PipelineConfig(
        query_pipeline="busemann",
        curvature=1.0,
        stage_sizes=(args.pool, 50),
    )

    # ── Run eval ──────────────────────────────────────────────────────────
    for tier_label, qs in [("story", story_queries), ("narrative", narr_queries),
                            ("all", story_queries + narr_queries)]:
        if not qs:
            continue
        print(f"\n{'='*60}")
        print(f"TIER: {tier_label.upper()}  ({len(qs)} queries)")
        print(f"{'='*60}")

        results_cosine, results_busemann = [], []

        for q in qs:
            ev = enc.encode([q["text"]], tiers=[q["tier"]])

            candidates = run_search(
                client, COLLECTION,
                ev.dense[0], ev.poincare[0], ev.tangent[0],
                config=eval_cfg, limit=args.pool,
            )
            if not candidates:
                results_cosine.append({"retrieved": [],             "relevant": q["relevant"]})
                results_busemann.append({"retrieved": [],           "relevant": q["relevant"]})
                continue

            cand_ids     = [c.id for c in candidates]
            cand_str_ids = [int_to_strid.get(cid, "") for cid in cand_ids]
            cand_embs    = np.stack([
                embs[id_to_idx[sid]] if sid in id_to_idx else np.zeros(1024, dtype=np.float32)
                for sid in cand_str_ids
            ])

            # Cosine ANN order (stage-1 only)
            results_cosine.append({
                "retrieved": cand_ids[:args.k],
                "relevant":  q["relevant"],
            })

            # Busemann rerank (stage-2)
            s    = busemann_scores(head, cand_embs, ev.poincare[0], device)
            idx  = np.argsort(s)[::-1][:args.k]
            results_busemann.append({
                "retrieved": [cand_ids[i] for i in idx],
                "relevant":  q["relevant"],
            })

        k = args.k
        print_table(f"cosine ANN (stage-1, pool={args.pool})", mean_metrics(results_cosine,   k), k)
        print_table(f"busemann   (trained,  pool={args.pool})", mean_metrics(results_busemann, k), k)


if __name__ == "__main__":
    main()
