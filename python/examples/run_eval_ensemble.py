#!/usr/bin/env python3
"""
run_eval_ensemble.py — Ensemble of v2 + v3 projection heads.

Since both heads share frozen pplx-embed (dense ANN is identical), we:
  1. Run dense ANN once → top-1000 candidates
  2. Re-score each candidate with both heads via precomputed embeddings
  3. Average Busemann scores → rerank → top-k

No re-ingestion needed.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/run_eval_ensemble.py \
        --v2 data/trained/projection_head_v2.pt \
        --v3 data/trained/projection_head_v3.pt
"""
import sys, argparse, random
from pathlib import Path
from collections import defaultdict

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import numpy as np
import torch

WOS_JSONL  = root / "qdrant/dev/testbench/data/wos/documents.jsonl"
EMBED_PATH = root / "data/wos/embeddings.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v2",   required=True)
    p.add_argument("--v3",   required=True)
    p.add_argument("--wos",  default=str(WOS_JSONL))
    p.add_argument("--emb",  default=str(EMBED_PATH))
    p.add_argument("--areas",type=int, default=404)
    p.add_argument("--k",    type=int, default=10)
    p.add_argument("--pool", type=int, default=1000)
    p.add_argument("--url",  default="http://localhost:6333")
    p.add_argument("--alpha",type=float, default=0.5, help="weight for v2 (1-alpha for v3)")
    return p.parse_args()


def busemann_scores(head, emb_np: np.ndarray, query_poincare: np.ndarray,
                    device: str) -> np.ndarray:
    """Score a batch of candidate embeddings against a query using head's Busemann."""
    t = torch.from_numpy(emb_np).to(device)          # [N, 1024]
    tiers = ["content"] * len(emb_np)
    with torch.no_grad():
        poincare, _ = head(t, tiers=tiers)            # [N, 128]

    q = torch.from_numpy(query_poincare).to(device)   # [128]
    q_norm = q / q.norm().clamp(min=1e-8)             # ideal point

    norm_sq  = (poincare * poincare).sum(-1)           # [N]
    depth    = torch.log((1.0 - norm_sq).clamp(min=1e-8))
    diff     = poincare - q_norm.unsqueeze(0)          # [N, 128]
    dist_sq  = (diff * diff).sum(-1).clamp(min=1e-8)
    scores   = depth - torch.log(dist_sq)              # [N]
    return scores.cpu().numpy()


def mean_metrics(results, k):
    from hyperbolic.eval.metrics import precision_at_k, recall_at_k, reciprocal_rank, average_precision, ndcg_at_k
    ps = [precision_at_k(r["retrieved"], r["relevant"], k) for r in results]
    rs = [recall_at_k(r["retrieved"], r["relevant"], k) for r in results]
    ms = [reciprocal_rank(r["retrieved"], r["relevant"]) for r in results]
    aps = [average_precision(r["retrieved"], r["relevant"]) for r in results]
    ns = [ndcg_at_k(r["retrieved"], r["relevant"], k) for r in results]
    return {
        f"P@{k}": np.mean(ps), f"R@{k}": np.mean(rs),
        "MRR": np.mean(ms), "MAP": np.mean(aps), f"NDCG@{k}": np.mean(ns),
    }


def print_table(label, metrics, k):
    print(f"\n  {label}")
    print(f"  P@{k}={metrics[f'P@{k}']:.3f}  R@{k}={metrics[f'R@{k}']:.3f}  "
          f"MRR={metrics['MRR']:.3f}  MAP={metrics['MAP']:.3f}  "
          f"NDCG@{k}={metrics[f'NDCG@{k}']:.3f}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from train_projection import load_wos_all, load_embeddings
    from hyperbolic.training.projection_head import ProjectionHead
    from hyperbolic.client import HyperbolicClient
    from hyperbolic.encoder.trained_encoder import TrainedEncoder
    from hyperbolic.pipeline.config import BUSEMANN_PIPELINE
    from hyperbolic.search.strategies import run_search

    # ── Load corpus + queries ────────────────────────────────────────
    raw_docs = load_wos_all(args.wos, n_areas=args.areas)
    id_to_int  = {d["id"]: i for i, d in enumerate(raw_docs)}
    int_to_strid = {i: d["id"] for i, d in enumerate(raw_docs)}

    content_by_area   = defaultdict(list)
    content_by_domain = defaultdict(list)
    stories, narratives = [], []
    for d in raw_docs:
        key = (d["domain"], d["area"])
        if d["tier"] == "content":
            content_by_area[key].append(id_to_int[d["id"]])
            content_by_domain[d["domain"]].append(id_to_int[d["id"]])
        elif d["tier"] == "story":
            stories.append(d)
        elif d["tier"] == "narrative":
            narratives.append(d)

    queries = []
    for s in stories:
        rel = content_by_area.get((s["domain"], s["area"]), [])
        if rel: queries.append({"text": s["text"], "tier": "story", "relevant": rel})
    for n in narratives:
        rel = content_by_domain.get(n["domain"], [])
        if rel: queries.append({"text": n["text"], "tier": "narrative", "relevant": rel})

    print(f"Corpus: {len(raw_docs)} docs | Queries: {len(queries)}")

    # ── Load precomputed embeddings ──────────────────────────────────
    emb_map = load_embeddings(args.emb)  # id → [1024]

    # ── Load both heads ──────────────────────────────────────────────
    head_v2 = ProjectionHead.load(args.v2, device=device)
    head_v3 = ProjectionHead.load(args.v3, device=device)
    head_v2.eval(); head_v3.eval()
    print(f"Loaded v2: {args.v2}")
    print(f"Loaded v3: {args.v3}")

    # ── Encoders — dense is shared (frozen pplx-embed), poincare differs per head ──
    enc_v2 = TrainedEncoder(args.v2, device=device)
    enc_v3 = TrainedEncoder(args.v3, device=device)
    enc = enc_v3  # use v3 for dense ANN (identical to v2)
    client = HyperbolicClient.connect(args.url)
    collection = "eval_wos_trained"  # current collection (v4, but dense vectors are shared)

    # ── Eval ─────────────────────────────────────────────────────────
    for tier_label in ("story", "narrative", "all"):
        qs = [q for q in queries if tier_label == "all" or q["tier"] == tier_label]
        if not qs: continue

        print(f"\n{'='*60}")
        print(f"TIER: {tier_label.upper()}  ({len(qs)} queries)")
        print(f"{'='*60}")

        results_v2, results_v3, results_ens = [], [], []

        for q in qs:
            ev_v2 = enc_v2.encode([q["text"]], tiers=[q["tier"]])
            ev_v3 = enc_v3.encode([q["text"]], tiers=[q["tier"]])

            # Stage-1: dense ANN → pool candidates (dense vectors identical across heads)
            candidates = run_search(
                client, collection,
                ev_v3.dense[0], ev_v3.poincare[0], ev_v3.tangent[0],
                config=BUSEMANN_PIPELINE, limit=args.pool,
            )
            if not candidates:
                for r in (results_v2, results_v3, results_ens):
                    r.append({"retrieved": [], "relevant": q["relevant"]})
                continue

            cand_ids     = [c.id for c in candidates]
            cand_str_ids = [int_to_strid.get(cid, "") for cid in cand_ids]
            cand_embs    = np.stack([emb_map.get(sid, np.zeros(1024, dtype=np.float32))
                                     for sid in cand_str_ids])

            # Score candidates using each head with its own query poincare
            s_v2 = busemann_scores(head_v2, cand_embs, ev_v2.poincare[0], device)
            s_v3 = busemann_scores(head_v3, cand_embs, ev_v3.poincare[0], device)
            s_ens = args.alpha * s_v2 + (1 - args.alpha) * s_v3

            def top_k(scores):
                idx = np.argsort(scores)[::-1][:args.k]
                return [cand_ids[i] for i in idx]

            results_v2.append( {"retrieved": top_k(s_v2),  "relevant": q["relevant"]})
            results_v3.append( {"retrieved": top_k(s_v3),  "relevant": q["relevant"]})
            results_ens.append({"retrieved": top_k(s_ens), "relevant": q["relevant"]})

        k = args.k
        print_table(f"v2 alone     (α=1.0)", mean_metrics(results_v2,  k), k)
        print_table(f"v3 alone     (α=0.0)", mean_metrics(results_v3,  k), k)
        print_table(f"ensemble     (α={args.alpha})", mean_metrics(results_ens, k), k)


if __name__ == "__main__":
    main()
