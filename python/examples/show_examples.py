#!/usr/bin/env python3
"""
show_examples.py — Show example queries and their top-k retrieved docs.
Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/show_examples.py --head data/trained/projection_head_v3.pt
"""
import sys, argparse, random
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-story", type=int, default=2)
    p.add_argument("--n-narrative", type=int, default=1)
    p.add_argument("--url", default="http://localhost:6333")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    from train_projection import load_wos_all
    from hyperbolic.client import HyperbolicClient
    from hyperbolic.encoder.trained_encoder import TrainedEncoder
    from hyperbolic.pipeline.config import BUSEMANN_PIPELINE
    from hyperbolic.search.strategies import run_search
    from collections import defaultdict

    raw_docs = load_wos_all(root / "qdrant/dev/testbench/data/wos/documents.jsonl", n_areas=404)
    id_to_int = {d["id"]: i for i, d in enumerate(raw_docs)}
    int_to_doc = {i: d for i, d in enumerate(raw_docs)}

    content_by_area = defaultdict(list)
    content_by_domain = defaultdict(list)
    stories, narratives = [], []
    for d in raw_docs:
        key = (d["domain"], d["area"])
        if d["tier"] == "content":
            content_by_area[key].append(d)
            content_by_domain[d["domain"]].append(d)
        elif d["tier"] == "story":
            stories.append(d)
        elif d["tier"] == "narrative":
            narratives.append(d)

    rng = random.Random(args.seed)
    enc = TrainedEncoder(args.head)
    client = HyperbolicClient.connect(args.url)
    collection = "eval_wos_trained"

    def show_query(q_doc, tier, relevant_ids):
        print(f"\n{'═'*70}")
        print(f"QUERY [{tier.upper()}]")
        print(f"  Domain: {q_doc['domain']}  Area: {q_doc.get('area','—')}")
        print(f"  Text: {q_doc['text'][:300]}")
        print(f"  Relevant docs: {len(relevant_ids)}")
        print(f"{'─'*70}")

        ev = enc.encode([q_doc["text"]], tiers=[tier])
        results = run_search(
            client, collection,
            ev.dense[0], ev.poincare[0], ev.tangent[0],
            config=BUSEMANN_PIPELINE, limit=args.k,
        )

        hits = 0
        for rank, r in enumerate(results, 1):
            doc = int_to_doc.get(r.id)
            if not doc:
                continue
            is_rel = r.id in set(relevant_ids)
            hits += is_rel
            marker = "✓" if is_rel else "✗"
            print(f"  [{rank}] {marker} [{doc['tier'].upper():8}] score={r.score:.3f}")
            print(f"       Domain:{doc['domain']} Area:{doc.get('area','—')}")
            print(f"       {doc['text'][:180]}")
        print(f"\n  P@{args.k} = {hits/args.k:.2f}  ({hits}/{args.k} relevant)")

    # Story examples
    story_sample = rng.sample(stories, min(args.n_story, len(stories)))
    for s in story_sample:
        key = (s["domain"], s["area"])
        rel = [id_to_int[d["id"]] for d in content_by_area.get(key, [])]
        if rel:
            show_query(s, "story", rel)

    # Narrative examples
    narr_sample = rng.sample(narratives, min(args.n_narrative, len(narratives)))
    for n in narr_sample:
        rel = [id_to_int[d["id"]] for d in content_by_domain.get(n["domain"], [])]
        if rel:
            show_query(n, "narrative", rel)

if __name__ == "__main__":
    main()
