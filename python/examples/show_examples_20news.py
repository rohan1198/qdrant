#!/usr/bin/env python3
"""
show_examples_20news.py — Show example queries and retrieved docs for 20 Newsgroups.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/show_examples_20news.py \
        --head data/trained/projection_head_v3.pt
"""
import sys, json, argparse, random
from pathlib import Path
from collections import defaultdict

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

NEWSGROUPS = [
    'alt.atheism',
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x',
    'misc.forsale',
    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
    'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
    'soc.religion.christian',
    'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
    'talk.religion.misc',
]
TOP = ['alt', 'comp', 'misc', 'rec', 'sci', 'soc', 'talk']


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head",        required=True)
    p.add_argument("--k",           type=int, default=5)
    p.add_argument("--n-story",     type=int, default=3)
    p.add_argument("--n-narrative", type=int, default=2)
    p.add_argument("--url",         default="http://localhost:6333")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--jsonl",       default=str(root / "data/20news/documents.jsonl"))
    args = p.parse_args()

    from hyperbolic.client import HyperbolicClient
    from hyperbolic.encoder.trained_encoder import TrainedEncoder
    from hyperbolic.pipeline.config import PipelineConfig
    from hyperbolic.search.strategies import run_search

    raw_docs = [json.loads(l) for l in open(args.jsonl)]
    for i, d in enumerate(raw_docs):
        d["_int_id"] = i

    int_to_doc      = {d["_int_id"]: d for d in raw_docs}
    content_by_ng:     dict[int, list[int]] = defaultdict(list)
    content_by_domain: dict[int, list[int]] = defaultdict(list)
    stories, narratives = [], []

    for d in raw_docs:
        if d["tier"] == "content":
            content_by_ng[d["area"]].append(d["_int_id"])
            content_by_domain[d["domain"]].append(d["_int_id"])
        elif d["tier"] == "story":
            stories.append(d)
        elif d["tier"] == "narrative":
            narratives.append(d)

    cfg = PipelineConfig(query_pipeline="busemann", curvature=1.0, stage_sizes=(1000, 50))
    enc    = TrainedEncoder(args.head)
    client = HyperbolicClient.connect(args.url)

    def show_query(q_doc, tier, relevant_ids):
        ng_name  = NEWSGROUPS[q_doc["area"]] if tier == "story" else TOP[q_doc["domain"]]
        rel_set  = set(relevant_ids)

        print(f"\n{'═'*72}")
        print(f"QUERY [{tier.upper()}]  —  {ng_name}")
        print(f"  Relevant docs: {len(relevant_ids)}")
        print(f"  Query text:")
        # Show first 400 chars of query
        for line in q_doc["text"][:400].split("\n"):
            if line.strip():
                print(f"    {line.strip()[:120]}")
        print(f"{'─'*72}")

        ev = enc.encode([q_doc["text"]], tiers=[tier])
        results = run_search(
            client, "eval_20news_trained",
            ev.dense[0], ev.poincare[0], ev.tangent[0],
            config=cfg, limit=args.k,
        )

        hits = 0
        for rank, r in enumerate(results, 1):
            doc = int_to_doc.get(r.id)
            if not doc:
                continue
            is_rel = r.id in rel_set
            hits  += is_rel
            marker = "✓" if is_rel else "✗"
            ng = NEWSGROUPS[doc["area"]] if doc["area"] >= 0 else TOP[doc["domain"]]
            print(f"  [{rank}] {marker}  score={r.score:.3f}  [{ng}]")
            # Show first non-empty line of the doc
            snippet = next((l.strip() for l in doc["text"].split("\n") if len(l.strip()) > 20), doc["text"][:120])
            print(f"       {snippet[:120]}")
        print(f"\n  P@{args.k} = {hits/args.k:.2f}  ({hits}/{args.k} relevant)")

    rng = random.Random(args.seed)

    story_sample = rng.sample(stories, min(args.n_story, len(stories)))
    for s in story_sample:
        rel = content_by_ng.get(s["area"], [])
        if rel:
            show_query(s, "story", rel)

    narr_sample = rng.sample(narratives, min(args.n_narrative, len(narratives)))
    for n in narr_sample:
        rel = content_by_domain.get(n["domain"], [])
        if rel:
            show_query(n, "narrative", rel)


if __name__ == "__main__":
    main()
