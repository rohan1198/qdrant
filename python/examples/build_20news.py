#!/usr/bin/env python3
"""
build_20news.py — Convert 20 Newsgroups to our 3-tier JSONL format.

Hierarchy:
  narrative (7)  : top group  (alt, comp, misc, rec, sci, soc, talk)
  story    (20)  : newsgroup  (alt.atheism, comp.graphics, ...)
  content  (~18k): individual post

Story docs are synthesized by concatenating the 3 longest clean posts
from each newsgroup (gives a denser topic signal than a single post).

Narrative docs are synthesized by concatenating one story excerpt per
subgroup (gives a broad multi-topic overview of the top group).

Output: data/20news/documents.jsonl
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

root = Path(__file__).parent.parent
OUT  = root / "data" / "20news" / "documents.jsonl"


# ---------------------------------------------------------------------------
# Group → (domain, area) mapping
# ---------------------------------------------------------------------------

TOP = ['alt', 'comp', 'misc', 'rec', 'sci', 'soc', 'talk']
TOP_IDX = {t: i for i, t in enumerate(TOP)}

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
NG_IDX  = {ng: i for i, ng in enumerate(NEWSGROUPS)}
NG_TOP  = {ng: TOP_IDX[ng.split('.')[0]] for ng in NEWSGROUPS}

# subgroups per top group (for narrative synthesis)
TOP_TO_NG: dict[int, list[int]] = defaultdict(list)
for ng, idx in NG_IDX.items():
    TOP_TO_NG[NG_TOP[ng]].append(idx)


def clean(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def pick_longest(posts: list[str], n: int = 3, min_len: int = 100) -> list[str]:
    """Return up to n longest posts above min_len chars."""
    filtered = [p for p in posts if len(p.strip()) >= min_len]
    return sorted(filtered, key=len, reverse=True)[:n]


def main():
    from sklearn.datasets import fetch_20newsgroups

    print("Downloading 20 Newsgroups (remove headers/footers/quotes)…")
    data = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
    )
    print(f"  {len(data.data)} posts across {len(data.target_names)} newsgroups")

    # Bucket posts by newsgroup index
    posts_by_ng: dict[int, list[str]] = defaultdict(list)
    for text, target in zip(data.data, data.target):
        posts_by_ng[target].append(clean(text))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    docs = []
    content_per_ng: dict[int, list[dict]] = defaultdict(list)

    # ── Content tier ────────────────────────────────────────────────────
    for ng_idx, ng_name in enumerate(NEWSGROUPS):
        dom = NG_TOP[ng_name]
        for post_idx, text in enumerate(posts_by_ng[ng_idx]):
            if not text.strip():
                continue
            doc = {
                "id":     f"20news_{ng_idx}_{post_idx}",
                "text":   text[:2000],          # cap at ~500 tokens
                "tier":   "content",
                "domain": dom,
                "area":   ng_idx,
            }
            docs.append(doc)
            content_per_ng[ng_idx].append(doc)

    print(f"  Content: {len(docs)} posts")

    # ── Story tier (one synthetic doc per newsgroup) ─────────────────────
    story_docs = []
    story_texts: dict[int, str] = {}
    for ng_idx, ng_name in enumerate(NEWSGROUPS):
        dom = NG_TOP[ng_name]
        posts = posts_by_ng[ng_idx]
        excerpts = pick_longest(posts, n=3, min_len=150)
        if not excerpts:
            excerpts = [p for p in posts if p.strip()][:3]
        combined = "\n\n".join(e[:600] for e in excerpts)
        story_texts[ng_idx] = combined
        doc = {
            "id":     f"20news_story_{ng_idx}",
            "text":   combined,
            "tier":   "story",
            "domain": dom,
            "area":   ng_idx,
        }
        docs.append(doc)
        story_docs.append(doc)

    print(f"  Story: {len(story_docs)} synthetic docs")

    # ── Narrative tier (one synthetic doc per top group) ─────────────────
    narr_docs = []
    for dom_idx, top in enumerate(TOP):
        ng_indices = TOP_TO_NG[dom_idx]
        # One excerpt from each subgroup
        parts = []
        for ng_idx in ng_indices:
            txt = story_texts.get(ng_idx, "")
            parts.append(txt[:400])
        combined = "\n\n".join(parts)
        doc = {
            "id":     f"20news_narrative_{dom_idx}",
            "text":   combined,
            "tier":   "narrative",
            "domain": dom_idx,
            "area":   -1,           # spans all areas in domain
        }
        docs.append(doc)
        narr_docs.append(doc)

    print(f"  Narrative: {len(narr_docs)} synthetic docs")
    print(f"  Total: {len(docs)} docs")

    with open(OUT, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    print(f"\nSaved → {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")

    # Quick sanity check
    print("\nSanity check:")
    for dom_idx, top in enumerate(TOP):
        ng_count = len(TOP_TO_NG[dom_idx])
        c_count  = sum(len(content_per_ng[i]) for i in TOP_TO_NG[dom_idx])
        print(f"  {top}: {ng_count} newsgroups, {c_count} content posts")


if __name__ == "__main__":
    main()
