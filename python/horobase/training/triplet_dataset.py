"""
TripletDataset — samples (anchor, positive, negative) from a WOS-style hierarchy.

Triplet types:
  story_descendant:
    anchor   = story-tier doc (mid)
    positive = content doc in that story's area
    negative = content doc from a DIFFERENT area (same domain preferred)

  narrative_descendant:
    anchor   = narrative-tier doc (root)
    positive = content doc anywhere under that narrative
    negative = content doc from a DIFFERENT domain

Both loss heads (InfoNCE + Busemann-triplet) use the same triplets.
The batch is built so each anchor appears once with one positive and
N_NEG hard negatives drawn from other docs.

Embeddings are pre-computed and passed in as emb_map (id → np.ndarray)
so no model inference happens at training time — each epoch is ~30s
instead of ~60min.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class HierarchyTripletDataset(Dataset):
    """
    Builds triplets from a flat list of raw_docs with tier/domain/area fields.

    Parameters
    ----------
    raw_docs : list[dict]
        Each doc has: id, text, tier, domain, area
    emb_map : dict[str, np.ndarray]
        Pre-computed embeddings keyed by doc id.
    mode : "story" | "narrative" | "mixed"
        Which anchor tier to sample from.
    n_neg : int
        Hard negatives per anchor (same domain, different area).
    seed : int
    """

    def __init__(
        self,
        raw_docs: list[dict],
        emb_map: dict[str, np.ndarray],
        mode: str = "mixed",
        n_neg: int = 7,
        seed: int = 42,
    ):
        self.mode    = mode
        self.n_neg   = n_neg
        self.emb_map = emb_map
        self.rng     = random.Random(seed)

        # Index docs by tier and (domain, area)
        self._content_by_area:   dict[tuple, list[dict]] = defaultdict(list)
        self._content_by_domain: dict[int,   list[dict]] = defaultdict(list)
        self._stories_by_domain: dict[int,   list[dict]] = defaultdict(list)
        self._stories:    list[dict] = []
        self._narratives: list[dict] = []

        for d in raw_docs:
            if d["id"] not in emb_map:
                continue  # skip docs without embeddings
            t   = d["tier"]
            key = (d["domain"], d["area"])
            if t == "content":
                self._content_by_area[key].append(d)
                self._content_by_domain[d["domain"]].append(d)
            elif t == "story":
                self._stories.append(d)
                self._stories_by_domain[d["domain"]].append(d)
            elif t == "narrative":
                self._narratives.append(d)

        self._triplets = self._build_triplets()

    # ------------------------------------------------------------------
    def _build_triplets(self) -> list[dict]:
        triplets = []

        if self.mode in ("story", "mixed"):
            for story in self._stories:
                key      = (story["domain"], story["area"])
                positives = self._content_by_area.get(key, [])
                if not positives:
                    continue
                # Hard negatives: same-domain/different-area content + sibling stories
                # + sample of cross-domain content (teaches explicit domain boundaries)
                content_negs = [
                    d for d in self._content_by_domain.get(story["domain"], [])
                    if (d["domain"], d["area"]) != key
                ]
                story_negs = [
                    d for d in self._stories_by_domain.get(story["domain"], [])
                    if (d["domain"], d["area"]) != key
                ]
                cross_domain = [
                    d for dom, docs in self._content_by_domain.items()
                    if dom != story["domain"]
                    for d in docs
                ]
                # Sample cross-domain to avoid overwhelming same-domain negatives
                cross_sample = self.rng.sample(cross_domain, min(len(content_negs), len(cross_domain)))
                neg_pool = content_negs + story_negs + cross_sample
                if not neg_pool:
                    continue
                for pos in positives:
                    negs = self.rng.sample(neg_pool, min(self.n_neg, len(neg_pool)))
                    triplets.append({
                        "anchor_id":   story["id"],
                        "anchor_tier": "story",
                        "positive_id": pos["id"],
                        "negative_ids": [n["id"] for n in negs],
                    })

        if self.mode in ("narrative", "mixed"):
            for narr in self._narratives:
                domain    = narr["domain"]
                positives = self._content_by_domain.get(domain, [])
                if not positives:
                    continue
                # Hard negatives: content + stories from different domains
                # Stories from other domains are hardest — same abstraction level, wrong branch
                content_negs = [
                    d for dom, docs in self._content_by_domain.items()
                    if dom != domain for d in docs
                ]
                story_negs = [
                    d for dom, docs in self._stories_by_domain.items()
                    if dom != domain for d in docs
                ]
                neg_pool = content_negs + story_negs
                if not neg_pool:
                    continue
                pos_sample = self.rng.sample(positives, min(200, len(positives)))
                for pos in pos_sample:
                    negs = self.rng.sample(neg_pool, min(self.n_neg, len(neg_pool)))
                    triplets.append({
                        "anchor_id":   narr["id"],
                        "anchor_tier": "narrative",
                        "positive_id": pos["id"],
                        "negative_ids": [n["id"] for n in negs],
                    })

        self.rng.shuffle(triplets)
        return triplets

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, idx: int) -> dict:
        t = self._triplets[idx]
        return {
            "anchor_emb":    self.emb_map[t["anchor_id"]],
            "anchor_tier":   t["anchor_tier"],
            "positive_emb":  self.emb_map[t["positive_id"]],
            "negative_embs": np.stack([self.emb_map[nid] for nid in t["negative_ids"]]),
        }


def collate_triplets(batch: list[dict]) -> dict:
    """Collate a list of triplet dicts into tensors ready for the projection head."""
    anchor_tiers = [b["anchor_tier"] for b in batch]

    # Stack embeddings into tensors
    anchor_embs  = torch.from_numpy(np.stack([b["anchor_emb"]   for b in batch]))
    positive_embs = torch.from_numpy(np.stack([b["positive_emb"] for b in batch]))

    # Pad negatives to same N_NEG within batch (repeat last if short)
    max_neg = max(b["negative_embs"].shape[0] for b in batch)
    neg_list = []
    for b in batch:
        neg = b["negative_embs"]
        if neg.shape[0] < max_neg:
            pad = np.repeat(neg[-1:], max_neg - neg.shape[0], axis=0)
            neg = np.concatenate([neg, pad], axis=0)
        neg_list.append(neg)
    negative_embs = torch.from_numpy(np.stack(neg_list))  # [B, N, 1024]

    return {
        "anchor_embs":   anchor_embs,    # [B, 1024]
        "anchor_tiers":  anchor_tiers,   # list[str]
        "positive_embs": positive_embs,  # [B, 1024]
        "negative_embs": negative_embs,  # [B, N, 1024]
    }
