#!/usr/bin/env python3
"""
train_projection.py — Train a ProjectionHead on the WOS hierarchy.

What this trains:
  A small MLP (1024 → 512 → 128, ~590K params) on top of frozen pplx-embed.
  Loss = InfoNCE (pulls descendants together in dense space so cosine ANN works)
       + Busemann-triplet (places descendants deeper toward anchor's ideal point)

Best practices applied:
  - Pre-computed embeddings — frozen encoder runs once, not at every step
  - AdamW (Adam + decoupled weight decay) — better than plain Adam
  - OneCycleLR with 10% warmup — robust to LR choice, avoids early overfitting
  - Train/val split (90/10 by area) — val loss is the early-stopping signal
  - Early stopping (patience=3) — stops when val loss plateaus, not at fixed epochs
  - Dropout(0.1) in projection head — prevents co-adaptation of neurons
  - Label smoothing (0.05) on InfoNCE — prevents overconfidence on easy negatives
  - Hard negatives — same domain, different area (semantically close, wrong branch)
  - Gradient clipping (max_norm=1.0) — prevents gradient explosion

Usage:
    # Step 1: embed once (fast, only needed once)
    CUDA_VISIBLE_DEVICES=1 python3 scripts/precompute_embeddings.py

    # Step 2: train (fast — each epoch ~30s instead of ~60min)
    python3 scripts/train_projection.py
    python3 scripts/train_projection.py --max-epochs 20 --patience 3
    CUDA_VISIBLE_DEVICES=1 python3 scripts/train_projection.py
"""
import sys
import argparse
import logging
import random
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WOS_JSONL  = root / "qdrant" / "dev" / "testbench" / "data" / "wos" / "documents.jsonl"
EMBED_PATH = root / "data" / "wos" / "embeddings.npz"
SAVE_DIR   = root / "data" / "trained"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wos",        default=str(WOS_JSONL))
    p.add_argument("--embeddings", default=str(EMBED_PATH), help="Pre-computed embeddings npz")
    p.add_argument("--areas",      type=int,   default=404,   help="WOS areas (404 = all)")
    p.add_argument("--val-frac",   type=float, default=0.10,  help="Fraction of areas held out for val")
    p.add_argument("--max-epochs", type=int,   default=20,    help="Upper bound — early stopping usually fires first")
    p.add_argument("--patience",   type=int,   default=3,     help="Early stop after N epochs no val improvement")
    p.add_argument("--batch",      type=int,   default=128,   help="Batch size (can be large — no encoder inference)")
    p.add_argument("--n-neg",      type=int,   default=7,     help="Hard negatives per triplet")
    p.add_argument("--lr",         type=float, default=3e-4,  help="OneCycleLR peak LR")
    p.add_argument("--temp",       type=float, default=0.07,  help="InfoNCE temperature")
    p.add_argument("--lambda-bus", type=float, default=0.5,   help="Busemann loss weight")
    p.add_argument("--margin",     type=float, default=0.5,   help="Busemann triplet margin")
    p.add_argument("--hidden",     type=int,   default=512)
    p.add_argument("--dim",        type=int,   default=128,   help="Poincaré output dim")
    p.add_argument("--curvature",  type=float, default=1.0)
    p.add_argument("--mode",       default="mixed",           help="story|narrative|mixed")
    p.add_argument("--save",       default=str(SAVE_DIR / "projection_head.pt"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wos_all(jsonl_path: str, n_areas: int = 404) -> list[dict]:
    """Load WOS docs, sampling up to n_areas areas."""
    import json
    from collections import defaultdict

    TIER_MAP = {"leaf": "content", "mid": "story", "root": "narrative"}
    rng = random.Random(42)

    leaves_by_area: dict = defaultdict(list)
    mids:  list = []
    roots: list = []
    all_areas: set = set()

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            key = (d["domain"], d["area"])
            all_areas.add(key)
            d["tier"] = TIER_MAP.get(d["tier"], d["tier"])
            if d["tier"] == "content":
                leaves_by_area[key].append(d)
            elif d["tier"] == "story":
                mids.append(d)
            elif d["tier"] == "narrative":
                roots.append(d)

    sampled     = rng.sample(sorted(all_areas), min(n_areas, len(all_areas)))
    sampled_set = set(sampled)

    docs = list(roots)
    docs += [d for d in mids if (d["domain"], d["area"]) in sampled_set]
    for key in sampled:
        docs += leaves_by_area.get(key, [])

    logger.info(
        "Loaded %d docs from %d areas  (%d content, %d story, %d narrative)",
        len(docs), len(sampled),
        sum(1 for d in docs if d["tier"] == "content"),
        sum(1 for d in docs if d["tier"] == "story"),
        sum(1 for d in docs if d["tier"] == "narrative"),
    )
    return docs


def load_embeddings(npz_path: str) -> dict[str, np.ndarray]:
    """Load pre-computed embeddings from npz → {id: embedding}."""
    logger.info("Loading pre-computed embeddings from %s", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    ids        = data["ids"]
    embeddings = data["embeddings"]
    emb_map = {str(ids[i]): embeddings[i] for i in range(len(ids))}
    logger.info("Loaded %d embeddings, dim=%d", len(emb_map), embeddings.shape[1])
    return emb_map


def train_val_split(raw_docs: list[dict], val_frac: float, seed: int = 42):
    """Split by area so val areas are entirely unseen during training."""
    from collections import defaultdict
    rng = random.Random(seed)

    areas = sorted({(d["domain"], d["area"]) for d in raw_docs if d["tier"] != "narrative"})
    rng.shuffle(areas)
    n_val = int(len(areas) * val_frac)  # 0 when val_frac=0 — no val split
    val_areas   = set(areas[:n_val])
    train_areas = set(areas[n_val:])

    def belongs(d, area_set):
        if d["tier"] == "narrative":
            return True  # included in both splits
        return (d["domain"], d["area"]) in area_set

    train_docs = [d for d in raw_docs if belongs(d, train_areas)]
    val_docs   = [d for d in raw_docs if belongs(d, val_areas)]
    logger.info(
        "Split: %d train docs (%d areas) | %d val docs (%d areas)",
        len(train_docs), len(train_areas), len(val_docs), len(val_areas),
    )
    return train_docs, val_docs


# ---------------------------------------------------------------------------
# One epoch — no encoder inference, pure head forward/backward
# ---------------------------------------------------------------------------

def run_epoch(head, loader, device, optimizer, scheduler, args, train: bool):
    """Run one train or val epoch. Returns (avg_total, avg_infonce, avg_busemann)."""
    from hyperbolic.training.losses import combined_loss

    head.train(train)
    total_info = total_bus = total_loss = n_batches = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            anc_dense = batch["anchor_embs"].to(device)    # [B, 1024]
            pos_dense = batch["positive_embs"].to(device)  # [B, 1024]
            neg_dense = batch["negative_embs"].to(device)  # [B, N, 1024]
            anchor_tiers = batch["anchor_tiers"]

            B, N, _ = neg_dense.shape

            anc_p, anc_proj = head(anc_dense, tiers=anchor_tiers)
            pos_p, pos_proj = head(pos_dense, tiers=["content"] * B)
            neg_p_flat, neg_proj_flat = head(
                neg_dense.reshape(B * N, -1), tiers=["content"] * (B * N)
            )
            neg_p    = neg_p_flat.view(B, N, -1)
            neg_proj = neg_proj_flat.view(B, N, -1)

            loss, comps = combined_loss(
                anc_proj, pos_proj, neg_proj,
                anc_p, pos_p, neg_p,
                temperature=args.temp,
                curvature=args.curvature,
                margin=args.margin,
                lambda_bus=args.lambda_bus,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_info += comps["infonce"]
            total_bus  += comps["busemann"]
            total_loss += comps["total"]
            n_batches  += 1

    n = max(n_batches, 1)
    return total_loss / n, total_info / n, total_bus / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────
    raw_docs = load_wos_all(args.wos, n_areas=args.areas)

    # Check for pre-computed embeddings — required
    embed_path = Path(args.embeddings)
    if not embed_path.exists():
        logger.error(
            "Pre-computed embeddings not found at %s\n"
            "Run first:  CUDA_VISIBLE_DEVICES=1 python3 scripts/precompute_embeddings.py",
            embed_path,
        )
        sys.exit(1)

    emb_map = load_embeddings(args.embeddings)

    train_docs, val_docs = train_val_split(raw_docs, val_frac=args.val_frac)

    from hyperbolic.training.triplet_dataset import HierarchyTripletDataset, collate_triplets
    train_ds = HierarchyTripletDataset(train_docs, emb_map=emb_map, mode=args.mode, n_neg=args.n_neg, seed=42)
    val_ds   = HierarchyTripletDataset(val_docs,   emb_map=emb_map, mode=args.mode, n_neg=args.n_neg, seed=99)
    logger.info("Triplets: %d train | %d val", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_triplets, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              collate_fn=collate_triplets, num_workers=4, pin_memory=True)

    # ── Projection head ───────────────────────────────────────────────
    from hyperbolic.training.projection_head import ProjectionHead
    head = ProjectionHead(
        input_dim=1024, hidden_dim=args.hidden,
        output_dim=args.dim, curvature=args.curvature,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    logger.info("ProjectionHead: %d params (hidden=%d, out=%d)", n_params, args.hidden, args.dim)

    # ── Optimizer + scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )

    total_steps = args.max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )
    logger.info(
        "Optimizer: AdamW(lr=%.2e, wd=1e-4) | OneCycleLR(warmup=10%%, max_steps=%d)",
        args.lr, total_steps,
    )

    # ── Training loop with early stopping ────────────────────────────
    best_val      = float("inf")
    patience      = 0
    best_path     = args.save
    has_val_data  = len(val_ds) > 0   # False when val_frac=0 or no val triplets

    logger.info("Starting training (max_epochs=%d, patience=%d)", args.max_epochs, args.patience)
    logger.info("─" * 70)

    for epoch in range(1, args.max_epochs + 1):
        tr_loss, tr_info, tr_bus = run_epoch(
            head, train_loader, device, optimizer, scheduler, args, train=True
        )
        vl_loss, vl_info, vl_bus = run_epoch(
            head, val_loader, device, None, None, args, train=False
        )

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %2d/%d  train=%.4f (info=%.4f bus=%.4f)  "
            "val=%.4f (info=%.4f bus=%.4f)  lr=%.2e",
            epoch, args.max_epochs,
            tr_loss, tr_info, tr_bus,
            vl_loss, vl_info, vl_bus,
            lr_now,
        )

        monitor = tr_loss if not has_val_data else vl_loss
        if monitor < best_val - 1e-4:
            best_val = monitor
            patience = 0
            head.save(best_path)
            logger.info("  ✓ New best %s=%.4f — saved to %s",
                        "train" if not has_val_data else "val", best_val, best_path)
        else:
            patience += 1
            logger.info("  ✗ No improvement (patience %d/%d)", patience, args.patience)
            if patience >= args.patience:
                logger.info("Early stopping at epoch %d (best=%.4f)", epoch, best_val)
                break

    logger.info("─" * 70)
    logger.info("Training complete. Best loss: %.4f", best_val)
    logger.info("Checkpoint: %s", best_path)
    logger.info("")
    logger.info("Next step — evaluate the trained encoder:")
    logger.info("  python3 scripts/run_eval_trained.py --head %s", best_path)


if __name__ == "__main__":
    main()
