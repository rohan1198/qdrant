#!/usr/bin/env python3
"""
precompute_embeddings.py — Embed all WOS docs once with the frozen pplx-embed model.

Saves: data/wos/embeddings.npz
  ids        : str array [N]       — doc ids
  embeddings : float32 array [N, 1024]

This only needs to run once. Training then loads the npz and never touches
the 0.6B model again, cutting each training epoch from ~60min to ~30s.

Usage:
    python3 scripts/precompute_embeddings.py
    CUDA_VISIBLE_DEVICES=1 python3 scripts/precompute_embeddings.py
"""
import sys
import logging
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WOS_JSONL  = root / "qdrant" / "dev" / "testbench" / "data" / "wos" / "documents.jsonl"
EMBED_OUT  = root / "data" / "wos" / "embeddings.npz"
BASE_MODEL = "perplexity-ai/pplx-embed-v1-0.6B"


def main():
    from train_projection import load_wos_all

    logger.info("Loading WOS docs...")
    raw_docs = load_wos_all(str(WOS_JSONL), n_areas=404)
    ids   = [d["id"]   for d in raw_docs]
    texts = [d["text"] for d in raw_docs]
    logger.info("%d docs to embed", len(texts))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading frozen encoder: %s  (device=%s)", BASE_MODEL, device)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(BASE_MODEL, trust_remote_code=True, device=device)

    logger.info("Embedding %d docs in batches of 32...", len(texts))
    embeddings = st.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    logger.info("Embeddings shape: %s", embeddings.shape)

    EMBED_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(EMBED_OUT), ids=np.array(ids, dtype=object), embeddings=embeddings)
    logger.info("Saved to %s", EMBED_OUT)
    logger.info("File size: %.1f MB", EMBED_OUT.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
