#!/usr/bin/env python3
"""
precompute_20news_embeddings.py — Embed all 20news docs with frozen pplx-embed.

Run build_20news.py first to create data/20news/documents.jsonl.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/precompute_20news_embeddings.py
"""
import sys, json, logging
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

JSONL_PATH = root / "data" / "20news" / "documents.jsonl"
EMBED_OUT  = root / "data" / "20news" / "embeddings.npz"
BASE_MODEL = "perplexity-ai/pplx-embed-v1-0.6B"


def main():
    if not JSONL_PATH.exists():
        logger.error("Missing %s — run build_20news.py first", JSONL_PATH)
        sys.exit(1)

    docs = [json.loads(l) for l in open(JSONL_PATH)]
    ids   = [d["id"]   for d in docs]
    texts = [d["text"] for d in docs]
    logger.info("%d docs to embed", len(texts))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading frozen encoder: %s  (device=%s)", BASE_MODEL, device)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(BASE_MODEL, trust_remote_code=True, device=device)

    logger.info("Embedding %d docs in batches of 32…", len(texts))
    embeddings = st.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    EMBED_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(EMBED_OUT), ids=np.array(ids, dtype=object), embeddings=embeddings)
    logger.info("Saved to %s  (%.1f MB)", EMBED_OUT, EMBED_OUT.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
