"""
Ingestor — text → encode → Qdrant upsert.

Encoder-agnostic: works with PCAEncoder or NeuralEncoder interchangeably.
Collection schema management is delegated entirely to HyperbolicClient.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from qdrant_client.http import models as rest

from hyperbolic.client import HyperbolicClient, DENSE_VECTOR, POINCARE_VECTOR, TANGENT_VECTOR
from hyperbolic.encoder.base import Encoder
from hyperbolic.math.poincare import (
    compute_busemann_depths, compute_focal_direction, alpha_precompute_batch
)

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Input document for ingestion."""
    id: int
    text: str
    tier: str = "leaf"          # root | mid | leaf (or: theme | narrative | story | content)
    domain: str = ""
    area: str = ""
    item_type: str = "content"
    parent_ids: list[str] = None
    child_ids: list[str] = None
    created_at: Optional[str] = None
    extra_payload: dict[str, Any] = None

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.child_ids is None:
            self.child_ids = []
        if self.extra_payload is None:
            self.extra_payload = {}


class Ingestor:
    """
    Encodes documents and upserts to a unified Qdrant collection.

    Usage::

        ingestor = Ingestor(client, encoder, collection="uk_hyperbolic")
        ingestor.ingest(documents, batch_size=64)
    """

    def __init__(
        self,
        client: HyperbolicClient,
        encoder: Encoder,
        collection: str,
    ):
        self._client = client
        self._encoder = encoder
        self._collection = collection

    def ingest(
        self,
        documents: list[Document],
        batch_size: int = 64,
        curvature: Optional[float] = None,
    ) -> None:
        c = curvature or self._encoder.curvature

        # Encode in batches to avoid OOM on large corpora
        all_dense, all_poincare, all_tangent = [], [], []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [d.text for d in batch]
            tiers = [d.tier for d in batch]

            # PCAEncoder accepts tiers kwarg; NeuralEncoder ignores extras
            try:
                encoded = self._encoder.encode(texts, tiers=tiers)   # type: ignore[call-arg]
            except TypeError:
                encoded = self._encoder.encode(texts)

            all_dense.append(encoded.dense)
            all_poincare.append(encoded.poincare)
            all_tangent.append(encoded.tangent)

        dense_all = np.concatenate(all_dense, axis=0)
        poincare_all = np.concatenate(all_poincare, axis=0)
        tangent_all = np.concatenate(all_tangent, axis=0)

        # Compute batch-level metadata
        focal = compute_focal_direction(poincare_all, c)
        depths = compute_busemann_depths(poincare_all, c)
        alphas = alpha_precompute_batch(poincare_all, c)

        # Build PointStructs
        points: list[rest.PointStruct] = []
        for i, doc in enumerate(documents):
            payload = {
                "item_type": doc.item_type,
                "tier": doc.tier,
                "domain": doc.domain,
                "area": doc.area,
                "parent_ids": doc.parent_ids,
                "child_ids": doc.child_ids,
                "point_id": str(doc.id),
                "busemann_depth": float(depths[i]),
                "alpha": float(alphas[i]),
                **doc.extra_payload,
            }
            if doc.created_at:
                payload["created_at"] = doc.created_at

            points.append(rest.PointStruct(
                id=doc.id,
                vector={
                    DENSE_VECTOR:   dense_all[i].tolist(),
                    POINCARE_VECTOR: poincare_all[i].tolist(),
                    TANGENT_VECTOR:  tangent_all[i].tolist(),
                },
                payload=payload,
            ))

        self._client.upsert(self._collection, points, batch_size=256)
        logger.info("Ingested %d documents into %r", len(documents), self._collection)
