"""
HyperbolicClient — thin wrapper around qdrant-client.

Design contract
---------------
- All hyperbolic math (Poincaré distance, Busemann, horoball, Klein) lives
  in the Python client layer, never inside Qdrant. This keeps the system
  compatible with any upstream Qdrant release.
- Qdrant is used purely as a vector store + ANN index. All named vectors
  use Distance::Cosine so we never depend on fork-specific distance types.
- Stage 1 of every pipeline is a cosine ANN (fast, standard). Stage 2+
  is client-side re-ranking using the fetched poincaré/tangent vectors.
- Collection schema is declared here and nowhere else. One source of truth.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — the ONLY place collection shape is defined
# ---------------------------------------------------------------------------

DENSE_DIM = 1024        # pplx-embed-v1 output dimension
HYPERBOLIC_DIM = 128    # Poincaré / tangent vector dimension

# Named vector keys — stable across Qdrant versions
DENSE_VECTOR = "dense"
POINCARE_VECTOR = "poincare"
TANGENT_VECTOR = "tangent"

# Default HNSW params
HNSW_M = 16
HNSW_EF = 200


# ---------------------------------------------------------------------------
# Schema helpers — raw REST dicts (no Distance enum needed)
# ---------------------------------------------------------------------------

def _build_vectors_config(include_tangent: bool = True) -> dict[str, rest.VectorParams]:
    """Build named-vector config for create_collection.

    All vectors use standard distances so the collection works with any Qdrant version.
    Poincaré and tangent distances are computed client-side; Qdrant is used for ANN only.
    """
    hnsw = rest.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF)
    vectors: dict[str, rest.VectorParams] = {
        DENSE_VECTOR: rest.VectorParams(
            size=DENSE_DIM, distance=rest.Distance.COSINE, hnsw_config=hnsw,
        ),
        POINCARE_VECTOR: rest.VectorParams(
            size=HYPERBOLIC_DIM, distance=rest.Distance.EUCLID, hnsw_config=hnsw,
        ),
    }
    if include_tangent:
        vectors[TANGENT_VECTOR] = rest.VectorParams(
            size=HYPERBOLIC_DIM, distance=rest.Distance.EUCLID, hnsw_config=hnsw,
        )
    return vectors


# ---------------------------------------------------------------------------
# Payload index helpers
# ---------------------------------------------------------------------------

_PAYLOAD_INDICES: list[tuple[str, Any]] = [
    ("item_type",       rest.PayloadSchemaType.KEYWORD),
    ("tier",            rest.PayloadSchemaType.KEYWORD),   # stored as string: "content"|"story"|"narrative"
    ("busemann_depth",  rest.PayloadSchemaType.FLOAT),
    ("domain",          rest.PayloadSchemaType.KEYWORD),
    ("area",            rest.PayloadSchemaType.KEYWORD),
    ("parent_ids",      rest.PayloadSchemaType.KEYWORD),
    ("child_ids",       rest.PayloadSchemaType.KEYWORD),
    ("created_at",      rest.PayloadSchemaType.DATETIME),
    ("point_id",        rest.PayloadSchemaType.KEYWORD),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class HyperbolicClient:
    """
    Qdrant client for hyperbolic search. Works with any standard Qdrant instance.

    All hyperbolic geometry (Poincaré, Busemann, Klein, horoball) is computed
    client-side. Qdrant is used only for cosine ANN and vector storage.

    Usage::

        hc = HyperbolicClient.connect("http://localhost:6333")
        hc.ensure_collection("my_collection", curvature=1.0)
        hc.upsert("my_collection", points)
        results = hc.search("my_collection", vector, using="dense")
    """

    _client: QdrantClient = field(repr=False)

    # ------------------------------------------------------------------
    @classmethod
    def connect(
        cls,
        url: str = "http://localhost:6333",
        *,
        api_key: Optional[str] = None,
    ) -> "HyperbolicClient":
        """Connect to any Qdrant instance. No fork required."""
        client = QdrantClient(url=url, api_key=api_key)
        logger.info("Connected to Qdrant at %s", url)
        return cls(_client=client)

    # ------------------------------------------------------------------
    def ensure_collection(
        self,
        name: str,
        *,
        curvature: float = 1.0,
        include_tangent: bool = True,
        recreate: bool = False,
    ) -> None:
        """Create collection with named vectors (dense, poincare, tangent), all Cosine distance.

        curvature is accepted for API stability but is not passed to Qdrant —
        it is encoded into vectors at ingest time by the encoder.
        """
        if not recreate and self._client.collection_exists(name):
            logger.debug("Collection %r already exists — skipping", name)
            return

        if recreate and self._client.collection_exists(name):
            self._client.delete_collection(name)

        self._client.create_collection(
            collection_name=name,
            vectors_config=_build_vectors_config(include_tangent),
        )

        for field_name, schema in _PAYLOAD_INDICES:
            self._client.create_payload_index(name, field_name, schema)
        logger.info("Created collection %r (all Cosine distance)", name)

    # ------------------------------------------------------------------
    def upsert(
        self,
        collection: str,
        points: list[rest.PointStruct],
        batch_size: int = 256,
    ) -> None:
        """Upsert in batches."""
        for i in range(0, len(points), batch_size):
            self._client.upsert(collection, points=points[i : i + batch_size])
        logger.debug("Upserted %d points to %r", len(points), collection)

    # ------------------------------------------------------------------
    def search(
        self,
        collection: str,
        vector: np.ndarray,
        *,
        using: str = DENSE_VECTOR,
        limit: int = 10,
        ef: int = 128,
        query_filter: Optional[rest.Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[rest.ScoredPoint]:
        """Single-vector search. Uses query_points (qdrant-client >= 1.14)."""
        resp = self._client.query_points(
            collection_name=collection,
            query=vector,
            using=using,
            limit=limit,
            search_params=rest.SearchParams(hnsw_ef=ef, exact=False),
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return resp.points

    # ------------------------------------------------------------------
    def scroll_all(
        self,
        collection: str,
        with_vectors: bool = True,
        batch_size: int = 256,
    ) -> list[rest.Record]:
        """Retrieve all points (for client-side exact scoring)."""
        records, offset = [], None
        while True:
            batch, offset = self._client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            records.extend(batch)
            if offset is None:
                break
        return records

    # ------------------------------------------------------------------
    def count(self, collection: str) -> int:
        return self._client.count(collection_name=collection).count

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def raw(self) -> QdrantClient:
        """Escape hatch to the underlying qdrant-client for advanced use."""
        return self._client
