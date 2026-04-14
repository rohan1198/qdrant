#!/usr/bin/env python3
"""Unified, separate, and cosine collection creation for the testbench.

Creates 3 collection variants per dataset:
1. {dataset}_cosine       — cosine-only baseline, all tiers
2. {dataset}_unified      — named vectors (dense + poincare), all tiers
3. {dataset}_narratives / _stories / _content — separate per-tier

Each variant gets the same data with proper payload indices.
"""

import json
import time
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType,
    Filter, FieldCondition, MatchValue, Range,
)

# ID offset scheme for unified collections
CONTENT_ID_OFFSET = 0
STORY_ID_OFFSET = 100_000
NARRATIVE_ID_OFFSET = 200_000
THEME_ID_OFFSET = 300_000

TIER_OFFSETS = {
    "content": CONTENT_ID_OFFSET,
    "story": STORY_ID_OFFSET,
    "narrative": NARRATIVE_ID_OFFSET,
    "theme": THEME_ID_OFFSET,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rest_create_collection(qdrant_url: str, name: str, vectors_config: dict) -> None:
    """Delete existing collection via REST, then create new one with given vector config.

    vectors_config can be a flat dict (single vector) or nested dict (named vectors).
    Uses requests.put() against the Qdrant REST API.
    """
    url = f"{qdrant_url}/collections/{name}"

    del_resp = requests.delete(url)
    if del_resp.status_code not in (200, 404):
        print(
            f"  Warning: unexpected status {del_resp.status_code} "
            f"when deleting '{name}': {del_resp.text[:200]}"
        )

    resp = requests.put(url, json={"vectors": vectors_config})
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to create collection '{name}': "
            f"HTTP {resp.status_code} — {resp.text[:400]}"
        )
    print(f"  Created collection '{name}'")


def _create_payload_indices(qdrant_url: str, name: str) -> None:
    """Create payload indices for fast filtering on common fields.

    Indexed fields:
      item_type     (keyword)   — document type tag
      tier          (integer)   — numeric tier level
      busemann_depth (float)    — hyperbolic depth for range queries
      domain        (keyword)   — top-level subject domain
      area          (keyword)   — sub-domain / area label
      parent_ids    (keyword)   — IDs of parent nodes in hierarchy
      child_ids     (keyword)   — IDs of child nodes in hierarchy
      created_at    (datetime)  — ISO-8601 timestamp
      point_id      (keyword)   — stable string identifier
    """
    base_url = f"{qdrant_url}/collections/{name}/index"

    field_specs = [
        ("item_type",      "keyword"),
        ("tier",           "integer"),
        ("busemann_depth", "float"),
        ("domain",         "keyword"),
        ("area",           "keyword"),
        ("parent_ids",     "keyword"),
        ("child_ids",      "keyword"),
        ("created_at",     "datetime"),
        ("point_id",       "keyword"),
    ]

    for field_name, field_type in field_specs:
        resp = requests.put(
            base_url,
            json={"field_name": field_name, "field_schema": field_type},
        )
        if resp.status_code not in (200, 201):
            print(
                f"  Warning: index creation for '{field_name}' returned "
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
        else:
            print(f"    Index created: {field_name} ({field_type})")


# ---------------------------------------------------------------------------
# Collection creation
# ---------------------------------------------------------------------------


def create_cosine_collection(
    qdrant_url: str,
    dataset: str,
    dim: int = 1024,
) -> str:
    """Create {dataset}_cosine with a single dense vector using Cosine distance.

    Args:
        qdrant_url: Qdrant REST API URL (e.g. "http://localhost:6334").
        dataset:    Dataset prefix (e.g. "wos", "bgc", "eurlex").
        dim:        Dense vector dimension (default 1024).

    Returns:
        The collection name that was created.
    """
    name = f"{dataset}_cosine"
    vectors_config = {
        "size": dim,
        "distance": "Cosine",
        "hnsw_config": {"m": 16, "ef_construct": 200},
    }
    print(f"Creating cosine collection '{name}' (dim={dim}) ...")
    _rest_create_collection(qdrant_url, name, vectors_config)
    _create_payload_indices(qdrant_url, name)
    return name


def create_unified_collection(
    qdrant_url: str,
    dataset: str,
    dense_dim: int = 1024,
    poincare_dim: int = 128,
    curvature: float = 5.0,
    tangent_dim: int | None = None,
) -> str:
    """Create {dataset}_unified with named vectors: "dense" (Cosine) + "poincare" (Poincare).

    The poincare vector uses HNSW m=16, ef_construct=200 and stores the curvature
    parameter for the distance computation.  Falls back gracefully if the server
    does not support the curvature field.

    Args:
        qdrant_url:   Qdrant REST API URL.
        dataset:      Dataset prefix.
        dense_dim:    Dimension of the dense (cosine) vector (default 1024).
        poincare_dim: Dimension of the Poincaré ball vector (default 128).
        curvature:    Curvature parameter c for the Poincaré ball (default 5.0).

    Returns:
        The collection name that was created.
    """
    name = f"{dataset}_unified"
    url = f"{qdrant_url}/collections/{name}"

    # Delete if exists
    del_resp = requests.delete(url)
    if del_resp.status_code not in (200, 404):
        print(
            f"  Warning: unexpected status {del_resp.status_code} "
            f"when deleting '{name}': {del_resp.text[:200]}"
        )

    vectors_config: dict = {
        "dense": {
            "size": dense_dim,
            "distance": "Cosine",
            "hnsw_config": {"m": 16, "ef_construct": 200},
        },
        "poincare": {
            "size": poincare_dim,
            "distance": "Poincare",
            "hnsw_config": {"m": 16, "ef_construct": 200},
            "curvature": curvature,
        },
    }

    if tangent_dim is not None:
        vectors_config["tangent"] = {
            "size": tangent_dim,
            "distance": "Euclid",
        }

    body = {"vectors": vectors_config}

    print(
        f"Creating unified collection '{name}' "
        f"(dense={dense_dim}d Cosine, poincare={poincare_dim}d c={curvature}) ..."
    )
    resp = requests.put(url, json=body)
    if resp.status_code in (200, 201):
        print(f"  Created '{name}' (dense + poincare c={curvature})")
        _create_payload_indices(qdrant_url, name)
        return name

    # Retry without curvature if the server doesn't support it
    print(
        f"  Note: PUT returned {resp.status_code}, retrying without curvature field. "
        f"Response: {resp.text[:300]}"
    )
    body["vectors"]["poincare"].pop("curvature")
    resp2 = requests.put(url, json=body)
    if resp2.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to create unified collection '{name}': "
            f"HTTP {resp2.status_code} — {resp2.text[:400]}"
        )
    print(f"  Created '{name}' (curvature param unsupported — using server default)")
    _create_payload_indices(qdrant_url, name)
    return name


def create_separate_collections(
    qdrant_url: str,
    dataset: str,
    dense_dim: int = 1024,
    poincare_dim: int = 128,
    curvature: float = 5.0,
) -> dict[str, str]:
    """Create per-tier collections: {dataset}_narratives, _stories, _content.

    Each collection holds named vectors ("dense" + "poincare") for a single tier.
    Falls back gracefully if the server does not support the curvature field.

    Args:
        qdrant_url:   Qdrant REST API URL.
        dataset:      Dataset prefix.
        dense_dim:    Dimension of the dense vector (default 1024).
        poincare_dim: Dimension of the Poincaré ball vector (default 128).
        curvature:    Curvature parameter c (default 5.0).

    Returns:
        Dict mapping tier name -> collection name:
        {"narrative": "{dataset}_narratives",
         "story":     "{dataset}_stories",
         "content":   "{dataset}_content"}
    """
    tier_suffixes = {
        "narrative": "narratives",
        "story":     "stories",
        "content":   "content",
    }

    result: dict[str, str] = {}

    for tier, suffix in tier_suffixes.items():
        coll_name = f"{dataset}_{suffix}"

        url = f"{qdrant_url}/collections/{coll_name}"

        # Delete if exists
        del_resp = requests.delete(url)
        if del_resp.status_code not in (200, 404):
            print(
                f"  Warning: unexpected status {del_resp.status_code} "
                f"when deleting '{coll_name}': {del_resp.text[:200]}"
            )

        body = {
            "vectors": {
                "dense": {
                    "size": dense_dim,
                    "distance": "Cosine",
                    "hnsw_config": {"m": 16, "ef_construct": 200},
                },
                "poincare": {
                    "size": poincare_dim,
                    "distance": "Poincare",
                    "hnsw_config": {"m": 16, "ef_construct": 200},
                    "curvature": curvature,
                },
            },
        }

        print(
            f"Creating per-tier collection '{coll_name}' "
            f"(tier={tier}, dense={dense_dim}d, poincare={poincare_dim}d c={curvature}) ..."
        )
        resp = requests.put(url, json=body)
        if resp.status_code in (200, 201):
            print(f"  Created '{coll_name}' (dense + poincare c={curvature})")
        else:
            # Retry without curvature
            print(
                f"  Note: PUT returned {resp.status_code}, retrying without curvature. "
                f"Response: {resp.text[:300]}"
            )
            body["vectors"]["poincare"].pop("curvature")
            resp2 = requests.put(url, json=body)
            if resp2.status_code not in (200, 201):
                raise RuntimeError(
                    f"Failed to create collection '{coll_name}': "
                    f"HTTP {resp2.status_code} — {resp2.text[:400]}"
                )
            print(f"  Created '{coll_name}' (curvature param unsupported)")

        _create_payload_indices(qdrant_url, coll_name)
        result[tier] = coll_name

    return result


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def populate_collection(
    client: QdrantClient,
    collection_name: str,
    points: list[dict],
    batch_size: int = 256,
) -> None:
    """Upsert points into a collection in batches.

    Each point dict must have:
      id              (int)          — Qdrant point ID
      dense_vector    (list/ndarray) — dense embedding
      poincare_vector (list/ndarray, optional) — Poincaré ball embedding
      payload         (dict)         — arbitrary metadata

    If poincare_vector is present the point is upserted with named vectors
    {"dense": ..., "poincare": ...}; otherwise the single dense_vector is used.

    Args:
        client:          QdrantClient instance.
        collection_name: Target collection name.
        points:          List of point dicts (see above).
        batch_size:      Number of points per upsert batch (default 256).
    """
    total = len(points)
    n_batches = (total + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc=f"Upserting {collection_name}"):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = points[start:end]

        qdrant_points = []
        for pt in batch:
            pt_id = pt["id"]
            payload = pt.get("payload", {})

            dense_vec = pt["dense_vector"]
            if isinstance(dense_vec, np.ndarray):
                dense_vec = dense_vec.tolist()

            poincare_vec = pt.get("poincare_vector")
            tangent_vec = pt.get("tangent_vector")

            if poincare_vec is not None:
                if isinstance(poincare_vec, np.ndarray):
                    poincare_vec = poincare_vec.tolist()
                vector = {"dense": dense_vec, "poincare": poincare_vec}
                if tangent_vec is not None:
                    if isinstance(tangent_vec, np.ndarray):
                        tangent_vec = tangent_vec.tolist()
                    vector["tangent"] = tangent_vec
            else:
                vector = dense_vec

            qdrant_points.append(
                PointStruct(id=pt_id, vector=vector, payload=payload)
            )

        client.upsert(collection_name=collection_name, points=qdrant_points)

    print(f"  Upserted {total} points into '{collection_name}'")


# ---------------------------------------------------------------------------
# Point building
# ---------------------------------------------------------------------------


def build_points_from_data(
    documents: list[dict],
    cosine_vectors: np.ndarray,
    poincare_vectors: np.ndarray | None,
    busemann_depths: np.ndarray | None,
    alpha_values: np.ndarray | None,
    tangent_vectors: np.ndarray | None = None,
    tier_map: dict[int, str] | None = None,
    synthetic_timestamps: list[str] | None = None,
) -> list[dict]:
    """Convert raw arrays into point dicts suitable for populate_collection().

    Assigns integer point IDs using TIER_OFFSETS based on the "tier" field in
    each document (or the tier_map override).  Builds a rich payload with all
    standard fields used by the testbench.

    Args:
        documents:            List of document dicts (loaded from documents.jsonl).
                              Each should have at minimum: "domain", "area",
                              "tier" (if tier_map is None), and optionally
                              "hierarchy_path", "parent_ids", "child_ids",
                              "item_type", "point_id", "all_paths".
        cosine_vectors:       Dense embeddings, shape (N, dense_dim).
        poincare_vectors:     Poincaré ball embeddings, shape (N, poincare_dim),
                              or None for cosine-only collections.
        busemann_depths:      Busemann depth per point, shape (N,), or None.
        alpha_values:         Conformal factor (alpha = 1/(1-c||x||²)) per point,
                              shape (N,), or None.
        tier_map:             Optional {doc_index: tier_name} override.  When
                              provided, supersedes doc["tier"].
        synthetic_timestamps: Optional list of ISO-8601 strings, one per document.
                              Auto-generated as "2024-01-{i%28+1:02d}T00:00:00Z"
                              when None.

    Returns:
        List of point dicts, each with keys:
          id, dense_vector, poincare_vector (if poincare_vectors given), payload.
    """
    n = len(documents)
    result: list[dict] = []

    # Track per-tier counters so IDs are assigned sequentially within each tier
    tier_counters: dict[str, int] = {}

    for i, doc in enumerate(documents):
        # Resolve tier
        if tier_map is not None and i in tier_map:
            tier = tier_map[i]
        else:
            tier = doc.get("tier", "content")

        # Assign point ID using offset + local counter
        offset = TIER_OFFSETS.get(tier, CONTENT_ID_OFFSET)
        local_idx = tier_counters.get(tier, 0)
        tier_counters[tier] = local_idx + 1
        point_id = offset + local_idx

        # Dense vector
        dense_vec = cosine_vectors[i]

        # Poincaré vector (optional)
        poincare_vec = poincare_vectors[i] if poincare_vectors is not None else None

        # Busemann depth
        busemann_depth = float(busemann_depths[i]) if busemann_depths is not None else None

        # Alpha (conformal factor)
        alpha = float(alpha_values[i]) if alpha_values is not None else None

        # Timestamp
        if synthetic_timestamps is not None and i < len(synthetic_timestamps):
            created_at = synthetic_timestamps[i]
        else:
            day = (i % 28) + 1
            created_at = f"2024-01-{day:02d}T00:00:00Z"

        # Hierarchy path
        domain = doc.get("domain", "")
        area = doc.get("area", "")
        hierarchy_path = doc.get("hierarchy_path", "")
        if not hierarchy_path:
            if domain and area:
                hierarchy_path = f"{domain}/{area}"
            elif domain:
                hierarchy_path = domain

        # Stable string ID (used by some query patterns)
        stable_point_id = doc.get("point_id", str(point_id))

        # Build payload
        payload: dict = {
            "item_type": doc.get("item_type", "document"),
            "tier": tier,
            "domain": domain,
            "area": area,
            "hierarchy_path": hierarchy_path,
            "all_paths": doc.get("all_paths", [hierarchy_path] if hierarchy_path else []),
            "parent_ids": doc.get("parent_ids", []),
            "child_ids": doc.get("child_ids", []),
            "point_id": stable_point_id,
            "created_at": created_at,
        }

        if busemann_depth is not None:
            payload["busemann_depth"] = busemann_depth

        if alpha is not None:
            payload["alpha"] = alpha

        # Carry through any extra doc fields that don't conflict
        for key in ("title", "abstract", "label", "labels", "source_ids"):
            if key in doc:
                payload[key] = doc[key]

        pt: dict = {
            "id":           point_id,
            "dense_vector": dense_vec,
            "payload":      payload,
        }
        if poincare_vec is not None:
            pt["poincare_vector"] = poincare_vec
        if tangent_vectors is not None:
            pt["tangent_vector"] = tangent_vectors[i].tolist()

        result.append(pt)

    return result
