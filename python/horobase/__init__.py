"""
hyperbolic — Python adapter layer for Qdrant with Poincaré distance support.

Architecture:
  client.py          HyperbolicClient (feature-detect, schema, fallbacks)
  encoder/base.py    Encoder ABC + EncodedVectors
  encoder/pca        PCAEncoder (SE's approach — no training needed)
  encoder/neural     NeuralEncoder (learnable curvatures, product manifold)
  math/poincare.py   All Poincaré ball ops (single source of truth)
  pipeline/config    PipelineConfig dataclass + presets
  pipeline/ingest    Ingestor (text -> encode -> upsert)
  search/strategies  Search pipelines (alpha, tangent, combined, klein)
  search/fusion      Fusion strategies (rrf, busemann, linear_alpha, horo)
"""
