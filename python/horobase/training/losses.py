"""
Training losses for hyperbolic projection learning.

Two complementary objectives:

InfoNCE (contrastive):
  Pulls anchor closer to positive in DENSE space so cosine ANN (stage-1)
  surfaces descendants. Temperature-scaled cross-entropy over in-batch negatives.
  This is the standard NT-Xent / SimCSE loss.

Busemann triplet:
  Ensures the Poincaré projection places descendants *deeper toward* the
  anchor's ideal point than non-descendants. Pure geometry loss — operates
  entirely in the Poincaré ball, independent of InfoNCE.

Combined loss:
  L = L_InfoNCE + lambda_bus * L_Busemann
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(
    anchor: torch.Tensor,      # [B, D]  L2-normalised dense embeddings
    positive: torch.Tensor,    # [B, D]
    negatives: torch.Tensor,   # [B, N, D]
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss.

    For each anchor, the positive should score higher than all N negatives.
    Negatives come from two sources:
      1. The explicit per-sample negatives (different area / domain)
      2. Other positives in the batch (in-batch negatives — free, diverse)

    We concatenate both so the effective negative count is N + B-1.
    """
    B, D = anchor.shape
    N    = negatives.shape[1]

    # Explicit negatives: [B, N, D] → [B*N, D]
    neg_flat = negatives.reshape(B * N, D)

    # All candidates = positive + explicit negs + other anchors' positives
    # Shape: [B, 1+N+B-1, D] built via similarity matrix

    # Similarity of each anchor to its own positive: [B]
    pos_sim = (anchor * positive).sum(-1) / temperature  # [B]

    # Similarity of each anchor to all positives in batch: [B, B]
    # Diagonal = pos_sim (same anchor-positive pair)
    all_pos_sim = anchor @ positive.T / temperature  # [B, B]

    # Similarity of each anchor to explicit negatives: [B, N]
    all_neg_sim = torch.bmm(
        anchor.unsqueeze(1),              # [B, 1, D]
        negatives.transpose(1, 2),        # [B, D, N]
    ).squeeze(1) / temperature            # [B, N]

    # Concatenate: in-batch positives (all B) + explicit negatives (N)
    # For anchor i, positive i is the target (index i in the first block)
    logits = torch.cat([all_pos_sim, all_neg_sim], dim=1)  # [B, B+N]
    labels = torch.arange(B, device=anchor.device)         # diagonal is the target

    # Label smoothing: instead of one-hot [0,0,1,0,...] use [ε/(K-1), ..., 1-ε, ...]
    # Prevents the model from becoming overconfident on easy negatives early in training.
    return F.cross_entropy(logits, labels, label_smoothing=0.05)


def busemann_triplet_loss(
    anchor_poincare: torch.Tensor,    # [B, H]  inside Poincaré ball
    positive_poincare: torch.Tensor,  # [B, H]
    negatives_poincare: torch.Tensor, # [B, N, H]
    curvature: float = 1.0,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Busemann triplet loss.

    For each anchor q with ideal point ξ = q / ||q||, the positive d+
    should have a higher Busemann score than any negative d-.

    B(d) = log(1 - c||d||²) - log||d - ξ||²

    Loss = mean over negatives of max(0, margin - (B(d+) - B(d-)))
    """
    c = curvature
    B, N, H = negatives_poincare.shape

    # Ideal point: project anchor to ball boundary direction
    anchor_norm = torch.linalg.norm(anchor_poincare, dim=-1, keepdim=True).clamp(min=1e-8)
    xi = anchor_poincare / anchor_norm   # [B, H] — unit vector (boundary direction)

    def busemann_score(x: torch.Tensor) -> torch.Tensor:
        """x: [B, H] → [B] scores."""
        norm_sq = (x * x).sum(-1)                         # [B]
        depth   = torch.log((1.0 - c * norm_sq).clamp(min=1e-8))   # [B]
        # Distance to ideal point
        diff    = x - xi                                  # [B, H]
        dist_sq = (diff * diff).sum(-1).clamp(min=1e-8)  # [B]
        return depth - torch.log(dist_sq)                 # [B]

    def busemann_score_neg(x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, H] → [B, N] scores."""
        norm_sq = (x * x).sum(-1)                          # [B, N]
        depth   = torch.log((1.0 - c * norm_sq).clamp(min=1e-8))
        xi_exp  = xi.unsqueeze(1).expand_as(x)             # [B, N, H]
        diff    = x - xi_exp
        dist_sq = (diff * diff).sum(-1).clamp(min=1e-8)
        return depth - torch.log(dist_sq)                  # [B, N]

    pos_scores = busemann_score(positive_poincare)          # [B]
    neg_scores = busemann_score_neg(negatives_poincare)     # [B, N]

    # Triplet loss: pos should score >= neg + margin
    pos_exp = pos_scores.unsqueeze(1).expand_as(neg_scores)  # [B, N]
    losses  = F.relu(margin - (pos_exp - neg_scores))        # [B, N]
    return losses.mean()


def combined_loss(
    anchor_dense: torch.Tensor,
    positive_dense: torch.Tensor,
    negatives_dense: torch.Tensor,
    anchor_poincare: torch.Tensor,
    positive_poincare: torch.Tensor,
    negatives_poincare: torch.Tensor,
    temperature: float = 0.07,
    curvature: float = 1.0,
    margin: float = 0.5,
    lambda_bus: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    Combined InfoNCE + Busemann-triplet loss.

    Returns (total_loss, {component losses for logging}).
    """
    l_info = infonce_loss(anchor_dense, positive_dense, negatives_dense, temperature)
    l_bus  = busemann_triplet_loss(
        anchor_poincare, positive_poincare, negatives_poincare, curvature, margin
    )
    total = l_info + lambda_bus * l_bus
    return total, {"infonce": l_info.item(), "busemann": l_bus.item(), "total": total.item()}
