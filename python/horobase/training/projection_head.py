"""
ProjectionHead — trainable MLP that maps frozen transformer embeddings
into Poincaré ball positions.

Architecture:
    frozen pplx-embed (1024d, L2-normalised)
        ↓
    Linear(1024 → 512) + LayerNorm + GELU
        ↓
    Linear(512 → 128)
        ↓
    L2-normalise direction
        ↓
    Scale to tier radius  (explicit_radial: use tier label)
                        OR clamp to (0, max_norm) (natural_norm)
        ↓
    Poincaré position (128d, inside ball)

The head is small (~200K params) and trains in minutes on a 3090.
The frozen encoder means we never need more than ~4GB VRAM.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tier radii matching pca_encoder.py STATIC_TIER_SCALING
TIER_RADII = {
    "narrative": 0.45,
    "story":     0.70,
    "content":   0.90,
    # aliases
    "theme":     0.15,
    "mid":       0.70,
    "leaf":      0.90,
    "root":      0.45,
}
DEFAULT_RADIUS = 0.90


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head.

    Parameters
    ----------
    input_dim : int
        Dimension of frozen encoder output (1024 for pplx-embed).
    hidden_dim : int
        Hidden layer size.
    output_dim : int
        Poincaré ball dimension (128).
    curvature : float
        Ball curvature c. Radius clamped to 1/sqrt(c) - eps.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 128,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.curvature  = curvature
        self.output_dim = output_dim
        self._max_norm  = min(0.999, 1.0 / (curvature ** 0.5) - 1e-4)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),              # regularization — prevents co-adaptation
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(
        self,
        dense: torch.Tensor,           # [B, input_dim]  L2-normalised
        tiers: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (poincare, dense_proj):
          poincare   [B, output_dim]  — inside Poincaré ball
          dense_proj [B, output_dim]  — L2-normalised direction (for InfoNCE)
        """
        x = self.net(dense)                          # [B, output_dim]
        direction = F.normalize(x, dim=-1)           # unit sphere — semantic direction

        # Dense projection for InfoNCE is just the direction (unit norm = cosine friendly)
        dense_proj = direction

        # Poincaré position: scale direction to tier radius
        if tiers is not None:
            radii = torch.tensor(
                [TIER_RADII.get(t, DEFAULT_RADIUS) for t in tiers],
                dtype=torch.float32, device=dense.device,
            )  # [B]
        else:
            radii = torch.full((dense.shape[0],), DEFAULT_RADIUS,
                               dtype=torch.float32, device=dense.device)

        poincare = direction * radii.unsqueeze(-1)   # [B, output_dim]

        # Safety clamp
        norms = torch.linalg.norm(poincare, dim=-1, keepdim=True).clamp(min=1e-8)
        too_large = norms >= self._max_norm
        poincare  = torch.where(too_large, poincare * (self._max_norm / norms), poincare)

        return poincare, dense_proj

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "curvature":  self.curvature,
            "input_dim":  self.net[0].in_features,
            "hidden_dim": self.net[0].out_features,
            "output_dim": self.output_dim,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ProjectionHead":
        ckpt = torch.load(path, map_location=device)
        head = cls(
            input_dim  = ckpt["input_dim"],
            hidden_dim = ckpt["hidden_dim"],
            output_dim = ckpt["output_dim"],
            curvature  = ckpt["curvature"],
        )
        head.load_state_dict(ckpt["state_dict"])
        head.to(device)
        return head
