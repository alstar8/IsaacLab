# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Boosted residual decoder for OAT: tokens act like gradient-boosting stages.

The reconstruction from a token prefix of length k is an additive sum of per-token
residual corrections with geometrically decreasing amplitude (shrinkage):

    recon(k) = sum_{j=0..k-1} eta**j * delta_j(tokens[0..j])

Every prefix is a valid reconstruction (total decodability), later tokens can only
refine the earlier ones with shrinking magnitude, and the causal attention over
register positions means correction ``delta_j`` is conditioned on tokens 0..j only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from oat_tok.model.head import LinearHead
from oat_tok.model.linear import LinearLayer
from oat_tok.model.pos_emb import PositionalEmbeddingAdder


class BoostedResidualDecoder(nn.Module):
    """Additive coarse-to-fine decoder with a geometric shrinkage schedule."""

    def __init__(
        self,
        sample_dim: int,
        sample_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        latent_dim: int,
        latent_horizon: int,
        shrinkage: float = 0.85,
    ):
        super().__init__()
        self.latent_proj = LinearLayer(latent_dim, emb_dim)
        self.latent_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[latent_horizon])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=emb_dim // head_dim,
            dim_feedforward=4 * emb_dim,
            dropout=pdropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stages = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = LinearHead(emb_dim, sample_horizon * sample_dim)

        etas = shrinkage ** torch.arange(latent_horizon, dtype=torch.float32)
        self.register_buffer("etas", etas)

        self.sample_dim = sample_dim
        self.sample_horizon = sample_horizon
        self.latent_horizon = latent_horizon
        self.emb_dim = emb_dim
        self.shrinkage = shrinkage

    def _prefix_recons(self, latents: torch.Tensor) -> torch.Tensor:
        """Return reconstructions for every prefix length, shape [B, K, T, sample_dim]."""
        batch_size = latents.shape[0]
        x = self.latent_proj(latents)
        x = self.latent_pos_emb(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.stages(x, mask=mask, is_causal=True)  # (B, K, emb)
        deltas = self.head(h).view(batch_size, self.latent_horizon, self.sample_horizon, self.sample_dim)
        contributions = deltas * self.etas.view(1, -1, 1, 1)
        return torch.cumsum(contributions, dim=1)

    def forward(
        self,
        latents: torch.Tensor,
        eval_keep_k: list[int] | None = None,
        return_all_prefixes: bool = False,
    ) -> torch.Tensor:
        # latents: (B, K, latent_dim) -> samples: (B, T, sample_dim)
        prefix_recons = self._prefix_recons(latents)
        if return_all_prefixes:
            return prefix_recons
        if eval_keep_k is None:
            return prefix_recons[:, -1]
        keep = torch.as_tensor(eval_keep_k, device=latents.device, dtype=torch.long).clamp(
            1, self.latent_horizon
        )
        idx = (keep - 1).view(-1, 1, 1, 1).expand(-1, 1, self.sample_horizon, self.sample_dim)
        return prefix_recons.gather(1, idx).squeeze(1)
