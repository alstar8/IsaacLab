# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal action normalizer used by the vendored OAT tokenizer.

Replaces ``oat.model.common.normalizer.LinearNormalizer`` (which depends on zarr/hydra)
with a simple buffer-based affine normalizer ``x_norm = (x - offset) * scale``.

Two fit modes:

* ``"std"`` (default) — z-score per dim (offset=mean, scale=1/std). Robust to the heavy
  per-dim outliers seen in dexterous-manipulation action targets, where min-max wastes
  almost all dynamic range on a few extreme samples and leaves the bulk variance
  unrepresented (≈80% of variance unexplained after reconstruction).
* ``"minmax"`` — legacy mapping of the fitted range to [-1, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionNormalizer(nn.Module):
    """Affine normalizer: ``x_norm = (x - offset) * scale``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("offset", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))

    @torch.no_grad()
    def fit(self, data: torch.Tensor, eps: float = 1e-6, mode: str = "std") -> None:
        """Fit offset/scale from data of shape [..., dim]."""
        flat = data.reshape(-1, data.shape[-1])
        if mode == "std":
            mean = flat.mean(dim=0)
            std = torch.clamp(flat.std(dim=0), min=eps)
            self.offset.copy_(mean)
            self.scale.copy_(1.0 / std)
        elif mode == "minmax":
            dmin = flat.min(dim=0).values
            dmax = flat.max(dim=0).values
            center = (dmax + dmin) / 2
            half_range = torch.clamp((dmax - dmin) / 2, min=eps)
            self.offset.copy_(center)
            self.scale.copy_(1.0 / half_range)
        else:
            raise ValueError(f"unknown normalizer mode: {mode!r}")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) * self.scale

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale + self.offset
