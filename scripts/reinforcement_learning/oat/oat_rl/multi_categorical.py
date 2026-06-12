# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-categorical distribution for discrete-token PPO with RSL-RL."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rsl_rl.modules.distribution import Distribution


class _ArgmaxDeterministicOutput(nn.Module):
    """Export-friendly module that extracts argmax token ids from logits."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output.argmax(dim=-1).float()


class MultiCategoricalDistribution(Distribution):
    """Independent categorical distributions over ``output_dim`` discrete tokens.

    The MLP must output a tensor of shape ``[..., output_dim, num_categories]`` interpreted as
    unnormalized logits of ``output_dim`` independent categorical variables. Sampled outputs are
    float tensors of token indices with shape ``[..., output_dim]`` (float for compatibility with
    the RSL-RL rollout storage; consumers cast back to long).
    """

    def __init__(self, output_dim: int, num_categories: int) -> None:
        """Initialize the multi-categorical distribution module.

        Args:
            output_dim: Number of independent discrete tokens.
            num_categories: Number of categories per token (codebook size).
        """
        super().__init__(output_dim)
        self.num_categories = num_categories
        self._distribution: Categorical | None = None
        Categorical.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the categorical distributions from MLP logits."""
        self._distribution = Categorical(logits=mlp_output)

    def sample(self) -> torch.Tensor:
        """Sample token indices, returned as float tensor of shape ``[..., output_dim]``."""
        return self._distribution.sample().float()  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract argmax token indices from the MLP logits."""
        return mlp_output.argmax(dim=-1).float()

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that extracts argmax token ids from the logits."""
        return _ArgmaxDeterministicOutput()

    @property
    def input_dim(self) -> list[int]:
        """Return the input dimension required by the distribution: ``[output_dim, num_categories]``."""
        return [self.output_dim, self.num_categories]

    @property
    def mean(self) -> torch.Tensor:
        """Return the mode (argmax) of each categorical as a float tensor (mean proxy)."""
        return self._distribution.logits.argmax(dim=-1).float()  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the per-token entropy as a spread proxy (categoricals have no std)."""
        return self._distribution.entropy()  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy summed over tokens."""
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return the log-probabilities of the current distribution as a single-tensor tuple."""
        return (self._distribution.logits,)  # type: ignore

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log probability of token indices ``[..., output_dim]``, summed over tokens."""
        return self._distribution.log_prob(outputs.long()).sum(dim=-1)  # type: ignore

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute KL(old || new) between multi-categorical distributions, summed over tokens."""
        (old_logits,) = old_params
        (new_logits,) = new_params
        old_logp = F.log_softmax(old_logits, dim=-1)
        new_logp = F.log_softmax(new_logits, dim=-1)
        kl = (old_logp.exp() * (old_logp - new_logp)).sum(dim=-1)
        return kl.sum(dim=-1)
