# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prefix curriculum for coarse-to-fine OAT tokens.

PPO starts with only the first ``start_k`` token heads active (small action space, easy
exploration) and linearly activates the remaining heads over ``grow_iters`` iterations.
Inactive heads are excluded from log-prob, entropy and KL, and the env wrapper decodes
only the active prefix (the ordered tokenizer guarantees every prefix decodes well).

The active prefix length is shared between the distribution, the env wrapper and the
algorithm through a module-level state, since RSL-RL constructs these objects
independently and offers no clean way to thread a schedule through them.
"""

from __future__ import annotations

import torch

from oat_rl.bc_anchor_ppo import BCAnchorPPO
from oat_rl.multi_categorical import MultiCategoricalDistribution

_ACTIVE_K: int | None = None


def set_active_k(k: int | None) -> None:
    """Set the globally shared number of active token heads (None = all)."""
    global _ACTIVE_K
    _ACTIVE_K = k


def get_active_k() -> int | None:
    """Return the globally shared number of active token heads (None = all)."""
    return _ACTIVE_K


class PrefixCurriculumDistribution(MultiCategoricalDistribution):
    """Multi-categorical distribution that masks heads beyond the active prefix.

    Inactive heads still produce a (deterministic argmax) sample so rollout-storage shapes
    stay fixed, but they contribute nothing to log-prob, entropy or KL, so PPO neither
    reinforces nor explores them until the curriculum activates them.
    """

    def _k(self) -> int:
        active = get_active_k()
        return self.output_dim if active is None else min(active, self.output_dim)

    def sample(self) -> torch.Tensor:
        sampled = self._distribution.sample()  # type: ignore
        k = self._k()
        if k < self.output_dim:
            argmax = self._distribution.logits.argmax(dim=-1)  # type: ignore
            sampled = torch.cat([sampled[..., :k], argmax[..., k:]], dim=-1)
        return sampled.float()

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy()[..., : self._k()].sum(dim=-1)  # type: ignore

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(outputs.long())[..., : self._k()].sum(dim=-1)  # type: ignore

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        (old_logits,) = old_params
        (new_logits,) = new_params
        k = self._k()
        old_logp = torch.log_softmax(old_logits[..., :k, :], dim=-1)
        new_logp = torch.log_softmax(new_logits[..., :k, :], dim=-1)
        return (old_logp.exp() * (old_logp - new_logp)).sum(dim=-1).sum(dim=-1)


class PrefixCurriculumBCAnchorPPO(BCAnchorPPO):
    """BC-anchored PPO that grows the active token prefix on a linear schedule."""

    def __init__(
        self,
        *args,
        prefix_start: int = 4,
        prefix_end: int = 16,
        prefix_grow_iters: int = 600,
        **kwargs,
    ) -> None:
        """Initialize with a prefix schedule: linear ``prefix_start -> prefix_end``.

        Args:
            prefix_start: Active heads at iteration 0.
            prefix_end: Active heads after ``prefix_grow_iters`` (tokenizer horizon).
            prefix_grow_iters: Iterations over which the prefix grows.
        """
        super().__init__(*args, **kwargs)
        self.prefix_start = prefix_start
        self.prefix_end = prefix_end
        self.prefix_grow_iters = max(prefix_grow_iters, 1)
        set_active_k(prefix_start)

    def _scheduled_k(self, iteration: int) -> int:
        progress = min(iteration / self.prefix_grow_iters, 1.0)
        return int(round(self.prefix_start + (self.prefix_end - self.prefix_start) * progress))

    def update(self) -> dict[str, float]:
        # the rollout that filled the storage used the current active_k; keep it for the
        # minibatch updates (ratio consistency) and only advance the schedule afterwards
        stats = super().update()
        new_k = self._scheduled_k(self._bc_iter)
        set_active_k(new_k)
        stats["active_prefix"] = float(new_k)
        return stats
