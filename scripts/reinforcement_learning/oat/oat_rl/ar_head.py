# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Autoregressive multi-categorical head for coarse-to-fine OAT tokens.

Token k is predicted conditioned on the observation context and the already chosen
tokens 1..k-1 (GRU over token positions), matching the coarse-to-fine structure of the
ordered code: the fine heads can see which coarse motion was selected.

API contract with RSL-RL:
    * ``update(mlp_output)`` caches the observation context (the MLP now outputs a
      context vector instead of logits).
    * ``sample()`` runs the AR loop with sampling and caches the per-position logits
      that conditioned each choice.
    * ``log_prob(actions)`` re-runs the loop teacher-forced on ``actions`` and refreshes
      the cached logits, so ``params``/``entropy`` read afterwards correspond to the
      same conditioning prefix as the stored actions (required for PPO ratio/KL).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rsl_rl.modules.distribution import Distribution


class ARTokenDistribution(Distribution):
    """Autoregressive categorical distribution over ordered token positions."""

    def __init__(
        self,
        output_dim: int,
        num_categories: int,
        context_dim: int = 256,
        token_emb_dim: int = 128,
    ) -> None:
        """Initialize the AR token head.

        Args:
            output_dim: Number of token positions (latent horizon).
            num_categories: Codebook size per position.
            context_dim: Size of the observation context produced by the MLP trunk
                (also the GRU hidden size).
            token_emb_dim: Embedding size of previously chosen tokens.
        """
        super().__init__(output_dim)
        self.num_categories = num_categories
        self.context_dim = context_dim

        self.tok_emb = nn.Embedding(num_categories, token_emb_dim)
        self.start_emb = nn.Parameter(torch.zeros(token_emb_dim))
        self.pos_emb = nn.Parameter(torch.zeros(output_dim, token_emb_dim))
        self.cell = nn.GRUCell(token_emb_dim, context_dim)
        self.head = nn.Linear(context_dim, num_categories)
        nn.init.normal_(self.pos_emb, std=0.02)

        self._context: torch.Tensor | None = None
        self._logits: torch.Tensor | None = None
        Categorical.set_default_validate_args(False)

    # -- AR core -----------------------------------------------------------------

    def _ar_loop(self, context: torch.Tensor, teacher_tokens: torch.Tensor | None, greedy: bool):
        """Run the AR loop; returns (tokens [B, K], logits [B, K, C]).

        If ``teacher_tokens`` is given, conditioning prefixes come from it (teacher
        forcing); otherwise tokens are chosen greedily or by sampling.
        """
        batch = context.shape[0]
        h = context
        x = self.start_emb.expand(batch, -1)
        tokens: list[torch.Tensor] = []
        logits_all: list[torch.Tensor] = []
        for k in range(self.output_dim):
            h = self.cell(x + self.pos_emb[k].unsqueeze(0), h)
            logits_k = self.head(h)
            logits_all.append(logits_k)
            if teacher_tokens is not None:
                tok_k = teacher_tokens[:, k]
            elif greedy:
                tok_k = logits_k.argmax(dim=-1)
            else:
                tok_k = Categorical(logits=logits_k).sample()
            tokens.append(tok_k)
            x = self.tok_emb(tok_k)
        return torch.stack(tokens, dim=1), torch.stack(logits_all, dim=1)

    def teacher_forced_logits(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Return per-position logits [B, K, C] conditioned on the given token prefixes."""
        _, logits = self._ar_loop(context, tokens.long(), greedy=False)
        return logits

    # -- Distribution API ----------------------------------------------------------

    def update(self, mlp_output: torch.Tensor) -> None:
        """Cache the observation context vector produced by the MLP trunk."""
        self._context = mlp_output

    def sample(self) -> torch.Tensor:
        """Sample token sequences autoregressively; caches the conditioning logits."""
        tokens, logits = self._ar_loop(self._context, None, greedy=False)
        self._logits = logits
        return tokens.float()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Greedy AR decoding from the context (used for evaluation/inference)."""
        tokens, _ = self._ar_loop(mlp_output, None, greedy=True)
        return tokens.float()

    def as_deterministic_output_module(self) -> nn.Module:
        """Export module: greedy AR decode of the context."""
        return _GreedyARDecode(self)

    @property
    def input_dim(self) -> list[int]:
        """The MLP trunk outputs a context vector, not logits."""
        return [self.context_dim]

    @property
    def mean(self) -> torch.Tensor:
        """Mode proxy: argmax of the cached per-position logits."""
        return self._logits.argmax(dim=-1).float()

    @property
    def std(self) -> torch.Tensor:
        """Spread proxy: per-position conditional entropy."""
        return Categorical(logits=self._logits).entropy()

    @property
    def entropy(self) -> torch.Tensor:
        """Sum of per-position conditional entropies (AR joint entropy proxy)."""
        return Categorical(logits=self._logits).entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Per-position logits conditioned on the most recent sample/log_prob pass."""
        return (self._logits,)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Joint log-probability of token sequences (teacher-forced recompute)."""
        logits = self.teacher_forced_logits(self._context, outputs)
        self._logits = logits
        return Categorical(logits=logits).log_prob(outputs.long()).sum(dim=-1)

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """KL between old/new conditionals at the same (stored-action) prefixes."""
        (old_logits,) = old_params
        (new_logits,) = new_params
        old_logp = F.log_softmax(old_logits, dim=-1)
        new_logp = F.log_softmax(new_logits, dim=-1)
        kl = (old_logp.exp() * (old_logp - new_logp)).sum(dim=-1)
        return kl.sum(dim=-1)


class _GreedyARDecode(nn.Module):
    """Export-friendly greedy AR decoding module."""

    def __init__(self, dist: ARTokenDistribution) -> None:
        super().__init__()
        self.dist = dist

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return self.dist.deterministic_output(mlp_output)
