# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with an annealed behavior-cloning anchor for discrete OAT-token actions.

Adds ``lambda(t) * CE(actor logits, expert tokens)`` to the PPO loss, where expert tokens
are obtained by querying the frozen continuous expert on the rollout observations and
tokenizing its actions (DAgger-style online relabeling, so the anchor follows the visited
state distribution instead of a stale BC dataset). ``lambda(t)`` decays to zero so the
policy is protected from early-PPO degradation but free to surpass the expert later.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.algorithms.ppo import PPO


class BCAnchorPPO(PPO):
    """PPO with an annealed DAgger-style BC cross-entropy anchor on token logits."""

    def __init__(
        self,
        *args,
        bc_anchor_coef: float = 0.3,
        bc_anchor_iters: int = 400,
        bc_anchor_decay: str = "cosine",
        **kwargs,
    ) -> None:
        """Initialize PPO with BC-anchor parameters.

        Args:
            bc_anchor_coef: Initial weight of the BC cross-entropy anchor.
            bc_anchor_iters: Iterations over which the anchor decays to zero.
            bc_anchor_decay: Decay shape ("cosine", "linear" or "const").
        """
        super().__init__(*args, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("BCAnchorPPO does not support RND or symmetry configurations.")
        self.bc_anchor_coef = bc_anchor_coef
        self.bc_anchor_iters = bc_anchor_iters
        self.bc_anchor_decay = bc_anchor_decay
        self._bc_iter = 0
        self._expert_policy: Callable[[object], torch.Tensor] | None = None
        self._tokenizer = None

    def set_bc_anchor(self, expert_policy: Callable, tokenizer) -> None:
        """Attach the frozen continuous expert policy and the OAT tokenizer."""
        self._expert_policy = expert_policy
        self._tokenizer = tokenizer

    def current_bc_coef(self) -> float:
        """Annealed anchor weight at the current iteration."""
        if self._expert_policy is None or self.bc_anchor_coef <= 0.0:
            return 0.0
        if self.bc_anchor_decay == "const":
            return self.bc_anchor_coef
        progress = min(self._bc_iter / max(self.bc_anchor_iters, 1), 1.0)
        if self.bc_anchor_decay == "linear":
            return self.bc_anchor_coef * (1.0 - progress)
        if self.bc_anchor_decay == "cosine":
            return self.bc_anchor_coef * 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        raise ValueError(f"Unknown bc_anchor_decay: {self.bc_anchor_decay}")

    def _expert_token_targets(self, observations) -> torch.Tensor:
        """Query the expert on rollout observations and tokenize its actions. Returns [B, K] long."""
        with torch.no_grad():
            expert_actions = self._expert_policy(observations)
            tokens = self._tokenizer.tokenize(expert_actions.unsqueeze(1))
        return tokens.long()

    def update(self) -> dict[str, float]:
        """PPO update with the annealed BC anchor added to the loss.

        Mirrors :meth:`PPO.update` for the non-recurrent, non-RND, non-symmetry path.
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_bc_loss = 0.0
        bc_coef = self.current_bc_coef()

        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("BCAnchorPPO does not support recurrent models.")
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(self.actor.output_distribution_params)
            entropy = self.actor.output_entropy

            # adaptive learning rate on KL (same as parent)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # annealed BC anchor: CE between current token logits and expert tokens at visited states
            if bc_coef > 0.0:
                expert_tokens = self._expert_token_targets(batch.observations)  # [B, K]
                logits = self.actor.output_distribution_params[0]  # [B, K, C], requires grad
                b, k, c = logits.shape
                bc_loss = F.cross_entropy(logits.reshape(b * k, c), expert_tokens.reshape(b * k))
                loss = loss + bc_coef * bc_loss
                mean_bc_loss += bc_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self._bc_iter += 1

        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "bc_anchor": mean_bc_loss / num_updates,
            "bc_coef": bc_coef,
        }
