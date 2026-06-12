# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VecEnv wrapper that exposes a discrete-token action space decoded by an OAT tokenizer."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv


class OATTokenVecEnvWrapper(VecEnv):
    """Wraps an :class:`RslRlVecEnvWrapper` with OAT token actions.

    One policy step consumes ``latent_horizon`` token indices, decodes them with the OAT
    tokenizer into a chunk of ``chunk_horizon`` low-level actions and plays the chunk
    open-loop in the underlying environment. Rewards are summed over the chunk so episode
    returns stay comparable with the per-step baseline; dones are OR-ed.

    Note:
        If an episode resets mid-chunk, the remaining actions of the chunk are still applied
        to the freshly reset environment (standard open-loop chunking approximation).
    """

    def __init__(self, env, tokenizer) -> None:
        """Initialize the wrapper.

        Args:
            env: The inner :class:`RslRlVecEnvWrapper` instance.
            tokenizer: A trained, frozen :class:`OATTok` instance on the env device.
        """
        self.env = env
        self.tokenizer = tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        self.chunk_horizon = tokenizer.decoder.sample_horizon
        self.num_envs = env.num_envs
        self.num_actions = tokenizer.latent_horizon
        self.device = env.device
        self.max_episode_length = max(env.max_episode_length // self.chunk_horizon, 1)

    @property
    def cfg(self) -> object:
        """Return the configuration class instance of the underlying environment."""
        return self.env.cfg

    @property
    def unwrapped(self):
        """Return the bare environment underneath all wrappers."""
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer of the underlying environment."""
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.env.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.env.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        return self.env.reset()

    def get_observations(self) -> TensorDict:
        """Return the current observations of the environment."""
        return self.env.get_observations()

    def step(self, token_actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Decode token actions and play the resulting chunk in the underlying environment.

        Args:
            token_actions: Token indices of shape [num_envs, latent_horizon] (float or long).

        Returns:
            Tuple of (observations, summed rewards, OR-ed dones, merged extras).
        """
        with torch.no_grad():
            chunks = self.tokenizer.detokenize(token_actions.long())  # [B, T, action_dim]

        reward_sum = torch.zeros(self.num_envs, device=self.device)
        dones_any = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        time_outs_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        merged_log: dict = {}
        # success-rate accounting across substeps (value is mean over envs reset at that substep)
        success_weighted = 0.0
        success_episodes = 0

        obs = None
        for t in range(self.chunk_horizon):
            obs, rew, dones, extras = self.env.step(chunks[:, t].contiguous())
            reward_sum += rew
            dones_any |= dones
            if "time_outs" in extras:
                time_outs_any |= extras["time_outs"].bool()
            log_dict = extras.get("log")
            if log_dict:
                num_dones = int(dones.sum().item())
                sr = log_dict.get("Metrics/success_rate")
                if sr is not None and num_dones > 0:
                    success_weighted += float(sr) * num_dones
                    success_episodes += num_dones
                merged_log.update(log_dict)

        if success_episodes > 0:
            merged_log["Metrics/success_rate"] = success_weighted / success_episodes

        extras_out: dict = {"log": merged_log}
        extras_out["time_outs"] = time_outs_any

        return obs, reward_sum, dones_any, extras_out

    def close(self):  # noqa: D102
        return self.env.close()
