# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect action chunks from a trained RSL-RL policy for OAT tokenizer training.

Rolls out a checkpoint with stochastic actions and stores non-overlapping chunks of
``--chunk_horizon`` consecutive actions that do not cross episode resets. The output is a
``.pt`` file with a tensor of shape [num_chunks, chunk_horizon, action_dim].
"""

import argparse
import importlib.metadata as metadata
import os
import sys

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Collect action chunks from an RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=640, help="Number of policy steps to roll out.")
parser.add_argument("--chunk_horizon", type=int, default=8, help="Number of consecutive actions per chunk.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--output", type=str, required=True, help="Output .pt file for the chunk dataset.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect action chunks."""
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        resume_path = retrieve_file_path(args_cli.checkpoint)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        actor = runner.alg.actor
        actor.eval()

        num_envs = env.num_envs
        horizon = args_cli.chunk_horizon
        action_dim = env.num_actions
        device = env.unwrapped.device

        chunk_buffer = torch.zeros(num_envs, horizon, action_dim, device=device)
        fill_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        chunks: list[torch.Tensor] = []

        obs = env.get_observations()
        for step in range(args_cli.num_steps):
            with torch.inference_mode():
                actions = actor(obs, stochastic_output=True)
                obs, _, dones, _ = env.step(actions)

            # append actions into per-env chunk buffers
            chunk_buffer[torch.arange(num_envs, device=device), fill_count] = actions
            fill_count += 1

            # flush completed chunks
            full = fill_count == horizon
            if full.any():
                chunks.append(chunk_buffer[full].clone())
                fill_count[full] = 0

            # discard partial chunks of reset envs (action sequence broken by reset)
            done_mask = dones.bool()
            if done_mask.any():
                fill_count[done_mask] = 0

            if (step + 1) % 100 == 0:
                total = sum(c.shape[0] for c in chunks)
                print(f"[INFO] step {step + 1}/{args_cli.num_steps}, chunks collected: {total}")

        dataset = torch.cat(chunks, dim=0).cpu()
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
        torch.save(
            {
                "actions": dataset,
                "chunk_horizon": horizon,
                "action_dim": action_dim,
                "checkpoint": os.path.abspath(resume_path),
                "task": args_cli.task,
                "num_envs": num_envs,
                "num_steps": args_cli.num_steps,
            },
            args_cli.output,
        )
        print(f"[INFO] Saved {dataset.shape[0]} chunks of shape [{horizon}, {action_dim}] to: {args_cli.output}")
        print(f"[INFO] Action stats: min={dataset.min():.3f} max={dataset.max():.3f} std={dataset.std():.3f}")

        env.close()


if __name__ == "__main__":
    main()
