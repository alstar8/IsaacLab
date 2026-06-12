# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the closed-loop ceiling of an OAT tokenizer.

Rolls out a continuous-action expert policy but replaces every action by its tokenizer
round-trip ``detokenize(tokenize(a))`` before stepping the environment. This isolates the
performance lost purely to quantization (independent of any token-policy learning): if the
round-trip ceiling is already far below the expert, the tokenizer — not the RL policy — is the
bottleneck. Reports success rate and mean episode reward for the raw expert and the round-trip.
"""

import argparse
import importlib.metadata as metadata
import json
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.append(os.path.join(_SCRIPT_DIR, "..", "rsl_rl"))
import cli_args  # isort: skip

from oat_tok.tokenizer import load_oat_tokenizer

parser = argparse.ArgumentParser(description="Measure OAT tokenizer closed-loop ceiling.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=900, help="Env steps to roll out.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--tokenizer", type=str, required=True, help="OAT tokenizer checkpoint.")
parser.add_argument("--roundtrip", action="store_true", help="Apply detokenize(tokenize(a)) to expert actions.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def run_rollout(env, policy, tokenizer, num_steps, roundtrip):
    """Roll out the expert (optionally round-tripping actions) and aggregate SR / reward."""
    total_episodes = 0
    weighted_success = 0.0
    reward_sum = 0.0
    obs = env.get_observations()
    for _ in range(num_steps):
        with torch.inference_mode():
            action = policy(obs)
            if roundtrip:
                action = tokenizer.detokenize(tokenizer.tokenize(action.unsqueeze(1))).squeeze(1)
            obs, rew, dones, extras = env.step(action)
        reward_sum += rew.mean().item()
        num_dones = int(dones.sum().item())
        if num_dones > 0:
            sr = extras.get("log", {}).get("Metrics/success_rate")
            if sr is not None:
                weighted_success += float(sr) * num_dones
                total_episodes += num_dones
    sr = weighted_success / total_episodes if total_episodes > 0 else float("nan")
    return {"success_rate": sr, "episodes": total_episodes, "mean_step_reward": reward_sum / max(num_steps, 1)}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Measure tokenizer ceiling."""
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        resume_path = retrieve_file_path(args_cli.checkpoint)
        device = env_cfg.sim.device if env_cfg.sim.device is not None else "cuda:0"
        tokenizer, tok_config = load_oat_tokenizer(args_cli.tokenizer, device=device)
        print(f"[INFO] Loaded OAT tokenizer: {tok_config}")

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)

        raw = run_rollout(env, policy, tokenizer, args_cli.num_steps, roundtrip=False)
        print(f"[INFO] RAW expert:        {raw}")
        rt = run_rollout(env, policy, tokenizer, args_cli.num_steps, roundtrip=True)
        print(f"[INFO] ROUND-TRIP tokens: {rt}")

        summary = {
            "tokenizer": os.path.abspath(args_cli.tokenizer),
            "config": tok_config,
            "checkpoint": os.path.abspath(resume_path),
            "num_envs": env.num_envs,
            "num_steps": args_cli.num_steps,
            "raw_expert": raw,
            "roundtrip": rt,
        }
        out_dir = os.path.dirname(os.path.abspath(args_cli.tokenizer))
        with open(os.path.join(out_dir, "ceiling_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Saved ceiling summary to: {os.path.join(out_dir, 'ceiling_summary.json')}")
        env.close()


if __name__ == "__main__":
    main()
