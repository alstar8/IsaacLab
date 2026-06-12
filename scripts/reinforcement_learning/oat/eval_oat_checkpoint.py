# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an OAT-token PPO checkpoint: record a video and measure episode success rate."""

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

# local imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.append(os.path.join(_SCRIPT_DIR, "..", "rsl_rl"))
import cli_args  # isort: skip

from oat_rl.token_env_wrapper import OATTokenVecEnvWrapper
from oat_tok.tokenizer import load_oat_tokenizer

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate an OAT-token PPO checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during evaluation.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in env steps).")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=120, help="Number of macro (token) steps to evaluate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--tokenizer", type=str, required=True, help="Path to the trained OAT tokenizer checkpoint.")
parser.add_argument("--label", type=str, default=None, help="Optional label for the eval output folder.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate the OAT-token checkpoint."""
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        resume_path = retrieve_file_path(args_cli.checkpoint)
        run_dir = os.path.dirname(os.path.abspath(resume_path))
        checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
        label = args_cli.label or checkpoint_stem
        eval_dir = os.path.join(run_dir, "eval", label)
        os.makedirs(eval_dir, exist_ok=True)
        env_cfg.log_dir = eval_dir

        device = env_cfg.sim.device if env_cfg.sim.device is not None else "cuda:0"
        tokenizer, tok_config = load_oat_tokenizer(args_cli.tokenizer, device=device)
        print(f"[INFO] Loaded OAT tokenizer: {tok_config}")

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if args_cli.video:
            video_kwargs = {
                "video_folder": eval_dir,
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "name_prefix": f"eval-{label}",
                "disable_logger": True,
            }
            print(f"[INFO] Recording video to: {eval_dir}")
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        env = OATTokenVecEnvWrapper(env, tokenizer)

        train_cfg = agent_cfg.to_dict()
        train_cfg["actor"]["distribution_cfg"] = {
            "class_name": "oat_rl.multi_categorical:MultiCategoricalDistribution",
            "num_categories": tokenizer.codebook_size,
        }

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        total_episodes = 0
        weighted_success_sum = 0.0
        reward_sum = 0.0
        step_count = 0

        obs = env.get_observations()
        for _ in range(args_cli.num_steps):
            with torch.inference_mode():
                actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)
            with torch.inference_mode():
                policy.reset(dones)
            step_count += 1
            reward_sum += rew.mean().item()
            num_dones = int(dones.sum().item())
            if num_dones > 0:
                sr = extras.get("log", {}).get("Metrics/success_rate")
                if sr is not None:
                    weighted_success_sum += float(sr) * num_dones
                    total_episodes += num_dones

        success_rate = weighted_success_sum / total_episodes if total_episodes > 0 else float("nan")
        summary = {
            "checkpoint": os.path.abspath(resume_path),
            "tokenizer": os.path.abspath(args_cli.tokenizer),
            "task": args_cli.task,
            "num_envs": env.num_envs,
            "num_macro_steps": step_count,
            "env_steps": step_count * env.chunk_horizon,
            "episodes_finished": total_episodes,
            "success_rate": success_rate,
            "mean_macro_step_reward": reward_sum / max(step_count, 1),
        }
        summary_path = os.path.join(eval_dir, "eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("[INFO] Evaluation summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print(f"[INFO] Summary saved to: {summary_path}")

        env.close()


if __name__ == "__main__":
    main()
