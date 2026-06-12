# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint: record a video and measure episode success rate.

The script runs the policy for a fixed number of steps, records a video of the first
``--video_length`` steps and accumulates the per-episode success metric that the Dexsuite
``success_reward`` term flushes into ``extras["log"]["Metrics/success_rate"]`` on reset.

Results are written to ``<run_dir>/eval/<checkpoint_stem>/`` as a JSON summary and a video.
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

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # isort: skip

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint with video and success-rate metrics.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during evaluation.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=900, help="Total number of policy steps to evaluate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
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
    """Evaluate the checkpoint."""
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

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # episode/success accounting
        total_episodes = 0
        weighted_success_sum = 0.0
        reward_sum = 0.0
        step_count = 0

        obs = env.get_observations()
        for _ in range(args_cli.num_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, rew, dones, extras = env.step(actions)
                policy.reset(dones)
            step_count += 1
            reward_sum += rew.mean().item()
            num_dones = int(dones.sum().item())
            if num_dones > 0:
                log_dict = extras.get("log", {})
                sr = log_dict.get("Metrics/success_rate")
                if sr is not None:
                    weighted_success_sum += float(sr) * num_dones
                    total_episodes += num_dones

        success_rate = weighted_success_sum / total_episodes if total_episodes > 0 else float("nan")
        summary = {
            "checkpoint": os.path.abspath(resume_path),
            "task": args_cli.task,
            "num_envs": env.num_envs,
            "num_steps": step_count,
            "episodes_finished": total_episodes,
            "success_rate": success_rate,
            "mean_step_reward": reward_sum / max(step_count, 1),
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
