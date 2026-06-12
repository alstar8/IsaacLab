# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train PPO with discrete OAT-token actions on an Isaac Lab task.

The actor outputs a multi-categorical distribution over OAT token ids; a frozen pre-trained
OAT tokenizer decodes each token sequence into a chunk of low-level actions that is played
open-loop in the environment (see :class:`oat_rl.token_env_wrapper.OATTokenVecEnvWrapper`).
"""

import argparse
import importlib.metadata as metadata
import os
import sys
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml

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
parser = argparse.ArgumentParser(description="Train PPO with discrete OAT-token actions.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=400, help="Number of PPO iterations.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--tokenizer", type=str, required=True, help="Path to the trained OAT tokenizer checkpoint.")
parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume PPO from this token-policy checkpoint.")
parser.add_argument("--ppo_entropy_coef", type=float, default=None, help="Override PPO entropy coefficient.")
parser.add_argument("--ppo_lr", type=float, default=None, help="Override PPO learning rate.")
parser.add_argument("--ppo_schedule", type=str, default=None, help="Override PPO LR schedule (adaptive|fixed).")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train PPO over OAT tokens."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        agent_cfg.max_iterations = args_cli.max_iterations
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # logging directory
        experiment_name = args_cli.experiment_name or "dexsuite_kuka_allegro_oat"
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
        log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(f"[INFO] Logging experiment in directory: {log_dir}")
        env_cfg.log_dir = log_dir

        # load the frozen tokenizer
        device = env_cfg.sim.device if env_cfg.sim.device is not None else "cuda:0"
        tokenizer, tok_config = load_oat_tokenizer(args_cli.tokenizer, device=device)
        print(f"[INFO] Loaded OAT tokenizer: {tok_config}, codebook_size={tokenizer.codebook_size}")

        # create environment with token-action wrapper
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        env = OATTokenVecEnvWrapper(env, tokenizer)
        print(
            f"[INFO] Token env: num_actions={env.num_actions} tokens x {tokenizer.codebook_size} codes,"
            f" chunk_horizon={env.chunk_horizon}, macro episode length={env.max_episode_length}"
        )

        # build train config with multi-categorical actor distribution
        train_cfg = agent_cfg.to_dict()
        train_cfg["actor"]["distribution_cfg"] = {
            "class_name": "oat_rl.multi_categorical:MultiCategoricalDistribution",
            "num_categories": tokenizer.codebook_size,
        }
        if args_cli.ppo_entropy_coef is not None:
            train_cfg["algorithm"]["entropy_coef"] = args_cli.ppo_entropy_coef
        if args_cli.ppo_lr is not None:
            train_cfg["algorithm"]["learning_rate"] = args_cli.ppo_lr
        if args_cli.ppo_schedule is not None:
            train_cfg["algorithm"]["schedule"] = args_cli.ppo_schedule

        runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
        if args_cli.resume_checkpoint is not None:
            resume_path = os.path.abspath(args_cli.resume_checkpoint)
            print(f"[INFO] Resuming token policy from: {resume_path}")
            runner.load(resume_path)

        # dump configs for reproducibility
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), train_cfg)
        with open(os.path.join(log_dir, "params", "oat.txt"), "w") as f:
            f.write(f"tokenizer={os.path.abspath(args_cli.tokenizer)}\nconfig={tok_config}\n")

        start_time = time.time()
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        env.close()


if __name__ == "__main__":
    main()
