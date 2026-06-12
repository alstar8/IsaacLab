# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BC warm-start + PPO with an annealed DAgger-style BC anchor for OAT-token actions.

Identical to :mod:`bc_warmstart_ppo` except the PPO phase uses :class:`BCAnchorPPO`,
which adds an annealed ``lambda(t) * CE(actor logits, expert tokens)`` term so the early
PPO updates cannot destroy the behavior-cloned policy (the failure observed in Idea B:
entropy inflation degrades the policy from iteration ~10 and recovery takes ~500
iterations). Expert tokens are recomputed online from the frozen continuous baseline on
the visited states, so the anchor does not go stale as the state distribution shifts.
"""

import argparse
import importlib.metadata as metadata
import os
import sys
import time
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
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

from oat_rl.ar_head import ARTokenDistribution
from oat_rl.token_env_wrapper import OATTokenVecEnvWrapper
from oat_tok.tokenizer import load_oat_tokenizer

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="BC warm-start + anchored PPO for OAT-token actions.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=1000, help="Number of PPO iterations.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--tokenizer", type=str, required=True, help="Path to the trained OAT tokenizer checkpoint.")
parser.add_argument("--baseline_checkpoint", type=str, required=True, help="Continuous-action expert checkpoint.")
parser.add_argument("--bc_rollout_steps", type=int, default=128, help="Expert rollout steps for the BC dataset.")
parser.add_argument("--bc_epochs", type=int, default=40, help="Behavior-cloning epochs.")
parser.add_argument("--bc_batch_size", type=int, default=8192, help="Behavior-cloning minibatch size.")
parser.add_argument("--bc_lr", type=float, default=1.0e-3, help="Behavior-cloning learning rate.")
parser.add_argument("--bc_anchor_coef", type=float, default=0.3, help="Initial BC-anchor weight in the PPO loss.")
parser.add_argument("--bc_anchor_iters", type=int, default=400, help="Iterations over which the anchor decays to 0.")
parser.add_argument(
    "--bc_anchor_decay", type=str, default="cosine", choices=["cosine", "linear", "const"], help="Anchor decay shape."
)
parser.add_argument("--ppo_entropy_coef", type=float, default=None, help="Override PPO entropy coefficient.")
parser.add_argument("--ppo_lr", type=float, default=None, help="Override PPO learning rate.")
parser.add_argument("--ppo_schedule", type=str, default=None, help="Override PPO LR schedule (adaptive|fixed).")
parser.add_argument("--ppo_desired_kl", type=float, default=None, help="Override PPO desired KL (adaptive schedule).")
parser.add_argument("--ppo_gamma", type=float, default=None, help="Override PPO discount factor.")
parser.add_argument("--ppo_lam", type=float, default=None, help="Override PPO GAE lambda.")
parser.add_argument(
    "--token_prefix",
    type=int,
    default=None,
    help="Token-prefix budget: policy predicts only the first k tokens (ordered tokenizer required).",
)
parser.add_argument(
    "--prefix_curriculum",
    type=str,
    default=None,
    metavar="START:GROW_ITERS",
    help="Grow the active token prefix from START to the full horizon over GROW_ITERS iterations (e.g. 4:600).",
)
parser.add_argument(
    "--ar_head",
    action="store_true",
    default=False,
    help="Use the autoregressive token head (token k conditioned on tokens 1..k-1).",
)
parser.add_argument("--ar_context_dim", type=int, default=256, help="Context size of the AR head (MLP output).")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def collect_bc_dataset(inner_env, baseline_policy, actor, critic, tokenizer, num_steps, gamma):
    """Roll out the expert, returning (X [N, obs_dim], token targets [N, K], MC returns [N, 1])."""
    obs_groups = actor.obs_groups
    xs: list[torch.Tensor] = []
    toks: list[torch.Tensor] = []
    rews: list[torch.Tensor] = []
    dones_list: list[torch.Tensor] = []
    obs = inner_env.get_observations()
    for step in range(num_steps):
        with torch.inference_mode():
            action = baseline_policy(obs)  # [B, action_dim], deterministic expert
            x = torch.cat([obs[g] for g in obs_groups], dim=-1)  # actor input, pre-normalization
            target_tokens = tokenizer.tokenize(action.unsqueeze(1))  # [B, K]
        xs.append(x.clone())
        toks.append(target_tokens.clone())
        with torch.inference_mode():
            obs, rew, dones, _ = inner_env.step(action)
        rews.append(rew.clone())
        dones_list.append(dones.clone())
        if (step + 1) % 32 == 0:
            print(f"[BC] collected {step + 1}/{num_steps} steps")

    # discounted return-to-go (truncated bootstrap = 0 at the end of the rollout)
    rewards = torch.stack(rews, dim=0)  # [T, B]
    dones_t = torch.stack(dones_list, dim=0).float()  # [T, B]
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in range(num_steps - 1, -1, -1):
        running = rewards[t] + gamma * running * (1.0 - dones_t[t])
        returns[t] = running

    x_all = torch.cat([x.cpu() for x in xs], dim=0)
    tok_all = torch.cat([t.cpu() for t in toks], dim=0)
    ret_all = returns.reshape(-1, 1).cpu()
    return x_all, tok_all, ret_all


def _fit_normalizer(model, X, batch_size, device):
    """Fit an MLPModel's empirical observation normalizer on inputs X."""
    if not getattr(model, "obs_normalization", False):
        return
    model.obs_normalizer.train()
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            model.obs_normalizer.update(X[i : i + batch_size].to(device))
    model.obs_normalizer.eval()


def behavior_clone(actor, critic, X, tokens, returns, args, device):
    """Clone the token actor (CE on tokens) and warm-start the critic (MSE on MC returns)."""
    n = X.shape[0]
    _fit_normalizer(actor, X, args.bc_batch_size, device)
    _fit_normalizer(critic, X, args.bc_batch_size, device)

    optimizer = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=args.bc_lr)
    num_categories = tokens.max().item() + 1
    for epoch in range(args.bc_epochs):
        perm = torch.randperm(n)
        total_ce = 0.0
        total_acc = 0.0
        total_v = 0.0
        steps = 0
        for i in range(0, n, args.bc_batch_size):
            idx = perm[i : i + args.bc_batch_size]
            xb = X[idx].to(device)
            yb = tokens[idx].to(device).long()  # [b, K]
            vb = returns[idx].to(device)  # [b, 1]
            trunk_out = actor.mlp(actor.obs_normalizer(xb))
            if isinstance(actor.distribution, ARTokenDistribution):
                logits = actor.distribution.teacher_forced_logits(trunk_out, yb)  # [b, K, C]
            else:
                logits = trunk_out  # [b, K, C]
            b, k, c = logits.shape
            ce = F.cross_entropy(logits.reshape(b * k, c), yb.reshape(b * k))
            value = critic.mlp(critic.obs_normalizer(xb))  # [b, 1]
            v_loss = F.mse_loss(value, vb)
            loss = ce + 0.5 * v_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
            optimizer.step()
            with torch.no_grad():
                total_acc += (logits.argmax(-1) == yb).float().mean().item()
            total_ce += ce.item()
            total_v += v_loss.item()
            steps += 1
        print(
            f"[BC] epoch {epoch + 1}/{args.bc_epochs} ce={total_ce / steps:.4f}"
            f" token_acc={total_acc / steps:.3f} v_mse={total_v / steps:.3f} (C={num_categories})"
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """BC warm-start then anchored PPO over OAT tokens."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        agent_cfg.max_iterations = args_cli.max_iterations
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        experiment_name = args_cli.experiment_name or "dexsuite_kuka_allegro_oat_anchor"
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
        log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(f"[INFO] Logging experiment in directory: {log_dir}")
        env_cfg.log_dir = log_dir

        device = env_cfg.sim.device if env_cfg.sim.device is not None else "cuda:0"
        tokenizer, tok_config = load_oat_tokenizer(args_cli.tokenizer, device=device)
        print(f"[INFO] Loaded OAT tokenizer: {tok_config}, codebook_size={tokenizer.codebook_size}")

        # build environments: inner continuous env + token wrapper
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        inner_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        token_env = OATTokenVecEnvWrapper(
            inner_env,
            tokenizer,
            num_tokens=args_cli.token_prefix,
            dynamic_prefix=args_cli.prefix_curriculum is not None,
        )
        print(
            f"[INFO] Token env: num_actions={token_env.num_actions} tokens x {tokenizer.codebook_size} codes,"
            f" chunk_horizon={token_env.chunk_horizon}"
        )

        # token PPO runner (multi-categorical actor) with the BC-anchored PPO algorithm
        train_cfg = agent_cfg.to_dict()
        train_cfg["actor"]["distribution_cfg"] = {
            "class_name": "oat_rl.multi_categorical:MultiCategoricalDistribution",
            "num_categories": tokenizer.codebook_size,
        }
        train_cfg["algorithm"]["class_name"] = "oat_rl.bc_anchor_ppo:BCAnchorPPO"
        train_cfg["algorithm"]["bc_anchor_coef"] = args_cli.bc_anchor_coef
        train_cfg["algorithm"]["bc_anchor_iters"] = args_cli.bc_anchor_iters
        train_cfg["algorithm"]["bc_anchor_decay"] = args_cli.bc_anchor_decay
        if args_cli.ar_head:
            train_cfg["actor"]["distribution_cfg"] = {
                "class_name": "oat_rl.ar_head:ARTokenDistribution",
                "num_categories": tokenizer.codebook_size,
                "context_dim": args_cli.ar_context_dim,
            }
            print(f"[INFO] AR token head: context_dim={args_cli.ar_context_dim}")
        if args_cli.prefix_curriculum is not None:
            start_k, grow_iters = (int(v) for v in args_cli.prefix_curriculum.split(":"))
            train_cfg["actor"]["distribution_cfg"]["class_name"] = (
                "oat_rl.prefix_curriculum:PrefixCurriculumDistribution"
            )
            train_cfg["algorithm"]["class_name"] = "oat_rl.prefix_curriculum:PrefixCurriculumBCAnchorPPO"
            train_cfg["algorithm"]["prefix_start"] = start_k
            train_cfg["algorithm"]["prefix_end"] = token_env.num_actions
            train_cfg["algorithm"]["prefix_grow_iters"] = grow_iters
            print(f"[INFO] Prefix curriculum: {start_k} -> {token_env.num_actions} over {grow_iters} iters")
        if args_cli.ppo_entropy_coef is not None:
            train_cfg["algorithm"]["entropy_coef"] = args_cli.ppo_entropy_coef
        if args_cli.ppo_lr is not None:
            train_cfg["algorithm"]["learning_rate"] = args_cli.ppo_lr
        if args_cli.ppo_schedule is not None:
            train_cfg["algorithm"]["schedule"] = args_cli.ppo_schedule
        if args_cli.ppo_desired_kl is not None:
            train_cfg["algorithm"]["desired_kl"] = args_cli.ppo_desired_kl
        if args_cli.ppo_gamma is not None:
            train_cfg["algorithm"]["gamma"] = args_cli.ppo_gamma
        if args_cli.ppo_lam is not None:
            train_cfg["algorithm"]["lam"] = args_cli.ppo_lam
        print(
            f"[INFO] PPO algo: entropy_coef={train_cfg['algorithm']['entropy_coef']}"
            f" lr={train_cfg['algorithm']['learning_rate']} schedule={train_cfg['algorithm']['schedule']}"
            f" desired_kl={train_cfg['algorithm']['desired_kl']}"
            f" anchor(coef={args_cli.bc_anchor_coef}, iters={args_cli.bc_anchor_iters},"
            f" decay={args_cli.bc_anchor_decay})"
        )
        runner = OnPolicyRunner(token_env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
        actor = runner.alg.actor
        critic = runner.alg.critic
        gamma = float(train_cfg["algorithm"].get("gamma", 0.99))

        # baseline (continuous) expert runner on the inner env; kept alive for the anchor
        baseline_runner = OnPolicyRunner(inner_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        baseline_path = retrieve_file_path(args_cli.baseline_checkpoint)
        baseline_runner.load(baseline_path)
        baseline_policy = baseline_runner.get_inference_policy(device=device)
        print(f"[INFO] Loaded baseline expert from: {baseline_path}")

        # -- Phase 1: behavior cloning -------------------------------------------------
        print("[INFO] === BC phase ===")
        bc_start = time.time()
        X, tokens, returns = collect_bc_dataset(
            inner_env, baseline_policy, actor, critic, tokenizer, args_cli.bc_rollout_steps, gamma
        )
        tokens = tokens[:, : token_env.num_actions]  # truncate targets to the token-prefix budget
        print(
            f"[INFO] BC dataset: X={tuple(X.shape)} tokens={tuple(tokens.shape)}"
            f" returns(mean={returns.mean():.2f} std={returns.std():.2f})"
        )
        behavior_clone(actor, critic, X, tokens, returns, args_cli, device)
        print(f"[INFO] BC phase done in {time.time() - bc_start:.1f}s")
        del X, tokens, returns
        torch.cuda.empty_cache()

        # attach the anchor (expert stays loaded for online relabeling during PPO)
        runner.alg.set_bc_anchor(baseline_policy, tokenizer)

        # dump configs
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), train_cfg)
        with open(os.path.join(log_dir, "params", "oat.txt"), "w") as f:
            f.write(
                f"tokenizer={os.path.abspath(args_cli.tokenizer)}\nconfig={tok_config}\n"
                f"baseline={os.path.abspath(baseline_path)}\n"
                f"bc_rollout_steps={args_cli.bc_rollout_steps} bc_epochs={args_cli.bc_epochs}\n"
                f"bc_anchor_coef={args_cli.bc_anchor_coef} bc_anchor_iters={args_cli.bc_anchor_iters}"
                f" bc_anchor_decay={args_cli.bc_anchor_decay}\n"
            )

        # -- Phase 2: anchored PPO fine-tune --------------------------------------------
        print("[INFO] === PPO phase (BC anchor) ===")
        start_time = time.time()
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        token_env.close()


if __name__ == "__main__":
    main()
