# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the OAT action tokenizer on a collected action-chunk dataset.

Standalone torch training (no simulator). Saves the tokenizer checkpoint, a loss-curve
plot and a JSON summary with reconstruction MSE per kept-token count (the "ordered"
property of OAT: more tokens = finer reconstruction).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oat_tok.tokenizer import build_oat_tokenizer, save_oat_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .pt chunk dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoint and curves.")
    parser.add_argument("--num_registers", type=int, default=4, help="Number of latent tokens per chunk.")
    parser.add_argument("--fsq_levels", type=int, nargs="+", default=[8, 5, 5, 5], help="FSQ quantization levels.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5.0e-5, help="Peak learning rate.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.05, help="Final LR as fraction of peak (cosine).")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Linear LR warmup epochs.")
    parser.add_argument("--norm_mode", type=str, default="std", choices=["std", "minmax"], help="Normalizer fit mode.")
    parser.add_argument(
        "--token_dropout_mode",
        type=str,
        default="pow2",
        choices=["pow2", "uniform", "uniform_pow2", "linear_biased", "quadratic_biased", "cubic_biased", "disable"],
        help="Nested-dropout prefix sampling mode for the decoder.",
    )
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="single_pass",
        choices=["single_pass", "boosted"],
        help="Decoder architecture: nested-dropout transformer or boosted residual stages.",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.85,
        help="Geometric amplitude decay per token stage (boosted decoder only).",
    )
    parser.add_argument("--val_fraction", type=float, default=0.05, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device.")
    return parser.parse_args()


def lr_at(epoch: float, args: argparse.Namespace) -> float:
    """Linear warmup then cosine decay to ``min_lr_ratio * lr``."""
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / max(args.warmup_epochs, 1)
    progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    payload = torch.load(args.dataset, weights_only=True)
    actions: torch.Tensor = payload["actions"]
    horizon = payload["chunk_horizon"]
    action_dim = payload["action_dim"]
    print(f"[INFO] Dataset: {actions.shape[0]} chunks, horizon={horizon}, action_dim={action_dim}")

    # train/val split
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(actions.shape[0], generator=generator)
    num_val = int(actions.shape[0] * args.val_fraction)
    val_actions = actions[perm[:num_val]].to(device)
    train_actions = actions[perm[num_val:]].to(device)

    config = {
        "action_dim": action_dim,
        "chunk_horizon": horizon,
        "num_registers": args.num_registers,
        "fsq_levels": args.fsq_levels,
        "token_dropout_mode": args.token_dropout_mode,
        "decoder_type": args.decoder_type,
        "shrinkage": args.shrinkage,
    }
    tokenizer = build_oat_tokenizer(**config).to(device)
    tokenizer.normalizer.fit(train_actions, mode=args.norm_mode)
    optimizer = tokenizer.get_optimizer(learning_rate=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    # variance of normalized actions (z-score → ~1.0 per dim); used for R²-style metric
    with torch.no_grad():
        norm_var = tokenizer.normalizer.normalize(val_actions).var().item()

    num_train = train_actions.shape[0]
    steps_per_epoch = num_train // args.batch_size
    train_losses: list[float] = []
    val_losses: list[float] = []

    start = time.time()
    for epoch in range(args.epochs):
        tokenizer.train()
        lr = lr_at(epoch, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        epoch_perm = torch.randperm(num_train, device=device)
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            batch = train_actions[epoch_perm[i * args.batch_size : (i + 1) * args.batch_size]]
            loss = tokenizer(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / steps_per_epoch)

        # validation: full-token reconstruction MSE in normalized space
        tokenizer.eval()
        with torch.no_grad():
            val_loss = 0.0
            num_batches = 0
            for i in range(0, val_actions.shape[0], args.batch_size):
                batch = val_actions[i : i + args.batch_size]
                val_loss += tokenizer(batch).item()
                num_batches += 1
            val_losses.append(val_loss / max(num_batches, 1))

        print(
            f"[INFO] epoch {epoch + 1}/{args.epochs} lr={lr:.2e} train_mse={train_losses[-1]:.5f}"
            f" val_mse={val_losses[-1]:.5f} frac_var_unexplained={val_losses[-1] / norm_var:.4f}"
            f" elapsed={time.time() - start:.0f}s"
        )

    # ordered reconstruction: MSE vs number of kept tokens (unnormalized action space)
    tokenizer.eval()
    keep_k_mse: dict[str, float] = {}
    with torch.no_grad():
        for k in range(1, args.num_registers + 1):
            mse_sum = 0.0
            count = 0
            for i in range(0, val_actions.shape[0], args.batch_size):
                batch = val_actions[i : i + args.batch_size]
                recon = tokenizer.autoencode(batch, eval_keep_k=[k] * batch.shape[0])
                mse_sum += torch.mean((recon - batch) ** 2).item() * batch.shape[0]
                count += batch.shape[0]
            keep_k_mse[str(k)] = mse_sum / count

    ckpt_path = os.path.join(args.output_dir, "oat_tokenizer.pt")
    save_oat_tokenizer(tokenizer, ckpt_path, config)

    # loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, args.epochs + 1), train_losses, linewidth=2.0, label="train MSE (normalized)")
    ax.plot(range(1, args.epochs + 1), val_losses, linewidth=2.0, label="val MSE (normalized)")
    ax.set_title("OAT tokenizer reconstruction loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    curve_path = os.path.join(args.output_dir, "tokenizer_loss_curve.png")
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)

    summary = {
        "dataset": os.path.abspath(args.dataset),
        "config": config,
        "epochs": args.epochs,
        "lr": args.lr,
        "norm_mode": args.norm_mode,
        "final_train_mse_normalized": train_losses[-1],
        "final_val_mse_normalized": val_losses[-1],
        "norm_var": norm_var,
        "frac_var_unexplained": val_losses[-1] / norm_var,
        "val_mse_unnormalized_by_keep_k": keep_k_mse,
        "codebook_size": tokenizer.codebook_size,
        "num_parameters": sum(p.numel() for p in tokenizer.parameters()),
        "train_time_s": time.time() - start,
    }
    summary_path = os.path.join(args.output_dir, "tokenizer_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Checkpoint: {ckpt_path}")
    print(f"[INFO] Loss curve: {curve_path}")
    print(f"[INFO] Summary: {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
