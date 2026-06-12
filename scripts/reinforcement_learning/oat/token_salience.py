# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure per-token salience of an OAT tokenizer (no simulator needed).

For each register position k, replace token k with a random codebook entry while keeping
all other tokens intact, then measure the unnormalized action-reconstruction MSE delta
against the clean round-trip. This tests whether late tokens really carry only small
"refinement" information (the premise of a tail-first PPO curriculum).

Also reports the cumulative variant: randomize all tokens from position k onward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oat_tok.tokenizer import load_oat_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer checkpoint path.")
    parser.add_argument("--dataset", type=str, required=True, help="Action chunk dataset (.pt).")
    parser.add_argument("--num_samples", type=int, default=20000, help="Number of chunks to evaluate.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer, config = load_oat_tokenizer(args.tokenizer, device=str(device))
    num_registers = config["num_registers"]
    codebook_size = tokenizer.codebook_size
    print(f"[INFO] Tokenizer: {config}, codebook_size={codebook_size}")

    payload = torch.load(args.dataset, weights_only=True)
    actions: torch.Tensor = payload["actions"]
    perm = torch.randperm(actions.shape[0], generator=torch.Generator().manual_seed(args.seed))
    actions = actions[perm[: args.num_samples]].to(device)
    print(f"[INFO] Evaluating on {actions.shape[0]} chunks, action std={actions.std().item():.3f}")

    def mse_with_tokens(tokens: torch.Tensor, reference: torch.Tensor) -> float:
        total, count = 0.0, 0
        for i in range(0, tokens.shape[0], args.batch_size):
            recon = tokenizer.detokenize(tokens[i : i + args.batch_size])
            ref = reference[i : i + args.batch_size]
            total += torch.mean((recon - ref) ** 2).item() * ref.shape[0]
            count += ref.shape[0]
        return total / count

    with torch.no_grad():
        clean_tokens = []
        for i in range(0, actions.shape[0], args.batch_size):
            clean_tokens.append(tokenizer.tokenize(actions[i : i + args.batch_size]))
        clean_tokens = torch.cat(clean_tokens, dim=0)  # [N, K]

        roundtrip_mse = mse_with_tokens(clean_tokens, actions)
        action_var = actions.var().item()
        print(f"[INFO] Clean round-trip MSE: {roundtrip_mse:.4f} (action var {action_var:.3f})")

        single_pos: dict[str, float] = {}
        for k in range(num_registers):
            corrupted = clean_tokens.clone()
            corrupted[:, k] = torch.randint(0, codebook_size, (corrupted.shape[0],), device=device)
            single_pos[str(k)] = mse_with_tokens(corrupted, actions)

        tail_from: dict[str, float] = {}
        for k in range(num_registers):
            corrupted = clean_tokens.clone()
            n_tail = num_registers - k
            corrupted[:, k:] = torch.randint(0, codebook_size, (corrupted.shape[0], n_tail), device=device)
            tail_from[str(k)] = mse_with_tokens(corrupted, actions)

    print("\n[RESULT] MSE after randomizing SINGLE token k (clean round-trip = "
          f"{roundtrip_mse:.4f}):")
    for k in range(num_registers):
        delta = single_pos[str(k)] - roundtrip_mse
        print(f"  token {k:2d}: mse={single_pos[str(k)]:.4f}  delta={delta:+.4f}")

    print("\n[RESULT] MSE after randomizing ALL tokens from position k onward:")
    for k in range(num_registers):
        print(f"  from {k:2d}: mse={tail_from[str(k)]:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "tokenizer": os.path.abspath(args.tokenizer),
                    "config": config,
                    "num_samples": int(actions.shape[0]),
                    "action_var": action_var,
                    "roundtrip_mse": roundtrip_mse,
                    "single_token_randomized_mse": single_pos,
                    "tail_from_k_randomized_mse": tail_from,
                },
                f,
                indent=2,
            )
        print(f"\n[INFO] Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
