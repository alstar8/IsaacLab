# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reshape an OAT action-chunk dataset to a different (smaller) chunk horizon.

The collected dataset has shape [N, T0, action_dim] of non-overlapping chunks of T0
consecutive actions. Any divisor ``T`` of ``T0`` yields valid non-overlapping chunks of
length ``T`` by splitting each chunk into ``T0 // T`` consecutive sub-chunks, so we can
study several chunk horizons without re-running the simulator.

Also prints per-dimension action statistics (range, std) to diagnose normalization.
"""

from __future__ import annotations

import argparse
import os

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, help="Source .pt dataset [N, T0, A].")
    parser.add_argument("--chunk_horizon", type=int, required=True, help="Target chunk horizon T (divides T0).")
    parser.add_argument("--output", type=str, required=True, help="Output .pt path.")
    parser.add_argument("--stats", action="store_true", help="Print per-dim action statistics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = torch.load(args.dataset, weights_only=True)
    actions: torch.Tensor = payload["actions"]  # [N, T0, A]
    n, t0, a = actions.shape
    target = args.chunk_horizon
    if t0 % target != 0:
        raise ValueError(f"target chunk_horizon {target} must divide source horizon {t0}")

    reshaped = actions.reshape(n * (t0 // target), target, a)

    if args.stats:
        flat = actions.reshape(-1, a)
        dmin = flat.min(dim=0).values
        dmax = flat.max(dim=0).values
        std = flat.std(dim=0)
        rng = dmax - dmin
        print(f"[INFO] action_dim={a}, total transitions={flat.shape[0]}")
        print(f"[INFO] global: min={flat.min():.3f} max={flat.max():.3f} std={flat.std():.3f}")
        print(f"[INFO] per-dim range: min={rng.min():.3f} median={rng.median():.3f} max={rng.max():.3f}")
        print(f"[INFO] per-dim std:   min={std.min():.3f} median={std.median():.3f} max={std.max():.3f}")
        widest = torch.argsort(rng, descending=True)[:5].tolist()
        print(f"[INFO] widest dims (idx: range, std): " + ", ".join(f"{i}: {rng[i]:.2f}/{std[i]:.2f}" for i in widest))

    out = dict(payload)
    out["actions"] = reshaped
    out["chunk_horizon"] = target
    out["reshaped_from"] = {"source": os.path.abspath(args.dataset), "source_horizon": t0}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print(f"[INFO] Saved {reshaped.shape[0]} chunks of shape [{target}, {a}] to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
