# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OAT (Ordered Action Tokenization) tokenizer.

Adapted from https://github.com/Chaoqi-LIU/oat (MIT license). The hydra/zarr-based
checkpointing and ``LinearNormalizer`` are replaced with plain torch equivalents so the
tokenizer can be used inside the Isaac Lab training stack without extra dependencies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from oat_tok.decoder.single_pass_decoder import SinglePassDecoder
from oat_tok.encoder.register_encoder import RegisterEncoder
from oat_tok.normalizer import ActionNormalizer
from oat_tok.quantizer.fsq import FSQ


class OATTok(torch.nn.Module):
    """Ordered action tokenizer: encoder + FSQ quantizer + single-pass (nested-dropout) decoder."""

    def __init__(self, encoder: RegisterEncoder, decoder: SinglePassDecoder, quantizer: FSQ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.normalizer = ActionNormalizer(encoder.sample_emb.dim_in)
        self.latent_horizon = self.decoder.latent_horizon
        self.codebook_size = self.quantizer.codebook_size

    def get_optimizer(
        self, learning_rate: float, weight_decay: float, betas: tuple[float, float]
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay applied to >=2D parameters only."""
        decay_params = [p for _, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for _, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction MSE loss for an action-chunk batch of shape [B, T, action_dim]."""
        nsamples = self.normalizer.normalize(samples)
        latents = self.encoder(nsamples)
        latents, _ = self.quantizer(latents)
        recons = self.decoder(latents)
        return F.mse_loss(recons, nsamples)

    def encode(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode action chunks [B, T, action_dim] to (latents [B, K, d], token ids [B, K])."""
        nsamples = self.normalizer.normalize(samples)
        latents = self.encoder(nsamples)
        latents, tokens = self.quantizer(latents)
        return latents, tokens

    def decode(self, latents: torch.Tensor, eval_keep_k: list[int] | None = None) -> torch.Tensor:
        """Decode quantized latents [B, K, d] back to action chunks [B, T, action_dim]."""
        if eval_keep_k is None:
            eval_keep_k = [latents.shape[1]] * latents.shape[0]
        nsamples = self.decoder(latents, eval_keep_k=eval_keep_k)
        return self.normalizer.unnormalize(nsamples)

    def autoencode(self, samples: torch.Tensor, eval_keep_k: list[int] | None = None) -> torch.Tensor:
        """Round-trip encode/decode for reconstruction evaluation."""
        latents, _ = self.encode(samples)
        return self.decode(latents, eval_keep_k=eval_keep_k)

    def tokenize(self, samples: torch.Tensor) -> torch.Tensor:
        """Return token ids [B, K] for action chunks [B, T, action_dim]."""
        _, tokens = self.encode(samples)
        return tokens

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode token ids [B, K] back to action chunks [B, T, action_dim]."""
        latents = self.quantizer.indices_to_embedding(tokens)
        return self.decode(latents)


def build_oat_tokenizer(
    action_dim: int,
    chunk_horizon: int,
    num_registers: int = 4,
    emb_dim: int = 256,
    head_dim: int = 64,
    encoder_depth: int = 2,
    decoder_depth: int = 4,
    pdropout: float = 0.1,
    fsq_levels: list[int] | None = None,
) -> OATTok:
    """Build an :class:`OATTok` with the reference architecture from the OAT repo."""
    if fsq_levels is None:
        fsq_levels = [8, 5, 5, 5]
    latent_dim = len(fsq_levels)
    encoder = RegisterEncoder(
        sample_dim=action_dim,
        sample_horizon=chunk_horizon,
        emb_dim=emb_dim,
        head_dim=head_dim,
        depth=encoder_depth,
        pdropout=pdropout,
        latent_dim=latent_dim,
        num_registers=num_registers,
    )
    decoder = SinglePassDecoder(
        sample_dim=action_dim,
        sample_horizon=chunk_horizon,
        emb_dim=emb_dim,
        head_dim=head_dim,
        depth=decoder_depth,
        pdropout=pdropout,
        token_dropout_mode="pow2",
        use_causal_decoder=True,
        latent_dim=latent_dim,
        latent_horizon=num_registers,
    )
    quantizer = FSQ(levels=fsq_levels)
    return OATTok(encoder, decoder, quantizer)


def save_oat_tokenizer(tokenizer: OATTok, path: str, config: dict) -> None:
    """Save tokenizer weights together with the build config."""
    torch.save({"state_dict": tokenizer.state_dict(), "config": config}, path)


def load_oat_tokenizer(path: str, device: str = "cpu") -> tuple[OATTok, dict]:
    """Load a tokenizer saved with :func:`save_oat_tokenizer`."""
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload["config"]
    tokenizer = build_oat_tokenizer(**config)
    tokenizer.load_state_dict(payload["state_dict"])
    tokenizer.to(device)
    tokenizer.eval()
    return tokenizer, config
