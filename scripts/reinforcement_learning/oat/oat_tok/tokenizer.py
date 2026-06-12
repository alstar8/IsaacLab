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

from oat_tok.decoder.boosted_residual_decoder import BoostedResidualDecoder
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
        """Decode token ids [B, k] back to action chunks [B, T, action_dim].

        Accepts a token prefix (k < latent_horizon): missing positions are padded and the
        decoder is asked for the prefix-k reconstruction (safe-tail / anytime decoding).
        """
        prefix = tokens.shape[1]
        eval_keep_k = None
        if prefix < self.latent_horizon:
            pad = torch.zeros(
                tokens.shape[0], self.latent_horizon - prefix, dtype=tokens.dtype, device=tokens.device
            )
            tokens = torch.cat([tokens, pad], dim=1)
            eval_keep_k = [prefix] * tokens.shape[0]
        latents = self.quantizer.indices_to_embedding(tokens)
        return self.decode(latents, eval_keep_k=eval_keep_k)


class BoostedOATTok(OATTok):
    """OAT tokenizer with a boosted residual decoder trained on every prefix at once.

    The decoder returns reconstructions for all prefix lengths via a cumulative sum, so
    the training loss averages the MSE over all prefixes 1..K. This is the dense version
    of nested dropout: every token is explicitly trained to reduce the residual of the
    prefix before it.
    """

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        nsamples = self.normalizer.normalize(samples)
        latents = self.encoder(nsamples)
        latents, _ = self.quantizer(latents)
        prefix_recons = self.decoder(latents, return_all_prefixes=True)  # [B, K, T, D]
        target = nsamples.unsqueeze(1).expand_as(prefix_recons)
        return F.mse_loss(prefix_recons, target)


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
    token_dropout_mode: str = "pow2",
    decoder_type: str = "single_pass",
    shrinkage: float = 0.85,
) -> OATTok:
    """Build an :class:`OATTok` with the reference architecture from the OAT repo.

    Args:
        token_dropout_mode: Nested-dropout prefix sampling for the ``single_pass`` decoder.
            ``"pow2"`` trains only power-of-two prefixes (historical default); ``"uniform"``
            trains all prefix lengths 1..num_registers, which is the coarse-to-fine recipe
            from the OAT paper.
        decoder_type: ``"single_pass"`` (nested-dropout transformer decoder) or
            ``"boosted"`` (additive residual decoder with shrinkage; tokens act like
            gradient-boosting stages).
        shrinkage: Geometric amplitude decay per token stage for the ``boosted`` decoder.
    """
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
    quantizer = FSQ(levels=fsq_levels)
    if decoder_type == "boosted":
        boosted_decoder = BoostedResidualDecoder(
            sample_dim=action_dim,
            sample_horizon=chunk_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=decoder_depth,
            pdropout=pdropout,
            latent_dim=latent_dim,
            latent_horizon=num_registers,
            shrinkage=shrinkage,
        )
        return BoostedOATTok(encoder, boosted_decoder, quantizer)
    decoder = SinglePassDecoder(
        sample_dim=action_dim,
        sample_horizon=chunk_horizon,
        emb_dim=emb_dim,
        head_dim=head_dim,
        depth=decoder_depth,
        pdropout=pdropout,
        token_dropout_mode=token_dropout_mode,
        use_causal_decoder=True,
        latent_dim=latent_dim,
        latent_horizon=num_registers,
    )
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
