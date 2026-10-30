"""Prompt Relay's released temporal penalty, ported to Diffusers cross-attention.

Reference: GordonChen19/Wan2.2@482339441717aee0b1cb97a314c1e733557d1536.
The original model is Wan2.2; this adapter is explicitly a Wan2.1 port.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


def temporal_bias(latent_frames, key_length, groups, sigma=0.1448):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    bias = torch.zeros(latent_frames, key_length, dtype=torch.float32)
    used = set()
    for group in groups:
        start, end = group["start"], group["end"]
        tokens = group["tokens"]
        if not (0 <= start < end <= latent_frames) or end - start < 4:
            raise ValueError("Invalid interval or interval shorter than upstream's minimum")
        if not tokens or min(tokens) < 0 or max(tokens) >= key_length or used.intersection(tokens):
            raise ValueError("Invalid/overlapping token indices")
        used.update(tokens)
        midpoint = (start + end) // 2
        window = (end - start) // 2 - 2
        distance = (torch.arange(latent_frames).float() - midpoint).abs()
        cost = torch.relu(distance - window).square() / (2 * sigma**2)
        bias[:, tokens] = -cost[:, None]
    return bias


class RelayProcessor:
    """Query chunks keep memory bounded; unconditional CFG uses upstream attention."""
    def __init__(self, original, owner):
        self.original, self.owner = original, owner

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        owner = self.owner
        if not owner.conditional or owner.strength == 0:
            return self.original(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        if encoder_hidden_states is None or rotary_emb is not None or attn.add_k_proj is not None:
            raise ValueError("Relay adapter supports only Wan text cross-attention")
        if getattr(attn, "fused_projections", False):
            raise ValueError("Fused projections have not been validated")
        batch, queries, _ = hidden_states.shape
        if queries != owner.latent_frames * owner.tokens_per_frame:
            raise ValueError("Video token layout differs from plan")
        if encoder_hidden_states.shape[1] != owner.bias.shape[1] or batch != 1:
            raise ValueError("Unexpected context length or batch; only batch=1 is validated")
        q = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        k = attn.norm_k(attn.to_k(encoder_hidden_states)).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        v = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        frame_bias = owner.bias.to(device=q.device, dtype=q.dtype) * owner.strength
        output = torch.empty_like(q)
        for start in range(0, queries, owner.chunk_size):
            end = min(start + owner.chunk_size, queries)
            frames = torch.arange(start, end, device=q.device) // owner.tokens_per_frame
            mask = frame_bias[frames][None, None]
            if attention_mask is not None:
                raise ValueError("An extra cross-attention mask is unsupported")
            output[:, :, start:end] = F.scaled_dot_product_attention(q[:, :, start:end], k, v, attn_mask=mask)
        owner.routed_calls += 1
        result = output.transpose(1, 2).flatten(2)
        return attn.to_out[1](attn.to_out[0](result))


class RelayRouting:
    def __init__(self, transformer, bias, tokens_per_frame, strength=1.0, chunk_size=256):
        self.transformer = transformer
        self.bias = bias
        self.latent_frames = bias.shape[0]
        self.tokens_per_frame = tokens_per_frame
        self.strength = strength
        self.chunk_size = chunk_size
        self.conditional = False
        self.routed_calls = 0

    def __enter__(self):
        self.saved = [(b.attn2, b.attn2.processor) for b in self.transformer.blocks]
        self.old_context = self.transformer.cache_context
        self.had_instance_context = "cache_context" in self.transformer.__dict__
        self.instance_context = self.transformer.__dict__.get("cache_context")
        owner = self

        @contextlib.contextmanager
        def cache_context(name):
            before = owner.conditional
            owner.conditional = name == "cond"
            try:
                with owner.old_context(name):
                    yield
            finally:
                owner.conditional = before

        for attention, processor in self.saved:
            attention.set_processor(RelayProcessor(processor, self))
        self.transformer.cache_context = cache_context
        return self

    def __exit__(self, *_):
        for attention, processor in self.saved:
            attention.set_processor(processor)
        if self.had_instance_context:
            self.transformer.cache_context = self.instance_context
        else:
            del self.transformer.cache_context
        self.conditional = False

