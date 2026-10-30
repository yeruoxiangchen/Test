"""Seconds-based soft phase gates; equal chunked SDPA for routed and zero-bias control.

Adapts the existing Prompt Relay adapter's conditional-context mechanism.
The temporal penalty here is a fixed alternative, not the upstream minimum-window formula.
"""
import contextlib
import hashlib
import torch
import torch.nn.functional as F

from video_count_probe.core import digest, token_groups


def phase_bias(latent_frames, key_length, groups, sigma=.25, cap=8., frame_stride_seconds=4/24):
    if sigma <= 0 or cap <= 0 or frame_stride_seconds <= 0:
        raise ValueError('Invalid temporal gate parameters')
    times = torch.arange(latent_frames, dtype=torch.float32) * frame_stride_seconds
    bias = torch.zeros(latent_frames, key_length)
    used = set()
    for group in groups:
        start, end, tokens = group['start'], group['end'], group['tokens']
        if not 0 <= start < end <= float(times[-1]) or not tokens or min(tokens) < 0 or max(tokens) >= key_length or used.intersection(tokens):
            raise ValueError('Invalid/overlapping phase')
        used.update(tokens)
        distance = (start - times).clamp_min(0) + (times - end).clamp_min(0)
        bias[:, tokens] = -(distance.square() / (2 * sigma**2)).clamp_max(cap)[:, None]
    return bias


def routing_material(plan, model_dir):
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import T5TokenizerFast
    prompt = plan['conditions']['text_only']['prompt']
    if any(s['prompt'] != prompt for s in plan['conditions'].values()) or prompt_clean(prompt) != prompt:
        raise ValueError('Different prompts or cleaning changed character offsets')
    tokenizer = T5TokenizerFast.from_pretrained(model_dir, subfolder='tokenizer', local_files_only=True)
    encoded = tokenizer(prompt, truncation=False, return_offsets_mapping=True)
    if len(encoded['input_ids']) > 512:
        raise ValueError('Phase prompt would be truncated')
    groups = token_groups(encoded['offset_mapping'], plan['routing']['segments'])
    cfg = plan['routing']
    bias = phase_bias(cfg['latent_frames'], 512, groups, cfg['sigma_seconds'], cfg['maximum_penalty'])
    audit = {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
             'input_ids': encoded['input_ids'], 'groups': groups,
             'bias_sha256': hashlib.sha256(bias.numpy().tobytes()).hexdigest(),
             'shape': list(bias.shape), 'nonzero_entries': int((bias != 0).sum()),
             'global_tokens': sorted(set(range(512)) - {i for g in groups for i in g['tokens']})}
    audit['sha256'] = digest(audit)
    return bias, audit


class PhaseProcessor:
    def __init__(self, original, owner):
        self.original, self.owner = original, owner

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        owner = self.owner
        if not owner.conditional:
            owner.unconditional_calls += 1
            return self.original(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        if encoder_hidden_states is None or rotary_emb is not None or attn.add_k_proj is not None or attention_mask is not None:
            raise ValueError('Only unmasked text cross-attention supported')
        if getattr(attn, 'fused_projections', False):
            raise ValueError('Fused projections unsupported')
        batch, queries, _ = hidden_states.shape
        if batch != 1 or queries != owner.bias.shape[0] * owner.tokens_per_frame or encoder_hidden_states.shape[1] != owner.bias.shape[1]:
            raise ValueError('Unexpected video/text token layout')
        q = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        k = attn.norm_k(attn.to_k(encoder_hidden_states)).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        v = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        bias = owner.bias.to(device=q.device, dtype=q.dtype)
        if not owner.enabled:
            bias = torch.zeros_like(bias)
        output = torch.empty_like(q)
        for start in range(0, queries, owner.chunk_size):
            end = min(queries, start + owner.chunk_size)
            indices = torch.arange(start, end, device=q.device) // owner.tokens_per_frame
            mask = bias[indices][None, None]
            output[:, :, start:end] = F.scaled_dot_product_attention(q[:, :, start:end], k, v, attn_mask=mask)
        owner.conditional_calls += 1
        result = output.transpose(1, 2).flatten(2)
        return attn.to_out[1](attn.to_out[0](result))


class PhaseRouting:
    def __init__(self, transformer, bias, tokens_per_frame, enabled, chunk_size=1024):
        self.transformer, self.bias = transformer, bias
        self.tokens_per_frame, self.enabled, self.chunk_size = tokens_per_frame, enabled, chunk_size
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        self.conditional = False
        self.conditional_calls = self.unconditional_calls = 0

    def __enter__(self):
        self.saved = [(b.attn2, b.attn2.processor) for b in self.transformer.blocks]
        self.old_context = self.transformer.cache_context
        self.had_instance_context = 'cache_context' in self.transformer.__dict__
        self.instance_context = self.transformer.__dict__.get('cache_context')
        owner = self

        @contextlib.contextmanager
        def cache_context(name):
            if name not in ('cond', 'uncond'):
                raise ValueError('Unexpected CFG cache context')
            before = owner.conditional
            owner.conditional = name == 'cond'
            try:
                with owner.old_context(name):
                    yield
            finally:
                owner.conditional = before

        for attention, processor in self.saved:
            attention.set_processor(PhaseProcessor(processor, self))
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

    def validate_calls(self, steps):
        expected = len(self.transformer.blocks) * steps
        if self.conditional_calls != expected or self.unconditional_calls != expected:
            raise RuntimeError(f'Routing/CFG call mismatch: {self.conditional_calls}/{self.unconditional_calls}, expected {expected}')
        return {'enabled': self.enabled, 'conditional_calls': self.conditional_calls,
                'unconditional_calls': self.unconditional_calls, 'expected_each': expected,
                'conditional_processor': 'chunked_sdpa', 'unconditional_processor': 'native'}
