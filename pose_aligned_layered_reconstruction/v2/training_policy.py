"""Explicit diffusion and optimizer policies, including legacy resume defaults."""
import math

import torch


POLICY_DEFAULTS = {
    'init_weights': 'ema', 't_sampling': 'uniform', 'lr_schedule': 'cosine',
    'lr_schedule_steps': 0, 'ss_absolute_tolerance': None,
}


def sample_time(policy, device):
    if policy == 'uniform':
        return float(torch.rand((), device=device).clamp(.001, .999))
    if policy == 'logit_normal':
        # Exact V1 distribution: sigmoid(N(1, 1)); not uniform in diffusion time.
        return float(torch.sigmoid(torch.randn((), device=device) + 1.))
    raise ValueError(f'unknown diffusion time policy: {policy}')


def learning_rate_factor(args, step):
    if args.lr_schedule == 'constant':
        return 1.
    horizon = args.lr_schedule_steps or args.max_steps
    warmup = min(1., step / max(args.warmup_steps, 1))
    return warmup * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(step / horizon, 1.))))


def validate_resume_policy(saved, current):
    keys = ('learning_rate', 'geometry_lr_scale', 'warmup_steps', 'max_steps',
            'cross_view_weight', 'layer_weight', 'p_uncond', 'finetune_lora',
            'freeze_geometry', 'amp', *POLICY_DEFAULTS)
    for key in keys:
        if saved.get(key, POLICY_DEFAULTS.get(key)) != current.get(key, POLICY_DEFAULTS.get(key)):
            raise ValueError(f'resume training policy differs: {key}')
