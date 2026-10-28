"""Atomic joint checkpoints: geometry, adapters, EMA, optimizer and RNG."""
from dataclasses import asdict
import os
from pathlib import Path
import random

import numpy as np
import torch

from .contracts import CHECKPOINT_FORMAT


def checkpoint_condition_mode(checkpoint):
    """Legacy V2 checkpoints used the full layered branch exclusively."""
    saved = checkpoint.get('args', {}).get('condition_mode', 'full')
    if checkpoint.get('condition_mode', saved) != saved:
        raise ValueError('checkpoint condition_mode metadata disagrees with training args')
    return saved


def bound_state(model):
    return {name: p for name, p in model.named_parameters()
            if p.requires_grad or name.startswith(('target_lari.', 'geometry.'))}


def initialize_ema(model):
    return {name: p.detach().float().clone() for name, p in bound_state(model).items()}


@torch.no_grad()
def update_ema(model, ema, decay=.999):
    for name, value in bound_state(model).items():
        ema[name].lerp_(value.detach().float(), 1. - decay)


def save_checkpoint(path, *, model, config, stage, bindings, data_identity, optimizer,
                    ema, step, cursor, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': CHECKPOINT_FORMAT, 'stage': stage, 'geometry_config': asdict(config),
        'condition_mode': getattr(args, 'condition_mode', 'full'),
        'bindings': bindings, 'data_identity': data_identity, 'step': step, 'cursor': cursor,
        'model_state': {n: p.detach().cpu().clone() for n, p in bound_state(model).items()},
        'ema_state': {n: p.detach().cpu().clone() for n, p in ema.items()},
        'optimizer': optimizer.state_dict(), 'args': vars(args),
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}}
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('expected a V2 joint checkpoint')
    return value


@torch.no_grad()
def load_state(model, checkpoint, config, *, stage, bindings, ema=False):
    if checkpoint['geometry_config'] != asdict(config) or checkpoint['stage'] != stage:
        raise ValueError('V2 architecture/stage mismatch')
    if checkpoint['bindings'] != bindings:
        raise ValueError('frozen parent/base checkpoint bindings differ')
    if stage == 'ss':
        if getattr(model, 'condition_mode', 'full') != checkpoint_condition_mode(checkpoint):
            raise ValueError('SS condition_mode differs from checkpoint training policy')
        tolerance = checkpoint.get('args', {}).get('ss_absolute_tolerance')
        tolerance = config.absolute_tolerance if tolerance is None else tolerance
        if getattr(model, 'match_absolute_tolerance', config.absolute_tolerance) != tolerance:
            raise ValueError('SS query tolerance differs from checkpoint training policy')
    state = checkpoint['ema_state' if ema else 'model_state']
    parameters = bound_state(model)
    if state.keys() != parameters.keys():
        raise ValueError('V2 parameter set differs; do not load partial checkpoints')
    for name, p in parameters.items():
        if state[name].shape != p.shape:
            raise ValueError(f'V2 shape mismatch: {name}')
        p.copy_(state[name].to(p))


@torch.no_grad()
def load_geometry(model, path, config, ema=True):
    checkpoint = read_checkpoint(path)
    if checkpoint['geometry_config'] != asdict(config):
        raise ValueError('geometry config mismatch')
    state = checkpoint['ema_state' if ema else 'model_state']
    prefix = {'lari': '', 'ss': 'target_lari.', 'slat': 'geometry.'}[checkpoint['stage']]
    state = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
    model.load_state_dict(state, strict=True)


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if value['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(value['cuda'])
