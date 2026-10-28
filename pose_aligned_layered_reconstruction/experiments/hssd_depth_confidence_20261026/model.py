"""Exact SharedSS geometry adapter for crossed depth/confidence evidence.

Only the evidence source changes. The full archived appearance parent and the
new adapter's initialization, parameterization, and CFG paths remain identical.
"""
import torch

from ..hssd_shared_ss_20261026.model import SharedEvidenceSS, ResidualCFG, validate_evidence
from ..hssd_cell_query_20261023.model import NestedSS
from ..hssd_cell_query_20261023.common import tensor_hash
from ...v2.checkpoints import read_checkpoint, load_state
from ...v2.layer_predictor import GeometryConfig
from ...v2.ss import build_ss
from .common import ARMS, verify


def build(contract, device, checkpointing=False):
    if contract['arm'] not in ARMS:
        raise ValueError('Only the two new depth/confidence arms may be trained here')
    parent_binding = contract['parent_checkpoint']
    checkpoint = torch.load(verify(parent_binding), map_location='cpu', weights_only=False)
    parent_contract = checkpoint['contract']
    if (parent_contract['arm'] != 'appearance_control' or parent_contract['domain'] != 'occluded'
            or checkpoint['step'] != parent_contract['steps'] or checkpoint['step'] != 200):
        raise ValueError('Expected the complete old occluded appearance_control parent')

    # Resolve physical storage without rewriting the frozen logical bindings.
    base_checkpoint = read_checkpoint(verify(parent_contract['appearance_base']))
    saved = base_checkpoint['args']
    if saved['condition_mode'] != 'no_layer' or not saved['freeze_geometry']:
        raise ValueError('Residual base must be the trained, frozen-geometry no_layer model')
    config = GeometryConfig(**base_checkpoint['geometry_config'])
    sampler, base, parent_bindings = build_ss(config, device, saved['parent_ss30k'], saved['pretrained'],
        checkpointing, absolute_tolerance=saved.get('ss_absolute_tolerance'), condition_mode='no_layer')
    base.target_lari.requires_grad_(False)
    load_state(base, base_checkpoint, config, stage='ss', bindings=parent_bindings, ema=False)
    parent = NestedSS(base, parent_contract['arm']).to(device)
    if checkpoint['bindings'] != parent_bindings:
        raise ValueError('Old appearance parent dependencies changed')
    parent.residual.load_state_dict(checkpoint['residual'], strict=True)
    model = SharedEvidenceSS(parent, 'geometry_observed', seed=contract['seed'],
        channels=int(contract.get('residual_channels', 128))).to(device)
    model.public_arm = contract['arm']
    bindings = {'parent_checkpoint':parent_binding, 'parent_bindings':parent_bindings,
        'parent_residual_sha':tensor_hash(checkpoint['residual']),
        'parent_steps':checkpoint['step'], 'parent_arm':parent_contract['arm']}
    return sampler, model, saved, bindings
