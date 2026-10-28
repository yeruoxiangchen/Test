import torch
from ..hssd_cell_query_20261023.model import build as old_build, ResidualCFG, observations
from .common import FORMAT, verify, tensor_hash


def build(contract, device, checkpointing=False):
    # Retain the frozen appearance path, original point interface and residual capacity.
    sampler, model, saved, bindings = old_build({**contract, 'arm':'predicted_point'}, device, checkpointing)
    cp = torch.load(verify(contract['lari']), map_location='cpu', weights_only=False)
    if (cp['format'] != FORMAT or cp['contract']['kind'] != 'lari' or
        cp['contract']['arm'] != contract['arm'] or cp['contract']['protocol'] != contract['protocol'] or
        cp['step'] != cp['contract']['steps']):
        raise ValueError('Need completed matching new LaRI')
    model.base.target_lari.load_state_dict(cp['state'], strict=True)
    model.base.target_lari.eval().requires_grad_(False)
    if tensor_hash(model.base.target_lari.state_dict()) != tensor_hash(cp['state']):
        raise ValueError('LaRI injection differs')
    bindings = {'appearance':bindings, 'lari':contract['lari'], 'lari_state_sha':tensor_hash(cp['state'])}
    return sampler, model, saved, bindings
