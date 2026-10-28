"""A matched zero-initialized adapter above the complete frozen SS parent.

The parent includes the old 200-step appearance residual. New observation
features never replace that parent, and both arms train the same parameter set.
"""
import torch
from torch import nn
import torch.nn.functional as F

from ..._bootstrap import bootstrap_external_paths
from ...native_ss import volume_xyz_to_flow_tokens, flow_tokens_to_volume_xyz
from ..hssd_cell_query_20261023.model import (
    ResidualAdapter, build as build_parent, observations, rms,
)
from ..hssd_cell_query_20261023.common import tensor_hash


def validate_evidence(evidence, points, device):
    """The caller supplies separately bound, already projected observations."""
    if not isinstance(evidence, dict):
        raise ValueError('Geometry arm needs an explicit observation evidence cache')
    values = {}
    for key in ('features', 'weights', 'valid'):
        if key not in evidence or not torch.is_tensor(evidence[key]):
            raise ValueError('Observation evidence missing tensor: '+key)
        values[key] = evidence[key].to(device)
    features, weights, valid = (values[k] for k in ('features','weights','valid'))
    if (features.ndim != 3 or features.shape[1:] != (points,12)
            or features.shape[0] < 1 or weights.shape != features.shape[:2]
            or valid.shape != weights.shape or valid.dtype != torch.bool
            or not features.is_floating_point() or not weights.is_floating_point()
            or not torch.isfinite(features).all() or not torch.isfinite(weights).all()
            or (weights < 0).any() or (weights[~valid] != 0).any()):
        raise ValueError('Invalid geometry features/weights/validity')
    # Evidence fusion uses its own confidence, not appearance view weights.
    # All-invalid cells must be zero. Nonempty cells use normalized weights.
    mass = weights.float().sum(0)
    present = valid.any(0)
    if ((mass[~present] != 0).any()
            or (present.any() and not torch.allclose(mass[present], torch.ones_like(mass[present]), atol=1.e-5, rtol=1.e-5))):
        raise ValueError('Observation fusion weights must normalize over valid evidence')
    return {key:value.detach() for key,value in values.items()}


class SharedEvidenceSS(nn.Module):
    def __init__(self, parent, arm, *, seed, channels=128):
        super().__init__()
        if arm not in ('appearance_control','geometry_observed'):
            raise ValueError('Unknown matched-training arm: '+str(arm))
        self.parent = parent.eval().requires_grad_(False)
        self.arm = str(arm)
        self.appearance_arm = self.arm == 'appearance_control'
        core = parent.base.flow_core
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.residual = ResidualAdapter(core.model_channels, len(core.blocks), channels=channels)

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    @torch.no_grad()
    def prepare(self, sample, device, evidence=None):
        # Parent sees only its historical image/camera allowlist, never depth,
        # GT support, latent targets or the new geometry feature cache.
        old = self.parent.prepare(observations(sample), device)
        if self.appearance_arm:
            branch = {key:old[key] for key in ('features','weights','valid')}
        else:
            branch = validate_evidence(evidence, old['features'].shape[1], device)
        return {**{key:value.detach() for key,value in old.items()},
            **{'branch_'+key:value.detach() for key,value in branch.items()}}

    def forward(self, x, t, context, prepared, enabled=True):
        if not enabled:
            return self.parent(x,t,context,prepared,enabled=True)
        bootstrap_external_paths()
        from trellis.modules.spatial import patchify, unpatchify
        parent_layer = self.parent.residual.encode(prepared['features'],prepared['weights'],prepared['valid'])
        layer = self.residual.encode(prepared['branch_features'],prepared['branch_weights'],prepared['branch_valid'])
        base,core = self.parent.base,self.parent.base.flow_core
        h = volume_xyz_to_flow_tokens(patchify(x,core.patch_size))
        h = core.input_layer(h)+core.pos_emb[None]
        embedding = core.t_embedder(t)
        if core.share_mod:
            embedding = core.adaLN_modulation(embedding)
        h,embedding = h.type(core.dtype),embedding.type(core.dtype)
        for ctx in context if isinstance(context,list) else [context]:
            ctx = ctx.type(core.dtype)
            for i,block in enumerate(core.blocks):
                # Preserve every trained parent addition in its original order.
                h = h+base.block_condition(i,prepared['appearance']).to(h.dtype)
                h = h+base.layer_block_condition(i,prepared['appearance_layer']).to(h.dtype)
                h = h+self.parent.residual.projections[i](parent_layer).to(h.dtype)
                h = h+self.residual.projections[i](layer).to(h.dtype)
                h = block(h,embedding,ctx)
        h = F.layer_norm(h.type(x.dtype),h.shape[-1:])
        return unpatchify(flow_tokens_to_volume_xyz(core.out_layer(h)),core.patch_size).contiguous()

    @torch.no_grad()
    def metrics(self, prepared):
        layer = self.residual.encode(prepared['branch_features'],prepared['branch_weights'],prepared['branch_valid'])
        return {'residual_condition_rms':rms(layer),
            'residual_injection_rms':sum(rms(p(layer)) for p in self.residual.projections)/len(self.residual.projections),
            'input_feature_rms':rms(prepared['branch_features']),
            'branch_valid_fraction':float(prepared['branch_valid'].float().mean()),
            'geometry_evidence':not self.appearance_arm}


def build(contract, device, checkpointing=False):
    from .common import verify
    parent_binding = contract['parent_checkpoint']
    checkpoint = torch.load(verify(parent_binding),map_location='cpu',weights_only=False)
    parent_contract = checkpoint['contract']
    if (parent_contract['arm']!='appearance_control' or parent_contract['domain']!='occluded'
            or checkpoint['step']!=parent_contract['steps'] or checkpoint['step']!=200):
        raise ValueError('Expected the complete old occluded appearance_control parent')
    sampler,parent,saved,parent_bindings = build_parent(parent_contract,device,checkpointing=checkpointing)
    if checkpoint['bindings']!=parent_bindings:
        raise ValueError('Old appearance parent dependencies changed')
    parent.residual.load_state_dict(checkpoint['residual'],strict=True)
    model = SharedEvidenceSS(parent,contract['arm'],seed=contract['seed'],
        channels=int(contract.get('residual_channels',128))).to(device)
    bindings = {'parent_checkpoint':parent_binding,'parent_bindings':parent_bindings,
        'parent_residual_sha':tensor_hash(checkpoint['residual']),
        'parent_steps':checkpoint['step'],'parent_arm':parent_contract['arm']}
    return sampler,model,saved,bindings


class EvidenceCFG(nn.Module):
    def __init__(self, model, positive, prepared, enabled=True):
        super().__init__()
        self.model,self.positive,self.prepared,self.enabled = model,positive,prepared,enabled
        self.positive_calls = self.negative_calls = 0

    def forward(self, x, t, condition):
        positive = condition is self.positive or (torch.is_tensor(condition)
            and condition.data_ptr()==self.positive.data_ptr())
        if positive:
            self.positive_calls += 1
            return self.model(x,t,condition,self.prepared,self.enabled)
        self.negative_calls += 1
        # Match the archived parent's negative CFG path, with no observation
        # residual from either the old 200-step adapter or the new adapter.
        return self.model.parent.base.adapted_prediction(x,t,condition,None)[0]


ResidualCFG = EvidenceCFG
