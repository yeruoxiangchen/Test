"""Zero-initialized neighbor-state attention without modifying the frozen base.

Every object keeps its own SS/Shape state. Only noisy/predicted neighbor states
and image observations enter this module, never clean neighbor target latents.
"""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from ..geometry import dense_coords


def state_arrays(state, stage):
    if stage == 'ss':
        if state.shape[0] != 1:
            raise ValueError('One object per backbone forward')
        coords = dense_coords(state.shape[-1], with_batch=False).to(state.device)
        return state.flatten(2).transpose(1, 2)[0], coords, int(state.shape[-1])
    return state.feats, state.coords[:, 1:], 32


def points_to_world(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def summarize_state(state, sample, stage, views=None, bins=4):
    """Pool each instance separately, retaining occupied cell positions."""
    feats, coords, resolution = state_arrays(state, stage)
    local = (coords.float() + .5) / resolution - .5
    cells = (coords.long() * bins // resolution).clamp(0, bins - 1)
    ids = cells[:, 0] * bins**2 + cells[:, 1] * bins + cells[:, 2]
    count = torch.bincount(ids, minlength=bins**3).float()
    feature_sum = feats.detach().float().new_zeros((bins**3, feats.shape[-1]))
    point_sum = local.new_zeros((bins**3, 3))
    feature_sum.index_add_(0, ids, feats.detach().float())
    point_sum.index_add_(0, ids, local)
    active = count > 0
    pooled = feature_sum[active] / count[active, None]
    positions = point_sum[active] / count[active, None]
    image = sample['dino_features']
    if views is not None:
        image = image.index_select(0, views.to(image.device))
    return {'uid': sample['uid'], 'features': pooled,
            'world_points': points_to_world(positions, sample['object_to_world'].float()),
            'appearance': image.float().mean((0, 1)).detach()}


class NeighborAttention(nn.Module):
    def __init__(self, state_channels, condition_channels, model_channels, blocks,
                 width=128, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.query = nn.Linear(state_channels + condition_channels + 3, width)
        self.state = nn.Linear(state_channels + 3, width)
        self.appearance = nn.Linear(condition_channels, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.outputs = nn.ModuleDict({str(i): nn.Linear(width, model_channels) for i in blocks})
        for layer in self.outputs.values():
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def exact_zero(self):
        return all(torch.count_nonzero(p.detach()).item() == 0
                   for layer in self.outputs.values() for p in layer.parameters())

    def attend(self, condition, context):
        own, points, transform, neighbors = context
        inverse = torch.linalg.inv(transform.float())
        keys, positions = [], []
        for neighbor in neighbors:
            xyz = points_to_world(neighbor['world_points'].float(), inverse)
            positions.append(xyz)
            keys.append(self.state(torch.cat([neighbor['features'], xyz], -1)) +
                        self.appearance(neighbor['appearance'])[None])
        token = torch.cat(keys, 0)
        xyz = torch.cat(positions, 0)
        condition = condition.reshape(-1, condition.shape[-1])
        q = self.query(torch.cat([own.float(), condition.float(), points], -1))
        k, v = self.key(token), self.value(token)
        def heads(x):
            return x.reshape(1, len(x), self.heads, self.width // self.heads).transpose(1, 2)
        # Target-local distances: a common scene transform cancels out.
        bias = -torch.cdist(points.float(), xyz.float()).square()[None, None] / 2.
        attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v),
                                                  attn_mask=bias.to(q.dtype), dropout_p=0.)
        return attended.transpose(1, 2).reshape(len(q), self.width)


class RelationalBlockCondition(nn.Module):
    def __init__(self, original, *, stage, condition_channels, model_channels):
        super().__init__()
        # Preserve names of all original base parameters/checkpoints.
        self.projections = original.projections
        selected = [i for i in (5, 11, 17, 23, 29) if i < len(self.projections)]
        if not selected:
            selected = [len(self.projections) - 1]  # tiny CPU test backbones
        self.relation = NeighborAttention(8 if stage == 'ss' else 32,
                                         condition_channels, model_channels, selected)
        self.context = None
        self.cached = None

    def forward(self, index, condition):
        residual = self.projections[index](condition)
        if self.context is None or str(index) not in self.relation.outputs:
            return residual
        if self.cached is None:
            self.cached = self.relation.attend(condition, self.context)
        extra = self.relation.outputs[str(index)](self.cached).reshape_as(residual)
        return residual + extra

    def exact_zero(self):
        return (all(torch.count_nonzero(p.detach()).item() == 0
                    for layer in self.projections for p in layer.parameters())
                and self.relation.exact_zero())


def attach_relations(model):
    model.block_condition = RelationalBlockCondition(
        model.block_condition, stage=model.stage, condition_channels=model.condition_channels,
        model_channels=int(model.flow_core.model_channels)).to(next(model.parameters()).device)
    return model


@contextmanager
def bind_neighbors(model, state, sample, neighbors):
    block = model.block_condition
    if not isinstance(block, RelationalBlockCondition):
        yield
        return
    if block.context is not None:
        raise RuntimeError('Nested relation binding')
    if sample is not None and neighbors:
        if any(n['uid'] == sample['uid'] for n in neighbors):
            raise ValueError('Target cannot be its own neighbor')
        own, coords, resolution = state_arrays(state, model.stage)
        points = (coords.float() + .5) / resolution - .5
        block.context = (own.detach(), points, sample['object_to_world'], neighbors)
    try:
        yield
    finally:
        block.context = None
        block.cached = None
