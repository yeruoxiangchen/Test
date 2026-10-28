"""Equal-weight 3x3x3 midpoint quadrature inside each SS16 cell.

One change from the prior experiment: average the complete 12-channel geometry
summary over subcell queries, preserving the common kernel and fusion adapter.
No GT-dependent points, learned pooling, support renormalization, or max pooling.
"""
import torch
from ...v2.conditioning import query_geometry


def offsets(grid_transform, *, device, dtype=torch.float32, side=16):
    if side != 16:
        raise ValueError('This frozen experiment is SS16 only')
    a = torch.tensor([-1./(3*side), 0., 1./(3*side)], device=device, dtype=dtype)
    result = torch.stack(torch.meshgrid(a, a, a, indexing='ij'), -1).reshape(27, 3)
    if grid_transform == 'pixal3d_rotation':
        result = result @ result.new_tensor(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.))).T
    elif grid_transform != 'identity':
        raise ValueError('Unknown grid transform')
    return result


def geometry_features(sample, prediction, points, *, kind, chunk=256):
    if kind == 'point':
        return query_geometry(sample, prediction, points, absolute_tolerance=.02,
                              relative_tolerance=0.).features
    if kind != 'cell_mean_27' or chunk < 1:
        raise ValueError('Unknown geometry query or invalid chunk')
    delta = offsets(sample['grid_transform'], device=points.device, dtype=points.dtype)
    rows = []
    for start in range(0, len(points), chunk):
        centers = points[start:start+chunk]
        subpoints = (centers[:, None] + delta[None]).reshape(-1, 3)
        f = query_geometry(sample, prediction, subpoints, absolute_tolerance=.02,
                           relative_tolerance=0.).features.float()
        rows.append(f.reshape(len(f), len(centers), 27, 12).mean(2))
    return torch.cat(rows, 1)
