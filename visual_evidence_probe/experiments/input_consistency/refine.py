"""Differentiable proper similarity fitting to two observed silhouettes."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import PARAMS, folder, observations, binding, verify, write, read, emit
from .policy import select_checkpoint


def transform(raw):
    x, y, z = raw[:3].unbind()
    zero = raw.new_zeros(())
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    rotvec_scale = math.radians(PARAMS['rotation_max_degrees']) / (1 + torch.linalg.vector_norm(raw[:3]))
    rotation = torch.matrix_exp(skew * rotvec_scale)
    translation = PARAMS['translation_max_per_axis'] * torch.tanh(raw[3:6])
    signed = torch.tanh(raw[6])
    log_scale = torch.where(signed >= 0, math.log(PARAMS['scale_max']) * signed,
                           -math.log(PARAMS['scale_min']) * signed)
    scale = log_scale.exp()
    return rotation, translation, scale


def project_clip(vertices, k, w):
    """Exactly the frozen CV camera and pixel-center convention (full 518 image)."""
    vh = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    xyz = torch.matmul(vh[None], w.transpose(1, 2))
    depth = xyz[..., 2]
    x = 2 * (k[:, None, 0, 0] * xyz[..., 0] + (k[:, None, 0, 2] + .5) * depth) / 518 - depth
    y = 2 * (k[:, None, 1, 1] * xyz[..., 1] + (k[:, None, 1, 2] + .5) * depth) / 518 - depth
    near, far = .001, 100.
    z = (far + near) / (far - near) * depth - 2 * far * near / (far - near)
    return torch.stack((x, y, z, depth), dim=-1).contiguous()


def silhouettes(vertices, faces, k, w, context, side, soft=True):
    import nvdiffrast.torch as dr
    clip = project_clip(vertices, k, w)
    rast, _ = dr.rasterize(context, clip, faces, resolution=(side, side))
    if not soft:
        return rast[..., 3] > 0
    attr = vertices.new_ones((len(k), len(vertices), 1))
    coverage, _ = dr.interpolate(attr, rast, faces)
    return dr.antialias(coverage, rast, clip, faces)[..., 0].clamp(0, 1)


def fit(index, key, context, device):
    import trimesh
    from visual_evidence_probe.render import render, mask_iou
    path = folder(index) / 'refined' / f'{key}.json'
    if path.exists():
        record = read(path)
        verify(record['mesh'])
        return record
    rec, k, w, observed = observations(index)
    mesh = trimesh.load(verify(rec['donors'][key]), force='mesh', process=False)
    k, w = k[:2].to(device), w[:2].to(device)
    if not torch.equal(k[:, 0, 1], torch.zeros_like(k[:, 0, 1])):
        raise ValueError('frozen projection requires zero skew')
    v = torch.tensor(np.asarray(mesh.vertices, np.float32), device=device)
    f = torch.tensor(np.asarray(mesh.faces, np.int32), device=device).contiguous()
    target = F.interpolate(torch.tensor(observed, dtype=torch.float32, device=device)[:, None],
                           size=(PARAMS['side'], PARAMS['side']), mode='area')[:, 0]
    raw = torch.nn.Parameter(v.new_zeros(7))
    optimizer = torch.optim.Adam([raw], lr=PARAMS['lr'])
    snapshots, curve = [], []
    start = time.monotonic()

    def vertices():
        r, t, s = transform(raw)
        return s * (v @ r.T) + t

    def snapshot(step):
        with torch.no_grad():
            hard = silhouettes(vertices(), f, k, w, context, 518, soft=False).cpu().numpy()
            ious = mask_iou(hard, observed).tolist()
            r, t, s = transform(raw)
            matrix = np.eye(4)
            matrix[:3, :3] = (s * r).cpu().numpy()
            matrix[:3, 3] = t.cpu().numpy()
            snapshots.append(dict(step=step, ious=ious, transform=matrix.tolist(),
                                  rotation_degrees=float(torch.linalg.vector_norm(raw[:3]) /
                                    (1 + torch.linalg.vector_norm(raw[:3])) * PARAMS['rotation_max_degrees']),
                                  scale=float(s)))
            return hard

    initial_hard = snapshot(0)
    _, legacy = render(mesh, k, w, side=518, context=context)
    mismatch = float(np.mean(initial_hard != legacy))
    if mismatch > 1.e-5:
        raise ValueError(f'projection differs from frozen renderer: {mismatch}')
    initial_gradient_norm = None
    for step in range(1, PARAMS['steps'] + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = silhouettes(vertices(), f, k, w, context, PARAMS['side'])
        intersection = (pred * target).sum((1, 2))
        union = (pred + target - pred * target).sum((1, 2)).clamp_min(1)
        outside = (pred * (1 - target)).mean((1, 2))
        l1 = (pred - target).abs().mean((1, 2))
        loss = (1 - intersection / union + .15 * outside + .08 * l1).mean()
        loss.backward()
        norm = float(torch.linalg.vector_norm(raw.grad))
        if initial_gradient_norm is None:
            initial_gradient_norm = norm
        if not torch.isfinite(loss) or not torch.isfinite(raw.grad).all():
            raise ValueError('nonfinite similarity loss/gradient')
        optimizer.step()
        curve.append(dict(step=step, loss=float(loss.detach()), gradient_norm=norm))
        if step in PARAMS['checkpoints']:
            snapshot(step)
    chosen = select_checkpoint(snapshots, PARAMS['max_view_regression'])
    matrix = np.asarray(chosen['transform'])
    if np.linalg.det(matrix[:3, :3]) <= 0 or initial_gradient_norm <= 0:
        raise ValueError('invalid proper transform or no silhouette gradient')
    result = mesh.copy()
    result.apply_transform(matrix)
    if not np.array_equal(mesh.faces, result.faces):
        raise ValueError('global similarity changed topology')
    path.parent.mkdir(parents=True, exist_ok=True)
    result.export(path.with_suffix('.obj'))
    record = dict(index=index, uid=rec['uid'], donor=key, input_mesh=rec['donors'][key],
        mesh=binding(path.with_suffix('.obj')), uses_gt=False,
        chosen=chosen, initial=snapshots[0], snapshots=snapshots, curve=curve,
        renderer_mismatch_fraction=mismatch, initial_gradient_norm=initial_gradient_norm,
        elapsed_seconds=time.monotonic()-start, topology_unchanged=True)
    write(path, record)
    emit('refined_donor', index=index, donor=key, checkpoint=chosen['step'],
         before=snapshots[0]['ious'], after=chosen['ious'], seconds=record['elapsed_seconds'])
    return record
