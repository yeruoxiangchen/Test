"""Recorded deterministic sparse trilinear replacement for CUDA grid_sample backward."""
from pathlib import Path

import torch

from . import local_repair_latent as original


def sdf_at(values, positions, points, res=256):
    side = res + 1
    # Match align_corners=True/border, but gather only eight sparse corners per query.
    continuous = ((points + .5) * res).clamp(0, res)
    base = torch.floor(continuous).long().clamp(0, res - 1)
    fraction = continuous - base
    corners = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        device=points.device, dtype=torch.long)
    indices = base[:, None] + corners
    ids = (indices[..., 0] * side + indices[..., 1]) * side + indices[..., 2]
    source_ids = (positions[:, 0].long() * side + positions[:, 1].long()) * side + positions[:, 2].long()
    rows = torch.searchsorted(source_ids.contiguous(), ids.contiguous())
    safe_rows = rows.clamp_max(len(values) - 1)
    present = (rows < len(values)) & (source_ids[safe_rows] == ids)
    corner_values = torch.where(present, values[safe_rows], torch.ones_like(fraction[:, :1]))
    weights = torch.where(corners[None].bool(), fraction[:, None], 1 - fraction[:, None]).prod(-1)
    return (corner_values * weights).sum(-1)


def audit():
    torch.manual_seed(20261026)
    pos = torch.stack(torch.meshgrid(*[torch.arange(5)] * 3, indexing='ij'), -1).reshape(-1, 3)
    pos = pos[torch.rand(len(pos)) > .2]
    values = torch.randn(len(pos), requires_grad=True)
    points = torch.rand(30, 3) * 1.4 - .7
    old = original.sdf_at(values, pos, points, 4)
    old_grad = torch.autograd.grad(old.sum(), values)[0]
    new = sdf_at(values, pos, points, 4)
    new_grad = torch.autograd.grad(new.sum(), values)[0]
    error, gradient_error = float((old - new).abs().max()), float((old_grad - new_grad).abs().max())
    if error > 2.e-6 or gradient_error > 2.e-6:
        raise ValueError('trilinear parity failed')
    return dict(cpu_forward_max_abs=error, cpu_gradient_max_abs=gradient_error, passed=True)


old_register = original.register


def register():
    old_register()
    receipt = dict(format='local_slat_deterministic_interpolation_amendment.v1',
        source=dict(path=str(Path(__file__).resolve()), sha256=original.config.digest(__file__)),
        trigger='First backward failed: grid_sampler_3d_backward_cuda has no deterministic implementation.',
        change='Eight-corner sparse trilinear gather; same zero-SDF objective, align_corners/border convention, no weights or targets changed.',
        audit=audit(), original_failure_retained=True, deterministic_algorithms_still_enabled=True)
    path = original.OUT / 'deterministic_interpolation_amendment.json'
    if path.exists() and original.config.read(path) != receipt:
        raise ValueError('interpolation amendment changed')
    original.config.write(path, receipt)


def main():
    # Run the CPU equivalence audit before replacing the old callable.
    receipt = audit()
    old_sdf = original.sdf_at
    def register_with_audit():
        original.sdf_at = old_sdf
        try: register()
        finally: original.sdf_at = sdf_at
    original.register = register_with_audit
    original.sdf_at = sdf_at
    original.main()


if __name__ == '__main__': main()
