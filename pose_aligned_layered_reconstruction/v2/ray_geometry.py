"""Object-space rays that exactly invert the released projection convention."""
from dataclasses import dataclass
import math

import torch

from ..projection import project_queries


@dataclass
class CameraBundle:
    intrinsics: torch.Tensor
    w2c: torch.Tensor
    forward_sign: float
    image_size: tuple

    @classmethod
    def from_sample(cls, sample, device=None):
        k = sample['intrinsics'].to(device=device, dtype=torch.float32).clone()
        k[:, 1, 1] *= float(sample.get('layer_camera_y_sign', 1.))
        return cls.from_tensors(k, sample['extrinsics'].to(device=device),
                               sample['extrinsics_type'], sample['camera_forward_sign'],
                               sample['image_size'])

    @classmethod
    def from_tensors(cls, k, extrinsics, extrinsics_type, forward_sign, image_size):
        if extrinsics_type not in ('w2c', 'c2w') or forward_sign not in (-1., 1.):
            raise ValueError('invalid camera convention')
        w2c = extrinsics.float() if extrinsics_type == 'w2c' else torch.linalg.inv(extrinsics.float())
        if k.shape != (len(w2c), 3, 3) or w2c.shape[1:] != (4, 4):
            raise ValueError('camera dimensions differ')
        if not torch.isfinite(k).all() or not torch.isfinite(w2c).all():
            raise ValueError('non-finite camera')
        if (k[:, (0, 1), (0, 1)].abs() < 1.e-8).any():
            raise ValueError('zero focal length')
        if k[:, 0, 1].abs().max() > 1.e-5 or k[:, 1, 0].abs().max() > 1.e-5:
            raise ValueError('skewed intrinsics are not supported by the parent projector')
        rotation = w2c[:, :3, :3]
        eye = torch.eye(3, device=rotation.device).expand_as(rotation)
        with torch.autocast(k.device.type, enabled=False):
            if not torch.allclose(rotation @ rotation.transpose(1, 2), eye, atol=2.e-3):
                raise ValueError('extrinsics rotation must be orthonormal')
            if (torch.linalg.det(rotation) < .99).any():
                raise ValueError('extrinsics must use proper rotations')
        return cls(k.float(), w2c, float(forward_sign), tuple(map(int, image_size)))

    @property
    def centers(self):
        with torch.autocast(self.w2c.device.type, enabled=False):
            return -torch.einsum('vji,vj->vi', self.w2c[:, :3, :3], self.w2c[:, :3, 3])

    @property
    def center_depth(self):
        return self.w2c[:, 2, 3] * self.forward_sign

    def select(self, index):
        return CameraBundle(self.intrinsics[index], self.w2c[index], self.forward_sign, self.image_size)

    def project(self, points, patch_side):
        with torch.autocast(self.w2c.device.type, enabled=False):
            return project_queries(points, intrinsics=self.intrinsics, extrinsics=self.w2c,
                                   extrinsics_type='w2c', camera_forward_sign=self.forward_sign,
                                   image_height=self.image_size[0], image_width=self.image_size[1],
                                   patch_grid_side=patch_side)


@dataclass
class PatchRays:
    origins: torch.Tensor        # [V,P,3]
    axial_directions: torch.Tensor  # X = origin + forward_z * direction
    unit_directions: torch.Tensor
    pixels: torch.Tensor

    def points(self, forward_depth):
        return self.origins[:, :, None] + forward_depth[..., None] * self.axial_directions[:, :, None]


def make_patch_rays(cameras, patch_hw):
    ph, pw = map(int, patch_hw)
    height, width = cameras.image_size
    k = cameras.intrinsics
    y, x = torch.meshgrid(torch.arange(ph, device=k.device), torch.arange(pw, device=k.device), indexing='ij')
    u = (x.flatten().float() + .5) * width / pw - .5
    v = (y.flatten().float() + .5) * height / ph - .5
    dx = (u[None] - k[:, None, 0, 2]) / k[:, None, 0, 0]
    dy = (v[None] - k[:, None, 1, 2]) / k[:, None, 1, 1]
    dc = torch.stack((dx, dy, torch.full_like(dx, cameras.forward_sign)), -1)
    with torch.autocast(k.device.type, enabled=False):
        directions = torch.einsum('vji,vpj->vpi', cameras.w2c[:, :3, :3], dc)
    origins = cameras.centers[:, None].expand_as(directions)
    return PatchRays(origins, directions, torch.nn.functional.normalize(directions, dim=-1),
                     torch.stack((u, v), -1))


def candidate_depths(rays, cameras, count=32, extent=.5):
    """Ray/canonical-cube intervals, with a finite fallback for non-object rays."""
    if count < 2 or extent <= 0:
        raise ValueError('candidate count/extent must be positive')
    d, o = rays.axial_directions, rays.origins
    parallel = d.abs() < 1.e-7
    safe = torch.where(parallel, torch.ones_like(d), d)
    a, b = (-extent - o) / safe, (extent - o) / safe
    lower, upper = torch.minimum(a, b), torch.maximum(a, b)
    inside = o.abs() <= extent
    lower = torch.where(parallel, torch.where(inside, -torch.inf, torch.inf), lower)
    upper = torch.where(parallel, torch.where(inside, torch.inf, -torch.inf), upper)
    near = lower.amax(-1).clamp_min(1.e-4)
    far = upper.amin(-1)
    hits = far > near
    radius = math.sqrt(3) * extent
    center = cameras.center_depth[:, None].expand_as(near)
    near = torch.where(hits, near, (center - radius).clamp_min(1.e-4))
    far = torch.where(hits, far, torch.maximum(center + radius, near + .01))
    alpha = torch.linspace(0., 1., count, device=o.device)
    return near[..., None] + (far - near)[..., None] * alpha, hits
