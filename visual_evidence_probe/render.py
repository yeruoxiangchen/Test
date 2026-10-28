import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def render(mesh, k, w2c, *, side=518, context=None):
    """Fixed gray, camera-independent Lambert shading, no recentering/fitting."""
    import nvdiffrast.torch as dr
    device = torch.device('cuda:0')
    context = context or dr.RasterizeCudaContext(device=device)
    v = torch.tensor(np.asarray(mesh.vertices, np.float32), device=device)
    f = torch.tensor(np.asarray(mesh.faces, np.int32), device=device).contiguous()
    normals = torch.tensor(np.array(mesh.vertex_normals, np.float32), device=device)
    light = torch.tensor([.35, -.5, .8], device=device)
    light = light / torch.linalg.vector_norm(light)
    shade = .28 + .62 * (normals @ light).clamp_min(0)
    color = shade[:, None] * torch.tensor([.72, .78, .82], device=device)[None]
    vh = torch.cat((v, torch.ones_like(v[:, :1])), -1)
    images, masks = [], []
    for kval, w in zip(k, w2c):
        kval, w = kval.to(device), w.to(device)
        xyz = vh @ w.T
        depth = xyz[:, 2]
        x = 2 * (kval[0, 0] * xyz[:, 0] + (kval[0, 2] + .5) * depth) / 518 - depth
        y = 2 * (kval[1, 1] * xyz[:, 1] + (kval[1, 2] + .5) * depth) / 518 - depth
        near, far = .001, 100.
        z = (far + near) / (far - near) * depth - 2 * far * near / (far - near)
        clip = torch.stack((x, y, z, depth), -1)[None].contiguous()
        rast, _ = dr.rasterize(context, clip, f, resolution=(side, side))
        hit = rast[0, ..., 3] > 0
        rgb, _ = dr.interpolate(color[None].contiguous(), rast, f)
        rgb = rgb[0] * hit[..., None]
        images.append(Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)))
        masks.append(hit.cpu().numpy())
    return images, np.stack(masks)


def mask_iou(pred, gt):
    axes = tuple(range(1, pred.ndim))
    return np.logical_and(pred, gt).sum(axes) / np.maximum(np.logical_or(pred, gt).sum(axes), 1)
