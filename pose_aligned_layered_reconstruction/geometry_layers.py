"""Camera-exact depth peeling of a supervised target mesh (no model inference)."""
from __future__ import annotations

import torch


@torch.no_grad()
def peel_mesh(vertices, faces, intrinsics, extrinsics, *, image_size=(518, 518),
              raster_side=37, layers=5, camera_forward_sign=-1.0, context=None,
              view_chunk_size=1):
    import nvdiffrast.torch as dr

    device = vertices.device
    context = context or dr.RasterizeCudaContext(device=device)
    if view_chunk_size > 0 and len(intrinsics) > view_chunk_size:
        # Views are independent rasterizations; chunking changes memory only,
        # never the mesh, camera, resolution, peel count or label definition.
        chunks = [peel_mesh(vertices, faces, intrinsics[i:i + view_chunk_size],
                           extrinsics[i:i + view_chunk_size], image_size=image_size,
                           raster_side=raster_side, layers=layers,
                           camera_forward_sign=camera_forward_sign, context=context,
                           view_chunk_size=0)
                  for i in range(0, len(intrinsics), view_chunk_size)]
        return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
    vertices = vertices.float()
    faces = faces.to(device=device, dtype=torch.int32).contiguous()
    homogeneous = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=1)
    camera = torch.einsum("vij,nj->vni", extrinsics.float(), homogeneous)[..., :3]
    depth = camera[..., 2] * float(camera_forward_sign)
    if not bool((depth > 0.05).all()):
        raise ValueError("target geometry crosses the camera/near plane")
    height, width = image_size
    # Pixel (u,v) maps to ((u+.5)/W,(v+.5)/H). Raster array row zero is
    # NDC y=-1, so this intentionally does not apply an extra image flip.
    x = 2 * (intrinsics[:, None, 0, 0] * camera[..., 0]
             + (intrinsics[:, None, 0, 2] + .5) * depth) / width - depth
    y = 2 * (intrinsics[:, None, 1, 1] * camera[..., 1]
             + (intrinsics[:, None, 1, 2] + .5) * depth) / height - depth
    near, far = .01, 100.
    z = (far + near) / (far - near) * depth - 2 * far * near / (far - near)
    clip = torch.stack((x, y, z, depth), dim=-1).contiguous()
    surfaces = []
    # Extra peels permit removal of coincident triangles without consuming a
    # layer slot. Labels use exact camera-forward z, not Euclidean ray length.
    with dr.DepthPeeler(context, clip, faces, (raster_side, raster_side)) as peeler:
        for _ in range(layers + 3):
            rast, _ = peeler.rasterize_next_layer()
            values, _ = dr.interpolate(depth[..., None].contiguous(), rast, faces)
            surfaces.append(torch.where(rast[..., 3] > 0, values[..., 0], torch.inf))
    stack = torch.stack(surfaces, dim=1).sort(dim=1).values
    unique = torch.isfinite(stack)
    unique[:, 1:] &= (stack[:, 1:] - stack[:, :-1]) > 1.e-5
    stack = torch.where(unique, stack, torch.inf).sort(dim=1).values[:, :layers]
    valid = torch.isfinite(stack)
    return {
        "layer_depths_gt": torch.where(valid, stack, 0).cpu(),
        "ray_stop_index_gt": valid.sum(dim=1).cpu(),
        "layer_target_gt": valid.float().cpu(),
    }
