"""Render fixed textured scenes from exact posed cameras, with depth ownership.

JSON recipes name local mesh paths; assets are never downloaded or modified.
Textures use an explicit opaque policy shared by RGB and geometric labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..paired_dataset import sha256_file, canonical_json_sha256


def load_parts(path, transform=None):
    import trimesh
    scene = trimesh.load(path, force='scene', process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(matrix)
        if transform is not None:
            mesh.apply_transform(transform)
        if len(mesh.faces):
            materials = getattr(getattr(mesh.visual, 'material', None), 'materials', None)
            face_materials = getattr(mesh.visual, 'face_materials', None)
            if materials is not None and face_materials is not None:
                for index in np.unique(face_materials):
                    submesh = mesh.submesh([np.flatnonzero(np.asarray(face_materials) == index)], append=True)
                    submesh.visual.material = materials[int(index)]
                    parts.append(submesh)
            else:
                parts.append(mesh)
    if not parts:
        raise ValueError(f'empty mesh: {path}')
    return parts


def canonicalize(parts):
    vertices = np.concatenate([m.vertices for m in parts])
    low, high = vertices.min(0), vertices.max(0)
    scale = .98 / np.max(high - low)
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -(low + high) * .5 * scale
    for mesh in parts:
        mesh.apply_transform(transform)
    return transform


def orbit_cameras(seed, views=8, radius_range=(1.7, 4.8), trajectory='wide', image_side=518):
    rng = np.random.default_rng(seed)
    azimuth = rng.uniform(-np.pi, np.pi)
    spread = rng.uniform(.15, .65) if trajectory == 'narrow' else rng.uniform(1.5, 2 * np.pi)
    elevation = rng.uniform(-.3, .65)
    c2ws = []
    for i in range(views):
        theta = azimuth + spread * (i / max(views - 1, 1) - .5)
        phi = elevation + rng.uniform(-.25, .25)
        radius = rng.uniform(*radius_range)
        center = radius * np.array([np.cos(phi) * np.cos(theta), np.sin(phi), np.cos(phi) * np.sin(theta)])
        z = -center / np.linalg.norm(center)
        # No gravity requirement: the provisional up vector only constructs an
        # orthonormal camera; independent roll is applied afterwards.
        up = np.array([0., 1., 0.])
        if abs(z @ up) > .95:
            up = np.array([1., 0., 0.])
        x = np.cross(z, up); x /= np.linalg.norm(x)
        y = np.cross(z, x)  # image y points down; proper rotation
        roll = rng.uniform(-np.pi, np.pi)
        xr, yr = np.cos(roll) * x + np.sin(roll) * y, -np.sin(roll) * x + np.cos(roll) * y
        c2w = np.eye(4); c2w[:3, :3] = np.stack((xr, yr, z), 1); c2w[:3, 3] = center
        c2ws.append(c2w)
    focal = image_side * rng.uniform(1.1, 1.6)
    k = np.array([[focal, 0., (image_side - 1) / 2], [0., focal, (image_side - 1) / 2], [0., 0., 1.]])
    return np.repeat(k[None], views, 0).astype(np.float32), np.linalg.inv(np.stack(c2ws)).astype(np.float32)


def box_occluders(seed, count=2):
    import trimesh
    rng = np.random.default_rng(seed)
    meshes = []
    for _ in range(count):
        mesh = trimesh.creation.box(extents=rng.uniform(.15, .6, 3))
        rotation = trimesh.transformations.random_rotation_matrix(rng.random(3))
        rotation[:3, 3] = rng.uniform(-.65, .65, 3)
        mesh.apply_transform(rotation)
        mesh.visual.vertex_colors = np.tile(np.r_[rng.integers(40, 230, 3), 255], (len(mesh.vertices), 1))
        meshes.append(mesh)
    return meshes


def pack_scene(target_parts, other_parts, device):
    vertices, faces, uv, colors, owners, material_ids, textures, hashes = [], [], [], [], [], [], [], []
    offset = 0
    for i, mesh in enumerate(target_parts + other_parts):
        vertices.append(np.asarray(mesh.vertices, np.float32))
        faces.append(np.asarray(mesh.faces, np.int32) + offset)
        visual = mesh.visual
        tex = None
        tex_uv = getattr(visual, 'uv', None)
        material = getattr(visual, 'material', None)
        if material is not None:
            tex = getattr(material, 'image', None)
            if tex is None:
                tex = getattr(material, 'baseColorTexture', None)
        if tex is not None and tex_uv is not None:
            array = np.asarray(tex.convert('RGB'), dtype=np.float32) / 255.
            textures.append(torch.from_numpy(array[::-1].copy()).to(device))
            uv.append(np.asarray(tex_uv, np.float32))
            colors.append(np.ones((len(mesh.vertices), 3), np.float32))
            material_ids.append(np.full(len(mesh.faces), len(textures) - 1))
            hashes.append(canonical_json_sha256({'texture_shape': list(array.shape),
                'texture_bytes_sha256': __import__('hashlib').sha256(array.tobytes()).hexdigest()}))
        else:
            rgba = np.asarray(visual.to_color().vertex_colors, dtype=np.float32) if hasattr(visual, 'to_color') else np.asarray(visual.vertex_colors, dtype=np.float32)
            colors.append(rgba[:, :3] / 255.)
            uv.append(np.zeros((len(mesh.vertices), 2), np.float32))
            material_ids.append(np.full(len(mesh.faces), -1))
        owners.append(np.full(len(mesh.faces), i < len(target_parts)))
        offset += len(mesh.vertices)
    return {'vertices': torch.from_numpy(np.concatenate(vertices)).to(device),
            'faces': torch.from_numpy(np.concatenate(faces)).to(device),
            'uv': torch.from_numpy(np.concatenate(uv)).to(device),
            'colors': torch.from_numpy(np.concatenate(colors)).to(device),
            'owners': torch.from_numpy(np.concatenate(owners)).to(device),
            'material_ids': torch.from_numpy(np.concatenate(material_ids)).to(device),
            'textures': textures, 'texture_hashes': hashes}


@torch.no_grad()
def rasterize(scene, k, w2c, *, image_size=(518, 518), raster_side=518, layers=5,
              forward_sign=1., context=None, rgb=False, background_rgb=(.82, .82, .82)):
    import nvdiffrast.torch as dr
    device = scene['vertices'].device
    context = context or dr.RasterizeCudaContext(device=device)
    verts, faces = scene['vertices'], scene['faces'].int().contiguous()
    homogeneous = torch.cat((verts, torch.ones_like(verts[:, :1])), -1)
    results = []
    for view in range(len(k)):
        camera = homogeneous @ w2c[view].T
        depth = camera[:, 2] * forward_sign
        height, width = image_size
        x = 2 * (k[view, 0, 0] * camera[:, 0] + (k[view, 0, 2] + .5) * depth) / width - depth
        y = 2 * (k[view, 1, 1] * camera[:, 1] + (k[view, 1, 2] + .5) * depth) / height - depth
        near, far = .001, 100.
        z = (far + near) / (far - near) * depth - 2 * far * near / (far - near)
        clip = torch.stack((x, y, z, depth), -1)[None].contiguous()
        stack, ownership, first_rgb, first_mask = [], [], None, None
        # L+1 distinct intersections are enough to detect overflow; extra peels
        # remove coincident triangles without consuming geometric layer slots.
        with dr.DepthPeeler(context, clip, faces, (raster_side, raster_side)) as peeler:
            for layer in range(layers + 4):
                rast, _ = peeler.rasterize_next_layer()
                hit = rast[0, ..., 3] > 0
                triangle = (rast[0, ..., 3].long() - 1).clamp_min(0)
                interpolated, _ = dr.interpolate(depth[None, :, None].contiguous(), rast, faces)
                values = interpolated[0, ..., 0]
                valid = hit & (values > 0)
                stack.append(torch.where(valid, values, torch.inf))
                ownership.append(scene['owners'][triangle] & valid)
                if layer == 0:
                    first_mask = scene['owners'][triangle] & valid
                    if rgb:
                        color, _ = dr.interpolate(scene['colors'][None].contiguous(), rast, faces)
                        color = color[0]
                        if scene['textures']:
                            uv, _ = dr.interpolate(scene['uv'][None].contiguous(), rast, faces)
                            material = scene['material_ids'][triangle]
                            for tid, texture in enumerate(scene['textures']):
                                sampled = dr.texture(texture[None].contiguous(), uv.contiguous(), filter_mode='linear', boundary_mode='wrap')[0]
                                color = torch.where((material == tid)[..., None], sampled, color)
                        background = color.new_tensor(background_rgb).expand_as(color)
                        first_rgb = torch.where(valid[..., None], color, background).clamp(0, 1)
                if not hit.any():
                    break
        values, order = torch.stack(stack).sort(0)
        own = torch.gather(torch.stack(ownership), 0, order)
        distinct = torch.isfinite(values)
        distinct[1:] &= (values[1:] - values[:-1]) > 1.e-5
        values = torch.where(distinct, values, torch.inf)
        values, order = values.sort(0)
        own = torch.gather(own, 0, order)
        # A fully empty ray may stop peeling early; pad to the requested shape.
        if len(values) < layers + 1:
            n = layers + 1 - len(values)
            values = torch.cat((values, torch.full((n, raster_side, raster_side), torch.inf, device=device)))
            own = torch.cat((own, torch.zeros((n, raster_side, raster_side), dtype=torch.bool, device=device)))
        overflow = torch.isfinite(values[layers])
        valid = torch.isfinite(values[:layers])
        results.append({'layer_depths_gt': torch.where(valid, values[:layers], 0),
                        'layer_target_gt': (own[:layers] & valid).float(),
                        'ray_stop_index_gt': valid.sum(0), 'ray_overflow_gt': overflow,
                        'visible_target_mask': first_mask, **({'rgb': first_rgb} if rgb else {})})
    return {key: torch.stack([r[key] for r in results]).cpu() for key in results[0]}


def asset_bindings(parts, path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256_file(path),
            'geometry_sha256': __import__('hashlib').sha256(b''.join(
                np.asarray(m.vertices, np.float32).tobytes() + np.asarray(m.faces, np.int32).tobytes() for m in parts)).hexdigest()}


def render_recipe(recipe, output, device):
    import trimesh
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = recipe['target_mesh']
    target = load_parts(source)
    transform = canonicalize(target)
    k, w2c = orbit_cameras(recipe.get('seed', 0), recipe.get('views', 8),
        tuple(recipe.get('radius_range', [1.7, 4.8])), recipe.get('trajectory', 'wide'))
    other = []
    for record in recipe.get('occluders', []):
        other.extend(load_parts(record['mesh'], np.asarray(record.get('transform', np.eye(4)))))
    other.extend(box_occluders(recipe.get('seed', 0) + 17, recipe.get('box_count', 2)))
    scene = pack_scene(target, other, device)
    kt, wt = torch.from_numpy(k).to(device), torch.from_numpy(w2c).to(device)
    background = recipe.get('background_rgb', [.82, .82, .82])
    rendered = rasterize(scene, kt, wt, layers=1, rgb=True, background_rgb=background)
    labels = rasterize(scene, kt, wt, raster_side=37, layers=recipe.get('layers', 5))
    target_scene = pack_scene(target, [], device)
    clean = rasterize(target_scene, kt, wt, layers=1, rgb=True, background_rgb=background)
    amodal = clean['visible_target_mask']
    rows = []
    for i, rgb in enumerate(rendered['rgb']):
        image = Image.fromarray((rgb.numpy() * 255).round().astype(np.uint8))
        image.save(output / f'raw_{i:02d}.png')
        Image.fromarray((clean['rgb'][i].numpy() * 255).round().astype(np.uint8)).save(output / f'clean_{i:02d}.png')
        visible = rendered['visible_target_mask'][i].numpy()
        Image.fromarray((visible * 255).astype(np.uint8)).save(output / f'visible_{i:02d}.png')
        Image.fromarray((amodal[i].numpy() * 255).astype(np.uint8)).save(output / f'amodal_{i:02d}.png')
        rows.append({'raw': f'raw_{i:02d}.png', 'visible': f'visible_{i:02d}.png', 'amodal': f'amodal_{i:02d}.png'})
    for key in ('visible_target_mask',):
        labels.pop(key, None)
    np.savez_compressed(output / 'layers.npz', **{key: val.numpy() for key, val in labels.items()})
    np.savez_compressed(output / 'camera.npz', intrinsics=k, extrinsics=w2c)
    mesh = trimesh.util.concatenate(target)
    mesh.export(output / 'target_canonical.obj')
    visible_fraction = rendered['visible_target_mask'].sum((1, 2)).float() / amodal.sum((1, 2)).clamp_min(1)
    meta = {'format': 'posed_scene_source.v2', 'uid': recipe['uid'], 'object_uid': recipe['object_uid'],
        'scene_uid': recipe.get('scene_uid', recipe['uid']), 'trajectory_uid': recipe['uid'] + ':trajectory',
        'data_domain': 'full_render', 'views': rows, 'camera': 'camera.npz', 'layers': 'layers.npz',
        'camera_forward_sign': 1., 'extrinsics_type': 'w2c', 'layer_camera_y_sign': 1.,
        'depth_kind': 'camera_forward_z', 'label_geometry_kind': 'source_mesh',
        'target_mesh': 'target_canonical.obj', 'object_from_asset': transform.tolist(),
        'visible_fraction': visible_fraction.tolist(), 'recipe': recipe, 'recipe_hash': canonical_json_sha256(recipe),
        'asset_binding': asset_bindings(target, source), 'texture_hashes': scene['texture_hashes'],
        'occluder_bindings': [asset_bindings(load_parts(r['mesh']), r['mesh']) for r in recipe.get('occluders', [])],
        'alpha_policy': 'opaque_rgb_and_geometry', 'renderer': 'nvdiffrast_exact_depth_peeling',
        'crop_kind': 'full_frame_no_gt_bbox'}
    (output / 'scene.json').write_text(json.dumps(meta, indent=2))
    clean_labels = rasterize(target_scene, kt, wt, raster_side=37, layers=recipe.get('layers', 5))
    clean_labels.pop('visible_target_mask')
    np.savez_compressed(output / 'clean_layers.npz', **{key: value.numpy() for key, value in clean_labels.items()})
    clean_recipe = {**recipe, 'box_count': 0, 'occluders': []}
    clean_meta = {**meta, 'uid': meta['uid'] + ':clean', 'scene_uid': meta['scene_uid'] + ':clean',
        'layers': 'clean_layers.npz', 'visible_fraction': [1.] * len(rows),
        'recipe': clean_recipe, 'recipe_hash': canonical_json_sha256(clean_recipe),
        'views': [{'raw': f'clean_{i:02d}.png', 'visible': row['amodal'], 'amodal': row['amodal']}
                  for i, row in enumerate(rows)]}
    (output / 'clean_scene.json').write_text(json.dumps(clean_meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    device = torch.device('cuda:0' if args.device == 'cuda' else args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    render_recipe(json.loads(Path(args.recipe).read_text()), args.output_dir, device)


if __name__ == '__main__':
    main()
