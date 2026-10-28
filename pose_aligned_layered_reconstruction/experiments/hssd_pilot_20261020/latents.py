"""Official-compatible surface/feature encoding and native-frame round trips."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from ...pretrained import load_trellis_model, DEFAULT_PRETRAINED
from ...paired_encoder import build_frozen_dino_encoder
from ...mesh_metrics import surface_metrics, structure_metrics
from .common import plan, object_dir, binding, verify, write_json, event, identity, sha


def save_arrays(path, **arrays):
    if Path(path).exists():
        with np.load(path) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError('Immutable target changed: ' + str(path))
    else:
        np.savez_compressed(path, **arrays)


def coordinate_audit(row):
    import trimesh
    root = object_dir(row)
    mesh = trimesh.load(verify(row['mesh']), force='mesh', process=False)
    actual = np.load(root / 'blender_canonical_vertices.npy')
    expected = np.asarray(mesh.vertices)
    forward = cKDTree(expected).query(actual)[0].max()
    backward = cKDTree(actual).query(expected)[0].max()
    error = float(max(forward, backward))
    passed = error < 3.e-5
    result = {'passed': passed, 'max_symmetric_vertex_distance': error,
              'bound': 3.e-5, 'mesh': binding(root / 'target_canonical.ply'),
              'blender_vertices': binding(root / 'blender_canonical_vertices.npy'),
              'independent_readers': ['trimesh_GLTF', 'Blender_GLTF']}
    write_json(root / 'coordinate_audit.json', result)
    if not passed:
        raise ValueError(f'Independent Blender/trimesh geometry disagrees: {error}')
    return mesh


def surface_support(root):
    # Do not import the ABI-incompatible Open3D .18 / NumPy 2 combination in
    # the actively used training environment. Same algorithm, isolated CPU.
    output = root / 'support.npz'
    worker = Path(__file__).with_name('voxel_worker.py')
    receipt = Path(str(output) + '.json')
    if not receipt.exists():
        subprocess.run(['/home/zjr/anaconda3/envs/foundpose/bin/python', '-u', str(worker),
            '--mesh', str(root / 'target_canonical.ply'), '--output', str(output)], check=True)
    meta = json.loads(receipt.read_text())
    if (meta['mesh_sha256'] != sha(root / 'target_canonical.ply') or meta['worker_sha256'] != sha(worker)
            or meta['output_sha256'] != sha(output)):
        raise ValueError('Voxelizer receipt identity changed')
    with np.load(output) as data:
        return data['coords'].copy()


@torch.no_grad()
def extract_features(root, coords, device):
    transforms = json.loads((root / 'target_renders/transforms.json').read_text())
    if len(transforms['frames']) != 150:
        raise ValueError('Dense targets require all 150 views')
    pipeline = build_frozen_dino_encoder(device=device)
    model = pipeline.models['image_cond_model']
    positions = torch.from_numpy((coords.astype(np.float32) + .5) / 64 - .5).to(device)
    homogeneous = torch.cat((positions, torch.ones_like(positions[:, :1])), 1)
    accum = torch.zeros((len(coords), 1024), device=device)
    coverage = torch.zeros(len(coords), device=device, dtype=torch.int32)
    view_coverage = []
    mean = positions.new_tensor([.485, .456, .406])[None, :, None, None]
    std = positions.new_tensor([.229, .224, .225])[None, :, None, None]
    render_bindings = []
    for i, frame in enumerate(transforms['frames']):
        path = root / 'target_renders' / frame['file_path']
        rgba = np.asarray(Image.open(path).convert('RGBA').resize((518, 518), Image.Resampling.LANCZOS), np.float32) / 255.
        image = torch.from_numpy(rgba[:, :, :3] * rgba[:, :, 3:]).permute(2, 0, 1)[None].to(device)
        # Label producer intentionally does NOT use the layer-normalized
        # condition tokens returned by pipeline.encode_image's default.
        tokens = model((image - mean) / std, is_training=True)['x_prenorm']
        patches = tokens[:, model.num_register_tokens + 1:].transpose(1, 2).reshape(1, 1024, 37, 37)
        c2w = positions.new_tensor(frame['transform_matrix'])
        c2w[:3, 1:3] *= -1
        camera = homogeneous @ torch.linalg.inv(c2w).T
        if not bool((camera[:, 2] > 0).all()):
            raise ValueError('Behind-camera target support in dense orbit')
        focal = .5 / np.tan(frame['camera_angle_x'] * .5)
        uv = 2 * focal * camera[:, :2] / camera[:, 2:3]
        in_frame = (uv.abs() <= 1).all(1)
        coverage += in_frame.int()
        view_coverage.append(float(in_frame.float().mean()))
        # Surface voxel centers may extend slightly beyond the rendered mesh.
        # Official grid_sample uses zero padding; preserve that exact behavior.
        lifted = F.grid_sample(patches, uv[None, None], mode='bilinear', align_corners=False)[0, :, 0].T
        accum.add_(lifted)
        render_bindings.append(binding(path))
        if (i + 1) % 10 == 0:
            event(stage='features', uid=root.name, view=i + 1, total=150)
    # Official features are float16 on disk, reloaded as float32 by encoder.
    if int(coverage.min()) < 75:
        raise ValueError('A target surface voxel is outside over half the target views')
    write_json(root / 'feature_projection_audit.json', {'minimum_inframe_views_per_voxel': int(coverage.min()),
        'view_inframe_fractions': view_coverage, 'padding': 'official grid_sample zeros, no valid-view renormalization',
        'all_support_in_front_of_all_cameras': True})
    features = (accum / 150).half().float().cpu().numpy()
    del pipeline, model, accum, patches, tokens, lifted
    torch.cuda.empty_cache()
    return features, render_bindings


@torch.no_grad()
def encode_one(row):
    import trimesh
    root = object_dir(row)
    mesh = coordinate_audit(row)
    complete = root / 'latent_audit.json'
    if complete.exists():
        audit = json.loads(complete.read_text())
        for name in ['ss_target', 'slat_target', 'raw_mesh', 'decoded_mesh']:
            verify(audit[name])
        receipt = json.loads((root / 'feature_receipt.json').read_text())
        for b in [receipt['features'], receipt['support'], receipt['transforms'], *receipt['renders']]:
            verify(b)
        event(stage='latent_reused', uid=row['uid'], audited_mesh=sha(root / 'decoded_target.ply'))
        return
    device = torch.device('cuda:0')
    event(stage='voxelize', uid=row['uid'])
    coords = surface_support(root)
    event(stage='ss_encode', uid=row['uid'], voxels=len(coords))
    enc, eb = load_trellis_model('microsoft/TRELLIS-image-large', 'ss_enc_conv3d_16l8_fp16')
    dec, db = load_trellis_model(DEFAULT_PRETRAINED, 'ss_dec_conv3d_16l8_fp16')
    enc, dec = enc.to(device).eval(), dec.to(device).eval()
    occupancy = torch.zeros((1, 1, 64, 64, 64), device=device)
    xyz = torch.from_numpy(coords.astype(np.int64)).to(device)
    occupancy[0, 0, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = 1
    z = enc(occupancy, sample_posterior=False)
    decoded = dec(z)[0, 0] > 0
    gt = occupancy[0, 0] > 0
    iou = float((decoded & gt).sum() / (decoded | gt).sum().clamp_min(1))
    save_arrays(root / 'ss.npz', target=z[0].float().cpu().numpy(), official_coords=coords,
                roundtrip_iou=np.asarray(iou))
    ss_audit = {'iou': iou, 'passed': iou >= .90, 'surface_count': len(coords), 'encoder': eb, 'decoder': db,
                'voxelization': 'official Open3D triangle surface within fixed 64-cube; no fill'}
    write_json(root / 'ss_audit.json', ss_audit)
    del enc, dec, z, occupancy, decoded
    torch.cuda.empty_cache()
    if iou < .90:
        raise ValueError('SS roundtrip fails; do not train: ' + str(iou))
    feature_file = root / 'surface_features.npz'
    if not feature_file.exists():
        features, renders = extract_features(root, coords, device)
        np.savez_compressed(feature_file, indices=coords, patchtokens=features.astype(np.float16))
        write_json(root / 'feature_receipt.json', {'support': binding(root / 'support.npz'),
            'transforms': binding(root / 'target_renders/transforms.json'), 'renders': renders,
            'features': binding(feature_file), 'dino': 'dinov2_vitl14_reg x_prenorm no feature layernorm'})
    else:
        receipt = json.loads((root / 'feature_receipt.json').read_text())
        for entry in [receipt['features'], receipt['support'], receipt['transforms'], *receipt['renders']]:
            verify(entry)
        with np.load(feature_file) as data:
            if not np.array_equal(data['indices'], coords):
                raise ValueError('Cached support differs')
            features = data['patchtokens'].astype(np.float32)
    enc, eb = load_trellis_model('/data/zjr/models/microsoft_TRELLIS-image-large', 'slat_enc_swin8_B_64l8_fp16')
    enc = enc.to(device).eval()
    from trellis.modules import sparse as sp
    x = sp.SparseTensor(feats=torch.from_numpy(features).to(device),
        coords=torch.from_numpy(np.c_[np.zeros(len(coords), np.int32), coords]).to(device))
    z = enc(x, sample_posterior=False)
    if not torch.isfinite(z.feats).all():
        raise FloatingPointError('Non-finite SLat')
    output_coords = z.coords[:, 1:].cpu().numpy().astype(np.int32)
    if not np.array_equal(np.unique(output_coords, axis=0), coords):
        raise ValueError('SLat encoder changed support set')
    save_arrays(root / 'slat.npz', coords=output_coords, feats=z.feats.float().cpu().numpy())
    del enc, x, features
    torch.cuda.empty_cache()
    from ...frozen_slat import configure_determinism
    from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
    configure_determinism()
    memory_binding = install_memory_efficient_mesh()
    dec, db = load_trellis_model(DEFAULT_PRETRAINED, 'slat_dec_mesh_swin8_B_64l8m256c_fp16')
    dec = dec.to(device).eval()
    result = dec(z)[0]
    decoded_mesh = trimesh.Trimesh(vertices=result.vertices.cpu().numpy(), faces=result.faces.cpu().numpy(), process=False)
    decoded_mesh.export(root / 'decoded_target.ply')
    metrics = surface_metrics(decoded_mesh, mesh, count=20000)
    audit = {'uid': row['uid'], 'finite_and_nonempty': True, 'ss': ss_audit,
             'slat_encoder': eb, 'mesh_decoder': db, 'surface': metrics, 'decoder_runtime': memory_binding,
             'structure': structure_metrics(decoded_mesh),
             'coordinate': json.loads((root / 'coordinate_audit.json').read_text()),
             'ss_target': binding(root / 'ss.npz'), 'slat_target': binding(root / 'slat.npz'),
             'raw_mesh': row['mesh'], 'decoded_mesh': binding(root / 'decoded_target.ply'),
             'quality_gate': 'pending matched reference + multi-view visual review; not a learning PASS'}
    write_json(root / 'latent_audit.json', audit)
    event(stage='latent_complete', uid=row['uid'], ss_iou=iou, surface=metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--coordinates-only', action='store_true')
    a = p.parse_args()
    row = plan()['plan']['objects'][a.index]
    if a.coordinates_only:
        coordinate_audit(row)
    else:
        torch.set_num_threads(2)
        encode_one(row)


if __name__ == '__main__':
    main()
