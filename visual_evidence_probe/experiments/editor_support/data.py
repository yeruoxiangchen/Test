import numpy as np
import torch

from .config import BASE, PRIOR, INPUT, PARAMS, read


def load_decoder(device):
    from pose_aligned_layered_reconstruction.pretrained import load_trellis_model, DEFAULT_PRETRAINED
    from .mesh_backend import enable_checkpoint
    decoder, binding = load_trellis_model(DEFAULT_PRETRAINED, 'slat_dec_mesh_swin8_B_64l8m256c_fp16')
    expected = read(PRIOR / 'model_binding.json')['decoder']
    if binding != expected: raise ValueError('mesh decoder identity changed')
    decoder.eval().requires_grad_(False).to(device)
    checkpoints = enable_checkpoint(decoder)
    return decoder, dict(**binding, checkpoints=checkpoints, model_weights_frozen=True)


def load_latent(index, seed, device):
    from trellis.modules.sparse import SparseTensor
    p = PRIOR / 'latents' / f'{index:03d}_{seed}.pt'
    cache = torch.load(p, map_location=device, weights_only=False)
    if not read(p.with_suffix('.json'))['parity']: raise ValueError('original latent parity missing')
    return SparseTensor(feats=cache['features'], coords=cache['coords'])


def transform(index, seed, device):
    fit = read(BASE / 'objects' / f'{index:03d}' / f'fitted/{seed}.json')
    return torch.tensor(fit['chosen']['transform'], device=device, dtype=torch.float32)


def points_transform(points, matrix):
    points = points.to(device=matrix.device, dtype=matrix.dtype)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def observation(index, device):
    from PIL import Image
    p = BASE / 'input_crop_resolution_control/objects' / f'{index:03d}'
    r = read(p / 'observations.json')
    def physical(b):
        return str(b['path']).replace('/home/zjr/Tracker/visual_evidence_probe/outputs/',
            '/data/zjr/visual_evidence_probe/outputs/')
    images = [Image.open(physical(b)).convert('RGB') for b in r['images']]
    masks = np.stack([np.asarray(Image.open(physical(b)).convert('L')) > 127 for b in r['masks']])
    return r, torch.tensor(r['intrinsics'], device=device, dtype=torch.float32), \
        torch.tensor(r['extrinsics'], device=device, dtype=torch.float32), images, masks


def constraints(index, seed):
    with np.load(INPUT / 'objects' / f'{index:03d}' / str(seed) / 'automatic_constraints.npz') as a:
        c = {key: a[key] for key in a.files}
    # Fixed confidence ranking, then stable subsample. No GT or treatment-specific selection.
    ids = np.argsort(-c['confidence'], kind='stable')[:PARAMS['max_constraints']]
    return {key: (value[ids] if key in ('target', 'displacement', 'confidence', 'face_ids', 'bary', 'pixel') else value)
        for key, value in c.items()}


def feature_std(device):
    return torch.tensor(read(PRIOR / 'model_binding.json')['normalization']['std'], device=device)[None]


def active_tokens(z, native_points):
    from scipy.spatial import cKDTree
    centers = (z.coords[:, 1:].float().cpu().numpy() + .5) / 64 - .5
    distances = cKDTree(np.asarray(native_points)).query(centers, workers=2)[0]
    return torch.tensor(distances <= PARAMS['active_radius_voxels'] / 64, device=z.device)
