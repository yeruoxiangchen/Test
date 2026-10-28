"""Surface (not solid-filled) support -> frozen SS VAE -> round-trip audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..pretrained import load_trellis_model
from ..paired_dataset import sha256_file


@torch.no_grad()
def encode(mesh_path, output, device, pretrained='microsoft/TRELLIS-image-large', runtime=None):
    import trimesh
    if Path(output).exists():
        raise FileExistsError(output)
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if np.max(mesh.vertices) > .501 or np.min(mesh.vertices) < -.501:
        raise ValueError('SS targets must already be in the canonical unit cube')
    # Trimesh subdivision voxelization samples the surface; no fill() operation.
    points = mesh.voxelized(1. / 64., method='subdivide').points
    coords = np.unique(np.floor((points + .5) * 64).astype(np.int32), axis=0)
    coords = coords[((coords >= 0) & (coords < 64)).all(1)]
    if not len(coords):
        raise ValueError('empty surface voxel support')
    if runtime is None:
        encoder, enc_binding = load_trellis_model(pretrained, 'ss_enc_conv3d_16l8_fp16')
        decoder, dec_binding = load_trellis_model(pretrained, 'ss_dec_conv3d_16l8_fp16')
        encoder, decoder = encoder.to(device).eval(), decoder.to(device).eval()
        encoder.requires_grad_(False); decoder.requires_grad_(False)
    else:
        encoder, enc_binding, decoder, dec_binding = runtime
    occupancy = torch.zeros((1, 1, 64, 64, 64), device=device)
    index = torch.from_numpy(coords.astype(np.int64)).to(device)
    occupancy[0, 0, index[:, 0], index[:, 1], index[:, 2]] = 1.
    latent = encoder(occupancy.to(next(encoder.parameters()).dtype), sample_posterior=False)
    decoded = decoder(latent.to(next(decoder.parameters()).dtype))[0, 0] > 0
    gt = occupancy[0, 0] > 0
    iou = float((decoded & gt).sum() / (decoded | gt).sum().clamp_min(1))
    np.savez_compressed(output, target=latent[0].float().cpu().numpy(), official_coords=coords,
                        roundtrip_iou=np.asarray(iou))
    metadata = {'mesh': str(Path(mesh_path).resolve()), 'mesh_sha256': sha256_file(mesh_path),
        'encoder': enc_binding, 'decoder': dec_binding, 'roundtrip_iou': iou,
        'voxelization': 'trimesh_surface_subdivide_pitch_1_over_64_no_fill',
        'surface_count': len(coords), 'target_sha256': sha256_file(output)}
    Path(str(output) + '.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--pretrained', default='microsoft/TRELLIS-image-large')
    a = p.parse_args()
    print(json.dumps(encode(a.mesh, a.output, torch.device(a.device), a.pretrained)))


if __name__ == '__main__':
    main()
