"""Full real-decoder regression, including an archived near-limit bench."""
import argparse
import hashlib
import json
import numpy as np
import torch

from .common import *
from .tiled_native import LargeInputNative, TILE_ROWS
from ...frozen_slat import (configure_determinism, bootstrap_external_paths,
    load_frozen_slat30k, load_trellis_model, DEFAULT_PRETRAINED, module_state_sha256,
    DECODER_STATE_SHA256, master_noise, sparse_noise)
from ..hssd_operator_audit_20261025.kernel_policy import FixedNative
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh


def tensor_digest(tensor):
    if tensor is None:
        return None
    h = hashlib.sha256()
    h.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    for block in tensor.split(65536):
        h.update(block.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def sparse(payload, device):
    from trellis.modules import sparse as sp
    return sp.SparseTensor(feats=payload['feats'].to(device), coords=payload['coords'].to(device))


def decode(decoder, payload, device, policy, dest):
    import trimesh
    hashes = []
    handle = decoder.upsample[1].out_layers[0].register_forward_hook(
        lambda module, args, output: hashes.append({
            'features': tensor_digest(output.feats), 'coords': tensor_digest(output.coords)}))
    try:
        with policy:
            output = decoder(sparse(payload, device))[0]
        raw = {key: tensor_digest(getattr(output, key)) for key in ['vertices', 'faces', 'vertex_attrs']}
        mesh = output.to_trimesh(transform_pose=False)
        dest.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(dest)
        mesh = trimesh.load(dest, process=False, force='mesh')
        raw['export_vertices'] = hashlib.sha256(np.asarray(mesh.vertices).tobytes()).hexdigest()
        raw['export_faces'] = hashlib.sha256(np.asarray(mesh.faces).tobytes()).hexdigest()
        return {'raw': raw, 'target_convolution': hashes, 'mesh': binding(dest),
            'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
            'tiled_calls': getattr(policy, 'tiled_calls', [])}
    finally:
        handle.remove()


@torch.no_grad()
def main(kind):
    import trimesh
    configure_determinism(); bootstrap_external_paths(); torch.set_num_threads(2)
    install_memory_efficient_mesh()
    device = torch.device('cuda:0')
    check_gate('kernel_check')
    ident = freeze_runtime()
    dest = REPAIR/'regression'/kind
    event('decoder_regression_load', kind=kind)
    if kind == 'small':
        folder = OUT/'recovery_20261025/replay/full_all/yxc30k/hssd_102344049_i076/42'
        payload_path = folder/'sampled_slat.pt'
        old = read(folder/'mesh.json')
        if old['identity'] != ident:
            raise ValueError('Archived latent replay identity changed')
        payload = torch.load(payload_path, map_location='cpu', weights_only=True)
        decoder, model_binding = load_trellis_model(DEFAULT_PRETRAINED, 'slat_dec_mesh_swin8_B_64l8m256c_fp16')
        if module_state_sha256(decoder) != DECODER_STATE_SHA256 or model_binding != old['slat_binding']['decoder']:
            raise ValueError('Decoder tensors changed')
        decoder = decoder.to(device).eval().requires_grad_(False)
        tile_rows = 65536  # Ensure multiple blocks even on this small real mesh.
    else:
        from ..hssd_input_adaptation_20261025 import evaluate as e
        from .worker import cached_sample
        obj = next(o for o in base.plan()['plan']['objects'] if o['uid'].endswith('_i080'))
        case = ('crop2072_visible', 'appearance_control', obj, 42)
        folder = e.dest_for(*case)
        old = read(folder/'mesh.json')
        runtime = load_frozen_slat30k(device, evaluation_uids=[o['uid'] for o in base.plan()['plan']['objects']])
        if runtime['binding'] != old['slat_binding']:
            raise ValueError('Large regression model identity changed')
        _, sample, inputs = e.load_input(obj, case[0], {})
        support = read(folder/'ss.json')
        with np.load(verify(support['coords'])) as data:
            coords = data['coords']
        if len(coords) != 85267:
            raise ValueError('Expected the archived near-limit 85267-point bench')
        initial = sparse_noise(coords, master_noise(case[3], obj['original_index'], device))
        with FixedNative():
            latent, calls, cache = cached_sample(case, runtime, sample, initial, inputs)
        payload_path = verify(cache['latent'])
        payload = {'coords': latent.coords.cpu(), 'feats': latent.feats.cpu()}
        decoder = runtime['decoder']
        model_binding = runtime['binding']['decoder']
        del latent, initial, runtime
        torch.cuda.empty_cache()
        tile_rows = TILE_ROWS
    event('decoder_regression_original', kind=kind, support=len(payload['coords']))
    original = decode(decoder, payload, device, FixedNative(), dest/'original.ply')
    torch.cuda.empty_cache()
    event('decoder_regression_tiled', kind=kind)
    tiled = decode(decoder, payload, device,
                   LargeInputNative(tile_rows=tile_rows, force_signature=(192, 96)), dest/'tiled.ply')
    reference = trimesh.load(verify(old['mesh']), process=False, force='mesh')
    replay = trimesh.load(verify(tiled['mesh']), process=False, force='mesh')
    raw_exact = original['raw'] == tiled['raw']
    conv_exact = original['target_convolution'] == tiled['target_convolution']
    archive_exact = np.array_equal(reference.vertices, replay.vertices) and np.array_equal(reference.faces, replay.faces)
    passed = raw_exact and conv_exact and archive_exact and len(tiled['tiled_calls']) == 1
    row = {'time': now(), 'passed': passed, 'kind': kind, 'base_identity': ident,
        'sources': [binding(__file__), binding(HERE/'tiled_native.py'), binding(HERE/'worker.py')],
        'latent': binding(payload_path), 'model_binding': model_binding,
        'archive_mesh': old['mesh'], 'raw_exact': raw_exact, 'target_conv_exact': conv_exact,
        'archive_mesh_exact': bool(archive_exact), 'original': original, 'tiled': tiled,
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
    write(REPAIR/f'decoder_{kind}.json', row)
    if not passed:
        raise ValueError('Decoder regression differs; do not continue to formal results')
    event('decoder_regression_passed', kind=kind, raw_exact=raw_exact, conv_exact=conv_exact, archive_exact=bool(archive_exact))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('kind', choices=['small', 'large'])
    main(p.parse_args().kind)
