import argparse
import gc
import time

import torch

from .config import setup, OUT, write, now, emit, register


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpu', default='0'); a = p.parse_args()
    setup(a.gpu)
    from visual_evidence_probe.runtime import setup as runtime_setup
    from pose_aligned_layered_reconstruction.experiments.round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
    from pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025.kernel_policy import FixedNative
    from .data import load_decoder, load_latent
    from .mesh_backend import decode_tensor_mesh
    device = runtime_setup(); register('preflight'); install_memory_efficient_mesh()
    decoder, binding = load_decoder(device)
    z = load_latent(4, 42, device)
    with torch.no_grad(), FixedNative():
        old = decoder(z)[0]
        original_vertices, original_faces = old.vertices.clone(), old.faces.clone()
        del old
        vertices, faces = decode_tensor_mesh(decoder, z)
        parity = dict(vertices_bitwise_equal=torch.equal(vertices, original_vertices),
            faces_bitwise_equal=torch.equal(faces, original_faces),
            max_vertex_error=float((vertices - original_vertices).abs().max()))
        if not parity['vertices_bitwise_equal'] or not parity['faces_bitwise_equal']:
            raise ValueError(f'complete mesh CUDA parity failed: {parity}')
    del original_vertices, original_faces, vertices, faces
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    features = z.feats.detach().clone().requires_grad_(True)
    start = time.monotonic()
    vertices, faces = decode_tensor_mesh(decoder, z.replace(features))
    loss = vertices.square().mean()
    loss.backward()
    if features.grad is None or not torch.isfinite(features.grad).all() or features.grad.norm() == 0:
        raise ValueError('full mesh latent backward missing/nonfinite/zero')
    result = dict(time=now(), passed=True, parity=parity, decoder=binding,
        gradient_norm=float(features.grad.norm()), vertex_count=len(vertices), face_count=len(faces),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic() - start,
        dtype_unchanged=True, full_deform_and_weights=True)
    write(OUT / 'preflight.json', result); emit('preflight_passed', **{k:result[k] for k in ('gradient_norm', 'peak_allocated_bytes', 'elapsed_seconds')})


if __name__ == '__main__': main()
