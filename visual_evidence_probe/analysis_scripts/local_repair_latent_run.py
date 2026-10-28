"""Scope the historical inference-only GEMM policy strictly to inference."""
import gc
import os
from pathlib import Path

import torch

from . import local_repair_latent as original
from . import local_repair_latent_deterministic as deterministic

base_register = deterministic.old_register


def register():
    base_register()
    receipt = dict(format='local_slat_gradient_runtime_amendment.v1',
        source=dict(path=str(Path(__file__).resolve()), sha256=original.config.digest(__file__)),
        trigger='Backward GEMM tuner had only a forward descriptor because FixedNative is inference-only.',
        change='Scope FixedNative to no-grad latent reconstruction and mesh endpoint decoding only. Restore full original native GEMM tuner for differentiable SDF proxy.',
        tensor_precision_unchanged=True, original_failures_retained=True,
        torch_deterministic_algorithms=True,
        limitation='Native spconv custom CUDA backward is not certified bitwise deterministic across processes.')
    path = original.OUT / 'gradient_runtime_amendment.json'
    if path.exists() and original.config.read(path) != receipt:
        raise ValueError('gradient runtime amendment changed')
    original.config.write(path, receipt)


def run(indices, seeds, methods, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    original.config.setup()
    from visual_evidence_probe.runtime import setup, load_slat
    from pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025.kernel_policy import FixedNative
    import nvdiffrast.torch as dr
    device = setup()
    if not original.config.read(original.config.STORE / 'io_relocation.json')['cache_ready']:
        raise ValueError('data model caches not ready')
    original.register()
    context = dr.RasterizeCudaContext(device=device)
    with FixedNative() as policy:
        runtime = load_slat(device)
        original.config.write(original.OUT / 'model_binding.json', runtime['binding'])
        latents = {}
        for index in indices:
            for seed in seeds:
                latents[index, seed] = original.recreate(runtime, index, seed).to('cpu')
        original.config.write(original.OUT / 'reconstruction_kernel_policy.json', policy.manifest())
    decoder = runtime['decoder']
    del runtime; gc.collect(); torch.cuda.empty_cache()
    for block in decoder.blocks:
        block.use_checkpoint = True
    forward = decoder.forward
    def endpoint_forward(*args, **kwargs):
        if torch.is_grad_enabled():
            raise ValueError('Actual FlexiCubes extraction is an inference-only endpoint')
        with FixedNative():
            return forward(*args, **kwargs)
    decoder.forward = endpoint_forward
    for index in indices:
        for seed in seeds:
            z = latents[index, seed].to(device)
            for method in methods:
                original.optimize(decoder, z, index, seed, method, context)
                gc.collect(); torch.cuda.empty_cache()
            del z


def main():
    original.run = run
    deterministic.old_register = register
    deterministic.main()


if __name__ == '__main__': main()
