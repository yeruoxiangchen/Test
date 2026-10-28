"""Actual frozen-latent backward diagnostics, with all artifacts under /data."""
import argparse
import gc
import time

import torch

from . import config


def stats(value):
    if value is None:
        return dict(missing=True)
    detached = value.detach()
    finite = torch.isfinite(detached)
    clean = detached.float()[finite]
    return dict(missing=False, shape=list(detached.shape), dtype=str(detached.dtype),
        count=detached.numel(), finite=int(finite.sum()),
        nan=int(torch.isnan(detached).sum()), inf=int(torch.isinf(detached).sum()),
        nonzero=int((detached != 0).sum()),
        max_abs=float(clean.abs().max()) if clean.numel() else None,
        norm=float(clean.norm()) if clean.numel() else None)


def disable_checkpoint(decoder):
    for module in decoder.modules():
        if hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = False
    for block in decoder.upsample:
        if getattr(block, '_editor_mesh_checkpoint', False):
            block.forward = block.forward.__kwdefaults__['_original']
            block._editor_mesh_checkpoint = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--scales', default='1,1024')
    parser.add_argument('--checkpoint', choices=('on', 'off'), default='on')
    parser.add_argument('--anomaly', action='store_true')
    args = parser.parse_args()
    config.setup(args.gpu)
    from visual_evidence_probe.runtime import setup as runtime_setup
    from pose_aligned_layered_reconstruction.experiments.round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
    from .data import load_decoder, load_latent
    from .mesh_backend import decode_tensor_mesh
    device = runtime_setup()
    install_memory_efficient_mesh()
    decoder, binding = load_decoder(device)
    if args.checkpoint == 'off':
        disable_checkpoint(decoder)
    z = load_latent(4, 42, device)
    rows = []
    stamp = config.now().replace(':', '-')
    path = config.OUT / f'debug_backward_{stamp}.json'
    for scale in [float(value) for value in args.scales.split(',')]:
        hooks = []
        gradients = []

        def hook_for(name):
            def forward_hook(module, inputs, output):
                tensor = output if torch.is_tensor(output) else getattr(output, 'feats', None)
                if tensor is not None and tensor.requires_grad:
                    def gradient_hook(gradient):
                        row = dict(module=name, **stats(gradient))
                        gradients.append(row)
                        print(dict(stage='gradient_hook', scale=scale, **row), flush=True)
                    tensor.register_hook(gradient_hook)
            return forward_hook

        for name, module in decoder.named_modules():
            if (name in ('input_layer', 'out_layer')
                    or name.startswith('blocks.') and name.count('.') == 1
                    or name.startswith('upsample.') and name.count('.') == 1):
                hooks.append(module.register_forward_hook(hook_for(name)))
        features = z.feats.detach().clone().requires_grad_(True)
        print(dict(stage='start', scale=scale, input=stats(features), checkpoint=args.checkpoint), flush=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        row = dict(scale=scale, checkpoint=args.checkpoint, anomaly=args.anomaly)
        try:
            with torch.autograd.set_detect_anomaly(args.anomaly):
                vertices, faces = decode_tensor_mesh(decoder, z.replace(features))
                loss = vertices.square().mean()
                print(dict(stage='loss', scale=scale, loss=float(loss), vertices=stats(vertices)), flush=True)
                (loss * scale).backward()
            row.update(loss=float(loss), gradient=stats(features.grad), vertex_count=len(vertices),
                       face_count=len(faces))
            if features.grad is not None:
                row['unscaled_gradient'] = stats(features.grad / scale)
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
            print(dict(stage='error', **row), flush=True)
        finally:
            for handle in hooks:
                handle.remove()
        row.update(elapsed_seconds=time.monotonic() - start,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), hooks=gradients)
        rows.append(row)
        config.write(path, dict(time=config.now(), decoder=binding, results=rows,
                               deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        print(dict(stage='result', output=str(path), **{k: v for k, v in row.items() if k != 'hooks'}), flush=True)
        if 'vertices' in locals():
            del vertices, faces, loss
        del features
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
