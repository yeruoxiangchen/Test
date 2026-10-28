"""Memory scheduling for the unchanged formal data-preparation worker.

Run in its own process. Decoder reduction is installed from the separately
audited resource patch. The same frozen DINO instance resides on CPU between
encode_image calls; its original forward always runs on its original GPU.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


def emit(event, **fields):
    print(json.dumps({'memory_policy_event': event, **fields}, ensure_ascii=False), flush=True)


class DinoResidency:
    """Change weight residency only; retain dtype, eval state and original calls."""

    def __init__(self):
        self.encoders = []
        self.hooks = []
        self.encode_calls = 0
        self.decode_calls = 0

    @staticmethod
    def release_cache():
        # No synchronization, random draws or global allocator-limit changes.
        torch.cuda.empty_cache()

    def attach_encoder(self, pipeline, device):
        model = pipeline.models['image_cond_model']
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Memory scheduling requires the original frozen eval DINO')
        if getattr(pipeline, 'low_vram', False):
            raise ValueError('Expected the original paired encoder low_vram=False path')
        original_encode = pipeline.encode_image
        record = {'pipeline': pipeline, 'model': model, 'device': device,
                  'original_encode': original_encode}
        self.encoders.append(record)

        @wraps(original_encode)
        def encode_on_original_device(*args, **kwargs):
            # The base pipeline.device reads its model's current device, so
            # restore before the original routine builds/transfers its inputs.
            model.to(device)
            self.encode_calls += 1
            try:
                return original_encode(*args, **kwargs)
            finally:
                model.cpu()
                self.release_cache()

        pipeline.encode_image = encode_on_original_device
        model.cpu()
        self.release_cache()
        return pipeline

    def before_decode(self, module, inputs):
        self.decode_calls += 1
        for record in self.encoders:
            record['model'].cpu()
        self.release_cache()
        # Returning None leaves the decoder's original input tensors untouched.

    def attach_runtime(self, runtime):
        self.hooks.append(runtime['decoder'].register_forward_pre_hook(self.before_decode))
        return runtime

    def close(self):
        for hook in self.hooks:
            hook.remove()
        for record in self.encoders:
            record['pipeline'].encode_image = record['original_encode']
            record['model'].cpu()
        self.release_cache()

    def summary(self):
        return {'dino_instances': len(self.encoders), 'encode_calls': self.encode_calls,
                'decode_calls': self.decode_calls, 'dino_cpu_between_encode_calls': True,
                'dino_original_dtype_batch_and_forward': True,
                'decoder_inputs_and_model_weights_unchanged': True}


@contextmanager
def install_dino_residency(*, paired_module=None, slat_module=None):
    """Patch factory imports in this process; restore them on every exit path."""
    if paired_module is None:
        from ... import paired_encoder as paired_module
    if slat_module is None:
        from ... import frozen_slat as slat_module
    original_builder = paired_module.build_frozen_dino_encoder
    original_loader = slat_module.load_frozen_slat30k
    residency = DinoResidency()

    @wraps(original_builder)
    def build_encoder(*args, **kwargs):
        # The original builder exposes device as a required keyword-only arg.
        return residency.attach_encoder(original_builder(*args, **kwargs), kwargs['device'])

    @wraps(original_loader)
    def load_runtime(*args, **kwargs):
        return residency.attach_runtime(original_loader(*args, **kwargs))

    paired_module.build_frozen_dino_encoder = build_encoder
    slat_module.load_frozen_slat30k = load_runtime
    try:
        yield residency
    finally:
        paired_module.build_frozen_dino_encoder = original_builder
        slat_module.load_frozen_slat30k = original_loader
        residency.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from . import formal2k_data as original_data
    if not argv or '--help' in argv or '-h' in argv:
        sys.argv = [original_data.__name__, *argv]
        return original_data.main()
    if argv[0] != 'augment':
        raise ValueError('The memory resource wrapper only supports the frozen augment command')
    from .memory_efficient_mesh import install_memory_efficient_mesh
    patch_binding = install_memory_efficient_mesh(chunk_channels=1)
    source = Path(__file__).resolve()
    emit('installed', pid=os.getpid(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
         source={'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
         mesh_reduction=patch_binding, artificial_memory_cap=False,
         training_module_modified=False)
    sys.argv = [original_data.__name__, *argv]
    with install_dino_residency() as residency:
        try:
            original_data.main()
        except BaseException:
            emit('failed', **residency.summary())
            raise
        else:
            emit('complete', **residency.summary())


if __name__ == '__main__':
    main()
