"""Historical FixedNative forwards with the full tuner restored for backward.

The existing policy owns descriptor selection and validation.  Its restricted
tuner is installed only during each native convolution forward.  A finally
block restores both complete GEMM tuners before control returns to autograd.
Keep this context open across backward so checkpoint recomputation uses the
same fixed forward descriptors.  No installed spconv source is changed.
"""
from __future__ import annotations

import inspect
from threading import RLock

from pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025.kernel_policy import FixedNative


_LOCK = RLock()
_ACTIVE = []


def is_active():
    """Whether this process is inside a hybrid forward/backward context."""
    return bool(_ACTIVE)


class FixedForwardNative:
    """Use FixedNative's exact forward choice and full native backward tuner.

    Nested instances share the outer descriptor cache.  This process-global
    patch is intended for a single decoding thread, as is FixedNative itself.
    """
    def __init__(self, forced_descriptor=None):
        self.forced_descriptor = forced_descriptor
        self.calls = self.restorations = self.failures = 0
        self._entered = False

    def __enter__(self):
        _LOCK.acquire()
        try:
            if self._entered:
                raise RuntimeError('cannot reenter the same FixedForwardNative instance')
            self._entered = True
            if _ACTIVE:
                self._owner = _ACTIVE[0]
                if self.forced_descriptor != self._owner.forced_descriptor:
                    raise ValueError('nested forward policies require the same descriptor rule')
            else:
                import spconv.pytorch.ops as ops
                if (inspect.isfunction(ops.indice_conv)
                        and 'FixedNative.__enter__.' in ops.indice_conv.__qualname__):
                    raise RuntimeError('leave the inference-only FixedNative before entering FixedForwardNative')
                self._owner = self
                self._policy = FixedNative(self.forced_descriptor)
                self._policy.__enter__()
                self._fixed_forward = self._policy.ops.indice_conv
                self._full_algo, self._full_ops = self._policy.old_algo, self._policy.old_ops

                def hybrid_forward(*args, **kwargs):
                    self.calls += 1
                    try:
                        # spconv's autograd Function.forward executes under
                        # no_grad, so the existing policy's validation remains
                        # applicable even when its input needs gradients.
                        return self._fixed_forward(*args, **kwargs)
                    except BaseException:
                        self.failures += 1
                        raise
                    finally:
                        self._policy.algo.GEMM_CPP = self._full_algo
                        self._policy.ops.GEMM_CPP = self._full_ops
                        self.restorations += 1

                hybrid_forward._editor_forward_fixed_native = True
                self._policy.ops.indice_conv = hybrid_forward
            _ACTIVE.append(self)
            return self
        except BaseException:
            self._entered = False
            _LOCK.release()
            raise

    def __exit__(self, *exception):
        try:
            if not self._entered or not _ACTIVE or _ACTIVE[-1] is not self:
                raise RuntimeError('forward policy contexts exited out of order')
            if self._owner is self:
                self._policy.__exit__(*exception)
        finally:
            if _ACTIVE and _ACTIVE[-1] is self:
                _ACTIVE.pop()
            self._entered = False
            _LOCK.release()

    def manifest(self):
        owner = self._owner
        return dict(kind='fixed_native_forward_full_native_backward_v1',
            forward_descriptors=owner._policy.manifest(),
            forward_calls=owner.calls, full_tuner_finally_restorations=owner.restorations,
            failed_forward_calls=owner.failures, checkpoint_recomputation_uses_fixed_forward=True,
            original_backward_descriptors_retained=True)


def main():
    """GPU parity/backward check for the actual cached frozen30k latent."""
    import argparse
    import gc
    import time
    import torch
    from . import config
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--scale', type=float, default=1024.)
    args = parser.parse_args()
    config.setup(args.gpu)
    from visual_evidence_probe.runtime import setup as runtime_setup
    from pose_aligned_layered_reconstruction.experiments.round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
    from .data import load_decoder, load_latent
    from .mesh_backend import decode_tensor_mesh
    from .debug_backward import stats
    device = runtime_setup()
    install_memory_efficient_mesh()
    decoder, binding = load_decoder(device)
    z = load_latent(4, 42, device)
    with torch.no_grad(), FixedNative() as historical:
        original = decoder(z)[0]
        old_vertices, old_faces = original.vertices.clone(), original.faces.clone()
        reference_descriptors = historical.manifest()
    del original
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    features = z.feats.detach().clone().requires_grad_(True)
    result = dict(time=config.now(), decoder=binding, scale=args.scale,
                  historical_forward_descriptors=reference_descriptors)
    try:
        with FixedForwardNative() as policy:
            vertices, faces = decode_tensor_mesh(decoder, z.replace(features))
            parity = dict(vertices_bitwise_equal=torch.equal(vertices, old_vertices),
                          faces_bitwise_equal=torch.equal(faces, old_faces),
                          historical_vertex_count=len(old_vertices), vertex_count=len(vertices),
                          historical_face_count=len(old_faces), face_count=len(faces))
            if vertices.shape == old_vertices.shape:
                parity['max_vertex_error'] = float((vertices - old_vertices).abs().max())
            result['parity'] = parity
            print(dict(stage='hybrid_forward_parity', **parity), flush=True)
            if not parity['vertices_bitwise_equal'] or not parity['faces_bitwise_equal']:
                raise ValueError('hybrid native forward differs from historical FixedNative')
            loss = vertices.square().mean()
            (loss * args.scale).backward()
            result.update(loss=float(loss), gradient=stats(features.grad),
                          unscaled_gradient=stats(features.grad / args.scale) if features.grad is not None else None,
                          hybrid_policy=policy.manifest())
            if (features.grad is None or not torch.isfinite(features.grad).all()
                    or not torch.count_nonzero(features.grad)):
                raise ValueError('hybrid backward missing, nonfinite or all zero')
            import spconv.pytorch.ops as ops
            import spconv.algo as algo
            result['full_tuners_restored_before_context_exit'] = (
                algo.GEMM_CPP is policy._full_algo and ops.GEMM_CPP is policy._full_ops)
            if not result['full_tuners_restored_before_context_exit']:
                raise RuntimeError('full tuner not restored before backward completion')
        result['passed'] = True
    except BaseException as error:
        result.update(passed=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      context_inactive_after_exit=not is_active(),
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        output = config.OUT / f'forward_policy_preflight_{config.now().replace(":", "-")}.json'
        config.write(output, result)
        print(dict(stage='hybrid_preflight', output=str(output), **result), flush=True)


if __name__ == '__main__':
    main()
