"""Inference-only, process-local deterministic selection of native spconv GEMMs.

Keep the native convolution and tensor precision. For each channel/dtype/stride
signature, give the original C++ tuner exactly one compatible descriptor. Its
timing measurements therefore cannot select different kernels across processes.
Never patch the installed package or use this context for backward passes.
"""
import torch

PREFERRED_HALF = 'Turing_f16f16f16f16f16tnt_m64n64k64m32n32k32A1T1688_200_S1'


class FixedNative:
    def __init__(self, forced_descriptor=None):
        self.forced_descriptor = forced_descriptor
        self.cache, self.records = {}, {}

    def __enter__(self):
        import spconv.algo as algo
        import spconv.pytorch.ops as ops
        from spconv import constants
        if not ops.SPCONV_CPP_GEMM or not ops.ALL_WEIGHT_IS_KRSC or constants.SPCONV_ALLOW_TF32:
            raise ValueError('This policy requires the frozen native KRSC/CPP/no-TF32 path')
        self.algo, self.ops = algo, ops
        self.original, self.old_algo, self.old_ops = ops.indice_conv, algo.GEMM_CPP, ops.GEMM_CPP

        def fixed(features, filters, pairs, counts, num_activate_out, *args, **kwargs):
            from spconv.core import ConvAlgo
            selected_algo = kwargs.get('algo', args[2] if len(args) > 2 else ConvAlgo.Native)
            if selected_algo != ConvAlgo.Native or not features.is_cuda or torch.is_grad_enabled():
                raise ValueError('FixedNative supports CUDA native inference under no_grad only')
            if features.dtype != filters.dtype or not filters.is_contiguous():
                raise ValueError('Unexpected native tensor precision/layout')
            # C++ uses filters[:, kernel_position, :] as the GEMM B matrix.
            # Its leading stride includes the full kernel volume; retain it.
            matrix = filters.reshape(filters.shape[0], -1, filters.shape[-1])[:, 0, :]
            key = (str(features.dtype), features.shape[1], filters.shape[0], matrix.stride(0))
            if key not in self.cache:
                self.cache[key] = self._make_tuner(features, matrix)
            tuner, descriptor = self.cache[key]
            self.algo.GEMM_CPP = self.ops.GEMM_CPP = tuner
            result = self.original(features, filters, pairs, counts, num_activate_out, *args, **kwargs)
            record = self.records.setdefault(str(key), {'signature': list(key),
                'descriptor': descriptor, 'calls': 0, 'verified_tuned_descriptor': False})
            record['calls'] += 1
            if not record['verified_tuned_descriptor']:
                from spconv.core import AlgoHint
                from spconv.pytorch.cppcore import torch_tensor_to_tv
                dt = torch_tensor_to_tv(features).dtype
                tuned, exists = tuner.get_tuned_algo(dt, dt, dt,
                    [len(features), features.shape[1]], [result.shape[1], features.shape[1]],
                    [len(result), result.shape[1]], False, True, False,
                    torch.cuda.get_device_capability(), 1, [1], [], [1], AlgoHint.Fowrard.value)
                if exists:
                    if str(tuned.algo_desp) != descriptor or tuned.splitk != 1:
                        raise ValueError('Actual native algorithm differs from fixed policy')
                    record['verified_tuned_descriptor'] = True
            return result

        ops.indice_conv = fixed
        return self

    def _make_tuner(self, features, matrix):
        from spconv.pytorch.cppcore import torch_tensor_to_tv
        from cumm.gemm.algospec.core import ShuffleStrideType
        # get_all_available inspects dtype, architecture and leading strides;
        # it does not read values or filter on row count. Tiny probes suffice.
        a = torch.empty((1, features.shape[1]), device=features.device, dtype=features.dtype)
        c = torch.empty((1, matrix.shape[0]), device=features.device, dtype=features.dtype)
        dtype = torch_tensor_to_tv(a).dtype
        candidates = self.old_algo.get_all_available(
            torch_tensor_to_tv(a), torch_tensor_to_tv(matrix), torch_tensor_to_tv(c),
            False, True, False, torch.cuda.get_device_capability(),
            ShuffleStrideType.ShuffleAC.value, False)
        if not candidates:
            raise ValueError('No compatible native GEMM descriptor')
        by_name = {str(d): d for d in candidates}
        if self.forced_descriptor is not None:
            if self.forced_descriptor not in by_name:
                raise ValueError('Forced descriptor is incompatible with this convolution')
            chosen = by_name[self.forced_descriptor]
        elif features.dtype == torch.float16 and PREFERRED_HALF in by_name:
            chosen = by_name[PREFERRED_HALF]
        else:
            # Preserve tensor-dtype accumulation when available, then use a
            # stable descriptor order. Never choose according to elapsed time.
            chosen = min(candidates, key=lambda d: (int(d.dacc) != dtype,
                int(d.dcomp) != dtype, str(d)))
        return self.algo.GemmTunerSimple([chosen]), str(chosen)

    def __exit__(self, *args):
        self.ops.indice_conv = self.original
        self.algo.GEMM_CPP, self.ops.GEMM_CPP = self.old_algo, self.old_ops

    def manifest(self):
        return [self.records[k].copy() for k in sorted(self.records)]
