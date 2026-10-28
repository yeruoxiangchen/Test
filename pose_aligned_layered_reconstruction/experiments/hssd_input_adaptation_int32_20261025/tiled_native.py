"""Keep native GEMM math while bounding the input matrix's byte address range.

This is inference-only. It partitions the *input row addresses*, not the spatial
grid or channels. Every original indice pair is retained. For each output the
center mm and ordered non-center kernel contributions are unchanged.
"""
import inspect
import torch

from ..hssd_operator_audit_20261025.kernel_policy import FixedNative, PREFERRED_HALF

INT32_MAX = (1 << 31) - 1
TILE_ROWS = 524288


class LargeInputNative(FixedNative):
    def __init__(self, *, tile_rows=TILE_ROWS, force_signature=None):
        super().__init__()
        if tile_rows <= 0:
            raise ValueError('tile_rows must be positive')
        self.tile_rows = tile_rows
        # Test-only: exercise exactly the same path on a representable tensor.
        self.force_signature = force_signature
        self.tiled_calls = []

    def __enter__(self):
        super().__enter__()
        fixed = self.ops.indice_conv
        signature = inspect.signature(self.original)

        def dispatch(features, filters, pairs, counts, num_activate_out, *args, **kwargs):
            channels = (features.shape[1], filters.shape[0])
            oversized = features.numel() * features.element_size() >= INT32_MAX
            force = self.force_signature is not None and channels == self.force_signature
            if not (oversized or force):
                return fixed(features, filters, pairs, counts, num_activate_out, *args, **kwargs)
            bound = signature.bind(features, filters, pairs, counts, num_activate_out, *args, **kwargs)
            bound.apply_defaults()
            return self._tiled(**bound.arguments)

        self.ops.indice_conv = dispatch
        return self

    def _tiled(self, features, filters, indice_pairs, indice_pair_num,
               num_activate_out, inverse=False, subm=False, algo=None,
               timer=None, bias=None, act_alpha=0., act_beta=0., act_type=None):
        from cumm import tensorview as tv
        from spconv.core import ConvAlgo, AlgoHint
        from spconv.pytorch.cppcore import torch_tensor_to_tv as as_tv

        if (torch.is_grad_enabled() or not features.is_cuda or algo != ConvAlgo.Native
                or not subm or inverse or features.dtype != torch.float16
                or filters.dtype != features.dtype or not filters.is_contiguous()):
            raise ValueError('LargeInputNative requires CUDA fp16 native subm inference, forward only')
        if act_type is None:
            act_type = tv.gemm.Activation.None_
        features = features.contiguous()
        n, cin = features.shape
        cout = filters.shape[0]
        weights = filters.reshape(cout, -1, cin)
        kv = weights.shape[1]
        if kv != 27 or (cin, cout) != (192, 96) or num_activate_out != n:
            raise ValueError('Tiled path is restricted to the audited 192->96 3x3x3 subm decoder convolution')
        if n * cout * features.element_size() >= INT32_MAX:
            raise ValueError('Output also exceeds the native address limit; this repair only tiles input rows')
        if self.tile_rows * cin * features.element_size() >= INT32_MAX:
            raise ValueError('Input row tile exceeds native address limit')
        if (indice_pairs.dtype != torch.int32 or indice_pairs.shape[:2] != (2, kv)
                or len(indice_pair_num) != kv):
            raise ValueError('Unexpected native pair layout')

        center = kv // 2
        # Identical to TorchSpconvMatmul.indice_conv_init_gemm: do NOT tile
        # cuBLAS's center mm, or change its shape, transpose or accumulation.
        output = torch.mm(features, weights[:, center].T)
        c_tv = as_tv(output)
        counts = indice_pair_num.cpu().tolist()
        counts = [min(int(v), indice_pairs.shape[2]) for v in counts]
        arch = torch.cuda.get_device_capability(features.device)
        stream = self.ops.get_current_stream()
        matrix = weights[:, 0, :]
        key = (str(features.dtype), cin, cout, matrix.stride(0))
        if key not in self.cache:
            self.cache[key] = self._make_tuner(features, matrix)
        tuner, descriptor = self.cache[key]
        if descriptor != PREFERRED_HALF:
            raise ValueError('The tiled path must use the frozen fp16 descriptor')
        self.algo.GEMM_CPP = self.ops.GEMM_CPP = tuner
        tuned = None
        record = {'input_rows': n, 'input_channels': cin, 'output_channels': cout,
                  'input_bytes': features.numel() * features.element_size(),
                  'tile_rows': self.tile_rows, 'descriptor': descriptor,
                  'kernel_volume': kv, 'pairs_expected': 0, 'pairs_processed': 0,
                  'gemm_calls': 0, 'offsets': [], 'center_mm_unchanged': True}
        print('TILED_NATIVE_BEGIN', record, flush=True)
        for offset in range(kv):
            if offset == center:
                continue
            nhot = counts[kv - offset - 1] if offset > center else counts[offset]
            if nhot <= 0:
                continue
            source = indice_pairs[0, offset, :nhot]
            dest = indice_pairs[1, offset, :nhot]
            record['pairs_expected'] += nhot
            processed = 0
            b_tv = as_tv(weights[:, offset, :])
            for start in range(0, n, self.tile_rows):
                end = min(start + self.tile_rows, n)
                chosen = (source >= start) & (source < end)
                a_ids = (source[chosen] - start).contiguous()
                count = a_ids.numel()
                if not count:
                    continue
                c_ids = dest[chosen].contiguous()
                a_tv = as_tv(features[start:end])
                ai_tv, ci_tv = as_tv(a_ids), as_tv(c_ids)
                if tuned is None:
                    # The one-descriptor tuner profiles into a clone of C.
                    # This does not modify the already computed center term.
                    tuned, _ = tuner.tune_and_cache(a_tv, b_tv, c_tv,
                        False, True, False, arch, 1, ai_tv, tv.Tensor(), ci_tv,
                        AlgoHint.Fowrard.value, 1., 0., stream, 5, False)
                    if str(tuned.algo_desp) != descriptor or tuned.splitk != 1:
                        raise ValueError('Tiled kernel differs from frozen native policy')
                tuner.run_with_tuned_result(tuned, a_tv, b_tv, c_tv,
                    False, True, False, arch, stream, 1, ai_tv, tv.Tensor(), ci_tv,
                    AlgoHint.Fowrard.value, 1., 1.)
                processed += count
                record['gemm_calls'] += 1
            if processed != nhot:
                raise ValueError('Lost/out-of-range input pairs during tiling')
            record['pairs_processed'] += processed
            record['offsets'].append({'offset': offset, 'pairs': processed})

        has_act = act_type != tv.gemm.Activation.None_
        if bias is not None and has_act:
            self.ops.InferenceOps.bias_add_act_inplace(c_tv, as_tv(bias), act_type, act_alpha, act_beta, stream)
        elif bias is not None:
            self.ops.InferenceOps.bias_add_inplace(c_tv, as_tv(bias), stream)
        elif has_act:
            self.ops.InferenceOps.activation_inplace(c_tv, act_type, act_alpha, act_beta, stream)
        self.tiled_calls.append(record)
        old = self.records.setdefault(str(key), {'signature': list(key), 'descriptor': descriptor,
            'calls': 0, 'verified_tuned_descriptor': True})
        old['calls'] += 1
        old['verified_tuned_descriptor'] = True
        print('TILED_NATIVE_END', {k: v for k, v in record.items() if k != 'offsets'}, flush=True)
        return output
