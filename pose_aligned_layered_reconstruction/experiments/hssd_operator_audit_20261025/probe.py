"""Inspect native block inputs/outputs and GEMM choices without changing inference."""
import argparse
import contextlib
import time
from pathlib import Path

import torch

from ..._bootstrap import bootstrap_external_paths
from ...frozen_slat import configure_determinism, module_state_sha256, DECODER_STATE_SHA256
from ...pretrained import load_trellis_model, DEFAULT_PRETRAINED
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
from ..hssd_mesh_replay_20261025.tracing import Trace, tensor_meta, compare_traces, compare_event
from ..hssd_mesh_replay_20261025.replay import save_tensor_file, sparse_from, decode, GenerationTrace, environment
from .common import OUT, PREVIOUS, DEPTH, ASSETS, freeze, binding, verify, read, write, now


def find_event(trace, name):
    return next(e for e in trace['events'] if e['name'] == name)


def prepare():
    from safetensors import safe_open
    torch.set_num_threads(2)
    freeze()
    path = OUT / 'inputs.json'
    if path.exists():
        info = read(path)
        if info['protocol'] != binding(OUT / 'protocol.json'):
            raise ValueError('Changed operator inputs')
        verify(info['packet'])
        return info
    summary = read(PREVIOUS / 'summary.json')
    sources = {}
    traces = {}
    for key in ['same_a', 'fresh']:
        traces[key] = read(verify(summary['results'][key]['trace']))
        for name in ['flow/internal/input_layer', 'flow/internal/t_embedder', 'flow/internal/input_blocks.0']:
            event = find_event(traces[key], name)
            verify(event['file'])
            sources[key + '/' + name] = event['file']
    for name in ['flow/internal/input_layer', 'flow/internal/t_embedder']:
        if find_event(traces['same_a'], name)['tensors'] != find_event(traces['fresh'], name)['tensors']:
            raise ValueError('Historical block precursor inputs differ')
    config_path = ASSETS / 'slat_flow_img_dit_L_64l8p2_fp16.json'
    weights_path = ASSETS / 'slat_flow_img_dit_L_64l8p2_fp16.safetensors'
    expected = read(DEPTH / 'mesh/uniform/slat_binding.json')['flow']
    config_binding, weights_binding = binding(config_path), binding(weights_path)
    if (config_binding['sha256'], weights_binding['sha256']) != (expected['config_sha256'], expected['weights_sha256']):
        raise ValueError('Changed pretrained Flow assets')
    config = read(config_path)['args']
    if not (config['num_io_res_blocks'] == 2 and config['io_block_channels'] == [128]
            and config['model_channels'] == 1024 and config['use_fp16'] and not config.get('share_mod', False)):
        raise ValueError('Unexpected block architecture')
    with safe_open(str(weights_path), framework='pt', device='cpu') as f:
        state = {k.removeprefix('input_blocks.0.'): f.get_tensor(k) for k in f.keys() if k.startswith('input_blocks.0.')}
    x = torch.load(sources['same_a/flow/internal/input_layer']['path'], map_location='cpu', weights_only=True)
    t = torch.load(sources['same_a/flow/internal/t_embedder']['path'], map_location='cpu', weights_only=True)
    packet = {'coords': x['/coords'], 'feats': x['/feats'].half(), 'embedding': t['tensor'].half(),
        'state': state, 'kwargs': {'channels': 128, 'emb_channels': 1024, 'out_channels': 128}}
    info = {'protocol': binding(OUT / 'protocol.json'), 'sources': sources,
        'packet': save_tensor_file(OUT / 'block_inputs.pt', packet),
        'config': config_binding, 'base_weights': weights_binding,
        'casts': 'Same core.dtype fp16 casts as FrozenSLat30K.forward; input sparse cache is empty before first block.',
        'latent': summary['results']['same_a']['latent'], 'gt': read(PREVIOUS / 'inputs.json')['gt'],
        'historical_decoder_trace': summary['results']['fixed_decode_a']['trace']}
    write(path, info)
    print('PREPARED_OPERATOR_INPUTS', flush=True)
    return info


def build_block(packet, device='cuda:0'):
    bootstrap_external_paths()
    from trellis.models.structured_latent_flow import SparseResBlock3d
    from trellis.modules.utils import convert_module_to_f16
    block = SparseResBlock3d(**packet['kwargs'])
    block.apply(convert_module_to_f16)
    block.load_state_dict(packet['state'], strict=True)
    return block.eval().requires_grad_(False).to(device)


def valid_neighbors(pairs, counts, subm):
    count = counts.detach().cpu().tolist()
    kv = len(count)
    result = {}
    for i in range(kv):
        if subm and i == kv // 2:
            continue
        n = count[kv-i-1] if subm and i > kv//2 else count[i]
        n = min(int(n), pairs.shape[2])
        if n:
            # Never hash unused torch.empty tails: only valid ordered pairs.
            result[str(i)] = pairs[:, i, :n].detach().cpu().clone()
    return result


class NativeObserver:
    def __init__(self, trace=None):
        self.trace, self.label, self.records = trace, None, []

    def __enter__(self):
        import spconv.pytorch.ops as ops
        self.ops, self.original = ops, ops.indice_conv
        def observed(features, filters, pairs, counts, num_activate_out, *args, **kwargs):
            label = self.label or f'native_{len(self.records):03d}'
            subm = kwargs.get('subm', args[1] if len(args) > 1 else False)
            if self.trace is not None:
                self.trace.add(label + '/native_inputs', {'features': features, 'filters': filters})
                self.trace.add(label + '/neighbors', valid_neighbors(pairs, counts, subm))
            result = self.original(features, filters, pairs, counts, num_activate_out, *args, **kwargs)
            self.records.append(self.algorithm(label, features, filters, result))
            if self.trace is not None:
                self.trace.add(label + '/native_output', result)
            return result
        ops.indice_conv = observed
        return self

    def __exit__(self, *args):
        self.ops.indice_conv = self.original

    def algorithm(self, label, features, filters, output):
        from spconv.core import AlgoHint
        from cumm.gemm.algospec.core import ShuffleStrideType
        from spconv.pytorch.cppcore import torch_tensor_to_tv
        from spconv.algo import GEMM_CPP
        types = [torch_tensor_to_tv(v).dtype for v in [features, filters, output]]
        n, cin, cout = len(features), features.shape[1], output.shape[1]
        arch = tuple(torch.cuda.get_device_capability())
        tuned, exists = GEMM_CPP.get_tuned_algo(*types, [n, cin], [cout, cin], [len(output), cout],
            False, True, False, arch, ShuffleStrideType.ShuffleAC.value, [1], [], [1], AlgoHint.Fowrard.value)
        if not exists:
            raise RuntimeError('Cannot observe actual native forward GEMM cache')
        d = tuned.algo_desp
        return {'label': label, 'shape': [n, cin, cout], 'dtypes': types, 'descriptor': str(d),
            'splitk': tuned.splitk, 'arch': list(tuned.arch),
            'accumulator_dtype': int(d.dacc), 'compute_dtype': int(d.dcomp),
            'static': {'trans_a': d.trans_a, 'trans_b': d.trans_b, 'trans_c': d.trans_c,
                       'shuffle_type': int(d.shuffle_type)}}


def install_block_hooks(block, trace, observer):
    handles = []
    modules = {'block': block, 'emb_layers': block.emb_layers, 'emb_linear': block.emb_layers[1],
        'norm1': block.norm1, 'conv1': block.conv1, 'norm2': block.norm2, 'conv2': block.conv2,
        'skip': block.skip_connection}
    for name, module in modules.items():
        def before(m, args, name=name):
            trace.add(name + '/input', args)
            if name in ['conv1', 'conv2']:
                observer.label = name
        def after(m, args, output, name=name):
            trace.add(name + '/output', output)
            if name in ['conv1', 'conv2']:
                observer.label = None
        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    return handles


def check_completed(path):
    r = read(path)
    if r['protocol'] != binding(OUT / 'protocol.json'):
        raise ValueError('Changed completed protocol')
    for result in r['results']:
        verify(result['trace'])
        for event in read(result['trace']['path'])['events']:
            verify(event['file'])
        if 'mesh' in result:
            verify(result['mesh'])
    return r


@torch.no_grad()
def block_worker(worker_id):
    torch.set_num_threads(2)
    freeze()
    path = OUT / f'block_{worker_id}.json'
    if path.exists():
        check_completed(path)
        return
    info = read(OUT / 'inputs.json')
    packet = torch.load(verify(info['packet']), map_location='cpu', weights_only=True)
    configure_determinism()
    block = build_block(packet)
    state_sha = module_state_sha256(block)
    versions = {n: p._version for n, p in block.named_parameters()}
    folder = OUT / 'attempts' / f'block_{worker_id}_{time.time_ns()}'
    results = []
    for repeat in range(2):
        trace = Trace(folder / f'repeat_{repeat}')
        with NativeObserver(trace) as observer:
            handles = install_block_hooks(block, trace, observer)
            try:
                initial = sparse_from(packet, 'cuda:0')
                embedding = packet['embedding'].clone().to('cuda:0')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = block(initial, embedding)
            finally:
                for h in handles:
                    h.remove()
        result = {'repeat': repeat, 'trace': trace.finish(), 'algorithms': observer.records,
            'output_sha': tensor_meta(output.feats.cpu())}
        results.append(result)
        print('BLOCK_DONE', worker_id, repeat, [a['descriptor'] for a in observer.records], flush=True)
    if any(p._version != versions[n] or p.grad is not None for n, p in block.named_parameters()):
        raise ValueError('Block parameter mutation')
    write(path, {'protocol': binding(OUT / 'protocol.json'), 'worker': worker_id, 'results': results,
        'state_sha': state_sha, 'environment': environment(), 'time': now()})


@torch.no_grad()
def decoder_worker(worker_id):
    import trimesh
    torch.set_num_threads(2)
    freeze()
    path = OUT / f'decoder_{worker_id}.json'
    if path.exists():
        check_completed(path)
        return
    info = read(OUT / 'inputs.json')
    configure_determinism()
    patch = install_memory_efficient_mesh()
    decoder, db = load_trellis_model(DEFAULT_PRETRAINED, 'slat_dec_mesh_swin8_B_64l8m256c_fp16')
    decoder = decoder.eval().requires_grad_(False).to('cuda:0')
    state_sha = module_state_sha256(decoder)
    if state_sha != DECODER_STATE_SHA256:
        raise ValueError('Wrong mesh decoder tensors')
    latent = sparse_from(torch.load(verify(info['latent']), map_location='cpu', weights_only=True), 'cuda:0')
    target = trimesh.load(verify(info['gt']), force='mesh', process=False)
    folder = OUT / 'attempts' / f'decoder_{worker_id}_{time.time_ns()}'
    folder.mkdir(parents=True)
    trace = Trace(folder / 'tensors')
    # Reuse exact prior observer hooks, without loading the large Flow checkpoint.
    runtime = {'decoder': decoder, 'sampler': type('UnusedSampler', (), {'sample_once': None})()}
    hooks = GenerationTrace(runtime, trace)
    hooks.install_decoder()
    try:
        with NativeObserver() as observer:
            result = decode(runtime, latent, trace, folder, target)
    finally:
        hooks.close()
    result.update(trace=trace.finish(), algorithms=observer.records)
    write(path, {'protocol': binding(OUT / 'protocol.json'), 'worker': worker_id, 'results': [result],
        'state_sha': state_sha, 'decoder_binding': db, 'patch': patch, 'environment': environment(), 'time': now()})
    print('DECODER_DONE', worker_id, result['surface']['chamfer_l1'], flush=True)


def causal_comparison(left, right):
    result = compare_traces(left, right)
    a = {e['name']: e for e in read(left['path'])['events']}
    b = {e['name']: e for e in read(right['path'])['events']}
    causal = []
    for name in ['emb_linear', 'emb_layers', 'norm1', 'conv1', 'norm2', 'conv2', 'skip', 'block']:
        input_equal = a[name+'/input']['tensors'] == b[name+'/input']['tensors']
        output_equal = a[name+'/output']['tensors'] == b[name+'/output']['tensors']
        if input_equal and not output_equal:
            causal.append(name)
    result['same_input_different_output_modules'] = causal
    return result


def summarize():
    torch.set_num_threads(2)
    freeze()
    blocks = [check_completed(OUT / f'block_{i}.json') for i in range(3)]
    decoders = [check_completed(OUT / f'decoder_{i}.json') for i in range(2)]
    if len({b['state_sha'] for b in blocks}) != 1 or any(b['environment'] != blocks[0]['environment'] for b in blocks):
        raise ValueError('Unmatched block weights/environment')
    if any(d['state_sha'] != DECODER_STATE_SHA256 for d in decoders):
        raise ValueError('Decoder weight mismatch')
    comparisons = {}
    for i, block in enumerate(blocks):
        comparisons[f'block_{i}_within'] = causal_comparison(block['results'][0]['trace'], block['results'][1]['trace'])
    for i in [1, 2]:
        comparisons[f'block_0_vs_{i}'] = causal_comparison(blocks[0]['results'][0]['trace'], blocks[i]['results'][0]['trace'])
    decoder_comparison = compare_traces(decoders[0]['results'][0]['trace'], decoders[1]['results'][0]['trace'])
    old_trace = read(OUT / 'inputs.json')['historical_decoder_trace']
    old_decoder_comparisons = [compare_traces(old_trace, d['results'][0]['trace'], ('decoder/', 'mesh/')) for d in decoders]
    write(OUT / 'summary.json', {'time': now(), 'protocol': binding(OUT / 'protocol.json'),
        'blocks': blocks, 'decoders': decoders, 'block_comparisons': comparisons,
        'decoder_comparison': decoder_comparison, 'historical_decoder_comparisons': old_decoder_comparisons})
    for k, v in comparisons.items():
        print(k, 'exact', v['all_exact'], 'causal', v['same_input_different_output_modules'], 'first', v['first_difference'], flush=True)
    print('decoder exact', decoder_comparison['all_exact'], 'first', decoder_comparison['first_difference'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'block', 'decoder', 'summary'])
    p.add_argument('--id', type=int, choices=range(3), default=0)
    args = p.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'block': block_worker(args.id)
    elif args.action == 'decoder': decoder_worker(args.id)
    else: summarize()
