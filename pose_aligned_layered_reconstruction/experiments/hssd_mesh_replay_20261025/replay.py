"""Run the original frozen generation path with detached trace hooks."""
import argparse
import importlib.metadata
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from ...frozen_slat import (INPUT_KEYS, canonical_coords, load_frozen_slat30k,
    master_noise, sparse_noise, sample_mesh_latent, module_state_sha256)
from ...mesh_metrics import surface_metrics
from ...v2.dataset import SceneDataset, select_views
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
from .common import OUT, OLD, INDEX, SEED, binding, verify, read, write, locked, freeze, now
from .tracing import Trace, tensor_meta, compare_traces


def save_tensor_file(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)
    return binding(path)


def prepare():
    torch.set_num_threads(2)
    protocol = freeze()
    done = OUT / 'inputs.json'
    if done.exists():
        info = read(done)
        if info['protocol'] != binding(OUT / 'protocol.json'):
            raise ValueError('Changed prepared input protocol')
        verify(info['payload'])
        return info
    records = {a: read(verify(p)) for a, p in protocol['ss'].items()}
    coords = []
    for a, row in records.items():
        if (row['index'], row['seed'], row['object_uid'], row['arm'], row['success']) != (
                INDEX, SEED, protocol['object_uid'], a, True):
            raise ValueError('Changed SS identity')
        with np.load(verify(row['coords'])) as data:
            coords.append(data['coords'].copy())
    if not np.array_equal(*coords):
        raise ValueError('The two SS supports are not identical')
    sample = select_views(SceneDataset(verify(protocol['manifest']), require_labels=False)[INDEX], torch.arange(8))
    if sample['object_uid'] != protocol['object_uid']:
        raise ValueError('Wrong selected object')
    observations = {k: sample[k] for k in INPUT_KEYS}
    for k, v in observations.items():
        if torch.is_tensor(v):
            observations[k] = v.detach().cpu().clone()
    payload = {'sample': observations, 'coords': torch.from_numpy(canonical_coords(coords[0])),
        'uid': sample['uid'], 'object_uid': sample['object_uid']}
    info = {'protocol': binding(OUT / 'protocol.json'), 'source_sample': binding(SceneDataset(protocol['manifest']['path'], require_labels=False).rows[INDEX]['sample']),
        'payload': save_tensor_file(OUT / 'inputs.pt', payload), 'gt': sample['target_mesh_binding'],
        'ss_exact_between_arms': True, 'views': 8, 'active_coords': len(coords[0]),
        'model_input_keys': list(observations)}
    verify(info['gt'])
    locked(done, info)
    print('PREPARED ' + str(info['active_coords']) + ' identical SS coordinates', flush=True)
    return info


def sparse_from(payload, device):
    from trellis.modules.sparse import SparseTensor
    return SparseTensor(feats=payload['feats'].clone().to(device), coords=payload['coords'].clone().to(device))


def sparse_payload(value):
    return {'coords': value.coords.detach().cpu().clone(), 'feats': value.feats.detach().cpu().clone()}


def rng_meta():
    return {'cpu': tensor_meta(torch.get_rng_state()), 'cuda': tensor_meta(torch.cuda.get_rng_state())}


def cache_meta(value):
    return {'spatial_cache_keys': sorted(str(k) for k in value._spatial_cache),
        'spconv_indice_keys': sorted(str(k) for k in getattr(value.data, 'indice_dict', {})),
        'scale': list(value._scale)}


def environment():
    packages = {}
    for name in ['torch', 'spconv-cu120', 'spconv-cu124', 'spconv-cu126', 'spconv', 'flash-attn', 'trimesh', 'numpy']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'python': platform.python_version(), 'packages': packages, 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'],
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'variables': {k: os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG', 'SPCONV_ALGO',
            'ATTN_BACKEND', 'SPARSE_ATTN_BACKEND', 'PYTORCH_CUDA_ALLOC_CONF', 'TORCH_CUDA_ARCH_LIST']}}


class GenerationTrace:
    def __init__(self, runtime, trace):
        self.runtime, self.trace = runtime, trace
        self.handles = []
        self.call, self.step = -1, 0
        self.original_once = runtime['sampler'].sample_once

    def install_flow(self):
        model = self.runtime['model']
        core = model.flow_core

        def before(module, args):
            self.call += 1
            self.trace.add(f'flow/call_{self.call:03d}/input', {'x': args[0], 'time': args[1]})
            if self.call == 0:
                self.trace.add('flow/first_condition', args[2])

        def after(module, args, output):
            self.trace.add(f'flow/call_{self.call:03d}/output', output)

        self.handles.extend([model.register_forward_pre_hook(before), model.register_forward_hook(after)])
        modules = {'input_layer': core.input_layer, 't_embedder': core.t_embedder,
            'pos_embedder': core.pos_embedder, 'aggregator': model.aggregator, 'out_layer': core.out_layer}
        if core.share_mod:
            modules['adaLN_modulation'] = core.adaLN_modulation
        for name in ['input_blocks', 'blocks', 'out_blocks']:
            modules.update({f'{name}.{i}': m for i, m in enumerate(getattr(core, name))})
        for name, module in modules.items():
            def hook(m, args, output, name=name):
                if self.call == 0:
                    self.trace.add('flow/internal/' + name, output)
            self.handles.append(module.register_forward_hook(hook))

        def sample_once(model, x_t, t, t_prev, *args, **kwargs):
            i = self.step
            self.trace.add(f'euler/{i:03d}/input', x_t)
            result = self.original_once(model, x_t, t, t_prev, *args, **kwargs)
            self.trace.add(f'euler/{i:03d}/output', result.pred_x_prev)
            self.step += 1
            if self.step % 5 == 0:
                print(f'REPLAY step={self.step}/25', flush=True)
            return result
        self.runtime['sampler'].sample_once = sample_once

    def install_decoder(self):
        decoder = self.runtime['decoder']
        modules = {'input_layer': decoder.input_layer, 'pos_embedder': decoder.pos_embedder, 'out_layer': decoder.out_layer}
        for name in ['blocks', 'upsample']:
            modules.update({f'{name}.{i}': m for i, m in enumerate(getattr(decoder, name))})
        for name, module in modules.items():
            def hook(m, args, output, name=name):
                self.trace.add('decoder/' + name, output)
            self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.runtime['sampler'].sample_once = self.original_once


@torch.no_grad()
def decode(runtime, latent, trace, folder, gt):
    import trimesh
    trace.add('decoder/input', latent)
    caches = cache_meta(latent)
    raw = runtime['decoder'](latent)[0]
    trace.add('mesh/raw', {'vertices': raw.vertices, 'faces': raw.faces})
    mesh = raw.to_trimesh(transform_pose=False)
    trace.add('mesh/converted', {'vertices': torch.from_numpy(np.asarray(mesh.vertices).copy()),
        'faces': torch.from_numpy(np.asarray(mesh.faces).copy())})
    path = folder / 'mesh.ply'
    mesh.export(path)
    saved = trimesh.load(path, force='mesh', process=False)
    trace.add('mesh/saved', {'vertices': torch.from_numpy(np.asarray(saved.vertices).copy()),
        'faces': torch.from_numpy(np.asarray(saved.faces).copy())})
    result = {'mesh': binding(path), 'surface': surface_metrics(saved, gt),
        'vertex_count': len(saved.vertices), 'face_count': len(saved.faces),
        'bbox_min': np.asarray(saved.vertices).min(0).tolist(), 'bbox_max': np.asarray(saved.vertices).max(0).tolist(),
        'fitted_transform': False, 'decoder_input_cache': caches}
    return result


@torch.no_grad()
def one_run(runtime, name, parent, payload, initial, gt, fixed_latent=None):
    start = time.monotonic()
    folder = parent / name
    folder.mkdir()
    trace = Trace(folder / 'tensors')
    hooks = GenerationTrace(runtime, trace)
    rng_before = rng_meta()
    try:
        hooks.install_decoder()
        if fixed_latent is None:
            hooks.install_flow()
            noise = sparse_from(initial, 'cuda:0')
            trace.add('initial_noise', noise)
            trace.add('observations', payload['sample'])
            latent, calls = sample_mesh_latent(runtime, payload['sample'], noise)
            if hooks.step != 25 or calls['positive_calls'] != 25:
                raise ValueError('Incomplete sampler trajectory')
            trace.add('latent/denormalized', latent)
            latent_file = save_tensor_file(folder / 'latent.pt', sparse_payload(latent))
        else:
            latent = sparse_from(torch.load(verify(fixed_latent), map_location='cpu', weights_only=True), 'cuda:0')
            latent_file, calls = fixed_latent, None
        result = decode(runtime, latent, trace, folder, gt)
        result.update(name=name, trace=trace.finish(), latent=latent_file, calls=calls,
            elapsed_seconds=time.monotonic()-start, rng_before=rng_before, rng_after=rng_meta(),
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/(1 << 20))
        write(folder / 'result.json', result)
        print('REPLAY_COMPLETE ' + name + ' ' + str(result['surface']['chamfer_l1']), flush=True)
        return binding(folder / 'result.json')
    finally:
        hooks.close()


@torch.no_grad()
def worker(group):
    import trimesh
    torch.set_num_threads(2)
    protocol = freeze()
    info = read(OUT / 'inputs.json')
    if info['protocol'] != binding(OUT / 'protocol.json'):
        raise ValueError('Changed input protocol')
    complete = OUT / f'{group}_complete.json'
    if complete.exists():
        validate_complete(complete)
        print('REUSE_COMPLETED ' + group, flush=True)
        return
    folder = OUT / 'attempts' / (group + '_' + str(time.time_ns()))
    folder.mkdir(parents=True)
    print('LOAD_FROZEN_RUNTIME ' + now(), flush=True)
    patch = install_memory_efficient_mesh()
    runtime = load_frozen_slat30k(torch.device('cuda:0'))
    if runtime['binding'] != read(verify(protocol['slat'])):
        raise ValueError('SLat differs from historical run')
    env = environment()
    if not all(not m.training for net in ['model', 'decoder'] for m in runtime[net].modules()):
        raise ValueError('Non-eval module')
    if any(p.requires_grad for net in ['model', 'decoder'] for p in runtime[net].parameters()):
        raise ValueError('Trainable replay parameters')
    weights = {key: module_state_sha256(runtime[key]) for key in ['model', 'decoder']}
    versions = {key: {n: p._version for n, p in runtime[key].named_parameters()} for key in ['model', 'decoder']}
    if group == 'fresh':
        reference = read(OUT / 'same_complete.json')
        if reference['weights'] != weights or reference['environment'] != env or reference['patch'] != patch:
            raise ValueError('Fresh process weights/environment/patch differ')
    payload = torch.load(verify(info['payload']), map_location='cpu', weights_only=True)
    gt = trimesh.load(verify(info['gt']), force='mesh', process=False)
    noise_identity = OUT / 'noise.json'
    if not noise_identity.exists():
        if group != 'same':
            raise ValueError('Reference noise must be produced first')
        raw = sparse_noise(payload['coords'].numpy(), master_noise(SEED, INDEX, 'cuda:0'))
        locked(noise_identity, {'file': save_tensor_file(OUT / 'noise.pt', sparse_payload(raw)),
            'protocol': binding(OUT / 'protocol.json'), 'generator': 'original CUDA master_noise(seed=42,index=2)',
            'coords': tensor_meta(raw.coords.cpu()), 'feats': tensor_meta(raw.feats.cpu())})
        del raw
    noise_info = read(noise_identity)
    if noise_info['protocol'] != binding(OUT / 'protocol.json'):
        raise ValueError('Wrong fixed noise protocol')
    initial = torch.load(verify(noise_info['file']), map_location='cpu', weights_only=True)
    if not torch.equal(initial['coords'], payload['coords']):
        raise ValueError('Noise support changed')
    results = {}
    names = ['same_a', 'same_b'] if group == 'same' else ['fresh']
    for name in names:
        results[name] = one_run(runtime, name, folder, payload, initial, gt)
    if group == 'same':
        latent = read(results['same_a']['path'])['latent']
        for name in ['fixed_decode_a', 'fixed_decode_b']:
            results[name] = one_run(runtime, name, folder, payload, initial, gt, fixed_latent=latent)
    for key in ['model', 'decoder']:
        if any(p._version != versions[key][n] or p.grad is not None for n, p in runtime[key].named_parameters()):
            raise ValueError('Frozen parameter modified')
    write(complete, {'protocol': binding(OUT / 'protocol.json'), 'environment': env, 'weights': weights,
        'patch': patch, 'results': results, 'frozen_parameter_versions_unchanged': True, 'time': now()})


def validate_complete(path):
    complete = read(path)
    if complete['protocol'] != binding(OUT / 'protocol.json'):
        raise ValueError('Completed phase has another protocol')
    for row in complete['results'].values():
        result = read(verify(row))
        for key in ['trace', 'latent', 'mesh']:
            verify(result[key])
        for event in read(result['trace']['path'])['events']:
            verify(event['file'])
    return complete


def summarize():
    freeze()
    same, fresh = [validate_complete(OUT / f'{group}_complete.json') for group in ['same', 'fresh']]
    if same['environment'] != fresh['environment'] or same['weights'] != fresh['weights']:
        raise ValueError('Unmatched process identity')
    results = {k: read(v['path']) for done in [same, fresh] for k, v in done['results'].items()}
    pairs = [('same_a', 'same_b', None), ('same_a', 'fresh', None),
        ('fixed_decode_a', 'fixed_decode_b', ('decoder/', 'mesh/')),
        ('same_a', 'fixed_decode_a', ('decoder/', 'mesh/'))]
    comparisons = {}
    for a, b, prefixes in pairs:
        value = compare_traces(results[a]['trace'], results[b]['trace'], prefixes)
        value['cd_a'] = results[a]['surface']['chamfer_l1']
        value['cd_b'] = results[b]['surface']['chamfer_l1']
        value['cd_b_minus_a'] = value['cd_b'] - value['cd_a']
        comparisons[a + '_vs_' + b] = value
    value = {'time': now(), 'protocol': binding(OUT / 'protocol.json'), 'comparisons': comparisons,
        'results': results, 'environment': same['environment'], 'weights': same['weights'],
        'historical': {a: read(OLD / 'mesh' / a / '002/42.json')['surface'] for a in ['uniform', 'balanced']},
        'scope': 'bounded numerical diagnosis; trace synchronization may affect runtime scheduling; no quality or generalization claim'}
    write(OUT / 'summary.json', value)
    for key, c in comparisons.items():
        print(key, 'exact=', c['all_exact'], 'first=', c['first_difference'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'worker', 'summary'])
    p.add_argument('--group', choices=['same', 'fresh'])
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'summary':
        summarize()
    elif args.group:
        worker(args.group)
    else:
        p.error('worker needs --group')
