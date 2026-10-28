"""Intervene on the two observed algorithms using the same frozen block."""
import argparse
import time
from pathlib import Path

import torch

from .common import OUT, freeze, binding, verify, read, write, locked, now
from .probe import build_block, install_block_hooks, NativeObserver, causal_comparison
from .kernel_policy import FixedNative, PREFERRED_HALF
from ..hssd_mesh_replay_20261025.replay import sparse_from, environment
from ..hssd_mesh_replay_20261025.tracing import Trace, tensor_meta
from ...frozen_slat import configure_determinism, module_state_sha256


def semantic_comparison(left, right):
    """Separate sparse pair enumeration permutations from numerical differences."""
    result = causal_comparison(left, right)
    events = [{e['name']: e for e in read(verify(p))['events']} for p in [left, right]]
    neighbors = []
    for name in ['conv1/neighbors', 'conv2/neighbors']:
        values = [torch.load(verify(es[name]['file']), map_location='cpu', weights_only=True) for es in events]
        if set(values[0]) != set(values[1]):
            raise ValueError('Different sparse neighbor offsets')
        for key in values[0]:
            a, b = values[0][key], values[1][key]
            for v in [a, b]:
                if v.shape[0] != 2 or v.numel() == 0 or int(v.min()) < 0 or int(v.max()) >= 9092:
                    raise ValueError('Invalid active neighbor pair')
                # A fixed offset has one contribution per output location.
                if any(len(torch.unique(v[j])) != v.shape[1] for j in [0, 1]):
                    raise ValueError('Duplicate neighbor mapping for one kernel offset')
            ca = a[:, torch.argsort(a[0].long()*9092+a[1])]
            cb = b[:, torch.argsort(b[0].long()*9092+b[1])]
            neighbors.append({'event': name, 'offset': key, 'pairs': a.shape[1],
                'ordered_exact': torch.equal(a, b), 'pair_set_exact': torch.equal(ca, cb)})
    numerical = [r for r in result['differences'] if not r['name'].endswith('/neighbors')]
    result.update(neighbor_sets=neighbors, neighbor_sets_exact=all(r['pair_set_exact'] for r in neighbors),
        numerical_differences=numerical, first_numerical_difference=numerical[0] if numerical else None)
    result['semantic_all_exact'] = result['neighbor_sets_exact'] and not numerical
    return result


def protocol():
    freeze()
    blocks = [read(OUT / f'block_{i}.json') for i in [0, 1]]
    names = [b['results'][0]['algorithms'][0]['descriptor'] for b in blocks]
    if names[0] == names[1] or names[0] != PREFERRED_HALF:
        raise ValueError('Expected two distinct observed algorithms, with the frozen default first')
    p = {'format': 'hssd_native_kernel_intervention.v1',
        'operator_protocol': binding(OUT / 'protocol.json'),
        'observations': [binding(OUT / f'block_{i}.json') for i in [0, 1]],
        'inputs': binding(OUT / 'inputs.json'), 'descriptors': names,
        'sources': [binding(Path(__file__)), binding(Path(__file__).with_name('kernel_policy.py'))],
        'order': [0, 1, 0], 'precision': 'unchanged', 'weights': 'unchanged',
        'scope': 'causal single-block intervention; full-path repeatability is a separate smoke gate'}
    locked(OUT / 'kernel_protocol.json', p)
    return p


@torch.no_grad()
def run():
    torch.set_num_threads(2)
    p = protocol()
    complete = OUT / 'kernel_check.json'
    if complete.exists():
        r = read(complete)
        if r['protocol'] != binding(OUT / 'kernel_protocol.json') or not r['passed']:
            raise ValueError('Incompatible previous intervention')
        for row in r['results']:
            verify(row['trace'])
        return
    configure_determinism()
    packet = torch.load(verify(read(OUT / 'inputs.json')['packet']), map_location='cpu', weights_only=True)
    block = build_block(packet)
    expected = [read(OUT / f'block_{i}.json') for i in [0, 1]]
    if module_state_sha256(block) != expected[0]['state_sha']:
        raise ValueError('Changed intervention block weights')
    folder = OUT / 'attempts' / f'kernel_{time.time_ns()}'
    results = []
    for i, choice in enumerate(p['order']):
        trace = Trace(folder / str(i))
        with FixedNative(p['descriptors'][choice]) as fixed, NativeObserver(trace) as observer:
            handles = install_block_hooks(block, trace, observer)
            try:
                initial = sparse_from(packet, 'cuda:0')
                embedding = packet['embedding'].clone().to('cuda:0')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = block(initial, embedding)
            finally:
                for h in handles: h.remove()
        result = {'choice': choice, 'trace': trace.finish(), 'algorithms': observer.records,
            'policy': fixed.manifest(), 'output_sha': tensor_meta(output.feats.cpu())}
        if any(a['descriptor'] != p['descriptors'][choice] for a in observer.records):
            raise ValueError('Actual algorithm differs from forced descriptor')
        result['comparison_to_observed'] = semantic_comparison(expected[choice]['results'][0]['trace'], result['trace'])
        result['exact_observed_output'] = result['output_sha'] == expected[choice]['results'][0]['output_sha']
        results.append(result)
        print('KERNEL_INTERVENTION', i, choice, 'reproduced', result['exact_observed_output'], flush=True)
    exact_repeat = semantic_comparison(results[0]['trace'], results[2]['trace'])
    passed = all(r['comparison_to_observed']['semantic_all_exact'] and r['exact_observed_output'] for r in results) and exact_repeat['semantic_all_exact']
    write(complete, {'time': now(), 'protocol': binding(OUT / 'kernel_protocol.json'),
        'environment': environment(), 'results': results, 'repeat': exact_repeat, 'passed': passed})
    if not passed:
        raise RuntimeError('Algorithm intervention did not reproduce the observations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run'])
    parser.parse_args()
    run()
