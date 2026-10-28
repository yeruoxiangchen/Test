"""Detached CPU snapshots and ordered numerical comparisons, with no model edits."""
import hashlib
from pathlib import Path

import torch

from .common import binding, read, verify, write


def flatten(value, prefix=''):
    result = {}
    if torch.is_tensor(value):
        result[prefix or 'tensor'] = value.detach().to('cpu').contiguous().clone()
    elif hasattr(value, 'feats') and hasattr(value, 'coords'):
        result.update(flatten(value.coords, prefix + '/coords'))
        result.update(flatten(value.feats, prefix + '/feats'))
    elif isinstance(value, dict):
        for key, item in value.items():
            result.update(flatten(item, prefix + '/' + str(key)))
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            result.update(flatten(item, prefix + '/' + str(i)))
    return result


def tensor_meta(value):
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite traced tensor')
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(raw).hexdigest()}


class Trace:
    def __init__(self, folder, limit=2 << 30):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.events = []
        self.bytes = 0
        self.limit = limit

    def add(self, name, value):
        tensors = flatten(value)
        if not tensors:
            raise ValueError('Empty trace: ' + name)
        if any(x['name'] == name for x in self.events):
            raise ValueError('Duplicate trace event: ' + name)
        self.bytes += sum(v.numel() * v.element_size() for v in tensors.values())
        if self.bytes > self.limit:
            raise RuntimeError('Trace tensor byte limit exceeded')
        path = self.folder / f'{len(self.events):04d}.pt'
        torch.save(tensors, path)
        event = {'name': name, 'file': binding(path), 'tensors': {k: tensor_meta(v) for k, v in tensors.items()}}
        self.events.append(event)
        return event

    def finish(self):
        result = {'events': self.events, 'tensor_bytes': self.bytes}
        write(self.folder / 'trace.json', result)
        return binding(self.folder / 'trace.json')


def compare_event(left, right):
    if left['tensors'] == right['tensors']:
        return {'exact': True}
    a = torch.load(verify(left['file']), map_location='cpu', weights_only=True)
    b = torch.load(verify(right['file']), map_location='cpu', weights_only=True)
    if set(a) != set(b):
        return {'exact': False, 'schema_mismatch': True}
    differences = {}
    for key, x in a.items():
        y = b[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            differences[key] = {'shape_a': list(x.shape), 'shape_b': list(y.shape), 'dtype_a': str(x.dtype), 'dtype_b': str(y.dtype)}
        elif left['tensors'][key] != right['tensors'][key]:
            diff = (x.double() - y.double()).abs()
            differences[key] = {'different_values': int((x != y).sum()), 'total_values': x.numel(),
                'max_abs': float(diff.max()) if diff.numel() else 0., 'mean_abs': float(diff.mean()) if diff.numel() else 0.,
                'rms': float(diff.square().mean().sqrt()) if diff.numel() else 0.}
    return {'exact': False, 'differences': differences}


def compare_traces(left, right, prefixes=None):
    a, b = read(verify(left))['events'], read(verify(right))['events']
    if prefixes:
        a = [e for e in a if e['name'].startswith(prefixes)]
        b = [e for e in b if e['name'].startswith(prefixes)]
    if not a or [x['name'] for x in a] != [x['name'] for x in b]:
        raise ValueError('Missing/reordered comparison events')
    changes = []
    for x, y in zip(a, b):
        result = compare_event(x, y)
        if not result['exact']:
            changes.append({'name': x['name'], **result})
    return {'event_count': len(a), 'different_event_count': len(changes),
        'all_exact': not changes, 'first_difference': changes[0] if changes else None, 'differences': changes}
