"""Bounded GT/input coordinate diagnostic; no model training or output selection."""
import os
from .. import common as c


def main():
    os.environ.update(c.environment())
    from ..runner import choose_gpu
    gpu = choose_gpu()
    if gpu is None: raise RuntimeError('No candidate GPU has enough free memory for the bounded projection probe')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(16384 * 1024**2 / torch.cuda.get_device_properties(0).total_memory)
    from ..data_worker import projection_audit
    group = next(g for g in c.plan()['pilot_groups'] if g['both_native_ready'])
    capture = c.read(c.SOURCE / 'scenes' / group['scene'] / 'plan.json')['plan']
    inputs = {'scene': group['scene'], 'frames': [r['frame'] for r in group['selected_frames']]}
    result = {uid: projection_audit(inputs, uid, capture) for uid in (group['a'], group['b'])}
    assert all(r['visible_to_amodal_hit'] >= .75 for values in result.values() for r in values)
    c.write(c.OUT / 'smoke/projection_probe.json', {'time': c.now(), 'gpu': gpu, 'results': result,
            'role': 'GT/input calibration only; this val pair is never trained on',
            'implementation_sha256': c.sha(c.HERE / 'data_worker.py')})
    print(result)


if __name__ == '__main__': main()
