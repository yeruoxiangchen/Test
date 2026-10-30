from __future__ import annotations

import copy
from pathlib import Path

from video_count_probe.core import atomic_json, checked_plan, digest, file_sha, is_complete, read_json, writer_lock
from video_count_probe.wan22_action_probe.protocol import MODEL, under_data
from video_count_probe.wan22_squat_count.protocol import REST, SEEDS

ROOT = Path('/data/zjr/video_count_wan22_duration_20261029')
REFERENCE = Path('/data/zjr/video_count_wan22_squat_count_20261028/pilot')
CONDITIONS = ('a5', 'b5', 'c8')
NAMES = {'a5': 'A: 5s, fixed 4s deadline (reused)',
         'b5': 'B: 5s, no fixed deadline', 'c8': 'C: 8s, same prompt as B'}


def reference_data(reference):
    """Read only the selected three reference clips, retaining original identities."""
    reference = Path(reference).resolve()
    old = checked_plan(reference / 'plan.json')
    identity = read_json(reference / 'identity.json')
    labels = read_json(reference / 'annotations.json')
    if old.get('experiment') != 'wan22_squat_count_v1' or old['mode'] != 'pilot':
        raise ValueError('Expected the completed squat count pilot')
    if identity['plan_hash'] != old['plan_hash'] or labels['plan_hash'] != old['plan_hash']:
        raise ValueError('Reference plan/identity/annotation mismatch')
    if labels.get('rater_type') not in ('ai', 'human') or not labels.get('rater'):
        raise ValueError('Reference annotations require reviewer provenance')
    selected = [j for j in old['jobs'] if j['case_id'] == 'squat_n2' and j['method'] == 'phase_prompt']
    if len(selected) != 3 or {j['seed'] for j in selected} != set(SEEDS):
        raise ValueError('Expected all three two-squat phase-prompt references')
    by_id = {x['job_id']: x for x in labels['items']}
    if len(by_id) != len(labels['items']):
        raise ValueError('Duplicate reference annotation')
    from video_count_probe.wan22_squat_count.review import validate_annotation
    records, selected_labels, bindings = {}, {}, []
    for job in selected:
        if not is_complete(reference, job, identity):
            raise ValueError(f'Reference incomplete/corrupt: {job["id"]}')
        path = reference / 'records' / (job['id'] + '.json')
        record, label = read_json(path), by_id[job['id']]
        validate_annotation(label, record, old)
        if not label['reviewed']:
            raise ValueError('Reference clip has not been reviewed')
        records[job['seed']], selected_labels[job['seed']] = record, label
        bindings.append({'seed': job['seed'], 'id': job['id'], 'record_sha256': file_sha(path),
                         'video_sha256': record['video_sha256']})
    binding = {'root': str(reference), 'plan_hash': old['plan_hash'],
               'identity_sha256': file_sha(reference / 'identity.json'),
               'annotations_sha256': file_sha(reference / 'annotations.json'),
               'reviewer': {'rater': labels['rater'], 'rater_type': labels['rater_type']},
               'selected': sorted(bindings, key=lambda x: x['seed'])}
    return old, identity, records, selected_labels, binding


def free_prompt(old_prompt):
    prefix = 'During the first four seconds: first, '
    if old_prompt.count(prefix) != 1 or not old_prompt.endswith(REST):
        raise ValueError('Unexpected reference prompt; do not silently rewrite it')
    # Keep the scene and both full-cycle descriptions. Remove only fixed clock deadlines.
    return old_prompt.replace(prefix, 'First, ', 1)[:-len(REST)] + (
        'After completing the second squat, the person stands fully upright and still, '
        'with straight knees and both feet on the floor, without starting another squat.')


def build_plan(mode, reference=REFERENCE):
    if mode not in ('pilot', 'smoke'):
        raise ValueError(mode)
    old, identity, refs, _, binding = reference_data(reference)
    base = copy.deepcopy(old['config'])
    expected = {'height': 704, 'width': 1280, 'frames': 121, 'fps': 24, 'steps': 50,
                'guidance_scale': 5.0, 'flow_shift': 5.0, 'input_image': None, 'prompt_rewrite': False}
    if any(base.get(k) != v for k, v in expected.items()):
        raise ValueError('Reference sampling configuration changed')
    prompts = {r['prompt'] for r in refs.values()}
    if len(prompts) != 1:
        raise ValueError('Reference prompts differ across seeds')
    original = prompts.pop()
    conditions = {}
    for condition, frames in (('a5', 121), ('b5', 121), ('c8', 193)):
        config = {**base, 'frames': frames, 'steps': 2 if mode == 'smoke' and condition != 'a5' else 50}
        conditions[condition] = {'name': NAMES[condition], 'config': config, 'target_count': 2,
                                 'prompt': original if condition == 'a5' else free_prompt(original),
                                 'duration_seconds': (frames - 1) / config['fps'],
                                 'fixed_completion_deadline_seconds': 4 if condition == 'a5' else None}
    jobs = []
    for index, seed in enumerate(SEEDS):
        for condition in (('b5', 'c8') if index % 2 == 0 else ('c8', 'b5')):
            job = {'condition': condition, 'seed': seed}
            job['id'] = digest({'experiment': 'wan22_duration_v1', 'mode': mode, **job})[:16]
            jobs.append(job)
    if mode == 'smoke':
        jobs = jobs[:2]
    plan = {'schema': 1, 'experiment': 'wan22_duration_v1', 'mode': mode,
            'scientific_evaluation': mode == 'pilot', 'reference': binding,
            'reference_identity': identity, 'conditions': conditions, 'jobs': jobs, 'seeds': list(SEEDS),
            'reference_conditions_in_report': ['a5'] if mode == 'pilot' else [],
            'pairing': {'a5_vs_b5': 'Same geometry and seed; require identical initial-latent SHA256.',
                        'b5_vs_c8': 'Seed-matched only; different latent shapes. Never claim identical noise.'},
            'scoring': {'primary': ['exact_count', 'count_and_stop'],
                        'complete_cycle': 'Visible upright -> lowered hips/bent knees -> upright; initial partials are separate.',
                        'count_and_stop': 'Exactly two completed cycles, with no additional partial attempt after target completion.',
                        'last_second_rest': 'Diagnostic only for B/C: full final second upright, small hand motion/breathing allowed.',
                        'timing': 'Record each start/bottom/end frame; report target completion time without imposing a fixed deadline on B/C.'},
            'limits': ['Six new videos; three reused A clips and only three shared seeds. Descriptive diagnostic, not independent replication.',
                       'B removes explicit clock instructions; A/B is a prompt-constraint intervention, not proof of a single internal mechanism.',
                       'B/C changes clip length at identical spatial resolution/fps/prompt/sampling steps; noise tensors have different shapes.',
                       'Reused A annotations retain original provenance; new AI/human labels must be identified.',
                       'Longer clips may produce more repetitions without reliable requested-count control.',
                       'Two-step smoke checks both geometries/decoding; it does not guarantee 50-step peak memory or quality.',
                       'No training, temporal routing, prompt search, resolution fallback, or automatic expansion.']}
    plan['plan_hash'] = digest(plan)
    return plan


def verify_reference(plan):
    data = reference_data(plan['reference']['root'])
    if data[-1] != plan['reference'] or data[1] != plan['reference_identity']:
        raise ValueError('Frozen reference changed; use a new output root')
    return data


def prepare(root, reference=REFERENCE):
    root = Path(root)
    if root.resolve() == Path(reference).resolve().parent:
        raise ValueError('Output root must differ from the reference experiment')
    for mode in ('smoke', 'pilot'):
        plan = build_plan(mode, reference)
        run = root / mode
        with writer_lock(run / '.writer.lock'):
            path = run / 'plan.json'
            if path.exists() and read_json(path) != plan:
                raise RuntimeError('Frozen plan differs; use a new --root')
            if not path.exists():
                atomic_json(path, plan)
        print(f'{mode}: {len(plan["jobs"])} new videos; 121/193 frames; {path}', flush=True)


def scoring_plan(condition):
    config = condition['config']
    return {'scoring': {'last_frame': config['frames'] - 1,
                        'rest_start_frame': config['frames'] - 1 - config['fps']}}


def all_prompts(plan):
    return [plan['conditions']['b5']['config']['negative_prompt'], plan['conditions']['b5']['prompt']]


def code_hash():
    package = Path(__file__).parent
    shared = package.parent
    files = [*package.glob('*.py'), package / 'run.sh', *shared.glob('*.py')]
    for name in ('wan22_action_probe', 'wan22_squat_count'):
        files += list((shared / name).glob('*.py'))
    return digest({str(p.relative_to(shared)): file_sha(p) for p in sorted(set(files))})
