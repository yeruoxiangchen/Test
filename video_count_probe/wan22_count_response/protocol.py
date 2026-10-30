from __future__ import annotations

from pathlib import Path

from video_count_probe.core import atomic_json, checked_plan, digest, file_sha, is_complete, read_json, writer_lock
from video_count_probe.wan22_action_probe.protocol import MODEL, under_data
from video_count_probe.wan22_duration_probe.protocol import code_hash as duration_code_hash, scoring_plan
from video_count_probe.wan22_squat_count.review import validate_annotation

ROOT = Path('/data/zjr/video_count_wan22_count_response_20261029')
REFERENCE = Path('/data/zjr/video_count_wan22_duration_20261029/pilot')
SEEDS = (2026, 3407, 7919)
CONDITIONS = ('n1', 'n2', 'n3')
SCENE = ('A person in a red shirt is doing bodyweight squats. Wide full-body shot, fixed camera. '
         'The whole body, including the head and both feet, remains visible throughout. '
         'The person starts standing upright with both feet planted on the floor. ')


def count_prompt(n):
    if n not in (1, 2, 3):
        raise ValueError('Only preregistered requested counts 1/2/3 are supported')
    ordinals, words = ('first', 'second', 'third'), ('one', 'two', 'three')
    phases = ['First, the person bends both knees and lowers the hips into a squat, then rises back '
              'to a fully upright standing posture, completing the first squat. ']
    for i in range(1, n):
        phases.append('Next, the person bends both knees and lowers the hips again, then rises back '
                      f'to a fully upright standing posture, completing the {ordinals[i]} squat. ')
    return SCENE + ''.join(phases) + (
        f'This completes exactly {words[n - 1]} {"squat" if n == 1 else "squats"}. The person then stops. '
        f'After completing the {ordinals[n - 1]} squat, the person stands fully upright and still, '
        'with straight knees and both feet on the floor, without starting another squat.')


def reference_data(reference):
    reference = Path(reference).resolve()
    old = checked_plan(reference / 'plan.json')
    identity, labels = read_json(reference / 'identity.json'), read_json(reference / 'annotations.json')
    if old.get('experiment') != 'wan22_duration_v1' or old['mode'] != 'pilot':
        raise ValueError('Expected the completed duration pilot')
    if identity['plan_hash'] != old['plan_hash'] or labels['plan_hash'] != old['plan_hash']:
        raise ValueError('Reference identity/annotation plan mismatch')
    if identity['source_hash'] != duration_code_hash():
        raise ValueError('Original inference dependencies changed since reference generation')
    if labels.get('rater_type') not in ('ai', 'human') or not labels.get('rater'):
        raise ValueError('Reference reviewer provenance is missing')
    spec = old['conditions']['c8']
    expected = {'frames': 193, 'fps': 24, 'height': 704, 'width': 1280, 'steps': 50,
                'guidance_scale': 5.0, 'flow_shift': 5.0, 'input_image': None, 'prompt_rewrite': False}
    if spec['target_count'] != 2 or spec['prompt'] != count_prompt(2) or any(spec['config'].get(k) != v for k, v in expected.items()):
        raise ValueError('Reference does not match the frozen n=2 template/geometry')
    selected = [j for j in old['jobs'] if j['condition'] == 'c8']
    if len(selected) != 3 or {j['seed'] for j in selected} != set(SEEDS):
        raise ValueError('Need all three C8 reference seeds')
    by_id = {x['job_id']: x for x in labels['items']}
    if len(by_id) != len(labels['items']):
        raise ValueError('Duplicate reference annotation')
    records, chosen_labels, bindings = {}, {}, []
    for job in selected:
        if not is_complete(reference, job, identity):
            raise ValueError('Reference video missing/corrupt')
        path = reference / 'records' / (job['id'] + '.json')
        record, label = read_json(path), by_id[job['id']]
        if record['config'] != spec['config'] or record['prompt'] != spec['prompt']:
            raise ValueError('Reference configuration/prompt mismatch')
        if record['initial_latent_shape'] != [1, 48, 49, 44, 80]:
            raise ValueError('Reference noise geometry mismatch')
        validate_annotation(label, record, scoring_plan(spec))
        if not label['reviewed']:
            raise ValueError('Reference must be reviewed before reuse')
        records[job['seed']], chosen_labels[job['seed']] = record, label
        bindings.append({'seed': job['seed'], 'id': job['id'], 'record_sha256': file_sha(path),
                         'video_sha256': record['video_sha256']})
    binding = {'root': str(reference), 'plan_hash': old['plan_hash'],
               'identity_sha256': file_sha(reference / 'identity.json'),
               'annotations_sha256': file_sha(reference / 'annotations.json'),
               'reviewer': {k: labels[k] for k in ('rater', 'rater_type')},
               'selected': sorted(bindings, key=lambda x: x['seed'])}
    return old, identity, records, chosen_labels, binding


def build_plan(mode, reference=REFERENCE):
    if mode not in ('smoke', 'pilot'):
        raise ValueError(mode)
    old, identity, _, _, binding = reference_data(reference)
    config = {**old['conditions']['c8']['config'], 'steps': 2 if mode == 'smoke' else 50}
    conditions = {f'n{n}': {'name': f'{n} requested squats, 8s', 'target_count': n,
                            'config': dict(config), 'prompt': count_prompt(n)} for n in (1, 2, 3)}
    jobs = []
    # Alternate requested counts across seeds; fixed order, no outcome-driven selection.
    for i, seed in enumerate(SEEDS):
        for condition in (('n1', 'n3') if i % 2 == 0 else ('n3', 'n1')):
            job = {'condition': condition, 'seed': seed}
            job['id'] = digest({'experiment': 'wan22_count_response_v1', 'mode': mode, **job})[:16]
            jobs.append(job)
    if mode == 'smoke':
        jobs = [next(j for j in jobs if j['condition'] == 'n3')]
    plan = {'schema': 1, 'experiment': 'wan22_count_response_v1', 'mode': mode,
            'scientific_evaluation': mode == 'pilot', 'reference': binding, 'reference_identity': identity,
            'conditions': conditions, 'jobs': jobs, 'seeds': list(SEEDS),
            'reused_conditions': ['n2'] if mode == 'pilot' else [],
            'pairing': 'All targets use identical geometry and seed; new initial-latent SHA256 MUST equal reused C8/n2.',
            'scoring': {'primary': ['exact_count', 'count_and_stop'],
                        'cycle': 'upright -> visibly lowered hips/bent knees -> upright; no numerical depth threshold',
                        'partials': 'Initial/terminal partials separate; low-position bobbing does not create another cycle.',
                        'diagnostics': ['initial_upright', 'last_second_rest', 'observed_minus_requested',
                                        'paired_count_response', 'extra_partial_after_target'],
                        'no_fixed_deadline': True},
            'limits': ['Six new formal clips, three reused n2 clips; one action/model, three seeds, descriptive only.',
                       'Requested count and matching phase expansion change together; not a numeral-only intervention.',
                       'n2 already seen: this follow-up is exploratory, not an independent confirmatory replication.',
                       'Existing n2 annotation includes shallow cycles; retain provenance and depth sensitivity caveat.',
                       'No learned method, training, automatic expansion, prompt search, or resolution fallback.',
                       'Prior 8s allocated/reserved maxima were about 18,000/21,902 MiB; new prompts still require smoke.',
                       'Unreviewed outcomes remain unknown, not failures. AI review is not human validation.']}
    plan['plan_hash'] = digest(plan)
    return plan


def verify_reference(plan):
    data = reference_data(plan['reference']['root'])
    if data[-1] != plan['reference'] or data[1] != plan['reference_identity']:
        raise ValueError('Frozen reference changed; use a new output root')
    return data


def prepare(root, reference=REFERENCE):
    if Path(root).resolve() == Path(reference).resolve().parent:
        raise ValueError('Output root must differ from reference')
    for mode in ('smoke', 'pilot'):
        plan = build_plan(mode, reference)
        path = Path(root) / mode / 'plan.json'
        with writer_lock(path.parent / '.writer.lock'):
            if path.exists() and read_json(path) != plan:
                raise ValueError('Frozen plan differs; use a new output root')
            if not path.exists():
                atomic_json(path, plan)
        print(f'{mode}: {len(plan["jobs"])} new clips; 193 frames, 24fps; {path}', flush=True)


def all_prompts(plan):
    return [plan['conditions']['n1']['config']['negative_prompt'],
            *[plan['conditions'][c]['prompt'] for c in CONDITIONS]]


def code_hash():
    package = Path(__file__).parent
    return digest({'unchanged_duration_dependencies': duration_code_hash(),
                   'new_files': {p.name: file_sha(p) for p in sorted([*package.glob('*.py'), package / 'run.sh'])}})
