from __future__ import annotations

import copy
from pathlib import Path

from video_count_probe.core import atomic_json, checked_plan, digest, file_sha, read_json, writer_lock
from video_count_probe.wan22_action_probe.protocol import MODEL, MODEL_ID, under_data

ROOT = Path('/data/zjr/video_count_wan22_squat_count_20261028')
REFERENCE = Path('/data/zjr/video_count_wan22_action_20261028/pilot')
SEEDS = (2026, 3407, 7919)
METHODS = ('count_prompt', 'phase_prompt')
COMMON = ('A person in a red shirt is doing bodyweight squats. Wide full-body shot, fixed camera. '
          'The whole body, including the head and both feet, remains visible throughout. '
          'The person starts standing upright with both feet planted on the floor. ')
REST = ('During the final second, from 4 to 5 seconds, the person stands fully upright and still, '
        'with straight knees and both feet on the floor, without starting another squat.')


def prompts(count):
    if count not in (1, 2):
        raise ValueError('Only one or two squats fit this pilot')
    word = {1: 'one', 2: 'two'}[count]
    plural = 'squat' if count == 1 else 'squats'
    direct = (f'During the first four seconds, the person performs exactly {word} complete bodyweight {plural}, '
              'then stops. ')
    if count == 1:
        phase = ('During the first four seconds: first, the person bends both knees and lowers the hips into a squat; '
                 'next, the person rises back to a fully upright standing posture. '
                 'This completes exactly one squat. The person then stops. ')
    else:
        phase = ('During the first four seconds: first, the person bends both knees and lowers the hips into a squat, '
                 'then rises back to a fully upright standing posture, completing the first squat. '
                 'Next, the person bends both knees and lowers the hips again, then rises back to a fully upright '
                 'standing posture, completing the second squat. This completes exactly two squats. '
                 'The person then stops. ')
    return {'count_prompt': COMMON + direct + REST, 'phase_prompt': COMMON + phase + REST}


def reference_binding(reference):
    reference = Path(reference).resolve()
    plan = checked_plan(reference / 'plan.json')
    review_path = reference / 'ai_review_20261028/summary_ai.json'
    review = read_json(review_path)
    if review['plan_hash'] != plan['plan_hash'] or not review['ai_review_complete']:
        raise ValueError('Expected completed AI capability review bound to the prior plan')
    passed = [r['job_id'] for r in review['per_video'] if r['action'] == 'squat' and r['complete_cycle'] is True]
    if len(passed) != 2:
        raise ValueError('This protocol was selected from two complete squat examples')
    return {'root': str(reference), 'plan_hash': plan['plan_hash'],
            'review_sha256': file_sha(review_path), 'selection': 'Two squat capability clips passed AI review; not independent human validation.'}


def build_plan(mode, reference=REFERENCE):
    if mode not in ('pilot', 'smoke'):
        raise ValueError(mode)
    reference = Path(reference)
    binding = reference_binding(reference)
    config = copy.deepcopy(checked_plan(reference / 'plan.json')['config'])
    expected = {'model_id': MODEL_ID, 'height': 704, 'width': 1280, 'frames': 121, 'fps': 24,
                'steps': 50, 'guidance_scale': 5.0, 'flow_shift': 5.0, 'input_image': None, 'prompt_rewrite': False}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Prior model/geometry/sampling config changed')
    config['steps'] = 2 if mode == 'smoke' else 50
    cases = [{'id': f'squat_n{count}', 'action': 'squat', 'target_count': count,
              'prompts': prompts(count)} for count in (1, 2)]
    jobs = []
    for count_index, case in enumerate(cases):
        for seed_index, seed in enumerate(SEEDS):
            pair_id = digest({'case_id': case['id'], 'seed': seed, 'mode': mode})[:16]
            order = METHODS if (count_index + seed_index) % 2 == 0 else tuple(reversed(METHODS))
            for method in order:
                job = {'case_id': case['id'], 'seed': seed, 'method': method, 'pair_id': pair_id}
                job['id'] = digest({'experiment': 'wan22_squat_count_v1', 'mode': mode, **job})[:16]
                jobs.append(job)
    if mode == 'smoke':
        # Exercise both prompt branches and the actual pairing path at full geometry.
        jobs = jobs[:2]
    plan = {'schema': 1, 'experiment': 'wan22_squat_count_v1', 'mode': mode,
            'scientific_evaluation': mode == 'pilot', 'config': config, 'cases': cases, 'jobs': jobs,
            'reference': binding, 'methods': list(METHODS), 'seeds': list(SEEDS),
            'intervention': 'One whole-clip prompt per video. Textual phase descriptions only; no attention routing, temporal masking, training, or model changes.',
            'pairing': 'Same seed and explicit initial latent tensor within each count/method pair. Method order counterbalanced across the six pairs.',
            'scoring': {'complete_cycle': 'Upright -> clearly lowered hips and bent knees -> upright. Partial attempts are recorded separately.',
                        'rest_start_frame': 96, 'last_frame': 120,
                        'exact_count': 'Number of fully visible completed cycles equals the requested count.',
                        'stopped_after_target': 'Target reached; no further complete cycle or partial attempt after its completion.',
                        'final_rest': 'Upright throughout frames 96..120 without renewed squat/step/jump; small breathing or hand motion is allowed.',
                        'joint_count_stop': 'Exact count AND stopped_after_target AND final_rest AND target completed by frame 96.',
                        'initial_partial': 'Report separately; not silently counted as a complete cycle.',
                        'unknown': 'Unknown judgments remain null; never treat them as confirmed failure or success.'},
            'limits': ['12 formal videos / 6 pairs; descriptive feasibility comparison, not a method effectiveness claim.',
                       'This compares two prompt formulations, not a trained controller or temporal routing algorithm.',
                       'One action and three fresh seeds; no population-level claim or formal significance test.',
                       'Common framing and stop instructions differ from the previous capability screen; cross-screen differences are not model-only causal effects.',
                       'AI annotation must retain provenance; reviewed-but-uncertain is distinct from unreviewed.',
                       '2-step smoke is an execution check only; it never contributes to pilot metrics.']}
    plan['plan_hash'] = digest(plan)
    return plan


def prepare(root, reference=REFERENCE):
    for mode in ('smoke', 'pilot'):
        plan = build_plan(mode, reference)
        run = Path(root) / mode
        with writer_lock(run / '.writer.lock'):
            path = run / 'plan.json'
            if path.exists() and read_json(path) != plan:
                raise RuntimeError('Frozen plan differs; use a new --root')
            if not path.exists():
                atomic_json(path, plan)
        print(f'{mode}: {len(plan["jobs"])} videos / {len(plan["jobs"]) // 2} pairs; {run}', flush=True)


def verify_reference(plan):
    if reference_binding(plan['reference']['root']) != plan['reference']:
        raise ValueError('Capability-screen reference changed; do not mix protocols')


def code_hash():
    package = Path(__file__).parent
    shared = package.parent
    sources = [*package.glob('*.py'), package / 'run.sh',
               *(shared / 'wan22_action_probe').glob('*.py'),
               *(shared / name for name in ('core.py', 'runtime.py', 'cli.py', 'downloads.py'))]
    return digest({str(p.relative_to(shared)): file_sha(p) for p in sorted(sources)})


def all_prompts(plan):
    return list(dict.fromkeys([plan['config']['negative_prompt']] +
                             [p for c in plan['cases'] for p in c['prompts'].values()]))
