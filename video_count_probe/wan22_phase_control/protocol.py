from pathlib import Path

from video_count_probe.core import atomic_json, digest, file_sha, read_json, writer_lock
from video_count_probe.wan22_action_probe.protocol import MODEL, under_data
from video_count_probe.wan22_duration_probe.protocol import scoring_plan
from video_count_probe.wan22_count_response.protocol import REFERENCE, SEEDS, reference_data, verify_reference
from video_count_probe.wan22_count_response.protocol import code_hash as dependency_hash

ROOT = Path('/data/zjr/video_count_wan22_phase_control_20261030')
CONDITIONS = ('text_only', 'routed')
SCENE = ('A person in a red shirt performs exactly two bodyweight squats and then stops. '
         'Wide full-body shot, fixed camera. The head and both feet remain visible throughout. ')
PHASES = (
    (0., .5, 'From 0 to 0.5 seconds, the person stands fully upright and still with straight knees.'),
    (.5, 1.75, 'From 0.5 to 1.75 seconds, the person bends both knees and lowers the hips into the first squat.'),
    (1.75, 3., 'From 1.75 to 3 seconds, the person rises from the first squat to a fully upright standing posture.'),
    (3., 4.25, 'From 3 to 4.25 seconds, the person bends both knees and lowers the hips into the second squat.'),
    (4.25, 5.5, 'From 4.25 to 5.5 seconds, the person rises from the second squat to a fully upright standing posture.'),
    (5.5, 8., 'From 5.5 to 8 seconds, the person stands fully upright and still with straight knees, without starting another squat.'),
)


def prompt_and_spans():
    prompt, spans = SCENE, []
    for start, end, text in PHASES:
        begin = len(prompt)
        prompt += text
        spans.append({'start': start, 'end': end, 'char_start': begin, 'char_end': len(prompt)})
        prompt += ' '
    return prompt.rstrip(), spans


def build_plan(mode, reference=REFERENCE):
    if mode not in ('smoke', 'pilot'):
        raise ValueError(mode)
    old, identity, _, _, binding = reference_data(reference)
    config = {**old['conditions']['c8']['config'], 'steps': 2 if mode == 'smoke' else 50}
    prompt, spans = prompt_and_spans()
    conditions = {c: {'name': c, 'target_count': 2, 'config': dict(config), 'prompt': prompt,
                       'routing_enabled': c == 'routed'} for c in CONDITIONS}
    jobs = []
    for i, seed in enumerate(SEEDS[:1] if mode == 'smoke' else SEEDS):
        for condition in CONDITIONS if i % 2 == 0 else CONDITIONS[::-1]:
            job = {'condition': condition, 'seed': seed}
            job['id'] = digest({'experiment': 'wan22_phase_control_v1', 'mode': mode, **job})[:16]
            jobs.append(job)
    plan = {'schema': 1, 'experiment': 'wan22_phase_control_v1', 'mode': mode,
            'scientific_evaluation': mode == 'pilot', 'reference': binding, 'reference_identity': identity,
            'conditions': conditions, 'jobs': jobs, 'seeds': list(SEEDS), 'reused_conditions': [],
            'routing': {'segments': spans, 'sigma_seconds': .25, 'maximum_penalty': 8.,
                        'chunk_size': 1024, 'latent_frames': 49, 'tokens_per_frame': 880,
                        'latent_time': 'index * 4 / 24 seconds; causal VAE anchor approximation',
                        'bias': '-min(distance_outside_phase_seconds^2 / (2 * sigma_seconds^2), maximum_penalty)',
                        'global_special_padding_tokens': 'zero added bias',
                        'unconditional_cfg': 'unchanged native processor',
                        'control': 'same chunked SDPA and embeddings; zero added bias',
                        'layers': 'all 30 text cross-attention blocks, all denoising steps'},
            'pairing': 'Byte-identical positive/negative prompt embeddings and initial latent SHA within each seed.',
            'scoring': {'primary': ['exact_count', 'count_and_stop'],
                        'diagnostics': ['initial_upright', 'last_second_rest', 'cycle_times_seconds',
                                        'target_completion_seconds'], 'schedule_completion_seconds': 5.5},
            'limits': ['Six formal videos, one action/model, three previously observed seeds; exploratory feasibility only.',
                       'Two-step smoke checks execution and memory, not semantic quality.',
                       'Fixed heuristic routing inspired by Prompt Relay, not an originality claim or trained method.',
                       'Contextual T5 embeddings can carry other phases; token routing does not fully isolate semantics.',
                       'Latent time anchors approximate decoded time; VAE temporal receptive fields overlap.',
                       'No automatic training, expansion, parameter sweep or resolution/frame fallback.',
                       'Visual review must inspect cycles, partial attempts, freezing and discontinuities; missing labels stay unknown.']}
    plan['plan_hash'] = digest(plan)
    return plan


def prepare(root, reference=REFERENCE):
    if Path(root).resolve() == Path(reference).resolve().parent:
        raise ValueError('Output root must differ from reference')
    for mode in ('smoke', 'pilot'):
        plan = build_plan(mode, reference)
        path = Path(root) / mode / 'plan.json'
        with writer_lock(path.parent / '.writer.lock'):
            if path.exists() and read_json(path) != plan:
                raise ValueError('Frozen plan differs; use a new root')
            if not path.exists():
                atomic_json(path, plan)
        print(f'{mode}: {len(plan["jobs"])} videos; {path}', flush=True)


def all_prompts(plan):
    spec = plan['conditions']['text_only']
    return [spec['config']['negative_prompt'], spec['prompt']]


def code_hash():
    package = Path(__file__).parent
    return digest({'dependencies': dependency_hash(),
                   'files': {p.name: file_sha(p) for p in sorted([*package.glob('*.py'), package / 'run.sh'])}})
