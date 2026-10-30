from __future__ import annotations

from pathlib import Path

from video_count_probe.core import atomic_json, checked_plan, digest, file_sha, read_json, writer_lock

ROOT = Path('/data/zjr/video_count_wan22_action_20261028')
MODEL = Path('/data/zjr/models/Wan2.2-TI2V-5B-Diffusers')
MODEL_ID = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
REFERENCE = Path('/data/zjr/video_count_action_calibration_20261027/pilot')
NEGATIVE = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
PROMPTS = {
    'clap': 'A person in a blue shirt is clapping their hands. Fixed camera, both hands clearly visible.',
    'jumping_jack': 'A person in a green shirt is doing jumping jacks. Fixed camera, the whole body and both feet clearly visible.',
    'squat': 'A person in a red shirt is doing bodyweight squats. Fixed camera, the whole body and both feet clearly visible.',
}
CRITERIA = {
    'clap': 'One clap = visible palm contact followed by separation. Approach alone is not a cycle.',
    'jumping_jack': 'Closed stance with arms down -> feet apart with arms overhead -> closed stance with arms down. Half cycles do not count.',
    'squat': 'Upright -> visibly lowered hips and bent knees -> upright. Incomplete descents or rises do not count.',
}


def under_data(path):
    path = Path(path).resolve()
    if not path.is_relative_to('/data/zjr'):
        raise ValueError(f'Expected storage under /data/zjr/: {path}')
    return path


def code_hash():
    package = Path(__file__).parent
    shared = package.parent
    files = list(package.glob('*.py')) + [package / 'run.sh']
    files += [shared / n for n in ('core.py', 'cli.py', 'downloads.py', 'runtime.py', 'evaluation.py')]
    return digest({str(p.relative_to(shared)): file_sha(p) for p in sorted(files)})


def reference_binding(reference):
    reference = Path(reference).resolve()
    old = checked_plan(reference / 'plan.json')
    cases = {c['id']: c for c in old['cases']}
    for action, prompt in PROMPTS.items():
        if cases[action]['prompts']['short_action'] != prompt:
            raise ValueError('Previous short prompts changed')
    return {'root': str(reference), 'plan_hash': old['plan_hash'],
            'comparison': 'Same six action/seed labels and short prompts; not identical latent noise or a model-only causal contrast.',
            'changed_configuration': ['model and VAE', '704x1280 vs 480x832', '121 frames at 24 fps vs 81 at 16 fps',
                                      'CFG 5 vs 6', 'flow shift 5 vs 8', 'official negative prompt', 'CPU text encoding']}


def build_plan(mode, reference=REFERENCE):
    if mode not in ('smoke', 'pilot'):
        raise ValueError(mode)
    config = {'model_id': MODEL_ID, 'height': 704, 'width': 1280, 'frames': 121, 'fps': 24,
              'steps': 2 if mode == 'smoke' else 50, 'guidance_scale': 5.0, 'flow_shift': 5.0,
              'max_sequence_length': 512, 'negative_prompt': NEGATIVE,
              'transformer_dtype': 'bfloat16', 'vae_dtype': 'float32', 'text_dtype': 'bfloat16',
              'scheduler': 'UniPCMultistepScheduler', 'expand_timesteps': True,
              'text_encoding': 'CPU; cached before GPU allocation', 'offload': 'model_cpu_offload',
              'vae_tiling': True, 'vae_tiling_compat': 'local 0.35.1 first_chunk + unpatchify + clamp fix',
              'quantization': None, 'prompt_rewrite': False, 'input_image': None}
    cases = [{'id': a, 'action': a, 'prompt': p, 'criterion': CRITERIA[a]} for a, p in PROMPTS.items()]
    jobs = []
    for action in PROMPTS:
        for seed in (42, 123):
            job = {'case_id': action, 'seed': seed, 'method': 'short_action'}
            job['id'] = digest({'experiment': 'wan22_action_v1', 'mode': mode, **job})[:16]
            jobs.append(job)
    if mode == 'smoke':
        jobs = jobs[:1]
    plan = {'schema': 1, 'mode': mode, 'scientific_evaluation': mode == 'pilot', 'config': config,
            'cases': cases, 'jobs': jobs, 'reference': reference_binding(reference),
            'metric': 'At least one visibly complete requested action cycle; no exact count target.',
            'limits': ['Only six formal videos; no automatic count-control expansion or training.',
                       'Same seed does not mean matched initial noise across different VAEs/models.',
                       'Native model configuration changes several variables; comparison is descriptive.',
                       'Smoke uses full geometry and two steps only; no quality inference from smoke.',
                       'Review provenance must identify AI versus human; no conclusion from generation success alone.']}
    plan['plan_hash'] = digest(plan)
    return plan


def prepare(root, reference=REFERENCE):
    for mode in ('smoke', 'pilot'):
        plan = build_plan(mode, reference)
        run = Path(root) / mode
        with writer_lock(run / '.writer.lock'):
            path = run / 'plan.json'
            if path.exists() and read_json(path) != plan:
                raise RuntimeError('Existing plan differs; use a new --root')
            if not path.exists():
                atomic_json(path, plan)
        print(f'{mode}: {len(plan["jobs"])} videos; {run / "plan.json"}', flush=True)


def verify_reference(plan):
    if reference_binding(plan['reference']['root']) != plan['reference']:
        raise ValueError('Reference plan changed')
