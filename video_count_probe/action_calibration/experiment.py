"""Twelve uncounted clips, with frozen counted clips as descriptive references.

No downloads or changes to the original generation sources/results. The smoke
clip tests execution only; the pilot is a separate long-running stage.
"""
from __future__ import annotations

import argparse
import datetime
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

from video_count_probe.core import (
    atomic_json, checked_plan, digest, file_sha, is_complete, read_json,
    source_hash, writer_lock,
)

DEFAULT_ROOT = '/data/zjr/video_count_action_calibration_20261027'
DEFAULT_REFERENCE = '/data/zjr/video_count_probe_20261025/pilot'
DEFAULT_MODEL = '/data/zjr/models/Wan2.1-T2V-1.3B-Diffusers'
METHODS = ('short_action', 'uncounted_matched')
SHORT_PROMPTS = {
    'clap': 'A person in a blue shirt is clapping their hands. Fixed camera, both hands clearly visible.',
    'jumping_jack': 'A person in a green shirt is doing jumping jacks. Fixed camera, the whole body and both feet clearly visible.',
    'squat': 'A person in a red shirt is doing bodyweight squats. Fixed camera, the whole body and both feet clearly visible.',
}


def fingerprint():
    return digest({p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def configure_storage(root):
    root = Path(root).resolve()
    if not root.is_relative_to('/data/zjr'):
        raise ValueError('Outputs and caches must be under /data/zjr/')
    for key, name in {
        'TMPDIR': 'tmp', 'TORCH_HOME': 'torch', 'HF_HOME': 'huggingface',
        'HF_HUB_CACHE': 'huggingface/hub', 'TORCH_EXTENSIONS_DIR': 'extensions',
        'MPLCONFIGDIR': 'matplotlib', 'XDG_CACHE_HOME': 'xdg',
    }.items():
        path = root / 'cache' / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    # tempfile may already have been imported by a dependency.
    import tempfile
    tempfile.tempdir = os.environ['TMPDIR']


def reference_state(reference):
    from video_count_probe.evaluation import completed_records
    reference = Path(reference).resolve()
    plan = checked_plan(reference / 'plan.json')
    records, errors = completed_records(reference, plan)
    if plan['mode'] != 'pilot' or errors or len(records) != 96:
        raise ValueError('The reference must be the verified 96-video pilot')
    identity = read_json(reference / 'identity.json')
    if source_hash() != identity['source_hash']:
        raise ValueError('Original generation source changed; preserve the frozen pilot implementation')
    annotation_path = reference / 'ai_review_20261027/annotations_ai.json'
    annotations = read_json(annotation_path)
    if annotations['plan_hash'] != plan['plan_hash'] or annotations['annotation_source'] != 'ai_visual_frame_review':
        raise ValueError('Expected the explicitly labelled pilot AI annotations')
    return plan, records, identity, annotation_path, annotations


def build_plan(reference, mode):
    if mode not in ('smoke', 'pilot'):
        raise ValueError('Expected smoke or pilot')
    old, records, identity, annotation_path, _ = reference_state(reference)
    cases = []
    for action in SHORT_PROMPTS:
        old_cases = [c for c in old['cases'] if c['action'] == action and c['schedule'] == 'regular']
        stripped = [c['count_prompt'].replace(f"exactly {c['target_count']} ", '', 1) for c in old_cases]
        if len(set(stripped)) != 1:
            raise ValueError('Count removal must yield the same matched prompt for all three targets')
        cases.append({'id': action, 'action': action, 'criterion': old_cases[0]['criterion'],
                      'prompts': {'short_action': SHORT_PROMPTS[action], 'uncounted_matched': stripped[0]}})
    config = dict(old['config'])
    if mode == 'smoke':
        config.update(height=240, width=416, steps=4)
    jobs = []
    for case in cases:
        for seed in (42, 123):
            methods = list(METHODS)
            random.Random(case['id'] + str(seed) + 'uncounted').shuffle(methods)
            for method in methods:
                job = {'case_id': case['id'], 'seed': seed, 'method': method}
                job['id'] = digest({'experiment': 'action_calibration_v1', 'mode': mode, **job})[:16]
                jobs.append(job)
    if mode == 'smoke':
        jobs = [j for j in jobs if j['case_id'] == 'clap' and j['seed'] == 42 and j['method'] == 'short_action']
    reference_jobs = [j for j in old['jobs'] if j['method'] == 'count_prompt' and j['case_id'].endswith('_regular')]
    plan = {
        'schema': 1, 'mode': mode, 'scientific_evaluation': mode == 'pilot', 'config': config,
        'cases': cases, 'jobs': jobs, 'methods': list(METHODS),
        'reference': {'root': str(Path(reference).resolve()), 'plan_hash': old['plan_hash'],
                      'identity': identity, 'annotations': str(annotation_path),
                      'annotations_sha256': file_sha(annotation_path),
                      'counted_records': {j['id']: {'job': j, 'record_sha256': file_sha(Path(reference) / 'records' / (j['id'] + '.json')),
                                                  'video_sha256': records[j['id']]['video_sha256'],
                                                  'initial_latent_sha256': records[j['id']]['initial_latent_sha256']}
                                          for j in reference_jobs}},
        'primary_question': 'Does the requested action have at least one visibly complete cycle, without an exact count target?',
        'contrasts': {
            'uncounted_matched_vs_counted': 'Only the phrase exactly N is removed from each original count_prompt; scene, cycle definition and all generation settings remain fixed.',
            'short_vs_matched': 'Tests a simpler prompt package. It changes scene wording and cycle explanation together, so an effect cannot be attributed to prompt length alone.',
        },
        'analysis_unit': 'Six action/seed units. One uncounted clip is not three independent samples when compared with targets 1/2/3.',
        'limitations': ['Descriptive capability check; two seeds per action and one model.',
                        'Complete-cycle presence is different from exact-count correctness.',
                        'No routing or training; do not infer the cause of failure from one contrast.',
                        'Prior reference annotations are AI reviews, not validated human ground truth.',
                        'Smoke uses four steps and is excluded from capability analysis.'],
    }
    plan['plan_hash'] = digest(plan)
    return plan


def prepare(root, reference):
    for mode in ('smoke', 'pilot'):
        plan = build_plan(reference, mode)
        run = Path(root) / mode
        with writer_lock(run / '.writer.lock'):
            path = run / 'plan.json'
            if path.exists() and read_json(path) != plan:
                raise RuntimeError(f'Frozen plan differs: {path}; use a new output root')
            if not path.exists():
                atomic_json(path, plan)
        print(f'{mode}: {len(plan["jobs"])} new videos; {run / "plan.json"}', flush=True)
    print('Pilot reuses 18 counted reference videos; no reference videos will be regenerated.', flush=True)


def verify_reference(plan):
    reference = plan['reference']
    old, records, identity, annotation_path, annotations = reference_state(reference['root'])
    if old['plan_hash'] != reference['plan_hash'] or identity != reference['identity']:
        raise ValueError('Reference identity changed')
    if file_sha(annotation_path) != reference['annotations_sha256']:
        raise ValueError('Reference annotations changed after plan freeze')
    for job_id, expected in reference['counted_records'].items():
        if file_sha(Path(reference['root']) / 'records' / (job_id + '.json')) != expected['record_sha256']:
            raise ValueError('Reference record changed')
    return old, records, annotations


def current_identity(plan, model_dir):
    from video_count_probe.runtime import model_receipt, versions
    receipt, packages = model_receipt(model_dir), versions()
    reference = plan['reference']['identity']
    if digest(receipt) != reference['model_receipt_hash'] or packages != reference['packages']:
        raise RuntimeError('Model receipt or packages differ from the counted reference pilot')
    if source_hash() != reference['source_hash']:
        raise RuntimeError('Shared generation code differs from the reference')
    return {'plan_hash': plan['plan_hash'], 'generation_source_hash': source_hash(),
            'calibration_source_hash': fingerprint(), 'model_receipt_hash': digest(receipt), 'packages': packages}


def check(root, model_dir):
    print('Checking pinned inputs and importing Torch / tokenizer ...', flush=True)
    faulthandler.dump_traceback_later(30, repeat=False)
    try:
        from transformers import T5TokenizerFast
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from diffusers import WanPipeline
        from types import SimpleNamespace
        import torch
        from video_count_probe.runtime import model_receipt
        plan = checked_plan(Path(root) / 'pilot/plan.json')
        verify_reference(plan)
        identity = current_identity(plan, model_dir)
        tokenizer = T5TokenizerFast.from_pretrained(model_dir, subfolder='tokenizer', local_files_only=True)
        lengths = {}
        for case in plan['cases']:
            for method, prompt in case['prompts'].items():
                if prompt_clean(prompt) != prompt:
                    raise ValueError('Prompt cleaning unexpectedly alters the prompt')
                n = len(tokenizer(prompt, add_special_tokens=True, truncation=False).input_ids)
                if n > 512:
                    raise ValueError('Prompt would be truncated')
                lengths[case['id'] + '/' + method] = n
        # Exercise the real Diffusers noise constructor on CPU, without weights.
        vae_config = read_json(Path(model_dir) / 'vae/config.json')
        transformer_config = read_json(Path(model_dir) / 'transformer/config.json')
        geometry = SimpleNamespace(vae_scale_factor_temporal=vae_config.get('scale_factor_temporal', 4),
                                   vae_scale_factor_spatial=vae_config.get('scale_factor_spatial', 8))
        noise_hashes = {}
        config = plan['config']
        for seed in (42, 123):
            latents = WanPipeline.prepare_latents(geometry, 1, transformer_config['in_channels'],
                                                  config['height'], config['width'], config['frames'],
                                                  torch.float32, torch.device('cpu'), torch.Generator(device='cpu').manual_seed(seed))
            sha = hashlib.sha256(latents.numpy().tobytes()).hexdigest()
            expected = {r['initial_latent_sha256'] for r in plan['reference']['counted_records'].values() if r['job']['seed'] == seed}
            if expected != {sha}:
                raise ValueError('CPU initial-noise check differs from reference')
            noise_hashes[str(seed)] = sha
        result = {'passed': True, 'model_revision': model_receipt(model_dir)['revision'],
                  'identity': identity, 'prompt_tokens': lengths, 'paired_initial_noise_sha256': noise_hashes,
                  'gpu_generation_started': False}
        atomic_json(Path(root) / 'check.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


def generate(root, mode, model_dir, limit=None, retry_failures=False, allow_busy=False):
    run = Path(root) / mode
    plan = checked_plan(run / 'plan.json')
    verify_reference(plan)
    identity = current_identity(plan, model_dir)
    config = plan['config']
    with writer_lock(run / '.writer.lock'):
        identity_path = run / 'identity.json'
        if identity_path.exists() and read_json(identity_path) != identity:
            raise RuntimeError('Run code/model/environment changed; use a new output root')
        pending = [j for j in plan['jobs'] if not is_complete(run, j, identity)]
        if limit is not None:
            if limit < 1:
                raise ValueError('--limit must be positive')
            pending = pending[:limit]
        if not pending:
            print('All requested videos already verified complete; model will not load.', flush=True)
            return
        for job in pending:
            error = run / 'errors' / (job['id'] + '.json')
            if error.exists() and not retry_failures:
                raise RuntimeError(f'Previous failure: {error}; inspect and use --retry-failures')
        from video_count_probe.runtime import check_gpu, encode_cached, load_pipe
        import torch
        import numpy as np
        from diffusers.utils import export_to_video
        gpu = check_gpu(allow_busy=allow_busy)
        # Use the original probe's shared GPU lock to prevent overlapping writers.
        with writer_lock(Path('/tmp') / ('video_count_probe_' + gpu['uuid'] + '.lock')):
            atomic_json(identity_path, identity)
            print(f'Loading model; {len(pending)} pending videos; output={run}', flush=True)
            torch.cuda.reset_peak_memory_stats()
            pipe = load_pipe(model_dir, config)
            cache = Path(root) / 'text_cache'
            negative = encode_cached(pipe, config['negative_prompt'], cache, identity)
            # Keep a process-level high-water mark across the per-generation
            # resets below, including model loading and text encoding.
            run_peak_allocated = torch.cuda.max_memory_allocated()
            run_peak_reserved = torch.cuda.max_memory_reserved()
            cases = {c['id']: c for c in plan['cases']}
            for index, job in enumerate(pending, 1):
                started = time.perf_counter()
                video = run / 'videos' / (job['id'] + '.mp4')
                partial = video.with_suffix('.partial.mp4')
                print(f'[{index}/{len(pending)}] {job["case_id"]} seed={job["seed"]} {job["method"]}', flush=True)
                try:
                    prompt = cases[job['case_id']]['prompts'][job['method']]
                    embeds = encode_cached(pipe, prompt, cache, identity)
                    generator = torch.Generator(device='cpu').manual_seed(job['seed'])
                    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels, config['height'], config['width'],
                                                   config['frames'], torch.float32, torch.device('cpu'), generator)
                    noise_sha = hashlib.sha256(latents.numpy().tobytes()).hexdigest()
                    if mode == 'pilot':
                        expected = {r['initial_latent_sha256'] for r in plan['reference']['counted_records'].values()
                                    if r['job']['seed'] == job['seed'] and r['job']['case_id'].startswith(job['case_id'] + '_n')}
                        if expected != {noise_sha}:
                            raise RuntimeError('Initial noise differs from the counted references')
                    torch.cuda.synchronize()
                    run_peak_allocated = max(run_peak_allocated, torch.cuda.max_memory_allocated())
                    run_peak_reserved = max(run_peak_reserved, torch.cuda.max_memory_reserved())
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    generation_start = time.perf_counter()
                    output = pipe(prompt_embeds=embeds, negative_prompt_embeds=negative,
                                  height=config['height'], width=config['width'], num_frames=config['frames'],
                                  num_inference_steps=config['steps'], guidance_scale=config['guidance_scale'],
                                  latents=latents, generator=generator, output_type='np').frames[0]
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - generation_start
                    if output.shape[0] != config['frames'] or not np.isfinite(output).all():
                        raise RuntimeError('Invalid decoded video')
                    video.parent.mkdir(parents=True, exist_ok=True)
                    export_to_video(output, str(partial), fps=config['fps'], macro_block_size=1)
                    os.replace(partial, video)
                    run_peak_allocated = max(run_peak_allocated, torch.cuda.max_memory_allocated())
                    run_peak_reserved = max(run_peak_reserved, torch.cuda.max_memory_reserved())
                    record = {'status': 'ok', 'job': job, 'identity': identity, 'prompt': prompt,
                              'initial_latent_sha256': noise_sha, 'routing': None, 'routed_calls': 0,
                              'video': str(video.relative_to(run)), 'video_sha256': file_sha(video),
                              'generation_seconds': elapsed, 'total_seconds': time.perf_counter() - started,
                              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                              'process_peak_allocated_mib': run_peak_allocated / 2**20,
                              'process_peak_reserved_mib': run_peak_reserved / 2**20,
                              'memory_scope': 'PyTorch allocator; process peaks include loading and prompt encoding through this video, exclude non-PyTorch CUDA allocations',
                              'gpu': gpu, 'frames': config['frames'], 'fps': config['fps']}
                    atomic_json(run / 'records' / (job['id'] + '.json'), record)
                    (run / 'errors' / (job['id'] + '.json')).unlink(missing_ok=True)
                    print(f'ok {job["id"]} generation={elapsed:.1f}s peak={record["peak_allocated_mib"]:.0f}MiB', flush=True)
                except BaseException as exc:
                    partial.unlink(missing_ok=True)
                    atomic_json(run / 'errors' / (job['id'] + '.json'),
                                {'job': job, 'identity': identity, 'exception': repr(exc), 'traceback': traceback.format_exc()})
                    raise


def ability_score(annotation):
    """Ambiguous exact count can still establish at least one complete cycle."""
    if not annotation['visually_usable']:
        return 'unusable'
    confirmed = annotation.get('complete_cycle_observed')
    if confirmed is not None:
        if not isinstance(confirmed, bool):
            raise ValueError('complete_cycle_observed must be bool or null')
        if confirmed and not annotation['action_present']:
            raise ValueError('Cannot confirm a cycle of an absent action')
        if not confirmed and not annotation['ambiguous'] and annotation['completion_times']:
            raise ValueError('Confirmed zero cycles conflicts with completion timestamps')
        if confirmed and annotation.get('count_range') is not None and annotation['count_range'][1] == 0:
            raise ValueError('Confirmed cycle conflicts with a zero count range')
        return 'complete_cycle' if confirmed else 'no_complete_cycle'
    if not annotation['ambiguous']:
        return 'complete_cycle' if annotation['completion_times'] else 'no_complete_cycle'
    bounds = annotation.get('count_range')
    if bounds is not None:
        if len(bounds) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in bounds) or bounds[0] > bounds[1]:
            raise ValueError('Invalid count range')
        if bounds[0] >= 1:
            if not annotation['action_present']:
                raise ValueError('A positive count range requires an action')
            return 'complete_cycle'
        if bounds[1] == 0:
            return 'no_complete_cycle'
    return 'uncertain'


def aggregate(rows):
    return {'n': len(rows), **{key: sum(r['ability'] == key for r in rows)
                             for key in ('complete_cycle', 'no_complete_cycle', 'uncertain', 'unusable')}}


def report(root, mode, annotation_paths=(), *, write=True):
    from video_count_probe.evaluation import completed_records, validate_annotation
    run = Path(root) / mode
    plan = checked_plan(run / 'plan.json')
    old, old_records, old_annotations = verify_reference(plan)
    records, errors = completed_records(run, plan)
    cases = {c['id']: c for c in plan['cases']}
    labels = {}
    for path in annotation_paths:
        payload = read_json(path)
        if payload['plan_hash'] != plan['plan_hash']:
            raise ValueError('Annotations belong to another plan')
        source = payload.get('annotation_source', 'unspecified')
        for annotation in payload['annotations']:
            if annotation.get('complete') is not True:
                continue
            job_id = annotation['job_id']
            if job_id in labels or job_id not in records:
                raise ValueError('Duplicate annotation or annotation without a verified video')
            case = cases[records[job_id]['job']['case_id']]
            validate_annotation(annotation, records[job_id], case, plan)
            labels[job_id] = {**annotation, 'annotation_source': source}
    rows = []
    for job_id, annotation in labels.items():
        job = records[job_id]['job']
        rows.append({**job, 'action': job['case_id'], 'ability': ability_score(annotation),
                     'annotation_source': annotation['annotation_source'], 'notes': annotation.get('notes', '')})
    old_cases = {c['id']: c for c in old['cases']}
    for annotation in old_annotations['annotations']:
        if annotation['job_id'] not in plan['reference']['counted_records']:
            continue
        record = old_records[annotation['job_id']]
        job = record['job']; case = old_cases[job['case_id']]
        validate_annotation(annotation, record, case, old)
        rows.append({**job, 'action': case['action'], 'method': f"count_n{case['target_count']}",
                     'ability': ability_score(annotation), 'annotation_source': 'ai_visual_frame_review',
                     'notes': annotation.get('notes', ''), 'reference_only': True})
    groups = {m: aggregate([r for r in rows if r['method'] == m]) for m in (*METHODS, 'count_n1', 'count_n2', 'count_n3')}
    result = {'plan_hash': plan['plan_hash'], 'expected': len(plan['jobs']), 'generated': len(records),
              'missing': len(plan['jobs']) - len(records), 'execution_errors': len(errors), 'annotated': len(labels),
              'assessment_complete': mode == 'pilot' and len(records) == len(labels) == len(plan['jobs']),
              'human_validated': False, 'reference_annotation_source': 'AI frame review',
              'metric': 'At least one visibly complete action cycle, not exact-count correctness',
              'methods': groups, 'per_video': rows, 'limitations': plan['limitations'],
              'interpretation': 'Six action/seed units per prompt. No automatic train/abandon threshold.'}
    if write:
        with writer_lock(run / '.writer.lock'):
            atomic_json(run / 'summary.json', result)
    print(f"generated={len(records)}/{len(plan['jobs'])} annotated={len(labels)} errors={len(errors)} assessment_complete={result['assessment_complete']}; {run/'summary.json'}", flush=True)
    if labels:
        print(json.dumps(groups, ensure_ascii=False, indent=2))
    return result


def review(root, mode):
    from video_count_probe.evaluation import completed_records, make_review
    from video_count_probe.review_tools.review_frames import sheet
    from PIL import Image
    import imageio_ffmpeg
    run = Path(root) / mode
    plan = checked_plan(run / 'plan.json')
    records, _ = completed_records(run, plan)
    if not records:
        return
    jobs = list(plan['jobs'])
    random.Random(271026).shuffle(jobs)
    cases = {c['id']: c for c in plan['cases']}
    config = plan['config']; width, height = config['width'], config['height']
    stride = width * height * 3
    items, mapping, template = [], [], []
    with writer_lock(run / '.writer.lock'):
        make_review(run)
        for blind_id, job in enumerate(jobs, 1):
            if job['id'] not in records:
                continue
            record = records[job['id']]
            destination = run / 'review_frames' / f'{blind_id:03d}.jpg'
            stamp = destination.with_suffix('.json')
            if not destination.exists() or not stamp.exists() or read_json(stamp)['video_sha256'] != record['video_sha256']:
                process = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-threads', '1', '-i', str(run / record['video']),
                                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', '-'], capture_output=True, check=True)
                if len(process.stdout) != config['frames'] * stride:
                    raise ValueError('Wrong decoded frame dimensions')
                images = [Image.frombytes('RGB', (width, height), process.stdout[i*stride:(i+1)*stride]) for i in range(config['frames'])]
                sheet(images, list(range(0, config['frames'], 2)), destination,
                      f"Blind {blind_id:03d} | action={job['case_id']} | 8 fps overview, original 16 fps")
                atomic_json(stamp, {'video_sha256': record['video_sha256']})
            items.append({'blind_id': blind_id, 'action': job['case_id'], 'criterion': cases[job['case_id']]['criterion'], 'sheet': str(destination)})
            mapping.append({'blind_id': blind_id, 'job_id': job['id'], 'video_sha256': record['video_sha256']})
            template.append({'job_id': job['id'], 'video_sha256': record['video_sha256'], 'complete': False,
                             'rater': '', 'visually_usable': None, 'action_present': None, 'ambiguous': None,
                             'complete_cycle_observed': None, 'completion_times': [], 'notes': ''})
        atomic_json(run / 'review_frames/blind_manifest.json', {'plan_hash': plan['plan_hash'], 'items': items})
        atomic_json(run / 'review_frames/private_mapping.json', {'plan_hash': plan['plan_hash'], 'items': mapping})
        atomic_json(run / 'review_frames/annotation_template.json', {'plan_hash': plan['plan_hash'], 'annotation_source': 'ai_visual_frame_review', 'annotations': template})
    print(f'Review frames ready: {run / "review_frames/blind_manifest.json"}', flush=True)
    print('Codex can review these frames after generation; human annotation is optional for this feasibility check.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'test', 'smoke', 'pilot', 'status', 'review', 'report'])
    parser.add_argument('--root', default=os.environ.get('VIDEO_ACTION_ROOT', DEFAULT_ROOT))
    parser.add_argument('--reference', default=os.environ.get('VIDEO_ACTION_REFERENCE', DEFAULT_REFERENCE))
    parser.add_argument('--model-dir', default=os.environ.get('VIDEO_COUNT_MODEL_DIR', DEFAULT_MODEL))
    parser.add_argument('--mode', choices=['pilot', 'smoke'], default='pilot')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-failures', action='store_true')
    parser.add_argument('--allow-busy', action='store_true',
                        help='Permit sharing the selected GPU; still require at least 16000 MiB free before loading')
    parser.add_argument('--annotations', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    configure_storage(args.root)
    faulthandler.enable()
    def terminate(signum, _):
        print(json.dumps({'event': 'termination_signal', 'signal': signal.Signals(signum).name,
                          'pid': os.getpid(), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}), flush=True)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    from video_count_probe.cli import stage_log
    with stage_log(args.root, 'action_' + args.command):
        if args.command == 'prepare':
            prepare(args.root, args.reference)
        elif args.command == 'test':
            import unittest
            suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / 'tests'))
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful():
                raise SystemExit(1)
        elif args.command == 'check':
            check(args.root, args.model_dir)
        elif args.command in ('smoke', 'pilot'):
            prepare(args.root, args.reference)
            generate(args.root, args.command, args.model_dir, args.limit, args.retry_failures, args.allow_busy)
            report(args.root, args.command, args.annotations)
            review(args.root, args.command)
        elif args.command == 'review':
            review(args.root, args.mode)
        else:
            report(args.root, args.mode, args.annotations, write=args.command != 'status')


if __name__ == '__main__':
    main()
