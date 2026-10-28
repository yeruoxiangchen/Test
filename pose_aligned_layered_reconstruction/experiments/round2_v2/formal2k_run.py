"""Freeze, supervise and monitor the single-GPU formal 2K comparison.

This process never initializes CUDA. Exactly one child performs data preparation
or training at a time. Run it in an independent systemd user service.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

PACKAGE = Path(__file__).resolve().parents[2]
WORKSPACE = PACKAGE.parent
PYTHON = '/home/zjr/anaconda3/envs/reconviagen/bin/python'
DEFAULT_RUN = PACKAGE / 'outputs/round2_v2_formal2k_20261018'
PROTOCOL = Path(__file__).with_name('FORMAL2K_PROTOCOL_20261018.md')
SERVICE = 'tracker-palr-v2-formal2k-20261018.service'
PREFIX = 'pose_aligned_layered_reconstruction.'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def journal(root, event, **fields):
    row = {'at_utc': utc(), 'event': event, **fields}
    with (root / 'journal.jsonl').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def verify_smoke(path):
    path = Path(path).resolve(strict=True)
    report = read(path)
    protocol_path = path.with_name('protocol.json')
    if not report.get('passed') or report.get('protocol_sha256') != sha(protocol_path):
        raise ValueError('A successful, bound real-GPU smoke report is required')
    protocol = read(protocol_path)
    if set(protocol['condition_modes']) != {'full', 'single_layer', 'no_layer'}:
        raise ValueError('Smoke must cover all three control modes')
    results = report['results']
    if (len(results) != 3 or {r['condition_mode'] for r in results} != set(protocol['condition_modes'])
        or not all(r.get('passed') and r['steps'] >= 3 for r in results)):
        raise ValueError('All three real-GPU control arms must pass at least three updates')
    for source, expected in protocol['code_sha256'].items():
        if sha(source) != expected:
            raise ValueError('Source changed since real-GPU acceptance: ' + source)
    return {'report': bind(path), 'protocol': bind(protocol_path)}


def freeze(args):
    root = Path(args.run_dir).resolve()
    if (root / 'protocol.json').exists():
        raise FileExistsError('Protocol already frozen; use run or monitor, never overwrite it.')
    data = read(root / 'data/plan.json')
    audit = read(root / 'data/clean_cache_audit.json')
    if (data['plan']['object_count'] != 2000 or data['plan']['variants'] != 2
        or data['plan']['final_sample_count'] != 6000 or not audit['passed'] or audit['count'] != 2064):
        raise ValueError('Complete 2K + Dev64 input audit is required before freeze')
    smoke_path = Path(args.smoke_report).resolve(strict=True)
    smoke = verify_smoke(smoke_path)
    sample_count = data['plan']['final_sample_count']
    train_manifest = root / f'data/train{sample_count}.json'
    geometry_checkpoint = root / 'lari/checkpoints/step_012000.pt'
    common = ['--manifest', str(train_manifest), '--grad_accum', '2', '--learning_rate', '0.0001',
              '--warmup_steps', '100', '--save_every', '500', '--validate_every', '500',
              '--seed', '20261018', '--amp', 'bf16', '--p_uncond', '0',
              '--lr_schedule', 'cosine', '--t_sampling', 'uniform',
              '--geometry_config', str(Path(__file__).with_name('configs') / 'geometry.json')]
    stages = []
    if data['plan']['variants']:
        stages.append({'name': 'prepare_occlusion', 'kind': 'data', 'expected': data['plan']['occluded_count'],
            'command': [PYTHON, '-u', '-m', PREFIX + 'experiments.round2_v2.formal2k_data',
                        'augment', '--output-dir', str(root / 'data'), '--device', 'cuda:0']})
    stages.append({'name': 'lari', 'kind': 'train', 'expected': 12000, 'sample_exposures': 24000,
        'command': [PYTHON, '-u', '-m', PREFIX + 'v2.train', '--stage', 'lari', *common,
                    '--max_steps', '12000', '--output_dir', str(root / 'lari'),
                    '--val_manifest', str(root / 'data/dev64.json'), '--val_samples', '8']})
    for mode in ('full', 'no_layer', 'single_layer'):
        name = 'ss_' + mode
        stages.append({'name': name, 'kind': 'train', 'condition_mode': mode,
            'expected': 9000, 'sample_exposures': 18000,
            'command': [PYTHON, '-u', '-m', PREFIX + 'v2.train', '--stage', 'ss', *common,
                        '--max_steps', '9000', '--output_dir', str(root / name),
                        '--init', str(geometry_checkpoint), '--init_weights', 'raw',
                        '--freeze_geometry', '--condition_mode', mode,
                        '--ss_absolute_tolerance', '0.02']})
    source_files = []
    for directory, children, files in os.walk(PACKAGE):
        children[:] = [name for name in children if name not in {'outputs', '__pycache__', '.git'}]
        source_files.extend(Path(directory) / name for name in files if name.endswith('.py'))
    source_files += [Path(__file__).with_name('configs') / 'geometry.json', PROTOCOL]
    sources = []
    for path in sorted(source_files):
        record = bind(path)
        snapshot = root / 'source_snapshot' / path.relative_to(PACKAGE)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
        record['snapshot'] = str(snapshot)
        sources.append(record)
    value = {
        'format': 'palr_formal2k_protocol.v1', 'frozen_at_utc': utc(), 'gpu': args.gpu,
        'world_size': 1, 'concurrent_gpu_children': 1, 'service': SERVICE,
        'data_plan': bind(root / 'data/plan.json'), 'data_identity': data['identity_hash'],
        'clean_audit': bind(root / 'data/clean_cache_audit.json'),
        'smoke_report': smoke['report'], 'smoke_protocol': smoke['protocol'],
        'protocol_document': bind(PROTOCOL),
        'source_code': sources, 'stages': stages,
        'training_manifest': str(train_manifest), 'sample_count': sample_count,
        'object_count': 2000, 'inference_weights': 'raw', 'final_checkpoint_only': True,
        'parent_reference': 'Unmodified SS30K EMA; frozen SLat30K EMA and decoder for every evaluation arm.',
        'comparison': {
            'full': 'Five predicted scene intersections from one frozen shared geometry frontend.',
            'single_layer': 'First-scene-intersection readout from the SAME frozen multilayer frontend; not a separately trained depth model.',
            'no_layer': 'Trainable raw/masked feature and visible-mask projection at known K/T queries, without predicted layer fields or layer supervision.',
            'matching': 'Same SS sample order, eight views, diffusion seed, updates, accumulation, optimizer and parent. Actual parameter/compute differences are reported.',
            'shared_frontend_training': 'Charged to full/single_layer; no_layer never consumes its outputs. Total compute is not claimed equal.',
        },
        'evaluation_protocol': {
            'status': 'frozen design; evaluation is a later stage, not automatically run by this training service',
            'development': 'Dev64 clean and pre-existing fixed Strong16; do not call either untouched final test.',
            'methods': ['parent', 'full', 'no_layer', 'single_layer'], 'expected_mesh_records': 640,
            'inference_seeds': [42, 43], 'ss_steps': 25, 'ss_cfg': 5,
            'ss_cfg_interval': [0.5, 1.0], 'ss_rescale_t': 3.0,
            'slat_sampler': {'steps': 25, 'cfg_strength': 5.0, 'cfg_interval': [0.5, 1.0],
                             'rescale_t': 3.0, 'guidance_rescale': 0.0},
            'slat_noise': 'seed*2000003+object_index*2017+7919, dense64cube, same values at shared coordinates',
            'surface_seed': 'seed*1009+object_index*9173; reload exported PLY for scores',
            'surface_points': 20000, 'fscore_thresholds': [0.01, 0.02, 0.05],
            'primary_frame': 'native; common object normalization and no per-mesh recentering/rescaling',
            'secondary_frame': 'GT-assisted proper SE3; no scale/reflection/deformation',
            'rigid_fit': {'registration_points': 4096, 'max_iterations': 35,
                'initialization': 'identity rotation with centroid translation plus 24 proper PCA-axis rotations',
                'candidate_selection': '3 lowest-registration-CD poses plus untouched identity; select by independent 20K Chamfer-L1',
                'registration_seed': '20261018+object_index*1009+seed*17; target seed +1'},
            'statistics': 'Average inference seeds within objects; paired object bootstrap 5000 draws with seed 20261018.',
            'confidence_intervals': '95% pairwise intervals; no family-wise significance claim without a separate correction',
            'exclusion': 'No reconstruction-score-based case exclusions; retain failures and complete cohort denominator.',
            'failed_evaluation': 'Retry identical parameters; if any scores remain unavailable, mark evaluation incomplete and report failures. Success-only statistics are diagnostic, not complete-cohort results.',
            'qualitative': 'Fixed cameras/material/scale; fixed effect-rank representatives and full cohort atlas.',
        },
        'evaluation_sources': [bind(PACKAGE / 'outputs/round2_v2_diagnostics_20261018/strong_data/manifest.json'),
                               bind(PACKAGE / 'outputs/round2_v2_heldout_mesh_20261018/alignment/protocol.json')],
        'scope': 'One fixed training seed for the first 2K mechanism comparison; repeated training and independent original-GT/AR evaluation remain future work.',
        'limitations': data['plan']['limitations'] + [
            'Fixed finite exposure budget is not proof of optimization convergence.',
            'Full formal model freezes geometry during SS; it differs from the prior joint-overfit V2.',
            'The no-layer visual projection has a small reported parameter-count difference.',
            'Strong-occlusion advantage is a hypothesis, not a frozen promised result.',
        ],
    }
    atomic(root / 'protocol.json', {'identity_hash': canonical(value), 'protocol': value})
    journal(root, 'protocol_frozen', identity_hash=canonical(value), gpu=args.gpu,
            train_objects=2000, train_samples=sample_count, stages=[s['name'] for s in stages])
    atomic(root / 'status.json', {'state': 'ready', 'updated_at_utc': utc(),
                               'identity_hash': canonical(value), 'gpu': args.gpu, 'current_stage': None,
                               'completed_stages': [], 'queue': [s['name'] for s in stages]})


def verify_protocol(root):
    envelope = read(root / 'protocol.json')
    plan = envelope['protocol']
    if canonical(plan) != envelope['identity_hash']:
        raise ValueError('Protocol hash mismatch')
    for record in [plan['data_plan'], plan['clean_audit'], plan['smoke_report'],
                   plan['smoke_protocol'], *plan['evaluation_sources'], *plan['source_code']]:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Frozen source changed: ' + record['path'])
    return plan, envelope['identity_hash']


def gpu_usage(index):
    result = subprocess.run(['nvidia-smi', f'--id={index}',
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    memory, utilization = (int(s.strip()) for s in result.stdout.strip().split(','))
    return {'memory_mib': memory, 'utilization_percent': utilization}


def tail_record(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        for line in reversed(stream.read().decode(errors='replace').splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def stage_progress(root, stage):
    if stage['kind'] == 'data':
        path = root / 'data/status.json'
        item = read(path) if path.exists() else {}
        item['progress'] = item.get('completed', item.get('reused_occlusion_count', 0))
        if item.get('state') == 'complete':
            item['progress'] = stage['expected']
    else:
        item = tail_record(root / stage['name'] / 'history.jsonl')
        item['progress'] = item.get('step', 0)
    item['expected'] = stage['expected']
    return item


def verify_stage_complete(root, stage, plan):
    if stage['kind'] == 'data':
        item = read(root / 'data/status.json')
        if (item['state'] != 'complete' or item['identity_hash'] != plan['data_identity']
            or item['sample_count'] != plan['sample_count'] or item['object_count'] != 2000
            or sha(item['manifest']['path']) != item['manifest']['sha256']):
            raise ValueError('Incomplete or mismatched data preparation')
        return {'manifest': item['manifest'], 'sample_count': item['sample_count']}
    folder = root / stage['name']
    item, config = read(folder / 'complete.json'), read(folder / 'run_config.json')
    # Compare the saved configuration with the actual frozen CLI, including
    # scalar types; --init is intentionally replaced by --resume on recovery.
    command = stage['command']
    for index, token in enumerate(command):
        if not token.startswith('--'):
            continue
        key = token[2:]
        if key == 'init' and config.get('resume'):
            continue
        expected = command[index + 1] if index + 1 < len(command) and not command[index + 1].startswith('--') else True
        actual = config.get(key)
        if isinstance(actual, (float, int)) and not isinstance(actual, bool) and expected is not True:
            matches = actual == float(expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f'Completed training option differs: {stage["name"]} {key}')
    if (item['step'] != stage['expected'] or config['max_steps'] != stage['expected']
        or item['data']['sha256'] != sha(plan['training_manifest'])
        or item['data'] != config['data_identity'] or item['data']['sample_count'] != plan['sample_count']
        or len(item['data']['object_uids']) != 2000 or config['effective_batch'] != 2
        or item['data']['world_size'] != 1 or config['max_samples'] != 0
        or config['cache_frozen_geometry'] or config['finetune_lora'] or config['no_gradient_checkpointing']):
        raise ValueError('Completed training identity/budget differs: ' + stage['name'])
    if stage['kind'] == 'train' and 'condition_mode' in stage:
        if config['condition_mode'] != stage['condition_mode'] or not config['freeze_geometry']:
            raise ValueError('Completed SS control mode differs')
        parent = config['bindings']['parent']
        expected_parent = read(plan['data_plan']['path'])['plan']['parent_audit']['ss30k']['checkpoint']
        if (parent['path'] != expected_parent['path'] or parent['sha256'] != expected_parent['sha256']
            or parent['weights'] != 'ema' or not parent['freeze_parent']):
            raise ValueError('SS parent differs from frozen data audit')
        smoke = read(plan['smoke_report']['path'])
        accepted = next(r for r in smoke['results'] if r['condition_mode'] == stage['condition_mode'])
        if config['trainable_parameter_count'] != accepted['trainable_parameter_count']:
            raise ValueError('SS trainable capacity differs from real-GPU acceptance')
        if set(config['init_adapter_hashes'] or {}) != {'layer_encoder', 'view_gate', 'layer_block_condition'}:
            raise ValueError('SS shared-adapter initial hashes are missing')
        if stage['name'] != 'ss_full':
            reference = read(root / 'ss_full/run_config.json')
            for key in ('data_identity', 'init_adapter_hashes'):
                if config[key] != reference[key]:
                    raise ValueError('SS arms differ in ' + key)
            if config['bindings']['stock'] != reference['bindings']['stock']:
                raise ValueError('SS arms differ in frozen stock model')
    return {'complete': bind(folder / 'complete.json'), 'run_config': bind(folder / 'run_config.json'),
            'checkpoint': bind(folder / 'checkpoints' / f"step_{stage['expected']:06d}.pt")}


def verify_receipts(root, plan, identity):
    receipts = {}
    for stage in plan['stages']:
        path = root / 'stage_receipts' / (stage['name'] + '.json')
        if not path.exists():
            continue
        envelope = read(path)
        receipt = envelope['receipt']
        if (canonical(receipt) != envelope['sha256'] or receipt['protocol_identity'] != identity
            or receipt['stage'] != stage['name']):
            raise ValueError('Stage receipt identity mismatch: ' + str(path))
        for record in receipt['artifacts'].values():
            if isinstance(record, dict) and 'path' in record and 'sha256' in record:
                if sha(record['path']) != record['sha256']:
                    raise ValueError('Sealed stage artifact changed: ' + record['path'])
        if stage['kind'] == 'data':
            if receipt['artifacts']['manifest']['path'] != plan['training_manifest']:
                raise ValueError('Data receipt points to a different training manifest')
        if stage['name'] == 'lari':
            for later in plan['stages']:
                if 'condition_mode' in later:
                    command = later['command']
                    if command[command.index('--init') + 1] != receipt['artifacts']['checkpoint']['path']:
                        raise ValueError('SS initialization differs from sealed LaRI checkpoint')
        receipts[stage['name']] = receipt
    return receipts


def seal_stage(root, stage, evidence, identity):
    path = root / 'stage_receipts' / (stage['name'] + '.json')
    receipt = {'protocol_identity': identity, 'stage': stage['name'],
               'sealed_at_utc': utc(), 'artifacts': evidence}
    # An exclusive hard link publishes a fully written receipt without ever
    # overwriting a previous seal, even after an interrupted supervisor.
    temporary = path.with_name(path.name + f'.pending-{os.getpid()}')
    atomic(temporary, {'sha256': canonical(receipt), 'receipt': receipt})
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return bind(path)


def run(args):
    root = Path(args.run_dir).resolve()
    plan, identity = verify_protocol(root)
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(plan['gpu']), V2_NPROC='1', WORLD_SIZE='1',
               CUDA_HOME='/usr/local/cuda-12.6', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONFAULTHANDLER='1',
               TORCH_CUDA_ARCH_LIST='8.6', SPCONV_ALGO='native',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               MPLCONFIGDIR=str(root / 'matplotlib_cache'))
    env['PATH'] = '/home/zjr/anaconda3/envs/reconviagen/bin:' + env.get('PATH', '')
    for key in ('RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    state = {'state': 'starting', 'identity_hash': identity, 'gpu': plan['gpu'],
             'supervisor_pid': os.getpid(), 'started_at_utc': utc(), 'completed_stages': [],
             'queue': [s['name'] for s in plan['stages']], 'current_stage': None, 'child_pid': None}
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def publish():
        state['updated_at_utc'] = utc()
        atomic(root / 'status.json', state)
    process = None
    journal(root, 'supervisor_started', pid=os.getpid(), gpu=plan['gpu'], resume=args.resume)
    try:
        for stage in plan['stages']:
            verify_protocol(root)
            receipts = verify_receipts(root, plan, identity)
            if stage['kind'] == 'train' and 'prepare_occlusion' not in receipts:
                raise ValueError('Training requires a sealed, complete data preparation receipt')
            if 'condition_mode' in stage and 'lari' not in receipts:
                raise ValueError('SS requires a sealed shared LaRI checkpoint')
            state.update(current_stage=stage['name'], state='checking_stage', child_pid=None)
            state['queue'] = state['queue'][1:]
            publish()
            completion = root / ('data/status.json' if stage['kind'] == 'data' else stage['name'] + '/complete.json')
            already_complete = completion.exists() and (stage['kind'] != 'data' or read(completion).get('state') == 'complete')
            if already_complete:
                if not args.resume:
                    raise ValueError('Completed stage already exists; explicit --resume required')
                if stage['name'] not in receipts:
                    raise ValueError('Completed stage has no immutable receipt; manual recovery audit required: ' + stage['name'])
                evidence = verify_stage_complete(root, stage, plan)
                state['completed_stages'].append({'name': stage['name'], 'reused_completed_stage': True, **evidence})
                publish()
                continue
            command = list(stage['command'])
            folder = root / stage['name']
            resume_checkpoint = None
            if stage['kind'] == 'train' and folder.exists():
                checkpoint = folder / 'checkpoints/last.pt'
                if not args.resume or not checkpoint.exists():
                    raise ValueError('Existing incomplete training needs explicit --resume and a checkpoint: ' + str(folder))
                if '--init' in command:
                    pos = command.index('--init')
                    del command[pos:pos+2]
                command += ['--resume', str(checkpoint)]
                resume_checkpoint = checkpoint
            idle = 0
            while idle < 2:
                if stop[0] or (root / 'STOP').exists():
                    raise InterruptedError('Stop requested before GPU stage')
                usage = gpu_usage(plan['gpu'])
                idle = idle + 1 if usage['memory_mib'] <= 64 and usage['utilization_percent'] <= 5 else 0
                state.update(state='waiting_for_fixed_gpu', last_gpu_usage=usage)
                publish()
                if idle < 2:
                    time.sleep(10)
            verify_protocol(root)
            verify_receipts(root, plan, identity)
            log = root / 'logs' / f"{stage['name']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            if resume_checkpoint is not None:
                resume_source = {'checkpoint': bind(resume_checkpoint),
                                 'previous_run_config': bind(folder / 'run_config.json')}
                launch_path = root / 'resume_sources' / f"{stage['name']}_{time.time_ns()}.json"
                atomic(launch_path, {**resume_source, 'previous_config_content': read(folder / 'run_config.json'),
                                    'protocol_identity': identity, 'recorded_at_utc': utc()})
                journal(root, 'resume_source_bound', name=stage['name'], record=bind(launch_path), **resume_source)
            with log.open('a') as stream:
                process = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                state.update(state='running', child_pid=process.pid, current_command=command,
                             current_log=str(log), stage_started_at_utc=utc())
                journal(root, 'stage_started', name=stage['name'], pid=process.pid, command=command, log=str(log))
                while process.poll() is None:
                    if stop[0] or (root / 'STOP').exists():
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise InterruptedError('Stop requested; last saved checkpoint retained')
                    state['progress'] = stage_progress(root, stage)
                    state['last_gpu_usage'] = gpu_usage(plan['gpu'])
                    publish()
                    time.sleep(10)
                code = process.returncode
            journal(root, 'stage_exited', name=stage['name'], exit_status=code)
            state.update(child_pid=None, last_exit_status=code)
            if code:
                raise RuntimeError(f"Stage {stage['name']} failed with exit {code}; following stages were not launched")
            verify_protocol(root)
            verify_receipts(root, plan, identity)
            evidence = verify_stage_complete(root, stage, plan)
            receipt = seal_stage(root, stage, evidence, identity)
            state['completed_stages'].append({'name': stage['name'], 'completed_at_utc': utc(),
                                             'receipt': receipt, **evidence})
            publish()
            process = None
        state.update(state='training_complete_pending_evaluation', current_stage=None, child_pid=None,
                     finished_at_utc=utc(), queue=[])
        atomic(root / 'training_matrix.json', {'identity_hash': identity, 'stages': state['completed_stages'],
               'state': state['state'], 'evaluation_complete': False, 'finished_at_utc': utc()})
        journal(root, 'all_training_stages_complete', evaluation_complete=False)
        publish()
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        state.update(state='stopped' if isinstance(error, InterruptedError) else 'failed',
                     error=str(error), traceback=traceback.format_exc(), child_pid=None)
        publish()
        journal(root, state['state'], error=str(error))
        raise


def monitor(args):
    root = Path(args.run_dir).resolve()
    while True:
        path = root / 'status.json'
        state = read(path) if path.exists() else {'state': 'not_started'}
        if not path.exists() and (root / 'data/progress.json').exists():
            audit = read(root / 'data/progress.json')
            state = {'state': 'preflight_data_audit', 'current_stage': audit.get('state'),
                     'updated_at_utc': audit.get('utc'), 'progress': audit,
                     'note': 'Formal background queue has not started yet.'}
        if args.json:
            print(json.dumps(state, ensure_ascii=False), flush=True)
        else:
            lines = [f"[{utc()}] Formal2K 单卡流水线", f"状态：{state['state']}  GPU：{state.get('gpu', '?')}  "
                     f"当前阶段：{state.get('current_stage') or '-'}", f"后台服务：{SERVICE}"]
            if state.get('updated_at_utc'):
                age = max(0., (datetime.now(timezone.utc) - datetime.fromisoformat(state['updated_at_utc'])).total_seconds())
                lines.append(f"最后心跳：{state['updated_at_utc']}（{age:.0f} 秒前）")
                if age > 120 and state['state'] in {'starting', 'checking_stage', 'running', 'waiting_for_fixed_gpu'}:
                    lines.append('注意：心跳已超过 120 秒，请用 systemctl 检查服务是否仍在运行。')
            protocol_path = root / 'protocol.json'
            if not protocol_path.exists() and state.get('progress'):
                progress = state['progress']
                lines.append(f"启动前数据审计：{progress.get('completed', 0)}/{progress.get('total', '?')}")
                lines.append('正式后台队列尚未启动。')
            if protocol_path.exists():
                plan = read(protocol_path)['protocol']
                lines.append(f"训练集：{plan['object_count']} 个物体 / {plan['sample_count']} 个固定场景样本")
                for stage in plan['stages']:
                    progress = stage_progress(root, stage)
                    unit = '遮挡样本' if stage['kind'] == 'data' else 'updates'
                    line = f"  {stage['name']}: {progress['progress']}/{stage['expected']} {unit}"
                    if 'loss' in progress:
                        line += f"  loss={progress['loss']:.6f}"
                    if 'flow_loss' in progress:
                        line += f"  flow={progress['flow_loss']:.6f}"
                    if 'object_group_exposures' in progress:
                        line += f"  场景曝光={progress['object_group_exposures']}"
                    lines.append(line)
            if state.get('current_log'):
                lines.append('当前日志：' + state['current_log'])
            if state.get('error'):
                lines.append('错误：' + state['error'])
            if args.watch and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print('\n'.join(lines), flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default=str(DEFAULT_RUN))
    sub = parser.add_subparsers(dest='action', required=True)
    frozen = sub.add_parser('freeze')
    frozen.add_argument('--gpu', type=int, required=True)
    frozen.add_argument('--smoke-report', required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--resume', action='store_true')
    watcher = sub.add_parser('monitor')
    watcher.add_argument('--watch', type=float, default=0)
    watcher.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.action == 'monitor' and (args.watch < 0 or 0 < args.watch < 1):
        parser.error('--watch must be 0 or at least one second')
    {'freeze': freeze, 'run': run, 'monitor': monitor}[args.action](args)


if __name__ == '__main__':
    main()
