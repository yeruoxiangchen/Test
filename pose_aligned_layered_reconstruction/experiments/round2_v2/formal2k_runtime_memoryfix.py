"""Data-memory scheduling amendment; preserve the frozen two-GPU training protocol."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
from . import formal2k_runtime_dual as dual
from .formal2k_run import (DEFAULT_RUN, PYTHON, WORKSPACE, utc, read, atomic,
    sha, bind, canonical, journal, verify_receipts, seal_stage, stage_progress)
from .formal2k_runtime_dual import (GPUS, gpu_usage, admission_ready,
    validate_rng_sidecars, verify_stage_complete, monitor)

AMENDMENT_NAME = 'runtime_amendments/003_data_memory.json'
ORIGINAL_DATA_MODULE = 'pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_data'
MEMORY_DATA_MODULE = ORIGINAL_DATA_MODULE + '_memory'
REGRESSION_RELATIVE = 'memory_fix/regression_gpu7/report.json'


def verify_regression(path, plan):
    """Accept the actual independent GPU regression, never a CPU-only proof."""
    from . import formal2k_data as data
    from . import memory_efficient_mesh as mesh
    report = read(path)
    if (report.get('passed') is not True or report.get('state') != 'complete'
            or report.get('data_plan_identity') != plan['data_identity']
            or report.get('artificial_memory_cap') is not False
            or report.get('production_data_modified') is not False
            or report.get('deterministic_algorithms') is not True
            or report.get('gpu') != 7 or report.get('cuda_visible_devices') != '7'):
        raise ValueError('A completed, successful GPU 7 memory regression for this data plan is required')
    for name, filename in (('regression', 'formal2k_memory_regression.py'),
                           ('reduction', 'memory_efficient_mesh.py'),
                           ('residency', 'formal2k_data_memory.py')):
        if report.get('sources', {}).get(name) != bind(Path(__file__).with_name(filename)):
            raise ValueError('GPU regression tested a different source: ' + filename)
    data_plan = read(plan['data_plan']['path'])['plan']
    if (report.get('original_mesh_sources') != mesh.original_source_bindings()
            or report.get('original_data_sources') != data.code_binding()
            or report.get('data_plan') != bind(plan['data_plan']['path'])
            or report.get('runtime_binding', {}).get('decoder') != data_plan['proxy_decoder_binding']
            or report.get('dino_weights') != data_plan['dino_weights']):
        raise ValueError('GPU regression original model/data bindings differ')
    cases = report.get('cases', [])
    if ([case.get('index') for case in cases] != [35, 65, 1918]
            or not all(case.get('passed') is True for case in cases)):
        raise ValueError('GPU regression must complete objects 35, 65 and 1918')
    for case in cases:
        expected = data_plan['clean_rows'][case['index']]
        historical = {'vertices': expected['target_mesh_vertices'], 'faces': expected['target_mesh_faces']}
        if (case.get('mesh_resolution') != 256 or case.get('mesh_success') is not True
                or case.get('vertices', 0) <= 0 or case.get('faces', 0) <= 0
                or case.get('historical_mesh_counts') != historical
                or case.get('historical_mesh_count_delta') !=
                    {key: case[key] - value for key, value in historical.items()}):
            raise ValueError('GPU regression lacks full-resolution success or historical-count disclosure')
    first = cases[0]
    if (first.get('same_run_original_mesh_counts') !=
            {'vertices': first['vertices'], 'faces': first['faces']}
            or first.get('mesh_tensor_fields_bitwise_equal') !=
            ['vertices', 'faces', 'vertex_attrs', 'face_normal', 'vertex_normal']
            or not first.get('production_cache_all_tensor_fields_bitwise_equal')
            or first.get('production_metadata_and_scene_outputs_equal') is not True):
        raise ValueError('GPU regression lacks original-mesh and production-cache equivalence')
    for case in cases[1:]:
        checks = case.get('reduction_checks', [])
        if (len(checks) != 1 or not all(
                check.get('bitwise_equal') is True and check.get('finite') is True
                and check.get('inputs_unchanged') is True and check.get('max_abs') == 0.
                and check.get('sdf_sign_flips') == 0 and check.get('reference_channel_chunk') == 2
                and check.get('rows', 0) > 0 and check.get('corners') == 8
                and check.get('channels') == 10 for check in checks)):
            raise ValueError('GPU regression lacks complete original reduction equivalence')
    return bind(path)


def verify_regression_artifacts(report):
    """Check large regression artifacts once at freeze, then seal the report."""
    records = [report['original_mesh35'], report['cases'][0]['production_receipt']]
    for case in report['cases']:
        records.append(case['scene'])
        row = case['cache_row']
        records.append({'path': row['sample'], 'sha256': row['sha256']})
    for record in records:
        if sha(record['path']) != record['sha256']:
            raise ValueError('GPU regression artifact changed: ' + record['path'])


def derived_plan(plan, amendment_record):
    # The parent already fixes world_size=2, grad_accum=1 and all training CLI.
    return {**plan, 'runtime_amendment': amendment_record}


def worker_command(command, stage):
    if stage['kind'] != 'data':
        return dual.worker_command(command, stage)
    if command[:4] != [PYTHON, '-u', '-m', ORIGINAL_DATA_MODULE]:
        raise ValueError('Expected the unchanged formal data module command')
    return [*command[:3], MEMORY_DATA_MODULE, *command[4:]]


def freeze_amendment(args):
    root = Path(args.run_dir).resolve()
    path = root / AMENDMENT_NAME
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if path.exists():
        raise FileExistsError('Runtime amendment is immutable and already exists')
    plan, identity = dual.verify_protocol(root)
    status = read(root / 'status.json')
    train_names = {stage['name'] for stage in plan['stages'] if stage['kind'] == 'train'}
    if (status['state'] != 'failed' or status.get('current_stage') != 'prepare_occlusion'
            or status.get('child_pid') or status.get('identity_hash') != identity
            or status.get('runtime_amendment', {}).get('identity_hash') != identity
            or status.get('completed_stages') or any((root / name).exists() for name in train_names)
            or list((root / 'stage_receipts').glob('*.json'))):
        raise ValueError('Memory revision requires the failed data stage before any training starts')
    failure = read(root / 'data/failure.json')
    if failure.get('command') != 'augment' or 'OutOfMemoryError' not in failure.get('exception', ''):
        raise ValueError('Expected preserved data-preparation OOM evidence')
    regression = verify_regression(args.regression_report or root / REGRESSION_RELATIVE, plan)
    verify_regression_artifacts(read(regression['path']))
    data_plan = read(plan['data_plan']['path'])['plan']
    progress = read(root / 'data/progress.json')
    if progress.get('identity_hash') != plan['data_identity'] or progress.get('completed') != 130:
        raise ValueError('Expected the failed attempt with 130 completed occlusion samples')
    completed = data_plan['occluded_rows'][:progress['completed']]
    expected = {f"{item['index']:05d}_{item['variant']:02d}.json": item
                for item in completed if 'reuse' not in item}
    receipt_paths = sorted((root / 'data/receipts').glob('*.json'))
    if {p.name for p in receipt_paths} != set(expected):
        raise ValueError('The successful data-receipt set differs from the frozen completed prefix')
    for receipt in receipt_paths:
        saved = read(receipt)
        if saved['identity_hash'] != plan['data_identity'] or saved['item'] != expected[receipt.name]:
            raise ValueError('Completed sample receipt identity differs: ' + str(receipt))
    history = []
    for source in (root / 'status.json', root / 'data/failure.json', root / 'data/runtime_binding.json',
                   root / 'data/progress.json', root / 'data/status.json', Path(status['current_log'])):
        source = source.resolve(strict=True)
        destination = root / 'runtime_amendments/003_data_memory_history' / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        history.append({**bind(destination), 'original_path': str(source)})
    sources = []
    for filename in ('formal2k_runtime_memoryfix.py', 'run_formal2k_memoryfix.sh',
                     'formal2k_data_memory.py', 'memory_efficient_mesh.py', 'formal2k_memory_regression.py'):
        source = Path(__file__).with_name(filename)
        record = bind(source)
        destination = root / 'runtime_amendments/003_data_memory_sources' / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources.append({**record, 'snapshot': str(destination)})
    cpu_reports = [bind(root / 'memory_fix' / filename) for filename in
                   ('dino_residency_cpu.json', 'cpu_channel_equivalence.json')]
    for record in cpu_reports:
        if read(record['path']).get('passed') is not True:
            raise ValueError('CPU resource-equivalence acceptance is incomplete')
    amendment = {'format': 'palr.formal2k.runtime_amendment.v1', 'created_at_utc': utc(),
        'parent_protocol': bind(root / 'protocol.json'),
        'parent_runtime_amendment': bind(root / dual.AMENDMENT_NAME), 'parent_runtime_identity': identity,
        'authorization': 'User requested repair of data preparation OOM; manual long-training launch remains with the user.',
        'reason': 'Original decoder scatter workspace OOM at object 65 after 130 completed samples.',
        'gpus': GPUS, 'data_gpus': [1], 'world_size': 2, 'grad_accum': 1, 'effective_batch': 2,
        'allocator_cap_mib': None, 'capacity_probe_required': False,
        'command_transform': {'from': ORIGINAL_DATA_MODULE, 'to': MEMORY_DATA_MODULE, 'stage_kind': 'data'},
        'memory_policy': {'mesh_channel_chunk': 1, 'dino_cpu_between_encode_calls': True,
                          'original_dino_forward_on_original_gpu': True},
        'unchanged': 'All training commands, model weights, precision, cohort, recipes, image/mesh resolution, labels, global batch and budgets; original admission/failure/resume guards.',
        'on_failure': 'Stop and preserve evidence; no automatic retry or sample filtering.',
        'failed_attempt_history': history, 'preserved_data_receipts': [bind(p) for p in receipt_paths],
        'completed_occlusion_count': len(completed), 'reused_occlusion_count': sum('reuse' in item for item in completed),
        'regression_report': regression, 'cpu_reports': cpu_reports,
        'original_mesh_sources': read(regression['path'])['original_mesh_sources'], 'source_code': sources}
    identity = canonical(amendment)
    atomic(path, {'identity_hash': identity, 'amendment': amendment})
    journal(root, 'runtime_amendment_frozen', identity_hash=identity, gpus=GPUS,
            automatic_launch=False, regression_report=regression)
    atomic(root / 'status.json', {'state': 'ready_manual_launch', 'identity_hash': identity,
        'gpu': '1,2', 'gpus': GPUS, 'active_gpus': [], 'child_pid': None, 'current_stage': None,
        'completed_stages': [], 'queue': [stage['name'] for stage in plan['stages']],
        'runtime_amendment': {**bind(path), 'identity_hash': identity}, 'updated_at_utc': utc(),
        'capacity_probe_required': False})


def verify_protocol(root):
    plan, identity = dual.verify_protocol(root)
    path = root / AMENDMENT_NAME
    envelope = read(path)
    amendment = envelope['amendment']
    if canonical(amendment) != envelope['identity_hash'] or amendment['parent_runtime_identity'] != identity:
        raise ValueError('Memory amendment identity mismatch')
    records = [amendment['parent_protocol'], amendment['parent_runtime_amendment'],
               *amendment['failed_attempt_history'], *amendment['preserved_data_receipts'],
               *amendment['source_code'], *amendment['cpu_reports'], amendment['regression_report'],
               *amendment['original_mesh_sources'].values()]
    for item in records:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Memory amendment source changed: ' + item['path'])
        if 'snapshot' in item and sha(item['snapshot']) != item['sha256']:
            raise ValueError('Memory source snapshot changed: ' + item['snapshot'])
    if (amendment['gpus'] != GPUS or amendment['data_gpus'] != [1]
            or amendment['world_size'] != 2 or amendment['grad_accum'] != 1
            or amendment['effective_batch'] != 2 or amendment['allocator_cap_mib'] is not None
            or amendment['capacity_probe_required'] is not False
            or amendment['command_transform'] != {'from': ORIGINAL_DATA_MODULE, 'to': MEMORY_DATA_MODULE, 'stage_kind': 'data'}
            or amendment['memory_policy'] != {'mesh_channel_chunk': 1, 'dino_cpu_between_encode_calls': True,
                                              'original_dino_forward_on_original_gpu': True}):
        raise ValueError('Fixed data-memory or training policy changed')
    verify_regression(amendment['regression_report']['path'], plan)
    return derived_plan(plan, {**bind(path), 'identity_hash': envelope['identity_hash']}), envelope['identity_hash']


def run(args):
    root = Path(args.run_dir).resolve()
    plan, identity = verify_protocol(root)
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_HOME='/usr/local/cuda-12.6',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONFAULTHANDLER='1',
               TORCH_CUDA_ARCH_LIST='8.6', SPCONV_ALGO='native',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               MPLCONFIGDIR=str(root / 'matplotlib_cache'))
    env['PATH'] = '/home/zjr/anaconda3/envs/reconviagen/bin:' + env.get('PATH', '')
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'V2_NPROC'):
        env.pop(key, None)
    state = {'state': 'starting', 'identity_hash': identity, 'gpu': '1,2', 'gpus': GPUS,
             'supervisor_pid': os.getpid(), 'started_at_utc': utc(), 'completed_stages': [],
             'queue': [s['name'] for s in plan['stages']], 'current_stage': None, 'child_pid': None,
             'runtime_amendment': plan['runtime_amendment'],
             'admission_policy': {'maximum_used_mib': 64, 'maximum_utilization_percent': 5,
                                  'torch_allocator_cap_mib': None, 'capacity_probe_required': False}}
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def publish():
        state['updated_at_utc'] = utc()
        atomic(root / 'status.json', state)
    process = None
    journal(root, 'supervisor_started', pid=os.getpid(), gpu=plan['gpu'], resume=args.resume,
            runtime_amendment=plan['runtime_amendment'])
    try:
        for stage in plan['stages']:
            verify_protocol(root)
            receipts = verify_receipts(root, plan, identity)
            if stage['kind'] == 'train' and 'prepare_occlusion' not in receipts:
                raise ValueError('Training requires a sealed, complete data preparation receipt')
            if 'condition_mode' in stage and 'lari' not in receipts:
                raise ValueError('SS requires a sealed shared LaRI checkpoint')
            state.update(current_stage=stage['name'], state='checking_stage', child_pid=None)
            active_gpus = [1] if stage['kind'] == 'data' else GPUS
            state['active_gpus'] = active_gpus
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
                validate_rng_sidecars(checkpoint)
                resume_checkpoint = checkpoint
            idle = 0
            while idle < 2:
                if stop[0] or (root / 'STOP').exists():
                    raise InterruptedError('Stop requested before GPU stage')
                usage = {str(gpu): gpu_usage(gpu) for gpu in active_gpus}
                idle = idle + 1 if admission_ready(usage) else 0
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
                                 'previous_run_config': bind(folder / 'run_config.json'),
                                 **validate_rng_sidecars(resume_checkpoint)}
                launch_path = root / 'resume_sources' / f"{stage['name']}_{time.time_ns()}.json"
                atomic(launch_path, {**resume_source, 'previous_config_content': read(folder / 'run_config.json'),
                                    'protocol_identity': identity, 'recorded_at_utc': utc()})
                journal(root, 'resume_source_bound', name=stage['name'], record=bind(launch_path), **resume_source)
            command = worker_command(command, stage)
            child_env = {**env, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, active_gpus))}
            with log.open('a') as stream:
                process = subprocess.Popen(command, cwd=WORKSPACE, env=child_env, stdout=stream,
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
                    state['last_gpu_usage'] = {str(gpu): gpu_usage(gpu) for gpu in active_gpus}
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
            evidence['runtime_amendment'] = bind(root / AMENDMENT_NAME)
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



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default=str(DEFAULT_RUN))
    sub = parser.add_subparsers(dest='action', required=True)
    freezer = sub.add_parser('freeze-amendment')
    freezer.add_argument('--regression-report', help='Successful isolated GPU regression; default RUN/memory_fix/regression_gpu7/report.json')
    runner = sub.add_parser('run')
    runner.add_argument('--resume', action='store_true')
    watcher = sub.add_parser('monitor')
    watcher.add_argument('--watch', type=float, default=0)
    watcher.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.action == 'monitor' and (args.watch < 0 or 0 < args.watch < 1):
        parser.error('--watch must be zero or at least one second')
    {'freeze-amendment': freeze_amendment, 'run': run, 'monitor': monitor}[args.action](args)


if __name__ == '__main__':
    main()
