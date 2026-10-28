"""Resumable single-GPU qualification queue and a CUDA-free status command."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

if __package__:
    from .common import OUT, PACKAGE, TRACKER, PYTHON, BLENDER, MODULE, plan, object_dir, write_json, now, event, binding, verify
else:
    # Direct entry avoids importing the package's Torch-based __init__ for
    # monitor/start/stop and for the long-lived process supervisor.
    from common import OUT, PACKAGE, TRACKER, PYTHON, BLENDER, MODULE, plan, object_dir, write_json, now, event, binding, verify
    sys.path.insert(0, str(TRACKER))


def command(module_name, /, **options):
    result = [PYTHON, '-u', '-m', MODULE + '.' + module_name]
    for key, value in options.items():
        result += ['--' + key.replace('_', '-'), str(value)]
    return result


def run_child(name, argv, record_command=None):
    for contract in ['launch_sources.json', 'frozen_protocol.json']:
        if not (OUT / contract).exists():
            continue
        for source in json.loads((OUT / contract).read_text())['sources']:
            verify(source)
    recorded = record_command or argv
    log = OUT / 'logs' / (name + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    done = OUT / 'completed' / (name + '.json')
    if done.exists():
        previous = json.loads(done.read_text())
        if previous['command'] != recorded:
            raise ValueError('Stage command changed: ' + name)
        return
    event(stage=name, command=argv, log=str(log))
    started = now()
    child = None
    try:
        with log.open('a') as stream:
            child = subprocess.Popen(argv, cwd=TRACKER, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                write_json(OUT / 'status.json', {'state': 'running', 'stage': name, 'pid': os.getpid(),
                    'child_pid': child.pid, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'updated_at': now(), 'started_at': started, 'log': str(log)})
                time.sleep(5)
        if child.returncode:
            raise RuntimeError(f'{name} failed with exit {child.returncode}; see {log}')
        write_json(done, {'command': recorded, 'completed_at': now()})
    except BaseException:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        raise


def prepare(limit=16):
    frozen = plan()
    for i, row in enumerate(frozen['plan']['objects'][:limit]):
        root = object_dir(row)
        for b in [row['source'], row['converted'], row['mesh']]:
            verify(b)
        run_child(f'{i:02d}_render', [BLENDER, '-b', '-t', '4', '--python-exit-code', '1', '-P',
            str(Path(__file__).with_name('render_targets.py')), '--', '--asset', str(root / 'asset.json')])
        run_child(f'{i:02d}_latent', command('latents', index=i))
        run_child(f'{i:02d}_visual', command('visuals', index=i))
        run_child(f'{i:02d}_scene', command('scenes', index=i))
    if limit == 16:
        run_child('manifests', [PYTHON, '-u', '-m', MODULE + '.scenes', '--manifests'])


def data_gate():
    log = OUT / 'logs/data_gate.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads((OUT / 'status.json').read_text()) if (OUT / 'status.json').exists() else {}
    write_json(OUT / 'status.json', {**status, 'state': 'running', 'stage': 'data_gate',
        'pid': os.getpid(), 'child_pid': None, 'updated_at': now(), 'log': str(log), 'error': None})
    results = []
    for row in plan()['plan']['objects']:
        root = object_dir(row)
        audit = json.loads((root / 'latent_audit.json').read_text())
        visual = json.loads((root / 'visual_audit.json').read_text())
        # Engineering thresholds frozen before the first target is decoded.
        # These are NOT published TRELLIS guarantees or test-set selection.
        passed = (audit['coordinate']['passed'] and audit['ss']['passed']
            and audit['surface']['chamfer_l1'] <= .015
            and audit['surface']['fscore_0p02'] >= .85
            and visual['min_source_silhouette_iou'] >= .95)
        results.append({'uid': row['uid'], 'passed': passed,
            'cd': audit['surface']['chamfer_l1'], 'f02': audit['surface']['fscore_0p02'],
            'projection_iou': visual['min_source_silhouette_iou']})
    write_json(OUT / 'data_gate.json', {'passed': all(r['passed'] for r in results), 'objects': results,
        'scope': 'numeric data qualification only; gallery and fit remain independent checks'})
    with log.open('a') as stream:
        stream.write(json.dumps({'time': now(), 'stage': 'data_gate', 'original_passed': all(r['passed'] for r in results),
                                'failed_objects': [r for r in results if not r['passed']]}) + '\n')
    if not all(r['passed'] for r in results):
        if (OUT / 'data_quality_review.json').exists():
            protocol = json.loads((OUT / 'frozen_protocol.json').read_text())
            if 'data_quality_review' not in protocol:
                raise RuntimeError('Surface review requires a recorded protocol amendment')
            verify(protocol['data_quality_review'])
            if __package__:
                from .quality_review import validate_review
            else:
                from quality_review import validate_review
            review = validate_review()
            if review['passed']:
                with log.open('a') as stream:
                    stream.write(json.dumps({'time': now(), 'stage': 'data_gate_review', 'passed': True,
                        'original_gate_passed': False, 'review': str(OUT / 'data_quality_review.json'),
                        'scope': 'all original16; fixed100K seed42/43/44; every seed passes unchanged thresholds'}) + '\n')
                event(stage='data_gate_review_accepted', original_gate_passed=False,
                      review=str(OUT / 'data_quality_review.json'))
                return
        raise RuntimeError('Data quality gate failed; no targets dropped and no training launched; '
                           'see data_gate.json and data_quality_review.json (if present)')


def train(name, stage, manifest, steps, init=None, mode='full', lora=False):
    output = OUT / name
    args = [PYTHON, '-u', '-m', 'pose_aligned_layered_reconstruction.v2.train',
        '--stage', stage, '--manifest', str(manifest), '--output_dir', str(output),
        '--max_steps', str(steps), '--max_samples', '16', '--grad_accum', '1',
        '--learning_rate', '0.0001', '--warmup_steps', '20', '--lr_schedule', 'constant',
        '--save_every', str(min(steps, 200)), '--validate_every', str(steps), '--p_uncond', '0',
        '--seed', '20261020', '--amp', 'bf16']
    if stage == 'ss':
        args += ['--freeze_geometry', '--condition_mode', mode]
    if lora:
        args += ['--finetune_lora']
    original_command = list(args)
    if init:
        original_command += ['--init', str(init), '--init_weights', 'raw']
    last = output / 'checkpoints/last.pt'
    # Keep completion command stable while allowing exact-cursor resume.
    completed = OUT / 'completed' / (name + '.json')
    if completed.exists():
        if json.loads(completed.read_text())['command'] != original_command:
            raise ValueError('Training definition changed')
        return last
    if last.exists():
        args += ['--resume', str(last)]
    elif init:
        args += ['--init', str(init), '--init_weights', 'raw']
    if output.exists() and not last.exists():
        failed = OUT / 'failed_starts'
        failed.mkdir(exist_ok=True)
        output.rename(failed / (name + '_' + str(time.time_ns())))
    run_child(name, args, record_command=original_command)
    return last


def evaluation(name, stage, manifest, checkpoint):
    out = OUT / 'evaluations' / name
    for mode in ['parent', 'candidate']:
        run_child(name + '_' + mode, command('evaluate', manifest=manifest, checkpoint=checkpoint,
            stage=stage, mode=mode, output=out))
    run_child(name + '_summary', [PYTHON, '-u', '-m', MODULE + '.evaluate', '--output', str(out), '--summarize'])
    return json.loads((out / 'summary.json').read_text())


def experiment():
    smoke_receipt = json.loads((OUT / 'smoke.json').read_text())
    if not smoke_receipt['passed']:
        raise RuntimeError('Real data smoke has not passed')
    for source in smoke_receipt['sources']:
        verify(source)
    freeze()
    prepare()
    data_gate()
    official = PACKAGE / 'outputs/round1_20260909/data_v2/train32.json'
    # Known official targets, same current architecture/capacity as new targets.
    lari = train('reference_lari', 'lari', official, 1000)
    slat = train('reference_slat', 'slat', official, 200, lari, lora=True)
    reference = evaluation('reference_gt_slat', 'slat', official, slat)
    if not reference['fit_advantage_established']:
        raise RuntimeError('Official-target positive control failed; investigate model/recipe before blaming HSSD')
    decisions = []
    for domain in ['clean', 'occluded']:
        manifest = OUT / f'{domain}16.json'
        lari = train(domain + '_lari', 'lari', manifest, 1000)
        slat = train(domain + '_slat', 'slat', manifest, 200, lari, lora=True)
        slat_fit = evaluation(domain + '_gt_slat', 'slat', manifest, slat)
        for mode in ['full', 'no_layer']:
            ss = train(domain + '_ss_' + mode, 'ss', manifest, 500, lari, mode=mode)
            result = evaluation(domain + '_ss_' + mode, 'ss', manifest, ss)
            decisions.append({'domain': domain, 'mode': mode, 'fit': result['fit_advantage_established']})
        from pose_aligned_layered_reconstruction.mesh_metrics import aggregate_mesh_records
        comparisons = []
        expected = None
        for arm, label in [('no_layer', 'parent'), ('full', 'candidate')]:
            folder = OUT / 'evaluations' / (domain + '_ss_' + arm)
            records = [json.loads(p.read_text()) for p in sorted(folder.glob('[0-9][0-9][0-9]/*_candidate.json'))]
            comparisons.extend([{**r, 'mode': label} for r in records])
            expected = sorted({r['uid'] for r in records})
        contrast = aggregate_mesh_records(comparisons, expected)
        contrast['comparison'] = 'full layered SS versus matched no_layer; same frozen SLat'
        write_json(OUT / 'evaluations' / (domain + '_full_vs_no_layer.json'), contrast)
        if not slat_fit['fit_advantage_established']:
            write_json(OUT / 'fit_gate.json', {'passed': False, 'failed_domain': domain, 'decisions': decisions})
            raise RuntimeError('SLat complete-sampling fit gate failed; stop before further stages')
    passed = all(r['fit'] for r in decisions if r['mode'] == 'full')
    write_json(OUT / 'fit_gate.json', {'passed': passed, 'decisions': decisions,
        'expansion_authorized': False, 'next': 'review galleries, full vs no_layer and held-out protocol before expansion'})
    if not passed:
        raise RuntimeError('Full SS branch did not establish mesh fitting gain; expansion remains closed')


def source_files():
    folders = [Path(__file__).parent, Path(__file__).parent / 'vendor', PACKAGE, PACKAGE / 'v2', PACKAGE / 'data_v2']
    files = sorted({p for folder in folders for p in folder.glob('*.py')})
    files.extend(Path(__file__).with_name(name) for name in ['PROTOCOL.md', 'run.sh'])
    return files


def freeze():
    path = OUT / 'frozen_protocol.json'
    if path.exists():
        for source in json.loads(path.read_text())['sources']:
            verify(source)
        return
    files = source_files()
    sources = [binding(p) for p in files]
    for p in files:
        snapshot = OUT / 'source_snapshot' / p.relative_to(PACKAGE)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, snapshot)
    write_json(path, {'frozen_at': now(), 'sources': sources, 'plan': binding(OUT / 'plan.json'),
        'smoke': binding(OUT / 'smoke.json'),
        'official_reference': binding(PACKAGE / 'outputs/round1_20260909/data_v2/train32.json'),
        'single_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'object_limit': 16,
        'no_automatic_expansion': True})


def smoke():
    # A completed smoke is immutable evidence referenced by frozen_protocol.
    # Re-entering start-preflight must not rewrite its timestamp/source identity.
    existing = OUT / 'smoke.json'
    if existing.exists():
        receipt = json.loads(existing.read_text())
        if not receipt['passed']:
            raise RuntimeError('Existing smoke did not pass')
        for source in receipt['sources']:
            verify(source)
        verify(receipt['manifest'])
        return
    from pose_aligned_layered_reconstruction.v2.dataset import write_manifest
    row = plan()['plan']['objects'][0]
    receipt = json.loads((object_dir(row) / 'scene_receipt.json').read_text())
    manifest = OUT / 'smoke_manifest.json'
    if not manifest.exists():
        write_manifest(manifest, list(receipt['rows'].values()),
                       {'scope': 'one new asset, same-camera clean/occluded input, execution smoke'}, OUT)
    lari = train('smoke_lari', 'lari', manifest, 2)
    full = train('smoke_ss_full', 'ss', manifest, 2, lari)
    train('smoke_ss_no_layer', 'ss', manifest, 2, lari, mode='no_layer')
    slat = train('smoke_slat', 'slat', manifest, 2, lari, lora=True)
    for stage, checkpoint in [('ss', full), ('slat', slat)]:
        for mode in ['parent', 'candidate']:
            run_child('smoke_eval_' + stage + '_' + mode, command('evaluate',
                manifest=manifest, checkpoint=checkpoint, stage=stage, mode=mode,
                output=OUT / 'smoke_evaluations' / stage, max_samples=1, seeds=42))
        run_child('smoke_eval_' + stage + '_summary', [PYTHON, '-u', '-m', MODULE + '.evaluate',
            '--output', str(OUT / 'smoke_evaluations' / stage), '--summarize'])
    histories = {}
    for name in ['smoke_lari', 'smoke_ss_full', 'smoke_ss_no_layer', 'smoke_slat']:
        values = [json.loads(line) for line in (OUT / name / 'history.jsonl').read_text().splitlines()]
        if len(values) != 2 or not all(v['gradient_norm'] > 0 for v in values):
            raise ValueError('Smoke lacks two updates with nonzero gradients: ' + name)
        histories[name] = values
    sources = [binding(p) for folder in [Path(__file__).parent, PACKAGE / 'v2', PACKAGE / 'data_v2']
               for p in sorted(folder.glob('*.py'))]
    sources += [binding(Path(__file__).with_name(name)) for name in ['PROTOCOL.md', 'run.sh']]
    write_json(OUT / 'smoke.json', {'passed': True, 'completed_at': now(),
        'sources': sources, 'manifest': binding(manifest), 'histories': histories,
        'mesh_sampling': '4 native-frame meshes: SS and GT-support SLat, parent/candidate, seed42, full25 steps',
        'scope': 'execution and gradient acceptance only; two updates do not establish fitting'})


def monitor():
    status_file = OUT / 'status.json'
    if status_file.exists():
        s = json.loads(status_file.read_text())
        from datetime import datetime, timezone
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(s['updated_at'])).total_seconds()
        print(f"[{s.get('updated_at')}] HSSD16 GPU {s.get('gpu')}\n状态：{s['state']}  阶段：{s.get('stage')}  心跳距今：{age:.0f} 秒")
        if s.get('error'):
            print('错误：' + s['error'])
        log = Path(s.get('log', '/nonexistent'))
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 6000))
                lines = stream.read().decode(errors='replace').splitlines()
            meaningful = [line for line in lines if 'PILOT_RENDER ' in line or line.startswith('{')]
            print('\n'.join((meaningful or lines)[-3:]))
            if s['state'] == 'failed':
                errors = [line for line in lines if 'Error:' in line or 'Exception:' in line]
                if errors:
                    print('失败原因：' + errors[-1])
            print('日志：' + str(log))
    else:
        print('队列尚未启动')
    rows = plan()['plan']['objects']
    print('目标：16；已编码：', sum((object_dir(r)/'latent_audit.json').exists() for r in rows),
          '；已构建场景输入：', sum((object_dir(r)/'scene_receipt.json').exists() for r in rows))
    for f in sorted(OUT.glob('evaluations/*/summary.json')):
        d = json.loads(f.read_text())
        print(f.parent.name, '拟合通过：', d['fit_advantage_established'],
              'CD改善：', d['paired']['chamfer_l1']['mean_improvement'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'qualify', 'start', 'start-preflight', 'stop', 'smoke', 'monitor', 'data-gate'])
    p.add_argument('--limit', type=int, default=16)
    p.add_argument('--watch', type=int, default=0)
    a = p.parse_args()
    if not 1 <= a.limit <= 16 or a.watch < 0:
        p.error('limit must be 1..16 and watch nonnegative')
    if a.action == 'stop':
        unit = 'tracker-palr-hssd16-20261020.service'
        active = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True)
        if active.stdout.strip() == 'active':
            subprocess.run(['systemctl', '--user', 'stop', unit], check=True)
            return
        status = json.loads((OUT / 'status.json').read_text())
        pid = int(status.get('child_pid', 0))
        proc = Path('/proc') / str(pid) / 'cmdline'
        if pid > 1 and proc.exists():
            cmd = proc.read_bytes().replace(b'\0', b' ').decode()
            if MODULE not in cmd and str(Path(__file__).parent) not in cmd:
                raise RuntimeError('Refuse to stop an unrelated/reused PID')
            os.killpg(pid, signal.SIGTERM)
            print('Stopped this pilot child only: ' + str(pid), flush=True)
        return
    if a.action in ('start', 'start-preflight'):
        if a.action == 'start':
            receipt = json.loads((OUT / 'smoke.json').read_text())
            if not receipt['passed']:
                raise RuntimeError('Real-data smoke required before service launch')
            for source in receipt['sources']:
                verify(source)
            freeze()
        # A prior interactive preparation must be finished/stopped before the
        # service takes over its output directory. Never race two writers.
        with (OUT / 'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        unit = 'tracker-palr-hssd16-20261020'
        active = subprocess.run(['systemctl', '--user', 'is-active', unit + '.service'], capture_output=True, text=True)
        if active.stdout.strip() == 'active':
            raise RuntimeError('HSSD service already running')
        launch = OUT / 'launch_sources.json'
        if launch.exists():
            for source in json.loads(launch.read_text())['sources']:
                verify(source)
        else:
            write_json(launch, {'created_at': now(), 'sources': [binding(p) for p in source_files()]})
        subprocess.run(['systemctl', '--user', 'reset-failed', unit + '.service'], capture_output=True)
        subprocess.run(['systemd-run', '--user', '--unit=' + unit, '--working-directory=' + str(TRACKER),
            '--setenv=HSSD_GPU=' + os.environ.get('CUDA_VISIBLE_DEVICES', '1'),
            '--property=KillMode=control-group', '--property=TimeoutStopSec=45',
            '/bin/bash', str(Path(__file__).with_name('run.sh')),
            'qualify' if a.action == 'start-preflight' else 'run'], check=True)
        print('Service: ' + unit + '.service', flush=True)
        return
    if a.action == 'monitor':
        while True:
            monitor()
            if not a.watch:
                return
            time.sleep(a.watch)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if a.action == 'prepare':
                prepare(a.limit)
            elif a.action == 'data-gate':
                data_gate()
            elif a.action == 'smoke':
                smoke()
            elif a.action == 'qualify':
                prepare(1)
                smoke()
                experiment()
            else:
                experiment()
            write_json(OUT / 'status.json', {'state': 'complete', 'stage': a.action, 'updated_at': now(),
                'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})
        except BaseException as error:
            status = json.loads((OUT / 'status.json').read_text()) if (OUT / 'status.json').exists() else {}
            write_json(OUT / 'status.json', {**status, 'state': 'failed', 'error': str(error), 'updated_at': now()})
            raise


if __name__ == '__main__':
    main()
