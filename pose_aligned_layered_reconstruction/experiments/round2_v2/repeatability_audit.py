"""Repeat four frozen mesh cases in a separate output, preserving all main data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[2]
MAIN = PACKAGE/'outputs/round2_v2_heldout_mesh_20261018'
OUT = MAIN/'runtime_repeatability'
TARGETS = ('007_42_raw', '018_43_raw', '012_43_raw', '024_43_raw')
PYTHON = '/home/zjr/anaconda3/envs/reconviagen/bin/python'
MODULE = 'pose_aligned_layered_reconstruction.experiments.round2_v2.heldout_mesh_eval'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha(path)}


def verify(item):
    if sha(item['path']) != item['sha256']:
        raise ValueError('changed bound file: '+item['path'])
    return Path(item['path'])


def save(path,value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def _copy_exact(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if sha(source)!=sha(destination):
            raise ValueError('existing copy differs: '+str(destination))
    else:
        shutil.copy2(source,destination)


def prepare():
    protocol_path = OUT/'protocol.json'
    if protocol_path.exists():
        protocol = read(protocol_path)
        verify(protocol['script']); verify(protocol['source_plan'])
        for item in protocol['source_records'].values():
            verify(item)
        return protocol
    payload = read(MAIN/'plan.json'); plan = payload['plan']
    identity = hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
    if identity != payload['identity_hash']:
        raise ValueError('source plan hash mismatch')
    names = [f"{row['index']:03d}_{case['seed']}_{case['mode']}" for row in plan['rows']
             if row['domain']=='clean' and 7<=row['object_index']<25 for case in row['cases']]
    if len(names)!=72 or not set(TARGETS)<=set(names):
        raise ValueError('audit subset must contain exactly 72 frozen records')
    source_records = {}; originals = {}
    for name in names:
        path = MAIN/'records'/(name+'.json')
        record = read(path)
        if not record['success'] or record['identity_hash']!=identity:
            raise ValueError('source subset is incomplete or mixed')
        verify(record['mesh']); verify(record['target_mesh'])
        source_records[name] = binding(path)
        if name in TARGETS:
            originals[name] = {'record':binding(path),'mesh':record['mesh'],'target_mesh':record['target_mesh']}
    OUT.mkdir(parents=True,exist_ok=True)
    _copy_exact(MAIN/'plan.json',OUT/'plan.json')
    for name in names:
        if name not in TARGETS:
            _copy_exact(MAIN/'records'/(name+'.json'),OUT/'records'/(name+'.json'))
    protocol = {'created_at_utc':utc(),'identity_hash':identity,'script':binding(__file__),
                'source_plan':binding(MAIN/'plan.json'),'copied_plan':binding(OUT/'plan.json'),
                'source_records':source_records,'original_targets':originals,
                'target_cases':list(TARGETS),'copied_records':68,'expected_records_after':72,
                'selection_reason':{'007_42_raw':'identical parent/V2 SS support natural control',
                    '018_43_raw':'identical parent/V2 SS support natural control',
                    '012_43_raw':'successful GPU0 case immediately preceding first runtime failure',
                    '024_43_raw':'successful GPU0 case immediately preceding second runtime failure'},
                'scope':'runtime repeatability diagnostic, not replacement or extra scientific model samples',
                'original_outputs_modified':False,'sampler_and_model_protocol_changed':False,
                'command':[PYTHON,'-u','-m',MODULE,'worker','--output',str(OUT),
                           '--start','7','--end','25','--domain','clean']}
    save(protocol_path,protocol)
    print(json.dumps({'repeatability_prepared':True,'output':str(OUT),'copied':68,'new_cases':list(TARGETS)}),flush=True)
    return protocol


def metric_comparison(left,right):
    if set(left['surface'])!=set(right['surface']):
        raise ValueError('metric schema changed')
    differences = {k:float(right['surface'][k]-left['surface'][k]) for k in left['surface']}
    return {'left_record_mesh':left['mesh'],'right_record_mesh':right['mesh'],
            'mesh_sha256_equal':left['mesh']['sha256']==right['mesh']['sha256'],
            'left_vertices':left['structure']['vertex_count'],'right_vertices':right['structure']['vertex_count'],
            'left_faces':left['structure']['face_count'],'right_faces':right['structure']['face_count'],
            'metric_delta_right_minus_left':differences,
            'max_absolute_metric_delta':max(abs(x) for x in differences.values())}


def compare():
    protocol = prepare()
    verify(protocol['copied_plan'])
    if len(list((OUT/'records').glob('*.json')))!=72:
        raise ValueError('repeatability subset is incomplete')
    comparisons = {}
    for name in TARGETS:
        original = read(verify(protocol['source_records'][name]))
        path = OUT/'records'/(name+'.json'); repeated = read(path)
        if not repeated['success'] or repeated['identity_hash']!=protocol['identity_hash']:
            raise ValueError('repeat failed/mixed identity')
        for key in ('index','object_index','uid','object_uid','domain','seed','mode','coordinates','target_mesh','flow'):
            if original[key]!=repeated[key]:
                raise ValueError('repeat altered scientific input/output protocol: '+key)
        verify(original['mesh']); verify(repeated['mesh'])
        result = {'original_record':protocol['source_records'][name],'repeat_record':binding(path),
                  'repeat_vs_original':metric_comparison(original,repeated)}
        if name in ('007_42_raw','018_43_raw'):
            parent_name = name.replace('_raw','_parent')
            parent = read(verify(protocol['source_records'][parent_name]))
            result['historical_parent_record'] = protocol['source_records'][parent_name]
            result['repeat_vs_historical_parent'] = metric_comparison(parent,repeated)
            result['original_vs_historical_parent'] = metric_comparison(parent,original)
        comparisons[name] = result
    result = {'completed_at_utc':utc(),'identity_hash':protocol['identity_hash'],
              'protocol':binding(OUT/'protocol.json'),'records_checked':72,'new_cases':4,
              'source_records_and_original_meshes_unchanged':True,'comparisons':comparisons,
              'interpretation':'observed reproducibility differences only; four selected controls do not bound all other cases',
              'failed_attempt_count':len(list((OUT/'failures').glob('*.json')))}
    # Recheck every original source plus every unchanged copied row after running.
    for name,item in protocol['source_records'].items():
        source = read(verify(item)); verify(source['mesh']); verify(source['target_mesh'])
        if name not in TARGETS and sha(OUT/'records'/(name+'.json'))!=item['sha256']:
            raise ValueError('copied reference record changed')
    save(OUT/'compare.json',result)
    print(json.dumps({'repeatability_complete':True,'comparisons':{n:v['repeat_vs_original'] for n,v in comparisons.items()}}),flush=True)
    return result


def _gpu_usage(gpu):
    text = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True)
    rows = {int(p[0]):(int(p[1]),int(p[2])) for line in text.splitlines() if (p:=line.split(','))}
    if gpu not in rows:
        raise ValueError('GPU does not exist')
    return rows[gpu]


def run_gpu_audit(gpu):
    gpu = int(gpu)
    if gpu not in range(6):
        raise ValueError('only GPUs 0–5 may be considered')
    protocol = prepare()
    checks = []
    for i in range(2):
        memory,utilization = _gpu_usage(gpu)
        checks.append({'at_utc':utc(),'gpu':gpu,'memory_mib':memory,'utilization_percent':utilization})
        if memory>32 or utilization>5:
            raise RuntimeError(f'GPU {gpu} is not idle; no worker launched')
        if i==0:
            time.sleep(10)
    name = f'repeat_gpu{gpu}_{time.time_ns()}'
    command = protocol['command']+['--name',name]
    log = OUT/(name+'.log')
    status = {'state':'running','name':name,'gpu':gpu,'idle_checks':checks,
              'command':command,'started_at_utc':utc(),'log':str(log)}
    save(OUT/'run_status.json',status)
    with log.open('x') as stream:
        result = subprocess.run(command,cwd=PACKAGE.parent,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=stream,stderr=subprocess.STDOUT)
    status.update(exit_status=result.returncode,finished_at_utc=utc(),state='complete' if result.returncode==0 else 'failed')
    save(OUT/'run_status.json',status)
    if result.returncode:
        raise RuntimeError('repeatability worker failed; evidence retained: '+str(log))
    return compare()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','compare','run'))
    parser.add_argument('--gpu',type=int)
    args = parser.parse_args()
    if args.action=='run':
        if args.gpu is None:
            parser.error('--gpu required for run')
        run_gpu_audit(args.gpu)
    elif args.action=='prepare':
        prepare()
    else:
        compare()


if __name__=='__main__':
    main()
