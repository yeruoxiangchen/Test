#!/usr/bin/env python3
"""Practical CoreCoder repair workflow: local inputs, delivery, resume and 100-task acceptance."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import run_harness_compare as harness
import run_scale_4gpu as scale

from screenshot_agent.config import ImageInput, Task
from screenshot_agent.control import atomic_json
from screenshot_agent.gpu_queue import QueuePolicy
from screenshot_agent.harness import PROTOCOL
from screenshot_agent.hidden_eval import validate_prepared
from screenshot_agent.scale_data import immutable_json, read, sha256
from screenshot_agent.workflow import METHOD, implementation, run_workflow_task
from screenshot_agent.workflow_config import WorkflowConfig, load_workflow_config
from screenshot_agent.workflow_view import export_delivery, render_batch, render_run

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path('/data/zjr/screenshot_code_agent/v4_dev10_20261029')
TERMINAL = {'verified','unresolved','error'}


def prepared_tasks(output):
    value = read(output/'prepared.json')
    rows = value['tasks']
    count = value.get('task_count', 100)
    if count not in {10, 100} or value['protocol'] != PROTOCOL or len(rows) != count or len({r['task_id'] for r in rows}) != count:
        raise ValueError('Run requires its declared fixed 10 or 100 distinct hidden-test tasks')
    for row in rows:
        if not row['task_id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in row['task_id']):
            raise ValueError('Invalid task identifier')
        if row['task_id'] in {'.','..'}:
            raise ValueError('Invalid task identifier')
    return rows


def prepare(c,args):
    # Reuse already prepared data, without touching its controller or results.
    prior = args.reuse_prepared
    if (args.output/'prepared.json').exists():
        rows = prepared_tasks(args.output)
        if len(rows) != getattr(args, 'limit', 100):
            raise ValueError('Task count changed for an existing run; use a new output directory')
        return
    if prior and (prior/'prepared.json').exists():
        rows = prepared_tasks(prior)
        limit = getattr(args, 'limit', 100)
        if len(rows) < limit:
            raise ValueError('Reused input manifest has fewer tasks than requested')
        rows = rows[:limit]
        selection = read(prior/'selection.json')
        if selection['sha256'] != sha256(args.source_plan):
            raise ValueError('Reused inputs come from a different source selection')
        for i,row in enumerate(rows):
            c.update(stage='validating_reused_inputs',validation_completed=i,validation_total=len(rows))
            validate_prepared(Path(row['task_file']).parent)
        selection = {**selection, 'selected_ids': [row['task_id'] for row in rows],
                     'selection_rule': 'first N entries of the existing ordered manifest; no score-based selection'}
        immutable_json(args.output/'selection.json',selection)
        immutable_json(args.output/'prepared.json', {**read(prior/'prepared.json'), 'tasks': rows,
                       'task_count': len(rows), 'selection_sha256': sha256(args.output/'selection.json')})
        atomic_json(args.output/'input_storage.json',{'reused_from':str(prior),'copied':False})
    else:
        if getattr(args, 'limit', 100) != 100:
            raise ValueError('10-task smoke requires --reuse-prepared pointing to the existing prepared inputs')
        harness.prepare(c,args)


def report_method(output):
    plan = output/'workflow_plan.json'
    method = read(plan)['method'] if plan.exists() else METHOD
    if method not in {'corecoder_retrieval_workflow_v1','corecoder_retrieval_workflow_v2',
                      'corecoder_repair_workflow_v3',METHOD}:
        raise ValueError('Unknown workflow method')
    return method


def result_rows(output):
    rows = []
    expected_method = report_method(output)
    for task in prepared_tasks(output):
        folder = output/'runs'/task['task_id']
        if (folder/'result.json').exists():
            row = read(folder/'result.json')
            if row['method'] != expected_method or row['protocol'] != PROTOCOL or row['task_id'] != task['task_id']:
                raise ValueError('Result does not match the workflow and input manifest')
        else:
            state = read(folder/'workflow.json') if (folder/'workflow.json').exists() else {}
            row = {'task_id':task['task_id'],'status':state.get('status','pending'),
                   'stage':state.get('stage'),'resolved':False}
        rows.append(row)
    return rows


def report(output):
    rows = result_rows(output)
    ended = [r for r in rows if r['status'] in TERMINAL]
    config_file = output/'workflow_config.json'
    known = config_file.exists() and read(config_file).get('workflow',{}).get('development_casebook',False)
    value = {'method':report_method(output),'protocol':PROTOCOL,'complete':len(ended)==len(rows),
        'development_regression': bool(known),
        'completed':len(ended),'total':len(rows),'resolved':sum(bool(r['resolved']) for r in ended),
        'runtime_failures':sum(bool(r.get('runtime_failure')) for r in ended),
        'zero_patch':sum(not r.get('changed_files') for r in ended),
        'status_counts':dict(Counter(r['status'] for r in rows)),
        'usage':{key:sum(r.get('usage',{}).get(key,0) for r in rows) for key in
                 ('code_model_calls','vision_model_calls','tool_calls','accounted_prompt_tokens','accounted_completion_tokens','elapsed_seconds')},
        'interpretation':f'固定{len(rows)}题，运行异常保留在分母。参考测试仅在Agent停止后评分。'
                         + ('本轮为工程小测，不代表100题成功率。' if len(rows)==10 else '')
                         + ('本轮使用已知失败反馈与开发验收检查，全部10题保留；不是未见任务泛化成绩。' if known else '')
                         + '不能与旧的可见参考测试19/100直接计算提升。',
        'tasks':rows}
    atomic_json(output/'report.json',value)
    render_batch(output,rows,len(rows))
    (output/'report.md').write_text(
        '# CoreCoder 工程修复流程验收\n\n'+value['interpretation']+'\n\n'
        f"已结束：{value['completed']}/{value['total']}；验证通过：{value['resolved']}/{value['total']}；"
        f"运行异常：{value['runtime_failures']}；零补丁：{value['zero_patch']}。\n\n"
        '逐题报告、补丁、测试反馈及调用统计见 report.html / report.json。\n')
    return value


def evaluate(c,args):
    tasks = prepared_tasks(args.output)
    config = load_workflow_config(args.config)
    config = config.model_copy(update={'code':config.code.model_copy(update={'base_url':f'http://127.0.0.1:{args.port}/v1'})})
    if config.workflow.execution_mode == 'linear' and len(tasks) == 10:
        validate_smoke_preflight(tasks, config, args.preflight_report)
    runtime = scale.frozen_runtime(args.output)
    immutable_json(args.output/'workflow_config.json',config.model_dump())
    plan = {'method':METHOD,'protocol':PROTOCOL,'prepared_sha256':sha256(args.output/'prepared.json'),
        'source_sha256':sha256(runtime/'source_state.json'),'config':config.model_dump(),
        'model':scale.verify_resource_receipts(),'framework_environment':harness.framework_environment(),
        'workers':args.workers,'gpu_policy':scale.gpu_policy(args)}
    immutable_json(args.output/'workflow_plan.json',plan)
    pending = []
    for i,task in enumerate(tasks):
        c.update(stage='validating_inputs',validation_completed=i,validation_total=len(tasks))
        validate_prepared(Path(task['task_file']).parent)
        folder = args.output/'runs'/task['task_id']
        if (folder/'result.json').exists():
            old = read(folder/'result.json')
            if old['method'] != METHOD or old['protocol'] != PROTOCOL or old['task_id'] != task['task_id']:
                raise ValueError('Mismatched task result')
            if read(folder/'config.json') != config.model_dump():
                raise ValueError('Configuration changed for an existing attempt')
            if old['status'] in TERMINAL:
                continue
        resume = folder.exists()
        if resume and not args.resume_interrupted:
            raise ValueError(f'Interrupted task retained: {folder}; use --resume-interrupted')
        if resume and not (folder/'workflow.json').exists():
            raise ValueError(f'Incomplete snapshot: {folder}; retain it and use a new output directory')
        pending.append((task,folder,resume))
    report(args.output)
    if not pending:
        return
    args.service_context_tokens, args.max_num_seqs, args.split = config.service_context_tokens,args.workers,'workflow'
    service,env = scale.start_service(c,args,runtime,'coder',read(runtime/'configs/scale_4gpu_resources.json'))
    active = []
    try:
        harness.capacity_check(c,args,config,env['SCA_CODE_API_KEY'])
        while pending or active:
            if service['proc'].poll() is not None:
                raise RuntimeError('Model service exited; completed results and interrupted workspaces retained')
            while pending and len(active) < args.workers:
                task,folder,resume = pending.pop(0)
                command = [sys.executable,'-u',str(runtime/'scripts/run_repair_workflow.py'),'task',
                    '--output',str(args.output),'--task',task['task_file'],'--config',str(args.output/'workflow_config.json'),
                    '--run-dir',str(folder),'--reader','ocr']
                if resume:
                    command.append('--resume-interrupted')
                entry = c.launch(command,'workflow_'+task['task_id'],env=env,cwd=runtime)
                active.append((entry,folder,time.monotonic()))
            for entry,folder,started in list(active):
                if time.monotonic()-started > config.limits.max_wall_seconds+1200:
                    raise RuntimeError('Task outer timeout; retain current workspace for explicit resume')
                if entry['proc'].poll() is not None:
                    if entry['proc'].returncode or not (folder/'result.json').exists():
                        raise RuntimeError('Task process failed; inspect '+entry['log'])
                    active.remove((entry,folder,started))
                    report(args.output)
                    if read(folder/'result.json').get('runtime_failure'):
                        raise RuntimeError('Task infrastructure/model error; stop scheduling new tasks and inspect '+str(folder))
            c.update(stage='evaluating',pending=len(pending),running=len(active),total=len(tasks))
            if active:
                time.sleep(1)
    finally:
        scale.terminate(service['proc'])
    report(args.output)


def validate_smoke_preflight(tasks, config, path):
    if not path:
        raise ValueError('10-task smoke requires --preflight-report from check_workflow_v3.py')
    checked = read(path)
    if (not checked.get('ready') or checked.get('implementation') != implementation()
            or checked.get('config') != config.model_dump()):
        raise ValueError('Public preflight failed or code/config changed; rerun the CPU preflight first')
    rows = checked['tasks']
    if ([r['task_id'] for r in rows] != [t['task_id'] for t in tasks]
            or any(r['task_sha256'] != sha256(Path(t['task_file'])) for r,t in zip(rows,tasks))):
        raise ValueError('Public preflight does not cover the same fixed ten inputs')


def single(args):
    resume = args.action == 'resume' or args.resume_interrupted
    if resume:
        if not args.run_dir:
            raise ValueError('--run-dir is required to resume')
        task = Task.from_file(args.run_dir/'task.json')
        config = WorkflowConfig.model_validate(read(args.run_dir/'config.json'))
        # Task/reader identity is frozen; credentials are read from the current environment.
        state = read(args.run_dir/'workflow.json')
        args.reader = state.get('reader','ocr')
    else:
        if args.task:
            task = Task.from_file(args.task)
        else:
            problem = args.problem_file.read_text() if args.problem_file else args.problem
            if not args.repo or not problem or not args.test_command:
                raise ValueError('Provide --task OR --repo, --problem/--problem-file and --test-command')
            task = Task(task_id=args.name,repo=str(args.repo.resolve()),problem=problem,test_command=args.test_command,
                images=[ImageInput(image_id=f'shot_{i+1}',path=str(p.resolve())) for i,p in enumerate(args.image)],
                protected_paths=args.protect or ['tests'])
        config = load_workflow_config(args.config)
    if task.task_id in {'.','..'}:
        raise ValueError('Task identifier must be a filename, not a directory reference')
    args.run_dir = args.run_dir or args.output/'requests'/task.task_id
    if not args.run_dir.resolve().is_relative_to('/data/zjr'):
        raise ValueError('Run directories must live under /data/zjr')
    if not args.local:
        result = run_workflow_task(task,config,args.run_dir,reader_name=args.reader,resume=resume)
    else:
        # This convenience option is explicit: it starts one TP4 model service.
        c = scale.Controller(args.output,'local_repair')
        try:
            runtime = scale.frozen_runtime(args.output)
            args.service_context_tokens,args.max_num_seqs,args.split = config.service_context_tokens,1,'workflow'
            configured = f'http://127.0.0.1:{args.port}/v1'
            if resume and config.code.base_url != configured:
                raise ValueError('Resume with the original --port')
            config = config.model_copy(update={'code':config.code.model_copy(update={'base_url':configured})})
            immutable_json(args.output/'single_config.json',config.model_dump())
            immutable_json(args.output/'single_task.json',task.model_dump())
            service,env = scale.start_service(c,args,runtime,'coder',read(runtime/'configs/scale_4gpu_resources.json'))
            command = [sys.executable,'-u',str(runtime/'scripts/run_repair_workflow.py'),'task',
                '--output',str(args.output),'--task',str(args.output/'single_task.json'),
                '--config',str(args.output/'single_config.json'),'--run-dir',str(args.run_dir),'--reader',args.reader]
            if resume:
                command.append('--resume-interrupted')
            entry = c.launch(command,'local_repair',env=env,cwd=runtime)
            c.wait(entry,config.limits.max_wall_seconds+1200)
            result = read(args.run_dir/'result.json')
            c.update(stage='completed',report=str(args.run_dir/'report.html'))
        finally:
            c.close()
    print(json.dumps({'result':result,'report':str(args.run_dir/'report.html')},ensure_ascii=False,indent=2))


def status(args):
    if args.run_dir:
        folder = args.run_dir
        state = read(folder/'workflow.json') if (folder/'workflow.json').exists() else {'status':'not_started'}
        state = {k:state.get(k) for k in ('task_id','status','stage','updated_at','retry_count')}
        state['usage'] = read(folder/'budget.json').get('usage',{}) if (folder/'budget.json').exists() else {}
        state['report'] = str(folder/'report.html')
        state['liveness_note'] = 'File progress only; use the launcher service journal for process liveness.'
    else:
        state = read(args.output/'status.json') if (args.output/'status.json').exists() else {'stage':'not_started'}
        state['controller_alive'] = scale.alive(state.get('pid'),state.get('start_ticks'))
        for child in state.get('children',[]):
            child['alive'] = scale.alive(child['pid'],child.get('start_ticks'))
        if (args.output/'prepared.json').exists():
            rows = result_rows(args.output)
            state['tasks'] = {'total':len(rows),'states':dict(Counter(r['status'] for r in rows)),
                              'resolved':sum(bool(r.get('resolved')) for r in rows)}
        state['report'] = str(args.output/'report.html')
    print(json.dumps(state,ensure_ascii=False,indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['repair','resume','task','demo','prepare','evaluate','all','status','report'])
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--source-plan',type=Path,default=harness.DEFAULT_SOURCE)
    p.add_argument('--reuse-prepared',type=Path,default=harness.DEFAULT_OUTPUT)
    p.add_argument('--limit',type=int,choices=[10,100],default=10,
                   help='Default: fixed first 10 prepared tasks; 100 requires explicit selection')
    p.add_argument('--preflight-report',type=Path,help='Successful CPU check of the same ten tasks and implementation')
    p.add_argument('--config',type=Path,default=ROOT/'configs/workflow-v4.json')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--task',type=Path)
    p.add_argument('--repo',type=Path)
    p.add_argument('--name',default='user_repair')
    problem = p.add_mutually_exclusive_group()
    problem.add_argument('--problem')
    problem.add_argument('--problem-file',type=Path)
    p.add_argument('--test-command')
    p.add_argument('--image',type=Path,action='append',default=[])
    p.add_argument('--protect',action='append',help='Read-only repository path; repeat for multiple paths')
    p.add_argument('--reader',choices=['ocr','vlm','oracle'],default='ocr')
    p.add_argument('--local',action='store_true',help='Explicitly start a local TP4 code model for this repair')
    p.add_argument('--gpus',default='2,5,6,7',help='Four physical indices, or auto to wait for any four idle GPUs')
    p.add_argument('--gpu-candidates',default='0,1,2,3,4,5,6,7',help='Physical GPU indices considered by auto selection')
    p.add_argument('--gpu-poll-seconds',type=float,default=10)
    p.add_argument('--gpu-stable-samples',type=int,default=2)
    p.add_argument('--workers',type=int,choices=[1,2,4],default=2)
    p.add_argument('--port',type=int,default=18876)
    p.add_argument('--gpu-memory-utilization',type=float,default=.8)
    p.add_argument('--gpu-headroom-mib',type=int,default=1536)
    p.add_argument('--allow-shared-gpus',action='store_true',
                   help='With --gpus auto, allow up to 2048 MiB existing use if the model budget plus headroom fits')
    p.add_argument('--wait-for-gpus',action='store_true')
    p.add_argument('--trace-server-signals',action='store_true')
    p.add_argument('--resume-interrupted',action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.gpus=='auto':
        QueuePolicy(candidates=tuple(args.gpu_candidates.split(',')),poll_seconds=args.gpu_poll_seconds,
            stable_samples=args.gpu_stable_samples,memory_fraction=args.gpu_memory_utilization,
            headroom_mib=args.gpu_headroom_mib,allow_shared_gpus=args.allow_shared_gpus).validate()
    args.output = harness.storage(args.output,create=args.action!='status')
    if args.action=='status':
        status(args)
    elif args.action=='report':
        if args.run_dir:
            render_run(args.run_dir)
            export_delivery(args.run_dir)
            print(args.run_dir/'report.html')
        else:
            value = report(args.output)
            print(json.dumps({k:v for k,v in value.items() if k!='tasks'},ensure_ascii=False))
    elif args.action=='demo':
        from screenshot_agent.workflow_demo import demo
        print(json.dumps(demo(args.output),ensure_ascii=False,indent=2))
    elif args.action in ('repair','resume','task'):
        if args.local:
            def cancel_local(signum,frame):
                raise KeyboardInterrupt('Cancellation requested')
            signal.signal(signal.SIGTERM,cancel_local)
        single(args)
    else:
        def cancel(signum,frame):
            raise KeyboardInterrupt('Cancellation requested')
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,cancel)
        c = scale.Controller(args.output,args.action)
        try:
            if args.action in ('prepare','all'):
                prepare(c,args)
            if args.action in ('evaluate','all'):
                evaluate(c,args)
            c.update(stage='prepared' if args.action=='prepare' else 'completed',finished_at=time.time())
        except KeyboardInterrupt:
            c.update(stage='paused',finished_at=time.time())
            raise SystemExit(130) from None
        except Exception as error:
            c.update(stage='failed',error=str(error),finished_at=time.time())
            raise
        finally:
            c.close()


if __name__=='__main__':
    main()
