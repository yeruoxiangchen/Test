#!/usr/bin/env python3
"""Supervise frozen, disjoint CodeOCR shards and archive their full-data report."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import zipfile
from pathlib import Path

from screenshot_agent.codeocr_benchmark import codeocr_report
from screenshot_agent.control import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress(output):
    state = load(output/'status.json') if (output/'status.json').exists() else {}
    workers = []
    for path in sorted(output.glob('shard_*/codeocr/status.json')):
        workers.append({'shard':path.parents[1].name,**load(path)})
    return {**state,'observed_completed':sum(w['completed'] for w in workers),'workers':workers}


def finalize(output, dataset, state):
    rows = [json.loads(s) for s in (dataset/'images.jsonl').read_text().splitlines() if s.strip()]
    records = []
    configs = []
    stats = []
    for folder in sorted(p for p in output.glob('shard_*') if p.is_dir()):
        report = load(folder/'codeocr/report.json')
        if not report['complete'] or load(folder/'launcher_status.json')['exit_code']!=0:
            raise ValueError('A worker did not complete and release its model')
        records.extend(report['records'])
        config = load(folder/'config.json')
        for kind in ['code','vision']:
            config[kind].pop('base_url',None)
        configs.append(config)
        stats.append({'shard':folder.name,**load(folder/'final_model_stats.json')})
    expected = {(row['sample_id'],reader) for row in rows for reader in ['ocr','vlm']}
    actual = [(r['sample_id'],r['reader']) for r in records]
    if set(actual)!=expected or len(actual)!=len(expected) or len(records)!=400:
        raise ValueError('Full expected 100 sources x 2 ratios x 2 readers were not covered exactly once')
    if any(c!=configs[0] for c in configs):
        raise ValueError('Workers used different non-endpoint configurations')
    changed = [n for n,v in state['source_sha256'].items() if sha(ROOT/n)!=v]
    if changed:
        raise ValueError(f'Runtime source changed while running: {changed}')
    for row in rows:
        if sha(dataset/row['image'])!=row['image_sha256'] or sha(dataset/row['reference'])!=row['source_sha256']:
            raise ValueError('A dataset image/reference changed during the experiment')
    records.sort(key=lambda r:(r['sample_id'],r['reader']))
    build = load(dataset/'build_report.json')
    combined = output/'combined'
    combined.mkdir(exist_ok=True)
    report = codeocr_report(build,records,['ocr','vlm'],combined)
    report.update(expected_readings=400,source_files=100,same_non_endpoint_config=True,
                  runtime_sources_unchanged=True,model_stats=stats)
    atomic_json(combined/'report.json',report)
    artifact = ROOT/'docs/artifacts/codeocr_full_v4'
    if artifact.exists():
        raise ValueError('Use a new archive version; an existing full-data archive will not be overwritten')
    artifact.mkdir(parents=True)
    for source,name in [(combined/'report.json','report.json'),(combined/'report.md','report.md'),
                        (dataset/'build_report.json','build_report.json'),(dataset/'images.jsonl','images.jsonl'),
                        (output/'source_state.json','source_state.json'),(output/'source_snapshot.zip','source_snapshot.zip'),
                        (ROOT/'CODEOCR_FULL_UPSTREAM.json','upstream.json')]:
        shutil.copy2(source,artifact/name)
    for folder in sorted(p for p in output.glob('shard_*') if p.is_dir()):
        target = artifact/folder.name
        target.mkdir()
        for name in ['config.json','launch.json','final_model_stats.json','launcher_status.json']:
            shutil.copy2(folder/name,target/name)
        for result in sorted((folder/'codeocr').glob('*/*/result.json')):
            relative = result.parent.relative_to(folder/'codeocr')
            trial = target/'readings'/relative
            trial.mkdir(parents=True)
            for name in ['result.json','prediction.txt','events.jsonl']:
                if (result.parent/name).exists():
                    shutil.copy2(result.parent/name,trial/name)
    with zipfile.ZipFile(artifact/'dataset.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(dataset.rglob('*')):
            if file.is_file():
                bundle.write(file,file.relative_to(dataset))
    lines = ['# CodeOCR 全量重建数据评测','', '使用作者公开的全部 100 份代码，来自 100 个仓库；1×/2×两档，共 200 张完整代码图像，OCR/VLM 合计 400 次读取。',
             '未按长度、语法或模型结果筛除样本；仓库分组先于图像变体。全部图片为合成数据。',
             '本轮覆盖完整重建源数据的两档输入；未运行 CodeOCR 其他下游任务和另外的退化档位，也不衡量代码修复成功率。','',
             '模型：Qwen3-VL-8B-Instruct，BF16、SDPA、贪心生成；每次输出上限 2048 token，服务上下文 16384。',
             'CPU 基线：RapidOCR 1.4.4 / ONNX Runtime；两个分片使用相同模型、参数、原图和指标。',
             '相比上一轮十文件试验，仅服务上下文由 8192 扩到 16384 以容纳较宽图片；输出上限未变。源码和数据在运行期冻结。','',
             '| 读取方式 | 档位 | 已评分 / 尝试 | 归一化 CER | 含请求失败的 CER¹ | WER | 行相似度 | JSON 合规 | 语法通过 | 平均秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'N/A' if x is None else f'{x:.2%}'

    for s in report['summary']:
        seconds='N/A' if s['mean_seconds'] is None else f"{s['mean_seconds']:.2f}"
        lines.append(f"| {s['reader']} | {s['ratio']} | {s['completed']}/{s['attempted']} | {fmt(s['normalized_cer_mean'])} | "
                     f"{fmt(s['cer_including_errors_mean'])} | {fmt(s['normalized_wer_mean'])} | {fmt(s['line_similarity_mean'])} | "
                     f"{s['schema_valid']}/{s['completed']} | {s['syntax_valid']}/{s['attempted']} | {seconds} |")
    peaks = ', '.join(f"{s['peak_cuda_memory_bytes']/2**30:.2f} GiB" for s in stats)
    lines += ['', '¹ 请求异常按 CER=1 纳入全尝试平均；输出格式异常保留原文并照常评分，不删除失败样本。OCR 的 JSON 合规列仅代表结构化工具返回。',
              '归一化 CER/WER/行相似度调用未修改的上游纯函数；忽略空白行与行尾空格，保留缩进。另存原始 CER 和精确匹配。',
              '语法仅用 ast.parse 检查，不执行这 100 份代码。完整程序是否正确仍需独立修复测试。','',
              '字体使用固定 Noto Sans Mono CJK。三份代码中的 ✅/❌/📊 使用固定 Noto Emoji 字体逐字形补齐；不替换参考字符、不截短代码，上游渲染源码未修改。',
              'API 标价配置为零，实际 GPU/电力/运行时间成本不包含在该价格中。',
              f"单分片 PyTorch 峰值已分配显存：{peaks}；驱动保留内存另计。",'',
              '完整报告、逐条预测、请求轨迹、数据与源码压缩包及 SHA-256 在 [证据目录](artifacts/codeocr_full_v4/report.json)。']
    document = ROOT/'docs/experiment_report_codeocr_full_v4.md'
    document.write_text('\n'.join(lines)+'\n')
    shutil.copy2(document,artifact/'experiment_report.md')
    atomic_json(artifact/'verification.json',{'readings':400,'source_files':100,'image_variants':200,
        'same_non_endpoint_config':True,'runtime_sources_unchanged':True,
        'sha256':{str(p.relative_to(artifact)):sha(p) for p in artifact.rglob('*') if p.is_file()}})
    return report


def run(output,dataset,gpus,port):
    build = load(dataset/'build_report.json')
    if build.get('selected_files')!=100 or not build.get('all_visible_glyphs_supported'):
        raise ValueError('Build and audit the complete 100-source dataset first')
    if output.exists():
        raise ValueError('Use a new run directory')
    used = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory = {int(line.split(',')[0]):int(line.split(',')[1]) for line in used.splitlines()}
    if not 1<=len(gpus)<=4 or len(set(gpus))!=len(gpus) or any(memory[g]>1000 for g in gpus):
        raise ValueError('Select at most four distinct idle GPUs')
    for index in range(len(gpus)):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',port+index))
    output.mkdir(parents=True)
    paths = [*ROOT.glob('src/**/*.py'),ROOT/'scripts/local_model_api.py',ROOT/'scripts/run_codeocr_local.sh',
             ROOT/'scripts/run_codeocr_full.py',ROOT/'configs/official-local.json']
    state = {'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'gpus':gpus,'port_base':port,
             'dataset':str(dataset),'manifest_sha256':sha(dataset/'images.jsonl'),'started_at':time.time()}
    atomic_json(output/'source_state.json',state)
    with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as snapshot:
        for p in paths:
            snapshot.write(p,p.relative_to(ROOT))
        for p in [*ROOT.glob('tests/*.py'),*ROOT.glob('requirements*.txt'),ROOT/'pyproject.toml',ROOT/'CODEOCR_FULL_UPSTREAM.json']:
            snapshot.write(p,p.relative_to(ROOT))
    children,logs = [],[]

    def stop(signum,frame):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        for index,gpu in enumerate(gpus):
            environment = dict(os.environ,SCA_GPU=str(gpu),SCA_PORT=str(port+index),SCA_SHARD_INDEX=str(index),
                SCA_SHARD_COUNT=str(len(gpus)),SCA_RUN_DIR=str(output/f'shard_{index}'),
                SCA_CODEOCR_MANIFEST=str(dataset/'images.jsonl'),SCA_MAX_CONTEXT='16384',SCA_MAX_OUTPUT='2048')
            log = (output/f'shard_{index}.log').open('w')
            logs.append(log)
            children.append(subprocess.Popen(['bash','scripts/run_codeocr_local.sh'],cwd=ROOT,env=environment,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(child.poll() is None for child in children):
            atomic_json(output/'status.json',{'status':'running','total':400,'worker_pids':[c.pid for c in children],
                                            'started_at':state['started_at']})
            info = progress(output)
            print(f"CodeOCR full: {info['observed_completed']}/400 readings; worker exit codes {[c.poll() for c in children]}",flush=True)
            if any(child.poll() not in (None,0) for child in children):
                raise RuntimeError('A worker exited; inspect its log. Completed results are preserved.')
            time.sleep(30)
        if any(c.returncode for c in children):
            raise RuntimeError('Worker failure; no complete-data claim will be made')
        report = finalize(output,dataset,state)
        atomic_json(output/'status.json',{'status':'complete','total':400,'completed':400,
            'errors':sum(r['status']=='error' for r in report['records']),'elapsed_seconds':time.time()-state['started_at'],
            'model_gpus_released':gpus,'report':str(ROOT/'docs/experiment_report_codeocr_full_v4.md')})
        print(json.dumps(load(output/'status.json'),ensure_ascii=False,indent=2),flush=True)
    except BaseException as error:
        atomic_json(output/'status.json',{'status':'cancelled' if isinstance(error,KeyboardInterrupt) else 'failed',
                                         'total':400,'error':str(error)})
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
        for log in logs:
            log.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('datasets/generated/codeocr_full_v4'))
    parser.add_argument('--gpus',default='6,7')
    parser.add_argument('--port-base',type=int,default=18780)
    parser.add_argument('--status',action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(progress(args.output.resolve()),ensure_ascii=False,indent=2))
    else:
        run(args.output.resolve(),args.dataset.resolve(),[int(s) for s in args.gpus.split(',')],args.port_base)
