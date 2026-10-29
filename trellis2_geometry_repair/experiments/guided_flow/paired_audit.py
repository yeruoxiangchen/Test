"""Post-seal, fixed-noise geometry audit; never changes method selections."""
import argparse
import statistics
import time
from collections import defaultdict
import torch

from .config import OUT,PARAMS,case,previous,read,write,digest,now,setup,lock_case
from ...runtime import load_model,decode_native
from ...geometry import transform
from ...metrics import surface_scores
from ..repair_diagnostic.data import evaluation_points

def aggregate(rows,key):
    by=defaultdict(list)
    for r in rows:by[r['uid']].append(r[key])
    per={k:statistics.mean(v) for k,v in by.items()}
    return dict(mean=statistics.mean(per.values()),by_object=per)

def one(index,seed,decoder):
    root=case(index,seed);path=root/'paired_geometry_audit.json'
    if path.exists():return read(path)
    method=read(root/'method_complete.json');diag=read(root/'diagnostic_complete.json')
    seal=read(root/'method_sealed.json')
    if method['GT_used'] or seal['decision_sha256']!=digest(root/'method_complete.json'):raise RuntimeError('unsealed method')
    if diag['domains']['paired_input']['query_ids_sha256']!=diag['domains']['oracle']['query_ids_sha256']:raise RuntimeError('nonpaired diagnostic queries')
    target=evaluation_points(index);matrix=torch.load(previous(index,seed)/'plan.pt',weights_only=False)['matrix'].cuda()
    result=read(root/'result.json');records=[];replay=[]
    with torch.no_grad():
        for noise in PARAMS['proposal_seeds']:
            scores={}
            for arm in ('prior','input','paired_input','oracle'):
                name=f'{arm}_n{noise}';payload=torch.load(root/'candidates'/f'{name}.pt',weights_only=False)
                v,f=decode_native(decoder,payload['features'].cuda(),payload['coords'].cuda());v=transform(v,matrix)
                scores[arm]=surface_scores(v.cpu().numpy(),f.cpu().numpy(),target)
                for endpoint in ('accepted','candidate'):
                    selection=result['selections'][arm][endpoint]
                    if selection['name']==name:
                        prior_score=result['endpoints'][arm][endpoint]['surface']
                        replay.append(dict(name=name,endpoint=endpoint,cd_abs_difference=abs(prior_score['chamfer_l1']-scores[arm]['chamfer_l1']),
                                           fscore_0p01_abs_difference=abs(prior_score['fscore_0p01']-scores[arm]['fscore_0p01'])))
                del v,f,payload
            p=scores['prior'];i=scores['input'];a=scores['paired_input'];o=scores['oracle']
            records.append(dict(index=index,seed=seed,noise=noise,uid=method['uid'],surface=scores,
                guided_vs_same_noise_prior=(p['chamfer_l1']-i['chamfer_l1'])/p['chamfer_l1'],
                oracle_vs_paired_input=(a['chamfer_l1']-o['chamfer_l1'])/a['chamfer_l1'],
                repeat_comparable=method['domains']['input']['target_queries']==diag['domains']['paired_input']['target_queries'],
                repeat_cd_absolute=abs(i['chamfer_l1']-a['chamfer_l1']),
                repeat_cd_relative=abs(i['chamfer_l1']-a['chamfer_l1'])/i['chamfer_l1']))
    value=dict(time=now(),index=index,seed=seed,records=records,OBJ_metric_replay=replay,
        native_float32_scoring=True,GT_used=True,diagnostic_only=True,selection_sha256=seal['decision_sha256'])
    write(path,value);print(dict(index=index,seed=seed,paired_geometry_audit=True),flush=True);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',default='5,6,10,11');a=p.parse_args();setup()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.8)
    decoder=load_model('shape_decoder',binding_root=OUT);records=[];replay=[];started=time.monotonic()
    for index in map(int,a.indices.split(',')):
        for seed in PARAMS['seeds']:
            with lock_case(index,seed):value=one(index,seed,decoder)
            records+=value['records'];replay+=value['OBJ_metric_replay']
    values=[r for r in records if r['repeat_comparable']]
    summary=dict(time=now(),complete=len(records)==16,records=records,
        guided_vs_same_noise_prior=aggregate(records,'guided_vs_same_noise_prior'),
        oracle_vs_paired_input=aggregate(records,'oracle_vs_paired_input'),
        same_target_repeat=aggregate(values,'repeat_cd_relative') if values else None,
        OBJ_replay_max_cd_difference=max((r['cd_abs_difference'] for r in replay),default=0.),
        OBJ_replay_max_fscore_difference=max((r['fscore_0p01_abs_difference'] for r in replay),default=0.),
        GT_used=True,diagnostic_only=True,selection_changed=False,seconds=time.monotonic()-started)
    write(OUT/'paired_geometry_audit.json',summary)
    pct=lambda x:f'{100*x:.4f}%'
    text=['# 同噪声配对几何审计','',f'时间：{now()}；共{len(records)}/16个case×noise配对。',
       '每个配对共享同一SS、编辑区和初始噪声，直接评估保存latent的原生float32解码；没有从两个噪声中重新按GT挑选。',
       f'输入引导相对同噪声prior的平均CD下降：{pct(summary["guided_vs_same_noise_prior"]["mean"])}。',
       f'oracle相对同噪声paired_input的平均CD下降：{pct(summary["oracle_vs_paired_input"]["mean"])}。',
       f'与主评估OBJ的最大CD重放差异：{summary["OBJ_replay_max_cd_difference"]:.10g}；F@.01差异：{summary["OBJ_replay_max_fscore_difference"]:.10g}。',
       '差异只作为序列化/浮点采样审计，主方法成绩仍以已封存OBJ评估为准。',
       'paired_input/oracle及此处GT评分均为诊断；原有方法选择与门限没有改变。', '']
    if values:text.insert(-1,f'完全同输入域的重复运行CD相对绝对差均值：{pct(summary["same_target_repeat"]["mean"])}；包含CUDA和离散解码敏感性，不是置信区间。')
    (OUT/'PAIRED_GEOMETRY_AUDIT.md').write_text('\n'.join(text));print({k:v for k,v in summary.items() if k!='records'},flush=True)

if __name__=='__main__':main()
