import argparse
from collections import defaultdict,Counter
import statistics
import torch

from .config import OUT,PARAMS,ARMS,DIAGNOSTIC,case,previous,original,read,write,digest,now
from ..local_regeneration.fields import keys,lookup
from ..local_regeneration.sampling import join_original

def grouped(rows,fn):
    values=defaultdict(list)
    for row in rows:
        value=fn(row)
        if value is not None:values[row['uid']].append(float(value))
    by={k:statistics.mean(v) for k,v in values.items()}
    return dict(mean=statistics.mean(by.values()) if by else None,by_object=by)

def build():
    rows=[read(case(i,s)/'result.json') for i in PARAMS['indices'] for s in PARAMS['seeds'] if (case(i,s)/'result.json').exists()]
    arms={};costs=[];failures=Counter();updates=Counter();reachability=[];repeats=[]
    for arm in ARMS:
        arms[arm]=dict(diagnostic_only=arm in DIAGNOSTIC,accepted=sum(r['selections'][arm]['accepted_edit'] for r in rows),endpoints={})
        for ep in ('accepted','candidate'):
            arms[arm]['endpoints'][ep]=dict(
                cd_gain=grouped(rows,lambda r:(r['baseline']['surface']['chamfer_l1']-r['endpoints'][arm][ep]['surface']['chamfer_l1'])/r['baseline']['surface']['chamfer_l1']),
                fscore_0p01_delta=grouped(rows,lambda r:r['endpoints'][arm][ep]['surface']['fscore_0p01']-r['baseline']['surface']['fscore_0p01']),
                fscore_0p02_delta=grouped(rows,lambda r:r['endpoints'][arm][ep]['surface']['fscore_0p02']-r['baseline']['surface']['fscore_0p02']),
                bad_error_gain=grouped(rows,lambda r:(r['baseline']['novel_clipped_error']['bad']-r['endpoints'][arm][ep]['novel_clipped_error']['bad'])/max(r['baseline']['novel_clipped_error']['bad'],1.e-9)),
                good_error_gain=grouped(rows,lambda r:(r['baseline']['novel_clipped_error']['good']-r['endpoints'][arm][ep]['novel_clipped_error']['good'])/max(r['baseline']['novel_clipped_error']['good'],1.e-9)))
    for row in rows:
        root=case(row['index'],row['seed']);method=read(root/'method_complete.json');generations=[]
        for mode in ('method','diagnostic'):generations +=read(root/f'generation_{mode}.json')['candidates']
        for spec in generations:
            for step in spec['guidance']:updates[(spec['arm'],'attempted')]+=1;updates[(spec['arm'],'accepted')]+=int(step.get('accepted',False))
            if spec['guidance'] and spec['guidance'][0].get('before'):
                route=spec['guidance'][0]['before']['routing'][0]
                reachability.append(dict(index=row['index'],seed=row['seed'],arm=spec['arm'],noise=spec['noise'],queries=route['queries'],coarse_reachable=route['reachable_parent'],outside_envelope=spec['domain']['target_outside_envelope']))
        paired=read(root/'diagnostic_complete.json')
        if method['domains']['input']['target_queries']==paired['domains']['paired_input']['target_queries']:
            for noise in PARAMS['proposal_seeds']:
                first=torch.load(root/'candidates'/f'input_n{noise}.pt',weights_only=False)
                second=torch.load(root/'candidates'/f'paired_input_n{noise}.pt',weights_only=False)
                difference=(first['features']-second['features'])/first['std']
                repeats.append(dict(index=row['index'],seed=row['seed'],noise=noise,identical_target_domain=True,max_normalized_delta=float(difference.abs().max()),rms_normalized_delta=float(difference.square().mean().sqrt())))
        for proposal in method['rows']:failures.update(k for k,v in proposal['decision']['checks'].items() if not v)
        costs.append(dict(index=row['index'],seed=row['seed'],budget=row['budget'],
            peak_gib=max(s['peak_gib'] for s in generations),
            full_bank_sampling_seconds=sum(s['seconds'] for s in generations if s['arm']=='prior'),
            guided_sampling_seconds=sum(s['seconds'] for s in generations if s['arm']=='input')))
    v2=grouped(rows,lambda r:(r['baseline']['surface']['chamfer_l1']-r['v2']['surface']['chamfer_l1'])/r['baseline']['surface']['chamfer_l1'])
    comparisons={}
    for control in ('prior','budget_prior'):
        comparisons[control]=grouped(rows,lambda r:(r['endpoints'][control]['accepted']['surface']['chamfer_l1']-r['endpoints']['input']['accepted']['surface']['chamfer_l1'])/r['endpoints'][control]['accepted']['surface']['chamfer_l1'])
    comparisons['v2']=grouped(rows,lambda r:(r['v2']['surface']['chamfer_l1']-r['endpoints']['input']['accepted']['surface']['chamfer_l1'])/r['v2']['surface']['chamfer_l1'])
    result=dict(time=now(),cases=len(rows),complete=len(rows)==8,arms=arms,records=rows,costs=costs,v2_delivered_cd_gain=v2,
        method_GT_used=False,development_only=True,failed_checks=dict(failures),coarse_reachability=reachability,numerical_repeats=repeats,input_incremental_cd_gain=comparisons,
        guidance_updates={arm:dict(attempted=updates[(arm,'attempted')],accepted=updates[(arm,'accepted')]) for arm in ('input',*DIAGNOSTIC)})
    write(OUT/'summary.json',result)
    pct=lambda x:'NA' if x is None else f'{100*x:.4f}%'
    text=['# V3 原生Shape Flow几何引导','',f'时间：{now()}；完成{len(rows)}/8案例。',
       '同一SS、三维编辑区、原生12步与参考图；输入选择封存后才构建paired/oracle诊断。所有交付均为原生latent直接解码。',
       '', '|组别|诊断组|接受数|交付CD下降|候选CD下降|交付F@.01变化|原正确域误差下降|', '|---|---|---:|---:|---:|---:|---:|']
    for arm,item in arms.items():
        e=item['endpoints'];f=e['accepted']['fscore_0p01_delta']['mean']
        text.append(f'|{arm}|{item["diagnostic_only"]}|{item["accepted"]}/{len(rows)}|{pct(e["accepted"]["cd_gain"]["mean"])}|{pct(e["candidate"]["cd_gain"]["mean"])}|{f:+.7f}|' +pct(e['accepted']['good_error_gain']['mean'])+'|' if f is not None else f'|{arm}|{item["diagnostic_only"]}|0/0|NA|NA|NA|NA|')
    text += ['',f'冻结V2交付平均CD下降：{pct(v2["mean"])}。所有变化相对相同输入配准stock base，按物体内2seeds平均后物体等权。',
      f'输入引导相对各自交付结果的增量CD下降：prior={pct(comparisons["prior"]["mean"])}, budget_prior={pct(comparisons["budget_prior"]["mean"])}, V2={pct(comparisons["v2"]["mean"])}。',
      'paired_input与oracle均用GT交点筛选共同射线，只作机制诊断，不能计入实际方法；actual input保留完整输入射线域。',
      '新视角旧正确/错误域由base固定；截断深度/轮廓误差与未截断深度、缺失比例在每例result.json分别保存。',
      '', '|案例|prior候选数（预算）|prior/input实测成本比|预算覆盖|全程峰值GiB|input采样秒|', '|---|---:|---:|---|---:|---:|']
    for c in costs:
        b=c['budget'];text.append(f'|{c["index"]:03d}/{c["seed"]}|{b["prior_candidates"]}|{b["ratio"]:.3f}|{b["sufficient"]}|{c["peak_gib"]:.2f}|{c["guided_sampling_seconds"]:.1f}|')
    text += ['',f'真实引导接受次数：{result["guidance_updates"]}。',f'实际方法候选未通过的检查：{dict(failures)}。',
      '预算用生成+原生评分的实测阶段成本选择前缀，无GT；离散超出或bank不足均公开。全bank预生成开销另列summary.json，不隐藏未用提案的实验成本。',
      '计时从冻结V2配准/证据/SS输入开始，排除了共享准备、模型初始化、导出及GT评估；不等于从原始图像到mesh的总延迟。',
      '每个引导时刻的32-grid目标覆盖与包络外目标数量见summary.json的coarse_reachability。GT目标缺少粗SS支持时，不能把oracle失败解释为Flow表示能力上限。',
      f'输入域与paired域完全一致时，保存数值重复检查{len(repeats)}条；它们用于识别CUDA数值波动，不是额外泛化样本。',
      ('同目标重复的最大normalized latent差异：'+f'{max(r["max_normalized_delta"] for r in repeats):.6f}，RMS最大'+f'{max(r["rms_normalized_delta"] for r in repeats):.6f}。小幅指标变化需要结合这种波动解释。') if repeats else '当前没有完全同域的重复对照。',
      '当前仍为已检查开发物体的渲染输入。010整体结构/配准误差、011大区域编辑应分开解释；损失下降和技术检查通过不代表几何改善。','']
    (OUT/'REPORT.md').write_text('\n'.join(text));print(dict(cases=len(rows),complete=result['complete']),flush=True);return result

def verify(indices=None,seeds=None):
    torch.set_num_threads(2);latents=0;composites=0;seen=set()
    partial=indices is not None or seeds is not None
    indices=PARAMS['indices'] if indices is None else indices;seeds=PARAMS['seeds'] if seeds is None else seeds
    for i in indices:
        for s in seeds:
            root=case(i,s);method=read(root/'method_complete.json');diag=read(root/'diagnostic_complete.json')
            if method['GT_used'] or digest(root/'method_complete.json')!=read(root/'method_sealed.json')['decision_sha256']:raise RuntimeError('input decisions changed')
            if diag['domains']['paired_input']['query_ids_sha256']!=diag['domains']['oracle']['query_ids_sha256']:raise RuntimeError('paired IDs differ')
            from .selection import budget_prefix
            prefix,budget=budget_prefix([r for r in method['rows'] if r['arm']=='prior'],sum(r['sampling_seconds']+r['score_seconds'] for r in method['rows'] if r['arm']=='input'))
            if budget!=method['budget'] or len(prefix)!=method['selections']['budget_prior']['candidates']:raise RuntimeError('budget allocation replay differs')
            plan=torch.load(previous(i,s)/'plan.pt',weights_only=False);env=torch.as_tensor(plan['roi']['envelope']);alpha=torch.as_tensor(plan['roi']['alpha'])
            old=torch.load(original(i,s)/'original_latent.pt',weights_only=False)
            support=torch.load(previous(i,s)/(PARAMS['support']+'.pt'),weights_only=False)['coords']
            for mode in ('method','diagnostic'):
                for spec in read(root/f'generation_{mode}.json')['candidates']:
                    payload=torch.load(spec['path'],weights_only=False)
                    if not torch.equal(payload['coords'],support):raise RuntimeError('coarse supports differ')
                    idx,present=join_original(old['coords'],payload['coords']);outside=~lookup(env,payload['coords'][:,1:]).bool()
                    if not bool(present[outside].all()) or not torch.equal(payload['features'][outside],old['features'][idx[outside]]):raise RuntimeError('exterior denormalized features differ')
                    if bool(spec['GT_used'])!=(spec['arm'] in DIAGNOSTIC):raise RuntimeError('GT labeling differs')
                    latents+=1
            base=torch.load(previous(i,s)/'base_fields.pt',weights_only=False);outside=lookup(alpha,base['coords'],16)==0
            bk=keys(base['coords'][outside])
            for selection in (method,diag):
                for arm,sel in selection['selections'].items():
                    for ep in ('accepted','candidate'):
                        if not (root/arm/f'{ep}.obj').is_file():raise RuntimeError('missing endpoint OBJ')
                        path=sel[ep].get('composite_fields')
                        if path and path not in seen:
                            state=torch.load(path,weights_only=False);k=keys(state['coords']);pos=torch.searchsorted(k,bk)
                            if not torch.equal(k[pos],bk) or not torch.equal(state['raw'][pos],base['raw'][outside]):raise RuntimeError('composite exterior fields differ')
                            seen.add(path);composites+=1
    result=dict(time=now(),passed=True,cases=len(indices)*len(seeds),partial=partial,latents=latents,composites=composites,method_seals_exact=True,paired_domains_exact=True,budget_allocation_exact=True)
    name='artifact_checks_'+('_'.join(map(str,indices)))+'_'+('_'.join(map(str,seeds)))+'.json' if partial else 'artifact_checks.json'
    write(OUT/'validation'/name,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--index',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.verify:print(verify([a.index] if a.index is not None else None,[a.seed] if a.seed is not None else None),flush=True)
    else:build()
