from collections import defaultdict,Counter
import numpy as np
from .config import OUT,PROFILE,PARAMS,ARMS,case,source,read,write,now


def grouped(rows,fn):
    values=defaultdict(list)
    for row in rows:
        x=fn(row)
        if x is not None and np.isfinite(x):values[row['uid']].append(x)
    by_object={uid:float(np.mean(v)) for uid,v in values.items()}
    return dict(mean=float(np.mean(list(by_object.values()))) if by_object else None,by_object=by_object)


def build():
    rows=[read(case(i,s)/'result.json') for i in PARAMS['indices'] for s in PARAMS['seeds'] if (case(i,s)/'result.json').exists()]
    arms={}
    for arm in ARMS:
        arms[arm]=dict(accepted=sum(r['selections'][arm]['accepted_edit'] for r in rows),endpoints={},
            accepted_kinds=dict(Counter(r['selections'][arm]['accepted']['kind'] for r in rows if r['selections'][arm]['accepted_edit'])))
        for ep in ('accepted','candidate'):
            arms[arm]['endpoints'][ep]=dict(
               full_cd_gain=grouped(rows,lambda r:(r['previous_baseline']['surface']['chamfer_l1']-r['endpoints'][arm][ep]['surface']['chamfer_l1'])/max(r['previous_baseline']['surface']['chamfer_l1'],1.e-8)),
               local_cd_gain=grouped(rows,lambda r:(r['registered_baseline']['surface']['chamfer_l1']-r['endpoints'][arm][ep]['surface']['chamfer_l1'])/max(r['registered_baseline']['surface']['chamfer_l1'],1.e-8)),
               fscore_0p01_delta=grouped(rows,lambda r:r['endpoints'][arm][ep]['surface']['fscore_0p01']-r['registered_baseline']['surface']['fscore_0p01']),
               fscore_0p02_delta=grouped(rows,lambda r:r['endpoints'][arm][ep]['surface']['fscore_0p02']-r['registered_baseline']['surface']['fscore_0p02']))
    pose=grouped(rows,lambda r:(r['previous_baseline']['surface']['chamfer_l1']-r['registered_baseline']['surface']['chamfer_l1'])/max(r['previous_baseline']['surface']['chamfer_l1'],1.e-8))
    failures=Counter();details=[]
    for row in rows:
        root=case(row['index'],row['seed']);selection=read(root/'input_complete.json')
        for candidate in selection['rows']:
            if not candidate.get('invalid'):
                failures.update(name for name,passed in candidate['decision']['checks'].items() if not passed)
        ss=read(root/'ss_complete.json');shape=read(root/'sampling_complete.json')
        domain=selection.get('scoring_domain',dict(global_verified_pixels=row['plan']['trusted_pixels'],local_verified_pixels=row['plan']['trusted_pixels']))
        details.append(dict(index=row['index'],seed=row['seed'],scoring_domain=domain,
            ss_requested=row['plan']['needs_ss'],anchors=row['plan'].get('verified_anchor_cells',0),
            heldout_photo_available=selection['baseline']['photo'] is not None,
            ss_coverage=[v.get('coverage') for v in ss['variants']],active_fraction=row['plan']['active_fraction'],
            seconds=dict(planning=row['plan']['seconds'],ss=ss.get('seconds',0),shape=shape.get('seconds',0),selection=selection.get('seconds',0)),
            selection_peak_gib=selection.get('peak_gib')))
    changed=sum(r['plan']['alignment']['changed'] for r in rows)
    value=dict(time=now(),profile=PROFILE,cases=len(rows),complete=len(rows)==8,arms=arms,registration_cd_gain=pose,registration_changed_cases=changed,records=rows,
               input_details=details,rejected_proposal_checks=dict(failures),
               development_only=True,GT_used_for_selection=False,composites_labeled=True)
    write(OUT/'summary.json',value)
    def pct(x):return 'NA' if x is None else f'{100*x:.4f}%'
    text=['# 原生 Flow 局部再生成','',f'版本：{PROFILE}；时间：{now()}；评估{len(rows)}/8。冻结模型，无模型微调。',
       '','相同输入、正向全局配准与共同新视角。Shape为固定SS参照，adaptive在需要时加入SS/Shape再生成。',
       'FDG场合成与单latent直接解码明确区分，最终输出保留对应原生候选。GT仅用于封存后的评估。',
       '',f'仅重新配准的平均CD变化：{pct(pose["mean"])}；这部分不算局部编辑增益。',
       f'实际更换全局变换：{changed}/{len(rows)}；保留同一变换时的微小CD差异来自缓存变换/OBJ序列化精度，不能声称是配准收益。','',
       '|组别|接受数|交付相对旧base CD下降|交付相对新配准base CD下降|候选相对新配准base CD下降|',
       '|---|---:|---:|---:|---:|']
    for arm,item in arms.items():
        e=item['endpoints'];text.append(f'|{arm}|{item["accepted"]}/{len(rows)}|{pct(e["accepted"]["full_cd_gain"]["mean"])}|{pct(e["accepted"]["local_cd_gain"]["mean"])}|{pct(e["candidate"]["local_cd_gain"]["mean"])}|')
    text+=['','## 配对案例','', '|案例|新配准base CD|Shape交付CD|Adaptive交付CD|Adaptive局部CD下降|F@.01变化|F@.02变化|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        b=row['registered_baseline']['surface'];a=row['endpoints']['adaptive']['accepted']['surface'];s=row['endpoints']['shape']['accepted']['surface']
        text.append(f'|{row["index"]:03d}/{row["seed"]}|{b["chamfer_l1"]:.8f}|{s["chamfer_l1"]:.8f}|{a["chamfer_l1"]:.8f}|{pct((b["chamfer_l1"]-a["chamfer_l1"])/b["chamfer_l1"])}|{a["fscore_0p01"]-b["fscore_0p01"]:+.5f}|{a["fscore_0p02"]-b["fscore_0p02"]:+.5f}|')
    text+=['','## 输入约束与成本','', '|案例|局部/全局确认射线|确认格点|SS后覆盖|编辑token比例|规划/SS/Shape/筛选 秒|筛选峰值GiB|',
           '|---|---:|---:|---|---:|---|---:|']
    for item in details:
        d=item['scoring_domain'];coverage=', '.join(f'{v["covered_after"]}/{v["requested"]}' for v in item['ss_coverage'] if v) or 'SS skipped'
        seconds='/'.join(f'{v:.1f}' for v in item['seconds'].values());peak=item['selection_peak_gib']
        text.append(f'|{item["index"]:03d}/{item["seed"]}|{d["local_verified_pixels"]}/{d["global_verified_pixels"]}|{item["anchors"]}|{coverage}|{pct(item["active_fraction"])}|{seconds}|{peak:.2f}|')
    text+=['',f'未通过的提案检查计数：{dict(failures)}。同一提案可同时未通过多项。',
           f'实际接受输出类型：{ {arm:arms[arm]["accepted_kinds"] for arm in ARMS} }。',
           f'可用heldout RGB验证的案例：{sum(item["heldout_photo_available"] for item in details)}/{len(details)}；没有有效支持的案例不能声称通过了光度验证。',
           '耗时为共享机器上的实际阶段时间，包含加载和多提案搜索，不等于单次编辑推理延时；评估与GPU技术检查另计。']
    text+=['','正值为改善。全部回退计入分母。候选、场合成及仅配准改善不能混称为直接latent修复成绩。',
       '本轮仍为已检查开发物体的渲染输入，不是独立泛化或真实拍摄基准。逐案例路由、证据像素、区域比例、配准选择及全部提案保留。','']
    (OUT/'REPORT.md').write_text('\n'.join(text));print(dict(cases=len(rows),complete=value['complete']),flush=True)
    return value


def verify(indices=None,seeds=None):
    import torch
    from .sampling import join_original
    from .fields import lookup,keys
    partial=indices is not None or seeds is not None
    indices=PARAMS['indices'] if indices is None else indices;seeds=PARAMS['seeds'] if seeds is None else seeds
    if not indices or not seeds:raise ValueError('empty artifact audit')
    torch.set_num_threads(2);latents=0;fields=0;supports=0
    for i in indices:
        for seed in seeds:
            root=case(i,seed);selected=read(root/'input_complete.json')
            if selected['gt_used']:raise RuntimeError('input decisions contaminated')
            plan=torch.load(root/'plan.pt',weights_only=False);envelope=torch.tensor(plan['roi']['envelope'])
            original=torch.load(source(i,seed)/'original_latent.pt',weights_only=False)
            core=torch.tensor(plan['roi']['core']);required=torch.tensor(plan['roi']['positive'])&core&torch.tensor(plan['roi']['hull'])
            base_outside=original['coords'][~lookup(core,original['coords'][:,1:]).bool()]
            for spec in read(root/'ss_complete.json')['variants']:
                payload=torch.load(spec['path'],weights_only=False);coords=payload['coords']
                outside=coords[~lookup(core,coords[:,1:]).bool()]
                if not torch.equal(keys(outside[:,1:]).sort().values,keys(base_outside[:,1:]).sort().values):raise RuntimeError('SS exterior support changed')
                if PARAMS.get('anchor_support',False):
                    occupied=torch.zeros_like(core);c=coords[:,1:].long();occupied[c[:,0],c[:,1],c[:,2]]=True
                    if not bool(occupied[required].all()):raise RuntimeError('saved SS omits verified anchors')
                supports+=1
            old_outside=~lookup(envelope,original['coords'][:,1:]).bool()
            for spec in read(root/'sampling_complete.json')['candidates']:
                payload=torch.load(spec['path'],weights_only=False);idx,present=join_original(original['coords'],payload['coords'])
                outside=~lookup(envelope,payload['coords'][:,1:]).bool()
                if not bool(present[outside].all()) or not torch.equal(payload['features'][outside],original['features'][idx[outside]]):
                    raise RuntimeError('exterior feature invariant failed')
                _,kept=join_original(payload['coords'],original['coords'])
                if not bool(kept[old_outside].all()):raise RuntimeError('old exterior SS dropped')
                latents+=1
            base=torch.load(root/'base_fields.pt',weights_only=False);alpha=torch.tensor(plan['roi']['alpha'])
            old=lookup(alpha,base['coords'],16)==0;old_keys=keys(base['coords'][old]);seen=set()
            for arm in ARMS:
                for ep in ('accepted','candidate'):
                    item=selected['selections'][arm][ep]
                    path=item.get('paired_composite_fields')
                    if path and path not in seen:
                        merged=torch.load(path,weights_only=False);mk=keys(merged['coords']);position=torch.searchsorted(mk,old_keys)
                        if not torch.equal(mk[position],old_keys) or not torch.equal(merged['raw'][position],base['raw'][old]):
                            raise RuntimeError('serialized FDG exterior differs')
                        seen.add(path);fields+=1
                    if not (root/arm/f'{ep}.obj').is_file():raise RuntimeError('missing endpoint OBJ')
    result=dict(time=now(),passed=True,candidate_latents=latents,serialized_composites=fields,ss_supports=supports,
                input_cases=len(indices)*len(seeds),indices=indices,seeds=seeds,partial=partial)
    name='artifact_checks_'+('_'.join(map(str,indices)))+'_'+('_'.join(map(str,seeds)))+'.json' if partial else 'artifact_checks.json'
    write(OUT/'validation'/name,result);print(result,flush=True);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');parser.add_argument('--index',type=int);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.verify:verify([args.index] if args.index is not None else None,[args.seed] if args.seed is not None else None)
    else:build()
