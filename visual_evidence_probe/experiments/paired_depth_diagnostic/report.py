import numpy as np
from .config import OUT,PARAMS,ARMS,folder,read,write,now
from visual_evidence_probe.experiments.dense_evidence.report import gain


def object_summary(values):
    grouped={}
    for uid,value in values:
        if value is not None and np.isfinite(value):grouped.setdefault(uid,[]).append(float(value))
    if not grouped:return dict(mean=None,ci95=None,objects=0,cases=0)
    means=np.array([np.mean(grouped[uid]) for uid in sorted(grouped)])
    rng=np.random.default_rng(PARAMS['random_seed'])
    interval=np.quantile(means[rng.integers(len(means),size=(5000,len(means)))].mean(1),[.025,.975]).tolist() if len(means)>1 else None
    return dict(mean=float(means.mean()),ci95=interval,objects=len(means),cases=sum(map(len,grouped.values())),
        improved_objects=int((means>1.e-12).sum()),worsened_objects=int((means< -1.e-12).sum()),
        values_by_object={uid:float(np.mean(values)) for uid,values in grouped.items()})


def build():
    records=[];missing=[];quality=[]
    for index in PARAMS['indices']:
        for seed in PARAMS['seeds']:
            root=folder(index,seed)
            if (root/'target_quality.json').exists():quality.append(read(root/'target_quality.json'))
            for arm in ARMS:
                path=root/arm/'result.json'
                if path.exists():records.append(read(path))
                else:missing.append(str(path))
    groups={}
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm];endpoints={}
        for endpoint in ('candidate','proposal','accepted'):
            scores={metric:object_summary([(r['uid'],gain(r,endpoint,metric)) for r in rows]) for metric in (
                'surface.chamfer_l1','surface.fscore_0p02','novel.local_target_depth_mae','novel.local_union_depth_mae')}
            scores['novel.local_depth_relative_reduction']=object_summary([(r['uid'],gain(r,endpoint,'novel.local_target_depth_mae',True)) for r in rows])
            endpoints[endpoint]=scores
        active=[r for r in rows if r['status']=='complete']
        groups[arm]=dict(cases=len(rows),accepted=sum(r['accepted_edit'] for r in rows),endpoints=endpoints,
            mean_seconds=float(np.mean([r['elapsed_seconds'] for r in active])) if active else None,
            rollbacks=sum(r['rollbacks'] for r in rows),max_peak_gib=max((r['peak_allocated_bytes']/2**30 for r in rows),default=0))
    indexed={(r['index'],r['seed'],r['arm']):r for r in records}
    contrasts={
        'accuracy_free':{'oracle_free':1,'estimated_free':-1},
        'accuracy_anchored':{'oracle_anchored':1,'estimated_anchored':-1},
        'constraint_estimated':{'estimated_anchored':1,'estimated_free':-1},
        'constraint_oracle':{'oracle_anchored':1,'oracle_free':-1},
        'interaction':{'oracle_anchored':1,'oracle_free':-1,'estimated_anchored':-1,'estimated_free':1}}
    pairs={}
    for name,weights in contrasts.items():
        pairs[name]={}
        for endpoint in ('candidate','proposal','accepted'):
            scores={}
            for metric in ('surface.chamfer_l1','novel.local_target_depth_mae'):
                values=[]
                for index in PARAMS['indices']:
                    for seed in PARAMS['seeds']:
                        rows=[(indexed.get((index,seed,arm)),weight) for arm,weight in weights.items()]
                        if any(r is None for r,_ in rows):continue
                        values.append((rows[0][0]['uid'],sum(gain(r,endpoint,metric)*weight for r,weight in rows)))
                scores[metric]=object_summary(values)
            pairs[name][endpoint]=scores
    interpretation='诊断尚未全部完成，不判定主因。'
    if not missing:
        evidence=[pairs[k][ep]['novel.local_target_depth_mae']['mean'] for k in ('accuracy_free','accuracy_anchored') for ep in ('candidate','proposal')]
        penalty=[pairs['constraint_oracle'][ep]['novel.local_target_depth_mae']['mean'] for ep in ('candidate','proposal')]
        better_evidence=all(v is not None and v>1.e-5 for v in evidence)
        costs_oracle=all(v is not None and v< -1.e-5 for v in penalty)
        if better_evidence and costs_oracle:interpretation='在这4个开发物体和固定预算下，正确深度在两种编辑器上均更好，同时新增保护在正确目标下也削弱纠错：证据误差和过度保护共同存在。'
        elif better_evidence:interpretation='在这4个开发物体和固定预算下，正确深度在两种编辑器上均更好，更支持先解决输入几何证据准确性；保护是否额外有害仍须看逐物体和回退。'
        elif costs_oracle:interpretation='当前结果显示新增保护在正确目标下仍削弱纠错，优先检查编辑参数化和约束方式；证据误差的作用未在所有对照中保持一致。'
        else:interpretation='四组结果尚未给出单一稳定主因。需结合逐物体、输入深度饱和比例和固定80步结果判断，不能只凭接受率判定编辑器能力。'
    replay=read(OUT/'numerical_replay_audit.json') if (OUT/'numerical_replay_audit.json').exists() else None
    value=dict(time=now(),complete=not missing,states=len(records),expected_states=32,records=records,missing=missing,
        groups=groups,contrasts=pairs,target_quality=quality,interpretation=interpretation,
        numerical_replay_audit=replay,
        privileged_diagnostic=True,all_arms_gt_pruned_support=True,params=PARAMS)
    write(OUT/'summary.json',value)
    text=['# 第九轮：同像素深度准确性与编辑器约束诊断','',f'时间：{now()}；结果：{len(records)}/32；完整：{not missing}。','',
        interpretation,'',
        '**四组均为含GT支持选择的机制诊断，不是可部署方法成绩。4个已检查物体×2 seeds；30K/解码器/SS固定。**',
        '监督像素、区域、训练视图、初始mesh、可更新行及目标归一化完全共享。只改变深度值与是否启用锚点/信任区域。','']
    for endpoint,label in (('candidate','各自训练目标选出的候选（oracle选择含GT）'),('proposal','固定80次尝试的输出'),('accepted','最终输入门接受后的结果（仍是诊断）')):
        text += [f'## {label}','','|组别|案例/接受|新视角局部深度下降|CD改善|','|---|---:|---:|---:|']
        for arm,d in groups.items():
            e=d['endpoints'][endpoint];ratio=e['novel.local_depth_relative_reduction']['mean'];cd=e['surface.chamfer_l1']['mean']
            text.append(f'|{arm}|{d["cases"]}/{d["accepted"]}|'+('NA' if ratio is None else f'{ratio:+.2%}')+'|'+('NA' if cd is None else f'{cd:+.8f}')+'|')
        text += ['','|配对对照|新视角局部深度改善差 / 95%区间|CD改善差 / 95%区间|','|---|---:|---:|']
        for name,pair in pairs.items():
            a,b=pair[endpoint]['novel.local_target_depth_mae'],pair[endpoint]['surface.chamfer_l1']
            text.append(f'|{name}|{a["mean"]} / {a["ci95"]}|{b["mean"]} / {b["ci95"]}|')
        text.append('')
    text += ['## 初始监督质量','','|案例|共同射线|原mesh clipped MAE|估计深度 clipped GT MAE|估计目标饱和比例|GT目标饱和比例|估计移动方向正确率|','|---|---:|---:|---:|---:|---:|---:|']
    for q in quality:text.append(f'|{q["index"]:03d}/{q["seed"]}|{q["pixels"]}|{q["original_clipped_gt_mae"]}|{q["estimated_clipped_gt_mae"]}|{q["estimated_training_clipped_fraction"]}|{q["oracle_training_clipped_fraction"]}|{q["estimated_correct_direction_fraction"]}|')
    if replay:
        text+=['','## 同进程数值重复对照','','另对005/42执行两次完全相同的estimated_free优化，不计入32组主方法结果。','',
            '|对照|latent逐位相同|CD差绝对值|新视角局部深度差绝对值|','|---|---|---:|---:|']
        for key,item in replay['comparisons'].items():text.append(f'|{key}|{item["features_bitwise_equal"]}|{item["absolute_cd_difference"]}|{item["absolute_novel_local_depth_difference"]}|')
        text+=['','该检查仅覆盖一个案例，不能作为所有案例的噪声上界。梯度数值、稀疏内核选择和离散拓扑可使微小初始差异放大，微小主效应需结合此对照解释。']
        text+=['',f'重复检查的CUDA同步模式：{replay.get("execution",{})}。首次异步重复检查在FlexiCubes体素索引中出现长度差1的异常，日志保留；两次控制以同一同步模式重试。原运行与控制的差异不能单独归因于随机数值噪声。']
    text += ['','## 范围与边界','',
        '- estimated组也用GT交点筛选共同像素，因此旧第七/八轮的估计组成绩不能直接复用。所有四组重新运行。',
        '- GT只改变训练输入相机的深度值；没有增加GT法线、GT轮廓，也没有读取新评估视角的GT进行优化或选择。',
        '- 四组共享estimated初始损失归一化，避免改变目标准确性同时改变整体优化尺度。',
        '- 锚点编辑器是表面锚点、外部法线保护及回退的联合干预；固定80次尝试不代表相同有效更新次数。',
        '- 主要看候选与固定80步结果；输入门拒绝正确几何不代表latent不能表达正确形状。',
        '- 当前像素支持、0.1深度损失截断、0.5std界限和80步预算构成有限诊断，oracle失败不能被解释成一般性的不可修复证明。',
        '- 四物体均已检查且按问题选取，bootstrap只描述本组差异，不是独立泛化验证。',
        '- 自动主因文字依据均值方向，仅是诊断线索，不是统计显著性或因果普适性声明；需与配对区间及数值重复对照一起阅读。',
        '- 同一原始世界坐标，没有GT ICP、按分支重新归一化或额外配准。','',f'详细数据：`{OUT/"summary.json"}`。','']
    (OUT/'REPORT.md').write_text('\n'.join(text));print(dict(complete=not missing,states=len(records),interpretation=interpretation),flush=True);return value


if __name__=='__main__':build()
