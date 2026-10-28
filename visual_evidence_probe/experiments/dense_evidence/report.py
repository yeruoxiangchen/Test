"""Factorial report. Seeds are averaged within objects; fallbacks remain."""
import argparse
from collections import Counter

import numpy as np

from .config import ARMS,OUT,PARAMS,folder,registry,read,write,now
from visual_evidence_probe.experiments.geometry_evidence.report import get,fmt


def object_summary(values):
    groups={}
    for key,value in values:
        if value is not None and np.isfinite(value):groups.setdefault(str(key),[]).append(float(value))
    if not groups:return dict(mean=None,ci95=None,objects=0,cases=0)
    means=np.array([np.mean(groups[key]) for key in sorted(groups)])
    rng=np.random.default_rng(PARAMS['random_seed'])
    # One physical real object has no population confidence interval.
    ci=np.quantile(means[rng.integers(len(means),size=(5000,len(means)))].mean(1),[.025,.975]).tolist() if len(means)>1 else None
    return dict(mean=float(means.mean()),ci95=ci,objects=len(means),cases=sum(map(len,groups.values())),
        improved_objects=int((means>1.e-12).sum()),worsened_objects=int((means< -1.e-12).sum()),
        values_by_object={key:float(np.mean(value)) for key,value in groups.items()})


def cluster(row):return 'ycbv_drill' if row['kind']=='real_scene' else row['uid']


def gain(row,endpoint,key,relative=False):
    before=get(row['original_metrics'],key);after=get(row['metrics'][endpoint],key)
    if before is None or after is None:return None
    value=after-before if 'fscore' in key else before-after
    return value/before if relative and before>1.e-8 else (None if relative else value)


METRICS=('surface.chamfer_l1','surface.fscore_0p02','input.local_target_depth_mae',
    'novel.local_target_depth_mae','input.local_union_depth_mae','novel.local_union_depth_mae',
    'novel.global_target_depth_mae')


def aggregate(rows):
    output={}
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm]
        endpoints={}
        for endpoint in ('proposal','candidate','accepted'):
            scores={key:object_summary([(cluster(r),gain(r,endpoint,key)) for r in subset]) for key in METRICS}
            for domain in ('input','novel'):
                scores[f'{domain}.local_depth_relative_reduction']=object_summary([
                    (cluster(r),gain(r,endpoint,f'{domain}.local_target_depth_mae',True)) for r in subset])
            endpoints[endpoint]=scores
        durations=[]
        for row in subset:
            source=read(folder(row['index'],row['seed'])/row['source']/'complete.json')
            if source['steps'] and not source.get('reused_sparse_trajectory'):
                durations.append(source['elapsed_seconds'])
        output[arm]=dict(cases=len(subset),states=dict(Counter(r['status'] for r in subset)),
            accepted_edits=sum(r['accepted_edit'] for r in subset),candidate_edits=sum(r['candidate_step']>0 for r in subset),
            reused_sparse_trajectories=sum(r['reused_sparse_trajectory'] for r in subset),
            unique_optimization_count=len(durations),mean_unique_optimization_seconds=float(np.mean(durations)) if durations else None,
            max_peak_gib=max((r['peak_allocated_bytes']/2**30 for r in subset),default=0),
            endpoints=endpoints)
    return output


def factorial(rows):
    indexed={(r['index'],r['seed'],r['arm']):r for r in rows}
    output={}
    contrasts={
        'density_under_legacy':{'dense_legacy':1,'sparse_legacy':-1},
        'density_under_observed':{'dense_observed':1,'sparse_observed':-1},
        'policy_under_sparse':{'sparse_observed':1,'sparse_legacy':-1},
        'policy_under_dense':{'dense_observed':1,'dense_legacy':-1},
        'interaction':{'dense_observed':1,'sparse_observed':-1,'dense_legacy':-1,'sparse_legacy':1}}
    for name,weights in contrasts.items():
        scores={}
        for key in METRICS:
            values=[]
            for index,seed in sorted({(r['index'],r['seed']) for r in rows}):
                group=[(indexed.get((index,seed,arm)),weight) for arm,weight in weights.items()]
                if any(r is None for r,_ in group):continue
                gs=[gain(r,'accepted',key) for r,_ in group]
                value=sum(g*weight for g,(_,weight) in zip(gs,group)) if all(g is not None for g in gs) else None
                values.append((cluster(group[0][0]),value))
            scores[key]=object_summary(values)
        output[name]=scores
    return output


def interpret(summary):
    group=summary['groups']['validation']['dense_observed']
    if group['cases']!=PARAMS['validation_object_count']*len(PARAMS['seeds']):
        return '新验证集尚未完整，不能给出总体结论。'
    metrics=group['endpoints']['accepted']
    depth=metrics['novel.local_depth_relative_reduction'];cd=metrics['surface.chamfer_l1']
    local_supported=depth['ci95'] is not None and depth['ci95'][0]>0
    global_supported=cd['ci95'] is not None and cd['ci95'][0]>0
    if local_supported and global_supported:
        return '在本轮16物体的局部修复验证上，实际接受结果同时出现新视角局部深度和全局CD改善；物体bootstrap区间均高于零。仍须扩大独立真实物体测试。'
    if local_supported:
        return '本轮支持新视角局部深度改善，但尚不能确认全局网格精度稳定提高。局部修复和整体精度必须分开判断。'
    return '当前证据尚不能确认该设置在新验证物体上稳定提高网格精度；接受率或训练目标下降不能替代几何收益。'


def build(final=False):
    cohort=registry()['rows'];records=[];missing=[];diagnostics=[]
    for obj in cohort:
        for seed in PARAMS['seeds']:
            root=folder(obj['index'],seed)
            if (root/'evidence_complete.json').exists():diagnostics.append(read(root/'evidence_complete.json'))
            for arm in ARMS:
                path=root/arm/'result.json'
                if not path.exists():missing.append(str(path));continue
                records.append(dict(read(path),arm=arm))
    if final and missing:raise ValueError(f'{len(missing)} arm results missing')
    groups={name:aggregate([r for r in records if r['split']==name]) for name in ('development','validation','real_pilot')}
    contrasts={name:factorial([r for r in records if r['split']==name]) for name in groups}
    real_groups={str(index):aggregate([r for r in records if r['index']==index]) for index in PARAMS['real_indices']}
    quality=read(OUT/'depth_quality.json')['records'] if (OUT/'depth_quality.json').exists() else []
    # Older development diagnostics used an unpenalized missing baseline ray.
    # Never mix those stale values into the current clipped/fixed-domain table.
    quality=[r for r in quality if r.get('metric_definition')=='clipped_fixed_missing_v2']
    value=dict(time=now(),complete=not missing,states=len(records),expected_states=len(cohort)*2*4,
        missing=missing,groups=groups,factorial=contrasts,real_groups=real_groups,
        depth_quality=quality,evidence_diagnostics=diagnostics,records=records,
        params=PARAMS,cluster_policy='average seeds within physical object; all three real groups are one drill',
        evaluation_frame='unchanged canonical frame; no GT alignment or ICP')
    value['interpretation']=interpret(value)
    write(OUT/'summary.json',value)
    text=['# 第七轮：稠密输入几何与接受策略的四组对照','',f'时间：{value["time"]}',
        f'结果：{len(records)}/{value["expected_states"]}；完整：{not missing}。','',value['interpretation'],'',
        '冻结原30K和网格解码器，在固定SS上局部编辑SLat；80步，上限0.5个训练标准差。',
        '稀疏组使用训练视图NCC匹配；稠密组保留全部NCC深度锚点，并添加通过置信度与跨视图一致性检查的DA3深度。',
        '每种证据只优化一次，两种接受策略共享0/20/40/80步候选。原策略先按训练损失选一个候选，再过旧门；新策略逐checkpoint检查观测轮廓、外部漂移与面积加权碎片。',
        '所有输入决策先封存再加载对应验证/真实GT。heldout RGB完全不进入深度预测和优化目标；只用于输入验证。',
        '正值为改善。先在物体内平均两个seed，再平均物体；95%区间为物体bootstrap。原样回退全部计入。','']
    for split,title in (('development','开发集：8个已检查物体'),('validation','验证集：16个新局部修复物体'),('real_pilot','真实试点：同一电钻的3组输入')):
        text += [f'## {title}','','|组别|接受/案例|新视角局部深度下降|CD改善|F@.02改善|','|---|---:|---:|---:|---:|']
        for arm in ARMS:
            group=groups[split][arm];e=group['endpoints']['accepted'];relative=e['novel.local_depth_relative_reduction']['mean']
            text.append(f'|{arm}|{group["accepted_edits"]}/{group["cases"]}|{relative:+.2%}|' if relative is not None else f'|{arm}|{group["accepted_edits"]}/{group["cases"]}|NA|')
            text[-1]+=f'{fmt(e["surface.chamfer_l1"]["mean"])}|{fmt(e["surface.fscore_0p02"]["mean"],4)}|'
        text += ['','训练选出候选（未必被接受）：','','|证据|候选/案例|新视角局部深度下降|CD改善|','|---|---:|---:|---:|']
        for arm in ('sparse_observed','dense_observed'):
            group=groups[split][arm];e=group['endpoints']['candidate'];relative=e['novel.local_depth_relative_reduction']['mean']
            rel='NA' if relative is None else f'{relative:+.2%}'
            text.append(f'|{arm}|{group["candidate_edits"]}/{group["cases"]}|{rel}|{fmt(e["surface.chamfer_l1"]["mean"])}|')
        text += ['','配对的接受结果差异：','','|对照|CD改善差 / 95% CI|新视角局部深度改善差 / 95% CI|','|---|---:|---:|']
        for name,pair in contrasts[split].items():
            cd,depth=pair['surface.chamfer_l1'],pair['novel.local_target_depth_mae']
            text.append(f'|{name}|{fmt(cd["mean"])} / {cd["ci95"]}|{fmt(depth["mean"])} / {depth["ci95"]}|')
        text.append('')
    text += ['## 真实电钻各输入组（每组两个seed）','','|组/index|策略|接受/案例|新视角局部深度下降|CD改善|','|---|---|---:|---:|---:|']
    for index in PARAMS['real_indices']:
        for arm in ('sparse_observed','dense_observed'):
            d=real_groups[str(index)][arm];e=d['endpoints']['accepted'];ratio=e['novel.local_depth_relative_reduction']['mean']
            relative='NA' if ratio is None else f'{ratio:+.2%}'
            text.append(f'|{index:03d}|{arm}|{d["accepted_edits"]}/{d["cases"]}|{relative}|{fmt(e["surface.chamfer_l1"]["mean"])}|')
    text += ['','## DA3单独的局部深度质量（GT仅用于事后评估）','',
        '|物体/seed|可靠覆盖率|原mesh MAE|DA3 MAE|射线改善比例|','|---|---:|---:|---:|---:|']
    for q in quality:
        local=q['local'];fraction=local.get('improved_fraction');improved='NA' if fraction is None else f'{fraction:.2%}'
        text.append(f'|{q["index"]:03d}/{q["seed"]}|{local["coverage"]:.2%}|{fmt(local["before_mae"])}|{fmt(local["after_mae"])}|{improved}|')
    text += ['','## 覆盖与耗时','','|物体/seed|NCC局部深度|混合局部深度|','|---|---:|---:|']
    for d in diagnostics:text.append(f'|{d["index"]:03d}/{d["seed"]}|{d["counts"]["sparse"]}|{d["counts"]["dense"]}|')
    text += ['','|组别|独立优化数|平均优化秒数|峰值GiB|复用稀疏轨迹|','|---|---:|---:|---:|---:|']
    for arm,d in aggregate(records).items():
        text.append(f'|{arm}|{d["unique_optimization_count"]}|{d["mean_unique_optimization_seconds"]}|{d["max_peak_gib"]:.2f}|{d["reused_sparse_trajectories"]}|')
    text += ['','## 解释边界','',
        '- 验证物体未参与本轮局部修复调参；它们来自既有Omni扫描基准，不等于原30K训练分布之外的新数据。',
        '- 真实试点只有一个物体、一个场景；三种遮挡输入和两个seed不可视为六个独立物体，不提供物体总体置信区间。',
        '- 真实BOP输入沿用GT物体位姿、完整包围框与模型尺度辅助，使用可见mask；不属于完全无标注的真实重建。',
        '- BOP完整框内部且可见mask之外的像素保持未知，不作为目标缺失的负样本。没有GT深度或GT表面用于编辑。',
        '- 真实组30K/NCC使用可见mask后的目标输入，DA3看同相机裁剪的raw场景，并只在可见目标前景取深度；输入图同时保存masked和raw版本。',
        '- 原30K真实输入直接使用声明的BOP规范坐标，没有ICP、GT刚体拟合或重新归一化；各组共用同一坐标和相机。',
        '- DA3使用给定相机和官方轨迹尺度适配；冻结网络输出经置信度与双向重投影检查。像素射线使用实际处理后的K，没有盲目缩放深度。',
        '- 稠密组是NCC锚点加DA3补充的混合证据；raw DA3误差及覆盖率单列在depth_quality.json。无新增局部深度时精确复用稀疏轨迹，耗时不把复用零值混入有效优化平均。',
        '- density_under_*对照表示更换证据来源为NCC+DA3，同时改变深度覆盖和准确性，不是单独控制采样密度的因果实验。',
        '- 原门比较原mesh轮廓；新门比较输入观测轮廓，允许训练深度改善且heldout非退化。新门不能证明GT几何一定改善。',
        '- 区域、可编辑行和SS均由原结果/输入封存；本实验不能证明对任意不可见缺失结构的补全。',
        '- heldout只对本轮DA3、匹配与编辑保留；原30K初始重建使用全部输入，因此不是完全没见过的图像。新渲染8视角仅供最终评估。',
        '- 本轮重新按训练视图生成NCC匹配，并移除先前额外0.04射线位移钳制；与第六轮不是完全重放。',
        '- 局部深度误差使用固定GT区域，缺失射线罚0.1，误差截断0.1；全局CD/F-score使用50000点固定种子采样，极小差异须结合深度检查。',
        '- 按实际接受结果判断可部署收益；训练选出候选和固定80步proposal仅供诊断，不作为最终方法成绩。',
        '- 开发物体005/seed42的稠密结果在上部出现额外短褶皱，局部深度下降而全局CD/F-score变差；见DEVELOPMENT_VISUAL_AUDIT.md。该单物体视觉检查不能外推到全体。',
        '- 优化时间含checkpoint保存，不含模型加载、DA3或最终评估；DA3各对象训练视图组合只推理一次并保留独立时间记录。','',
        '- CUDA异步错误后的同步重试会保留执行模式；当前时间是实际实验诊断时间，不能当作已充分加速的产品延迟。',
        f'完整数据：`{OUT/"summary.json"}`。']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(dict(complete=not missing,states=len(records),expected=value['expected_states']),flush=True)
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();build(a.final)


if __name__=='__main__':main()
