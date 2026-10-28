"""Post-closeout paired analysis. Never changes methods or measured results."""
import hashlib
from pathlib import Path
import numpy as np

from .config import OUT,PRIOR,STORE,ARMS,read,write,now,register
from visual_evidence_probe.experiments.dense_evidence.report import object_summary,gain


def main():
    summary=read(OUT/'summary.json');closeout=read(OUT/'closeout.json')
    if not summary['complete'] or not closeout['complete'] or summary['states']!=48:raise ValueError('cohort not complete')
    rows=summary['records'];indexed={(r['index'],r['seed'],r['arm']):r for r in rows}
    indices=sorted({r['index'] for r in rows});groups={};pairs={};objects=[]
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm];active=[r for r in selected if r['status']=='complete']
        cd_harm=[];depth_harm=[];times=[];rollbacks=[]
        for r in selected:
            case=dict(index=r['index'],seed=r['seed'],uid=r['uid'])
            if gain(r,'accepted','surface.chamfer_l1')< -1.e-12:cd_harm.append(case)
            if gain(r,'accepted','novel.local_target_depth_mae')< -1.e-12:depth_harm.append(case)
            if r['status']=='complete':
                if arm=='dense_reference':
                    source=read(PRIOR/'objects'/f'{r["index"]:03d}'/str(r['seed'])/'dense/complete.json')
                    times.append(source.get('effective_optimization_seconds',source['elapsed_seconds']))
                else:times.append(r['elapsed_seconds'])
            if arm!='dense_reference':
                source=read(OUT/'objects'/f'{r["index"]:03d}'/str(r['seed'])/arm/'complete.json')
                rollbacks.append(dict(**case,count=len(source['rollbacks'])))
        relative_cd=object_summary([(r['uid'],gain(r,'accepted','surface.chamfer_l1')/r['original_metrics']['surface']['chamfer_l1']) for r in selected])
        groups[arm]=dict(active_cases=len(active),no_evidence_cases=len(selected)-len(active),
            accepted=sum(r['accepted_edit'] for r in selected),cd_regression_cases=cd_harm,
            local_depth_regression_cases=depth_harm,relative_cd_improvement=relative_cd,
            active_mean_seconds=float(np.mean(times)),active_median_seconds=float(np.median(times)),
            active_range_seconds=[float(min(times)),float(max(times))],
            rollback_cases=rollbacks,rollback_case_count=sum(r['count']>0 for r in rollbacks))
    for left in ('anchored','anchored_reliable'):
        values={key:[] for key in ('cd','fscore','local_absolute','local_relative','cd_relative')}
        for index in indices:
            for seed in (42,43):
                a=indexed[index,seed,left];b=indexed[index,seed,'dense_reference'];uid=a['uid']
                values['cd'].append((uid,gain(a,'accepted','surface.chamfer_l1')-gain(b,'accepted','surface.chamfer_l1')))
                values['fscore'].append((uid,gain(a,'accepted','surface.fscore_0p02')-gain(b,'accepted','surface.fscore_0p02')))
                values['local_absolute'].append((uid,gain(a,'accepted','novel.local_target_depth_mae')-gain(b,'accepted','novel.local_target_depth_mae')))
                values['local_relative'].append((uid,gain(a,'accepted','novel.local_target_depth_mae',True)-gain(b,'accepted','novel.local_target_depth_mae',True)))
                values['cd_relative'].append((uid,(gain(a,'accepted','surface.chamfer_l1')-gain(b,'accepted','surface.chamfer_l1'))/a['original_metrics']['surface']['chamfer_l1']))
        stats={key:object_summary(value) for key,value in values.items()}
        stats['leave_one_object_out_cd']={uid:object_summary([(name,v) for name,v in values['cd'] if name!=uid]) for uid in sorted({v[0] for v in values['cd']})}
        pairs[left]=stats
    for index in indices:
        uid=indexed[index,42,'dense_reference']['uid'];item=dict(index=index,uid=uid,arms={})
        for arm in ARMS:
            case=[indexed[index,seed,arm] for seed in (42,43)]
            item['arms'][arm]=dict(accepted=sum(r['accepted_edit'] for r in case),
                cd_gain=float(np.mean([gain(r,'accepted','surface.chamfer_l1') for r in case])),
                local_relative_gain=float(np.mean([gain(r,'accepted','novel.local_target_depth_mae',True) for r in case])))
        objects.append(item)
    value=dict(time=now(),complete=True,states=48,groups=groups,paired_against_round7=pairs,objects=objects,
        statistical_unit='object; average two seeds first; 5000 object bootstrap samples',
        no_new_evaluation_or_alignment=True,measured_results_unchanged=True,
        visual_cases_inspected=['005_42','005_43','006_42','010_42'],
        verdict='Fewer CD regressions on this development set; stable extra accuracy over round7 is unproven.')
    write(OUT/'analysis.json',value)
    names=dict(dense_reference='第七轮稠密参考',anchored='三维锚点＋信任区域',anchored_reliable='再加输入可靠性权重')
    text=['# 第八轮完整结果分析','',f'时间：{value["time"]}','',
        '**这批开发样本上，新约束减少了CD退化，但尚未证明相对第七轮有稳定的额外精度收益。当前不应据此替换主基线或宣称整体mesh明显提升。**','',
        '8个已检查开发物体×2 seeds×3组=48条；原30K权重、解码器、SS固定，仅编辑局部latent。两物体没有有效局部深度证据，原样回退保留在统计中。所有方法使用同一初始mesh、相机及最终接受门。','',
        '## 接受后的实际结果','',
        '下表改善均相对同一个初始30K网格；新增模块是否有用，要继续与第七轮参考作配对比较。','',
        '|方法|接受/16|新视角局部深度误差相对下降|全局CD绝对下降|全局CD相对下降|F@.02绝对提高|',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        group=summary['groups'][arm];e=group['endpoints']['accepted'];g=groups[arm]
        text.append(f'|{names[arm]}|{group["accepted"]}/16|{e["novel.local_depth_relative_reduction"]["mean"]:.2%}|'
            f'{e["surface.chamfer_l1"]["mean"]:.8f}|{g["relative_cd_improvement"]["mean"]:.2%}|{e["surface.fscore_0p02"]["mean"]:.6f}|')
    text += ['','## 新增模块的配对增益','',
        '|相对第七轮参考|CD额外改善及95%区间|局部相对下降差及95%区间（百分点）|','|---|---:|---:|']
    for arm in ('anchored','anchored_reliable'):
        cd=pairs[arm]['cd'];d=pairs[arm]['local_relative']
        text.append(f'|{names[arm]}|{cd["mean"]:+.8f} / {cd["ci95"]}|{d["mean"]*100:+.3f} / {[round(x*100,3) for x in d["ci95"]]}|')
    text += ['','以上CD、局部深度和F-score的配对区间均跨零；当前不能确认新增模块稳定优于原参考。对象只有8个，且均已用于开发，区间不是独立泛化证明。','',
        '可靠性分支的物体平均相对深度下降为4.38%，高于参考的3.91%；但绝对MAE平均改善为0.00119292，略低于参考的0.00123619。两种平均方式对不同初始误差的物体赋予不同权重，不矛盾，也不能只挑4.38%来证明整体更准确。','',
        '## 改善集中在哪里','',
        '|物体|参考CD改善|锚点CD改善|可靠性CD改善|参考/锚点/可靠性接受数|','|---|---:|---:|---:|---:|']
    for item in objects:
        a,b,c=[item['arms'][arm] for arm in ARMS]
        text.append(f'|{item["uid"]}|{a["cd_gain"]:+.8f}|{b["cd_gain"]:+.8f}|{c["cd_gain"]:+.8f}|{a["accepted"]}/{b["accepted"]}/{c["accepted"]}|')
    text += ['','- 圆白菜：原参考CD变差，两条新分支转为小幅改善，是整体均值提升的最大来源；但上部短褶皱仍在，没有解决表面质量问题。',
        '- 胡萝卜：可靠性加权的物体平均局部深度由退化约0.77%转为改善约6.50%，说明抑制错误深度有价值；仍须看逐seed，不能把物体均值当作每次都改善。',
        '- 手提包：可靠性分支两个seed均回退，失去了原参考可接受的部分收益。新权重不能稳定判断什么时候应保留稠密证据。',
        '- 芦笋、护手霜和风筝：部分新结果相对原参考收益减少，存在保护与纠错幅度之间的权衡。',
        '- 棋子和无花果：全部保持原样，新增保护项没有补足输入几何证据覆盖。','']
    excluded='omni_brussels_sprout_002'
    for arm in ('anchored','anchored_reliable'):
        sensitivity=pairs[arm]['leave_one_object_out_cd'][excluded]
        text.append(f'移除圆白菜后，{names[arm]}相对参考的平均CD改善为{sensitivity["mean"]:+.8f}，95%区间{sensitivity["ci95"]}。这是均值敏感性诊断，不是删除不利样本后的主结果。')
    text += ['','## 稳健性、门槛和计算代价','',
        '|方法|接受结果CD退化案例|接受结果局部深度退化案例|触发回退的案例数|回退组总数|有效案例平均编辑秒数|','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        g=groups[arm];total=summary['groups'][arm]['total_rollbacks']
        text.append(f'|{names[arm]}|{len(g["cd_regression_cases"])}|{len(g["local_depth_regression_cases"])}|{g["rollback_case_count"]}|{total}|{g["active_mean_seconds"]:.2f}|')
    text += ['','案例按物体×seed计；主统计仍先在物体内平均seed。上述退化数仅描述此开发集，不能作为安全性保证。',
        '区域外漂移导致的训练候选拒绝从参考的2例降到两条新分支的0例，保护机制确实起作用；但接受数由9降到8，剩余主要卡在轮廓、拓扑或heldout证据，不能认为接受瓶颈已经解决。',
        '编辑耗时为80次尝试及checkpoint保存，不含原30K生成、DA3、指标和渲染。参考分支复用历史结果，需回溯effective_optimization_seconds排除缓存复用的假零耗时。各轮运行GPU与负载不同，时间只作工程参考，不能当作严格加速比较。','',
        '## 视觉与真实输入限制','',
        '人工检查了005/42、005/43、006/42和010/42的同相机对照。原物体基本形状均保留，但新结果与原参考的外观差异较小；圆白菜上部褶皱、胡萝卜上部局部起伏仍然可见。手提包可靠性分支的最终结果直接回退到原网格。不能仅凭CD改善宣称网格更平滑或细节更真实。',
        '本轮编辑实验没有新增真实物体抗遮挡成绩。电钻仅做输入审查：GT投影基本正确，但DA3有绝对深度偏差，原30K初始整体注册也较差；须先控制整体注册再评估局部修复。','',
        '## 下一步决策','',
        '保持第七轮为主参考，保留第八轮作为控制退化的候选机制。当前不建议直接长训或继续同时增加一批保护超参数。',
        '先做一个小型可区分原因的对照：在相同输入区域、训练视图及监督像素上，分别使用实际估计深度与正确GT深度诊断，对比原编辑器和锚点编辑器。严格配对使用有GT交点的共同像素；GT分支仅为诊断，不能算方法成绩。',
        '若正确深度下新增保护仍压制有效纠错，优先改编辑参数化或约束方式；若正确深度有效、实际估计无效，则优先解决输入证据的准确性和可靠性。获得明确机制收益后，再冻结新的未检查物体做独立验证。','',
        f'完整数据：`{OUT/"analysis.json"}`；同相机渲染：`{OUT/"figures/meshes"}`；原收尾结果保持原样。','']
    markdown='\n'.join(text);(OUT/'ANALYSIS.md').write_text(markdown)
    conclusion=STORE/'第八轮完整结果分析_20261028.md';conclusion.write_text(markdown)
    audit=STORE/'审计.md';before=audit.read_bytes();marker=b'<!-- constrained-evidence-round8-detailed-analysis-20261028 -->'
    if marker not in before:
        with audit.open('ab') as handle:handle.write(('\n\n# 8.1 第八轮完整结果分析\n\n'+marker.decode()+'\n\n'+markdown).encode())
        if audit.read_bytes()[:len(before)]!=before:raise ValueError('old audit contents changed')
    link=Path('/home/zjr/Tracker/visual_evidence_probe/第八轮完整结果分析.md')
    if not link.exists():link.symlink_to(conclusion)
    register('analysis')
    write(OUT/'analysis_receipt.json',dict(time=now(),analysis=str(conclusion),
        audit_preappend_sha256=hashlib.sha256(before).hexdigest(),prior_audit_contents_preserved=True,
        new_model_runs=0,original_experiment_outputs_unchanged=True))
    print(dict(time=value['time'],groups=groups,paired_against_round7=pairs,analysis=str(conclusion)),flush=True)


if __name__=='__main__':main()
