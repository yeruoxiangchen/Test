from collections import Counter
import numpy as np
from .config import OUT,PRIOR,PARAMS,ARMS,folder,read,write,now
from visual_evidence_probe.experiments.dense_evidence.report import object_summary,gain,fmt


def build():
    records=[];missing=[]
    for index in PARAMS['dev_indices']:
        for seed in PARAMS['seeds']:
            for arm in ARMS:
                path=folder(index,seed)/arm/'result.json'
                if path.exists():records.append(read(path))
                else:missing.append(str(path))
    groups={}
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        endpoints={}
        for endpoint in ('candidate','accepted','proposal'):
            metrics={}
            for key in ('surface.chamfer_l1','surface.fscore_0p02','novel.local_target_depth_mae','novel.local_union_depth_mae'):
                metrics[key]=object_summary([(r['uid'],gain(r,endpoint,key)) for r in rows])
            metrics['novel.local_depth_relative_reduction']=object_summary([(r['uid'],gain(r,endpoint,'novel.local_target_depth_mae',True)) for r in rows])
            endpoints[endpoint]=metrics
        durations=[r['elapsed_seconds'] for r in rows if not r['reused_round7'] and r['status']=='complete']
        reasons=Counter()
        for r in rows:
            if r['accepted_edit']:continue
            own=r['candidate_step'];row=next((d for d in r['decisions']['observed_decisions'] if d['step']==own),None)
            if row:
                for key in ('silhouette_safe','outside_safe','structure_safe','heldout_nonregression','enough_evidence'):
                    if not row[key]:reasons[key]+=1
        groups[arm]=dict(cases=len(rows),accepted=sum(r['accepted_edit'] for r in rows),
            candidate_edits=sum(r['candidate_step']>0 for r in rows),endpoints=endpoints,
            mean_seconds=float(np.mean(durations)) if durations else None,total_rollbacks=sum(r['rollbacks'] for r in rows),
            rejection_reasons=dict(reasons),max_peak_gib=max((r['peak_allocated_bytes']/2**30 for r in rows),default=0))
    paired={};paired_novel={}
    indexed={(r['index'],r['seed'],r['arm']):r for r in records}
    for left,right in (('anchored','dense_reference'),('anchored_reliable','anchored'),('anchored_reliable','dense_reference')):
        deltas={}
        for endpoint in ('candidate','accepted'):
            values=[]
            for index in PARAMS['dev_indices']:
                for seed in PARAMS['seeds']:
                    a=indexed.get((index,seed,left));b=indexed.get((index,seed,right))
                    if a and b:values.append((a['uid'],gain(a,endpoint,'surface.chamfer_l1')-gain(b,endpoint,'surface.chamfer_l1')))
            deltas[endpoint]=object_summary(values)
        paired[f'{left}_minus_{right}']=deltas
        values=[]
        for index in PARAMS['dev_indices']:
            for seed in PARAMS['seeds']:
                a=indexed.get((index,seed,left));b=indexed.get((index,seed,right))
                if a and b:values.append((a['uid'],gain(a,'accepted','novel.local_target_depth_mae')-gain(b,'accepted','novel.local_target_depth_mae')))
        paired_novel[f'{left}_minus_{right}']=object_summary(values)
    interpretation='当前只有部分开发案例，等待48组结果齐备后判断整体趋势。'
    if not missing:
        supported=[]
        for arm in ('anchored','anchored_reliable'):
            key=f'{arm}_minus_dense_reference';cd=paired[key]['accepted'];depth=paired_novel[key]
            if cd['ci95'] is not None and depth['ci95'] is not None and cd['ci95'][0]>0 and depth['ci95'][0]>0:supported.append(arm)
        interpretation=('开发物体上同时出现相对第七轮参考的全局CD和新视角局部深度改善：'+', '.join(supported)+'。仍需新的未检查物体独立验证。') if supported else '当前8物体开发集尚不能确认两条新分支相对第七轮参考同时稳定改善CD和局部深度；需结合配对区间、视觉缺陷和计算代价判断，不能仅按平均值宣布有效。'
    value=dict(time=now(),complete=not missing,states=len(records),expected_states=48,missing=missing,
        groups=groups,paired_cd_improvements=paired,paired_novel_improvements=paired_novel,
        interpretation=interpretation,records=records,params=PARAMS,
        scope='eight inspected development objects, no independent validation claim')
    write(OUT/'summary.json',value)
    text=['# 第八轮：三维锚点约束与输入可靠性加权','',f'时间：{now()}；结果：{len(records)}/48；完整：{not missing}。','',
        interpretation,'','8个已检查开发物体×2 seeds；30K权重及SS固定。三组共享原始结果、区域、输入相机与最终接受门。','',
        '|方法|接受/案例|候选局部深度下降|接受后局部深度下降|接受后CD改善|回退组数|平均编辑秒数|','|---|---:|---:|---:|---:|---:|---:|']
    for arm,d in groups.items():
        c=d['endpoints']['candidate']['novel.local_depth_relative_reduction']['mean'];a=d['endpoints']['accepted']['novel.local_depth_relative_reduction']['mean'];cd=d['endpoints']['accepted']['surface.chamfer_l1']['mean']
        text.append(f'|{arm}|{d["accepted"]}/{d["cases"]}|'+('NA' if c is None else f'{c:+.2%}')+'|'+('NA' if a is None else f'{a:+.2%}')+f'|{fmt(cd)}|{d["total_rollbacks"]}|{d["mean_seconds"]}|')
    text+=['','## 配对CD改善差（正值为左侧更好）','','|对照|候选差/95%区间|接受后差/95%区间|','|---|---:|---:|']
    for pair,d in paired.items():text.append(f'|{pair}|{fmt(d["candidate"]["mean"])} / {d["candidate"]["ci95"]}|{fmt(d["accepted"]["mean"])} / {d["accepted"]["ci95"]}|')
    text+=['','|对照|接受后新视角局部深度改善差/95%区间|','|---|---:|']
    for pair,d in paired_novel.items():text.append(f'|{pair}|{fmt(d["mean"])} / {d["ci95"]}|')
    text+=['','## 解释边界','',
        '- 全部是开发数据，已用于第六、七轮；不得称为第八轮独立验证。若开发表现有效，再冻结新的未检查物体。',
        '- dense_reference精确复用第七轮稠密分支；两条新分支各80次尝试更新，回退会减少有效更新次数，计时单列。',
        '- anchored同时引入表面锚点、外部法线保护和五步信任区域回退，是联合干预。',
        '- 表面距离搜索16个最近三角形重心邻居，再计算三角面片最近点；不是最近顶点距离，也不是全三角形最小值的严格认证。',
        '- reliability保留NCC锚点；深度权重只用训练RGB、相机与已有匹配。弱纹理保留弱权重；一致性/权重不等于正确性保证。',
        '- heldout仅用于最终门，不进入梯度、回退或权重构建。GT仅在单独评估进程访问。没有GT对齐、ICP或分支重新归一化。',
        '- 首先比较接受后的几何收益，同时检查褶皱、外部漂移、回退数量与时间；提高接受率本身不构成精度提升。','',
        '- 初次连续回退试跑暴露Adam状态备份共享问题；受影响的010/42完整留档在debug_optimizer_alias_v1，修正为每次恢复均深复制后重跑。其余已完成且零回退案例不受该问题影响。',
        '- 当前6项测试包含实际三角面片距离/梯度、heldout隔离，以及连续回退时动量与计数器恢复。最终收尾拒绝纳入存在回退却未标记修正实现的记录。','',
        f'电钻独立诊断：`{OUT/"REAL_INPUT_AUDIT.md"}`。完整数值：`{OUT/"summary.json"}`。']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n');print(dict(states=len(records),expected=48,complete=not missing),flush=True)
    return value


if __name__=='__main__':build()
