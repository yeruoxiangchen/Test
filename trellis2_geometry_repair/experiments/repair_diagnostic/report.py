from collections import defaultdict
from pathlib import Path
import numpy as np

from .config import OUT,PARAMS,ARMS,case,original,read,write,now


def mean_by_object(rows,fn):
    groups=defaultdict(list)
    for row in rows:
        value=fn(row)
        if value is not None and np.isfinite(value):groups[row['uid']].append(value)
    values={uid:float(np.mean(v)) for uid,v in groups.items()}
    return dict(mean=float(np.mean(list(values.values()))) if values else None,by_object=values)


def build():
    rows=[read(case(i,s)/'result.json') for i in PARAMS['indices'] for s in PARAMS['seeds'] if (case(i,s)/'result.json').exists()]
    audits=[read(case(i,s)/'input_audit.json') for i in PARAMS['indices'] for s in PARAMS['seeds'] if (case(i,s)/'input_audit.json').exists()]
    arms={}
    for arm in ARMS:
        arms[arm]=dict(oracle=arm.endswith('oracle'),endpoints={},accepted=sum(r['arms'][arm]['decisions']['accepted_step']>0 for r in rows))
        for ep in ('accepted','candidate','proposal'):
            cd=mean_by_object(rows,lambda r:(r['baseline']['surface']['chamfer_l1']-r['arms'][arm]['endpoints'][ep]['surface']['chamfer_l1'])/max(r['baseline']['surface']['chamfer_l1'],1.e-8))
            depth=mean_by_object(rows,lambda r:(r['baseline']['novel_depth']['bad']-r['arms'][arm]['endpoints'][ep]['novel_depth']['bad'])/max(r['baseline']['novel_depth']['bad'],1.e-8)
                if r['baseline']['novel_depth']['bad'] is not None else None)
            good=mean_by_object(rows,lambda r:r['arms'][arm]['endpoints'][ep]['novel_depth']['good']-r['baseline']['novel_depth']['good']
                if r['baseline']['novel_depth']['good'] is not None else None)
            arms[arm]['endpoints'][ep]=dict(cd_relative_gain=cd,bad_depth_relative_gain=depth,good_depth_damage=good)
        arms[arm]['mean_seconds']=float(np.mean([r['arms'][arm]['seconds'] for r in rows])) if rows else None
        arms[arm]['fitted_target_at_any_checkpoint']=sum(any(c['depth_gain']>=.1 for c in r['arms'][arm]['decisions']['checks']) for r in rows)
    value=dict(time=now(),cases=len(rows),expected_cases=8,complete=len(rows)==8,arms=arms,records=rows,
          audits=audits,gt_oracle_is_diagnostic_only=True,independent_generalization=False)
    write(OUT/'summary.json',value)
    def percent(x):return 'NA' if x is None else f'{100*x:.4f}%'
    text=['# TRELLIS.2 大误差梯度与纠错能力诊断','',f'更新：{now()}；评估 {len(rows)}/8；四组对照共32组。',
          '','同一原生latent、输入对齐和输入定义区域；GT oracle仅作机制诊断，不计入方法成绩。',
          '','|组别|接受数|实际交付CD下降|候选CD下降|末步CD下降|末步原错误像素深度下降|',
          '|---|---:|---:|---:|---:|---:|']
    for arm,item in arms.items():
        ep=item['endpoints'];text.append(f'|{arm}|{item["accepted"]}/{len(rows)}|{percent(ep["accepted"]["cd_relative_gain"]["mean"])}|{percent(ep["candidate"]["cd_relative_gain"]["mean"])}|{percent(ep["proposal"]["cd_relative_gain"]["mean"])}|{percent(ep["proposal"]["bad_depth_relative_gain"]["mean"])}|')
    if audits:
        text+=['','## 损失饱和与局部性','',
           f'初始目标射线中误差达到0.1截断值的比例，案例均值：{percent(float(np.mean([a["depth"]["clipped_fraction"] for a in audits])))}。',
           f'旧/新深度场零梯度比例均值：{percent(float(np.mean([a["old_zero_gradient_fraction"] for a in audits])))} / {percent(float(np.mean([a["new_zero_gradient_fraction"] for a in audits])))}。',
           '逐案例目标深度、中位值、原始MAE、梯度与激活token比例见input_audit.json；深度场有梯度不等于latent可达。']
    if rows:
        worse=sum(r['evidence_quality'].get('target_raw_mae',0)>r['evidence_quality'].get('base_raw_mae',0) for r in rows)
        text+=['','## 证据质量与解释','',f'在固定输入射线共同GT支持上，混合深度目标比原mesh更差的案例：{worse}/{len(rows)}。',
          'teacher尺度比与有符号偏差仅供GT事后审计，没有回灌实际输入组。',
          f'至少一个检查点达到10%训练目标下降的案例数：'+', '.join(f'{a}={arms[a]["fitted_target_at_any_checkpoint"]}/{len(rows)}' for a in ARMS)+'。',
          '对照定位当前参数化、证据和约束的瓶颈；不将某个优化器失败解释为整个表示理论上不可修复。']
    text+=['','正值为改善；全部回退计入交付分母。latent/mesh、candidate/proposal及oracle分别报告。',
           '所有图显示末步未过滤proposal，不能把oracle图或未通过门限的图当作实际交付。',
           '本轮复用已检查开发物体；完成状态不以指标变好为条件。','']
    review=Path(__file__).with_name('VISUAL_REVIEW.md')
    if review.exists():text+=['',review.read_text()]
    support_path=OUT/'support_audit.json'
    if support_path.exists():
        support=read(support_path)
        text+=['','## 固定SS支持的事后距离审计','',
          '在同一冻结配准下，统计目标点到原32网格占据单元并集的距离；此审计不回灌输入组。',
          f'保守邻域阈值：{support["conservative_distance_threshold"]:.6f}原生物体坐标单位。',
          '|案例|输入目标超出保守邻域|GT目标超出保守邻域|','|---|---:|---:|']
        for row in support['records']:
            a=row['teacher'].get('beyond_conservative_margin_fraction');b=row['gt'].get('beyond_conservative_margin_fraction')
            text.append(f'|{row["index"]:03d}/{row["seed"]}|{percent(a)}|{percent(b)}|')
        text+=['','支持外目标可能来自全局注册或粗结构错误，此检查没有将两者分离；支持内也不代表当前优化能到达。',
               '细分单元继承固定SS父单元，单靠小幅SLat特征更新无法无条件补齐远处新结构。',
               '若GT在支持内而输入目标在支持外，应先检查证据，不能盲目扩展SS追随错误目标。']
    (OUT/'REPORT.md').write_text('\n'.join(text));print(dict(evaluated=len(rows),complete=value['complete']),flush=True)
    return value


def verify():
    import torch
    checked=0
    for i in PARAMS['indices']:
        for seed in PARAMS['seeds']:
            saved=torch.load(original(i,seed)/'original_latent.pt',weights_only=False)
            with np.load(original(i,seed)/'evidence_sealed.npz') as a:active=torch.tensor(a['active'])
            for arm in ARMS:
                item=read(case(i,seed)/arm/'complete.json')
                if item['gt_used_in_training']!=arm.endswith('oracle'):raise RuntimeError('oracle label mismatch')
                for ep in ('accepted','candidate','proposal'):
                    path=case(i,seed)/arm/f'{ep}.obj'
                    if not path.is_file():raise RuntimeError('missing endpoint OBJ')
                    state=torch.load(case(i,seed)/arm/f'{ep}_state.pt',weights_only=False)
                    if arm.startswith('latent'):
                        if not torch.equal(state['coords'],saved['coords']):raise RuntimeError('SS changed')
                        if not torch.equal(state['features'][~active],saved['features'][~active]):raise RuntimeError('outside latent changed')
                        if not bool(torch.isfinite(state['features']).all()):raise RuntimeError('nonfinite latent')
                    elif not state['outside_vertices_exact'] or not bool(torch.isfinite(state['delta']).all()):
                        raise RuntimeError('mesh state invalid')
                    checked+=1
    receipt=dict(time=now(),passed=True,endpoints_checked=checked,oracle_labels_separate=True)
    write(OUT/'validation/artifact_checks.json',receipt);return receipt


if __name__=='__main__':build()
