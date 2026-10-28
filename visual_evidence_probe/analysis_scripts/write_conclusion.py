"""Write the reviewed interpretation; do not modify measurements or protocols."""
import numpy as np

from visual_evidence_probe.common import ROOT, OUT, read, write, now, binding


def main():
    raw = read(OUT / 'summary.json')
    crop_root = ROOT / 'outputs/correlated_views_cropped_control_20261025'
    crop = read(crop_root / 'summary.json')
    for root in (OUT, crop_root):
        if not read(root / 'status.json')['status'].startswith('complete'):
            raise ValueError('Require complete evaluations')
    means, by = crop['means'], crop['by_object']
    def group(arms):
        return {key: float(np.mean([means[a][key] for a in arms])) for key in means[arms[0]]}
    groups = [('两张原始观测', group(['real2'])), ('原图重复为八张', group(['repeat8'])),
              ('八张原始观测', group(['real8'])), ('两张＋GT灰色渲染（oracle）', group(['gt_render6'])),
              ('两张＋单一生成mesh供图（A/B均值）', group(['donor_a6','donor_b6'])),
              ('两张＋两个生成mesh混合（互补分配均值）', group(['mixed_ab6','mixed_ba6']))]
    base, coherent, mixed = groups[0][1], groups[-2][1], groups[-1][1]
    coherent_object = .5*(np.array(by['donor_a6']['cd'])+by['donor_b6']['cd'])
    mixed_object = .5*(np.array(by['mixed_ab6']['cd'])+by['mixed_ba6']['cd'])
    base_object = np.array(by['real2']['cd'])
    result = {'time':now(), 'status':'complete_reviewed', 'raw_results':binding(OUT/'summary.json'),
        'cropped_results':binding(crop_root/'summary.json'), 'raw_meshes':raw['records'],
        'cropped_meshes':crop['records'], 'new_model_trained':False, 'geometric_confidence_tested':False,
        'groups':dict(groups), 'coherent_cd_relative_to_real2_percent':100*(coherent['cd']/base['cd']-1),
        'coherent_f02_minus_real2_pp':100*(coherent['f02']-base['f02']),
        'mixed_cd_relative_to_coherent_percent':100*(mixed['cd']/coherent['cd']-1),
        'mixed_f02_minus_coherent_pp':100*(mixed['f02']-coherent['f02']),
        'coherent_worse_cd_objects':int((coherent_object>base_object).sum()),
        'mixed_worse_cd_objects':int((mixed_object>base_object).sum()),
        'mixed_better_than_coherent_cd_objects':int((mixed_object<coherent_object).sum()),
        'source_correlation_as_main_cause_established':False,
        'decision':'synthetic-input degradation observed in this small crop-conformant cohort; no stable benefit from mixing donor seeds; do not start new-module training on this evidence alone'}
    write(ROOT/'validation_summary.json',result)
    lines=['# 第一次验证结论', '', result['time'], '',
        '**结论：在当前30K模型及这批小样本上，生成视图会污染重建；但没有证据证明来源相关性是主因，混合两个生成结果也没有稳定收益。暂不据此启动相关性模块长训。**', '',
        '## 实验完成性', '',
        '- 第一轮：8个已检查HSSD对象×8组×2 seeds＝128个mesh，全部完成。',
        '- 发现第一轮完整画面缓存与30K训练裁剪契约不一致，因此保留归档，不作为主要机制证据。',
        '- 追加事后裁剪复核：原清单前4个对象×8组×2 seeds＝64个mesh，全部完成。',
        '- 两轮共192个结果mesh，均保存OBJ；未训练任何新模型，原30K权重保持冻结。',
        '- SS缓存实现与原实现正负分支逐值一致；相同输入两次完整mesh回放一致。',
        '- 原始相机GT轮廓核对通过，裁剪的相机仿射投影误差<1e-3像素。',
        '- 两项原协议CPU检查及一项裁剪图像/内参配对检查通过。', '',
        '## 符合训练裁剪后的结果', '',
        '以下为4对象、每对象2 seeds的均值。CD为规范坐标下50000表面采样点的对称L1距离；F阈值0.02。',
        '无旋转/平移/尺度拟合，无ICP，不剔除坏结果。', '',
        '| 输入 | CD↓ | F@.02↑ | SS表面占据IoU↑ |', '|---|---:|---:|---:|']
    for name,m in groups:
        lines.append(f"|{name}|{m['cd']:.6f}|{100*m['f02']:.2f}%|{m['ss_iou']:.4f}|")
    delta=crop['balanced_coherent_minus_mixed']['cd']
    lines += ['', '## 数据支持的判断', '',
        f"1. 单一生成来源相对两张原图，CD增加{result['coherent_cd_relative_to_real2_percent']:.2f}%，F下降{-result['coherent_f02_minus_real2_pp']:.2f}个百分点；4/4对象的CD均变差。",
        '2. GT灰色渲染使用同一个中性渲染器，平均结果优于生成mesh供图。这说明生成几何的错误值得研究，但不能由此排除全部输入风格或模型适配因素。',
        f"3. 混合两个生成结果相对单一来源，CD变化{result['mixed_cd_relative_to_coherent_percent']:+.2f}%，F变化{result['mixed_f02_minus_coherent_pp']:+.2f}个百分点。CD只有{result['mixed_better_than_coherent_cd_objects']}/4对象改善，指标方向不一致。",
        f"4. 单一来源减混合来源的配对CD差为{delta['mean']:+.8f}，同场景对象bootstrap区间{delta['within_scene_object_bootstrap_ci95']}；该区间只是描述性结果，不是跨场景统计保证。",
        '5. 柜架案例中，原始额外视图能恢复开放层架；生成来源及混合来源仍倾向封闭柜面。输入图和同相机mesh对照已保留。', '',
        '## 尚未验证的主张', '',
        '- 两张原始图未观察到的结构可能存在多种解释。相对GT的错误，不一定能仅凭这两张输入被识别；本轮不声称能恢复唯一隐藏结构。',
        '- 原30K没有校准的几何置信度输出，本轮不能声称“更自信却更错”。',
        '- 两个seed来自同一生成器、同两张输入，未被证明是独立来源。混合不同mesh还引入跨视图形状不一致，不能把本对照简单解释为严格去相关实验。',
        '- “存在生成输入污染”不等于“来源相关性是污染主因”，也不等于新增融合模块已经有效。',
        '- 四个对象来自一个已检查场景，且裁剪控制为事后追加；不能声称真实拍摄泛化或论文主表成绩。',
        '- 使用中性mesh渲染，未验证照片级NVS生成器、第二个重建骨干或跨生成器迁移。',
        '- 本轮测试native SS/SLat，不含原论文后续O2W/轮廓细化；输入一致性筛查或姿态尺度细化可能缓解部分污染，本轮未测试其实际修复收益。', '',
        '## 研究决策', '',
        '保留“生成内容如何被当作证据使用”这一问题线索，但目前不足以把来源相关性建模确定为论文核心。',
        '在任何新结构训练前，应先比较输入一致性筛查等强简单基线，并在第二个供图/重建模型上复核。若剩余收益空间不明确，应转向其他候选选题。', '',
        '## 文件', '',
        '- [完整审计](审计.md)',
        '- [裁剪复核数值](outputs/correlated_views_cropped_control_20261025/summary.json)',
        '- [裁剪复核图表](outputs/correlated_views_cropped_control_20261025/figures/paired_results.png)',
        '- [柜架输入图](outputs/correlated_views_cropped_control_20261025/objects/002/inputs.jpg)',
        '- [柜架mesh对照](outputs/correlated_views_cropped_control_20261025/objects/002/mesh_comparison.jpg)',
        '- [柜架OBJ结果](outputs/correlated_views_cropped_control_20261025/objects/002/results/)',
        '- [第一轮探索记录（含输入契约混杂）](outputs/correlated_views_20261025/REPORT.md)', '']
    (ROOT/'第一次验证结论.md').write_text('\n'.join(lines)+'\n')
    audit=ROOT/'审计.md'
    marker='## 最终复核结论（'+result['time']+'）'
    with audit.open('a') as f:
        f.write('\n'+marker+'\n\n')
        f.write(f"两轮共192个结果mesh全部完成。正确裁剪控制：基线CD={base['cd']:.8f}、F={base['f02']:.8f}；")
        f.write(f"单一来源均值CD={coherent['cd']:.8f}、F={coherent['f02']:.8f}；混合来源CD={mixed['cd']:.8f}、F={mixed['f02']:.8f}。\n\n")
        f.write('观察到生成输入污染；来源相关性作为主因尚未确立，混合来源无跨指标稳定收益。没有启动新训练。\n\n')
        f.write('详细判断、数据边界及全部产物见[第一次验证结论](第一次验证结论.md)。先前审计段落全部保留。\n')
    for root in (OUT,crop_root):
        state=read(root/'status.json')
        write(root/'status.json',{**state,'status':'complete_reviewed','review_time':result['time'],
                                  'review':str(ROOT/'第一次验证结论.md')})
    print(result)


if __name__=='__main__': main()
