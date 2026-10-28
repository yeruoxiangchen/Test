"""Publish the human-readable review after independent CPU audits succeed."""
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_oracle_finalize_common',HERE.parent/'common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
    dest=c.OUT/'review_20261026'
    analysis=c.read(dest/'analysis.json');verification=c.read(dest/'verification.json')
    summary=c.read(c.OUT/'summary.json');state=c.read(c.OUT/'status.json')
    if not analysis['verification']['passed'] or not verification['passed']:
        raise ValueError('Independent audits must pass before publishing review')
    if not analysis['identity']==verification['runtime_identity']==summary['identity']:
        raise ValueError('Review identities differ')
    notes=json.loads((HERE/'review_visual_notes.json').read_text())
    figures=[]
    for uid,obj in notes['objects'].items():
        for seed,note in obj['seeds'].items():
            name=f'{uid}_seed{seed}_visible_oracle.jpg'
            b=c.binding(c.OUT/'figures'/name)
            expected=next(x for x in summary['figures'] if x['uid']==uid and x['seed']==int(seed))
            if any(expected[k]!=b[k] for k in ('path','sha256')):raise ValueError('Reviewed image changed')
            figures.append({'uid':uid,'name':obj['name'],'seed':int(seed),'image':b,'observation':note})
    visual={'time':c.now(),'identity':summary['identity'],'reviewer':notes['reviewer'],
        'codex_review_completed':True,'human_review_completed':False,'sheets_viewed':12,
        'layout':notes['layout'],'figures':figures,'objects':notes['objects'],
        'scope':'All 12 sheets actually viewed; qualitative assessment, not a blinded human study.'}
    c.write(dest/'visual_review.json',visual)
    arms=('appearance_control','geometry_observed','geometry_oracle_visible')
    names={'appearance_control':'匹配外观','geometry_observed':'预测几何','geometry_oracle_visible':'可见首表面 oracle'}
    macro=analysis['macro']['cells'];d=analysis['macro']['contrasts']['oracle_minus_predicted']
    coordinate=verification['coordinate_audit']
    lines=['# 可见首表面 oracle：完整结果审阅','',f'审阅时间：{c.now()}。队列结束：{state["time"]}。',
        '', '**结论：当前 SS 注入分支能够利用更可靠的可见几何，观测质量是瓶颈之一；强遮挡长凳仍失败，不能把全部问题归于深度，也不能据此证明原 LaRI 多层路线已经成立。**',
        '', '本轮200步训练、12个新SS/mesh、24个旧引用均已完成，共36个逻辑case。12张对比图已逐张查看；独立统计、产物及坐标核验通过。本次没有启动新的GPU训练或重新生成mesh。',
        '', '## 核心指标','',
        '| 方法 | CD↓ | F@.01↑ | F@.02↑ | SS IoU↑ | 遮挡召回↑ | 自由空间违规↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        m=macro[arm]
        lines.append(f'| {names[arm]} | {m["chamfer_l1"]:.6f} | {100*m["fscore_0p01"]:.2f}% | {100*m["fscore_0p02"]:.2f}% | '+
            f'{100*m["ss_iou"]:.2f}% | {100*m["occluded_recall"]:.2f}% | {100*m["free_space_violation"]:.2f}% |')
    lines += ['',f'主对照 oracle−预测几何：F@.02 +{100*d["fscore_0p02"]:.3f}pp、F@.01 +{100*d["fscore_0p01"]:.3f}pp、'+
        f'SS IoU +{100*d["ss_iou"]:.3f}pp、遮挡召回 +{100*d["occluded_recall"]:.3f}pp，CD相对下降{-100*d["chamfer_l1"]/macro["geometry_observed"]["chamfer_l1"]:.3f}%。'+
        f'自由空间违规却增加{100*d["free_space_violation"]:.3f}pp，应保留这一负面结果。',
        '', 'CD、F@.01/.02和遮挡召回各10/12配对改善；自由空间违规仅5/12配对改善。三seed的宏F@.02增益分别+7.520/+8.427/+4.291pp；事后描述性去掉任一seed，增益仍为+5.906～7.974pp，主表保留全部seed。收益比上一轮更稳定，但椅子与高凳贡献了88.68%的净F增益。三个seed只是采样重复，不是独立训练/场景的统计证据。',
        '', '## 逐物体指标与实际 mesh','',
        '| 对象 | 外观 F@.02 | 预测几何 F@.02 | oracle F@.02 | oracle相对预测 |',
        '|---|---:|---:|---:|---:|']
    for uid,obj in notes['objects'].items():
        values=analysis['per_object'][uid]['cells']
        fs=[100*values[a]['fscore_0p02'] for a in arms]
        lines.append(f'| {obj["name"]} | {fs[0]:.2f}% | {fs[1]:.2f}% | {fs[2]:.2f}% | {fs[2]-fs[1]:+.2f}pp |')
    for uid,obj in notes['objects'].items():
        lines+=['',f'**{obj["name"]}**：{obj["summary"]}', '',
            '图：'+ '、'.join(f'[seed {seed}](../figures/{uid}_seed{seed}_visible_oracle.jpg)' for seed in (42,43,44))+'。']
    lines += ['', '长凳oracle F@.02只有12.35%、SS precision约4.49%；CD比预测组和外观组都更差，遮挡召回也仍低于外观组。它的大面积错误表面可能提高召回，同时增加假阳性。不能把召回单项增加写成成功补全。',
        '', '## 可以定位什么，尚不能定位什么','',
        '本轮条件严格是**全场景可见首交点深度＋目标可见mask**，包括已观察到的其他物体表面和首表面之前的空间；不是完整目标mesh/隐藏支持。原始archive只提取depth_z_world，训练读取仅含depth/valid的独立缓存。可见mask、相机和canonical frame仍有GT辅助。',
        '', 'Oracle既替换深度，也把有效置信度设为1，改变surface/free/unknown的覆盖和强度。因此可以说“可靠几何观测有用”，不能说“唯一问题已经确定为深度预测”，也不能把oracle作为RGB可部署方法成绩。',
        '', '长凳的oracle surface不再严格为零，但mean-surface>0.1的token只有7/4096，peak>0.3只有28/4096，71.17%的token仍unknown>0.9。原四帧仅约620个目标可见像素。GT深度没有增加真实可见信息；信息不足、16³汇聚、形状先验、训练预算及注入方式仍互相混杂。不能由一个失败物体断言整个架构无效。',
        '', '自由空间是软条件特征，当前训练只有velocity MSE，没有显式输出一致性损失。输入free从canonical距离0.03125外才开始非零，0.09375以外饱和；评估违规则用0.02，且输入与评估分别采用双线性/最近像素。两者的容差和用途不同，准确首表面也不保证违规指标单调。该结果不能单独归为坐标错误，但表明几何一致性尚未解决。',
        '', '## 公平性、坐标与产物核验','',
        '三臂同完整旧appearance父模型、同3164160新增参数及初始SHA、同输入与SS目标、200步对象/原index/时间t/噪声/学习率，均为每物体50曝光；SLat冻结。尾50步loss依次为0.0603464/0.0607532/0.0601896，末步或均值接近不代表mesh一致，也不能证明已完全拟合。',
        '',f'独立复核36条原始case和72个记录绑定，1980项统计检查＋5514项训练字段检查；最大数值差{analysis["verification"]["max_abs_error"]:.3g}。'+
        f'产物核验{verification["checks_passed"]}项、文件hash {verification["file_hashes_verified"]}个，含{verification["source_files_verified"]}个源码/脚本/协议；'+
        f'{len(verification["large_bindings_not_rehashed"])}个超过256MiB的已绑定大权重明确没有重新字节hash。',
        '',f'12对新canonical/world mesh共{coordinate["vertices_checked"]:,}顶点全量检查：面一致，最大world误差{coordinate["world_max_abs_error_m"]:.3g}米，'+
        f'canonical往返误差{coordinate["canonical_roundtrip_max_abs_error"]:.3g}。没有额外拟合旋转/平移/缩放或ICP。12个新SS/latent/mesh跨seed独立，且逐case不同于两个旧对照。',
        '', '40个渲染记录、120张单视角图、12张对比图及9个已知摆放GLB绑定通过，无画面裁剪。审阅没有重跑GPU、重抽样表面距离或重编码latent；独立统计核对的是原指标记录。Codex视觉审阅不记作人工审阅，原human_review_completed保持false。',
        '', '## 下一步：限定机制定位，然后进入未见场景','',
        '建议只再补齐一次深度×置信度2×2对照的两格，复用已有两格：GT depth＋预测confidence，以及预测depth＋有效confidence=1。保持当前4对象、200步预算、相同全部输入与采样；不修改tau、不删除长凳、不挑seed。它用于决定优先修深度还是置信度/覆盖，预先限定为最后一轮四对象机制定位，不将其当论文主结果。',
        '', '同时准备按场景和对象隔离的留出评估，冻结一版可由RGB产生的几何条件，以同输入、同预算的appearance为主基线。重点检查目标内部置信度覆盖、小物体表面误差相对网格容差，以及观察一致性；oracle仅作为诊断保留。当前证据不支持直接扩大原LaRI长训，也没有要求重做已通过的mesh→latent管线。',
        '', '以上为审阅后的建议，本轮未实现或启动下一组训练。',
        '', '证据：[独立统计](analysis.json)、[产物/坐标核验](verification.json)、[12图视觉记录](visual_review.json)、[原自动报告](../REPORT.md)、[冻结协议]('+str(c.HERE/'PROTOCOL.md')+')。','']
    (dest/'REPORT.md').write_text('\n'.join(lines))
    review={'time':c.now(),'identity':summary['identity'],'codex_review_completed':True,
        'human_review_completed':False,'artifacts_verified':True,'all_visuals_inspected':True,
        'training_set_diagnostic_only':True,'next_experiment_launched':False,
        'conclusion':'reliable visible geometry helps current SS injection; severe-occlusion bench and consistency remain unresolved',
        'analysis':c.binding(dest/'analysis.json'),'verification':c.binding(dest/'verification.json'),
        'visual_review':c.binding(dest/'visual_review.json'),'report':c.binding(dest/'REPORT.md'),
        'review_sources':[c.binding(HERE/name) for name in ('review_statistics.py','review_integrity.py','review_visual_notes.json','finalize_review.py')]}
    c.write(dest/'review.json',review)
    print(json.dumps({'review':str(dest/'REPORT.md'),'artifacts_verified':True,'sheets_viewed':12,
        'next_experiment_launched':False},ensure_ascii=False))


if __name__=='__main__':main()
