import numpy as np

from .config import OUT, PARAMS, SEEDS, folder, read, write, verify, now, check
from .scores import METHODS


def mean_defined(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def aggregate(records):
    by_object={}
    for r in records:by_object.setdefault(r['index'],[]).append(r)
    result={}
    for threshold in PARAMS['depth_thresholds']:
        result[str(threshold)]={}
        for method in METHODS:
            fields=('ap','auroc','ap_lift','precision_top20','recall_top20')
            objects=[]
            for i,rows in sorted(by_object.items()):
                values=[r['metrics'][str(threshold)] for r in rows]
                obj=dict(index=i,uid=rows[0]['uid'],records=len(rows),
                    defined_seeds=sum(v['common'][method]['defined'] for v in values),
                    coverage=float(np.mean([v['common_coverage'] for v in values])),
                    prevalence=mean_defined([v['common_error_rate'] for v in values]),
                    spearman=mean_defined([v['common_depth_error_spearman'][method] for v in values]))
                obj.update({key:mean_defined([v['common'][method][key] for v in values]) for key in fields})
                objects.append(obj)
            result[str(threshold)][method]=dict(objects=len(objects),
                defined_objects=sum(o['ap'] is not None for o in objects),
                mean_coverage=float(np.mean([o['coverage'] for o in objects])),
                prevalence=mean_defined([o['prevalence'] for o in objects]),
                spearman=mean_defined([o['spearman'] for o in objects]),
                **{field:mean_defined([o[field] for o in objects]) for field in fields},by_object=objects)
    return result


def paired(summary,candidate,reference):
    a=summary['0.02'][candidate]['by_object'];b=summary['0.02'][reference]['by_object']
    common=[(x,y) for x,y in zip(a,b) if x['ap'] is not None and y['ap'] is not None]
    if not common:return dict(objects=0,mean_ap_gain=None)
    values=np.array([x['ap']-y['ap'] for x,y in common])
    rng=np.random.default_rng(20261026)
    boot=values[rng.integers(len(values),size=(5000,len(values)))].mean(1)
    return dict(objects=len(values),mean_ap_gain=float(values.mean()),
        wins=int((values>1.e-9).sum()),losses=int((values < -1.e-9).sum()),
        object_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),deltas=values.tolist())


def full_summary(records):
    methods=('silhouette','boundary','direct_rgb','direct_dino','fused_full')
    result={}
    for method in methods:
        rows=[]
        for i in sorted({r['index'] for r in records}):
            scores=[r['metrics']['0.02']['full'][method] for r in records if r['index']==i]
            rows.append(dict(index=i,**{k:mean_defined([r[k] for r in scores])
                for k in ('ap','auroc','prevalence','ap_lift','precision_top20','recall_top20')}))
        result[method]=dict(by_object=rows,defined_objects=sum(r['ap'] is not None for r in rows),
            **{k:mean_defined([r[k] for r in rows]) for k in ('ap','auroc','prevalence','ap_lift','precision_top20','recall_top20')})
    return result


def report():
    check();natural=[];controls=[];counts={}
    for i in range(12):
        keys=[str(s) for s in SEEDS]+(['perfect_geometry','local_defect'] if i>=4 else [])
        for key in keys:
            path=folder(i)/'metrics'/f'{key}.json'
            r=read(path)
            if not r['success'] or r['uses_gt_for_scores']:raise ValueError('invalid/incomplete result')
            for field in ('maps','mesh','labels'):verify(r[field])
            if not (folder(i)/'figures'/f'{key}.jpg').exists():raise ValueError('missing figure')
            if not (folder(i)/'heat_meshes'/f'{key}_input_scores.ply').exists():raise ValueError('missing 3D score map')
            (controls if r['control'] else natural).append(r)
    if len(natural)!=24 or len(controls)!=16:raise ValueError('incomplete test')
    test=[r for r in natural if r['split']=='omni_test'];dev=[r for r in natural if r['split']=='hssd_development']
    summaries=dict(omni_new_objects=aggregate(test),hssd_development=aggregate(dev),
        local_defect_controls=aggregate([r for r in controls if r['key']=='local_defect']),
        perfect_geometry_controls=aggregate([r for r in controls if r['key']=='perfect_geometry']))
    comparisons={f'{c}_minus_{b}':paired(summaries['omni_new_objects'],c,b)
                 for c in ('direct_dino','cross_dino','fused_dino') for b in ('boundary','shuffled_dino')}
    zero_controls=all(r['metrics']['0.02']['interior_error_rate'] in (None,0.) for r in controls if r['key']=='perfect_geometry')
    if not zero_controls:raise ValueError('perfect geometry labels are not zero')
    result=dict(time=now(),natural_records=len(natural),control_records=len(controls),new_meshes=16,
        summaries=summaries,paired_comparisons=comparisons,correct_geometry_zero_error=True,
        omni_full_visible_summary=full_summary(test),
        scope='8 new probe objects, rendered scanned data; prior-paper objects, not new scene real-capture validation',
        no_model_training=True,all_successful=True,threshold_selection=False,hidden_surface_localization_tested=False)
    write(OUT/'summary.json',result)
    lines=['# 第三轮：全局对齐后的可见错误定位', '', f"完成时间：{result['time']}", '',
        '8个新增Omni物体×2seed；旧4个HSSD物体单独报告。新模型输出均按4张输入mask全局拟合。',
        '不训练、不用GT计算分数；GT仅用于封存输入分数后的camera-z深度错误评估。',
        '输入是扫描对象的渲染图，不是新的真实采集测试。不同域和不同视图数不能混作单一泛化结论。', '',
        '## 主结果：新Omni对象，内部共同支持区域，深度阈值0.02', '',
        '|分数|AP↑|AUROC↑|AP/误差率↑|Top20% precision↑|连续误差Spearman↑|有效对象|覆盖率|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'NA' if x is None else f'{x:.4f}'
    for method,v in summaries['omni_new_objects']['0.02'].items():
        lines.append(f"|{method}|{fmt(v['ap'])}|{fmt(v['auroc'])}|{fmt(v['ap_lift'])}|{fmt(v['precision_top20'])}|{fmt(v['spearman'])}|{v['defined_objects']}/8|{v['mean_coverage']:.3f}|")
    lines+=['', '同分数按组积分，常数分数AP等于误差率，AUROC=0.5。没有正/负样本的对象标NA，',
        '在完整数据中保留并报告有效对象数；均值仅对可定义指标计算，不能当作全部物体均有效。',
        '覆盖率是输入前景和预测前景重叠的内部区域中，有至少一个预测为共同可见参考图的比例。',
        '隐藏区域和没有跨图支持的区域不据此判断可靠。不同阈值与逐物体值见summary.json。', '',
        '## 对简单基线的物体级比较', '']
    for name,v in comparisons.items():lines.append(f'- {name}: {v}')
    lines+=['', '## 对照与解释', '',
        '- 正确GT几何控制的深度误差必须为零，非零RGB/DINO残差不等于真实几何错误。',
        '- 特征patch空间打乱保持特征内容，破坏几何对应；若原分数不优于它，不能声称正确利用了投影对应。',
        '- 人工局部形变是GT辅助构造，只作sanity，不能替代自然模型错误的证据。',
        '- 直接灰色渲染RGB/特征差异含材质/光照/渲染域影响，禁止直接将其作为几何错误标签。',
        '- 64³体素分数由输入证据投影汇总，未知顶点为灰色，不据此评价隐藏表面或定位缺失几何的唯一深度。', '',
        '## 产物与复现', '', '```bash', 'bash visual_evidence_probe/run_localization.sh prepare',
        'bash visual_evidence_probe/run_localization.sh worker --gpu 0 --indices 0,2,4,6,8,10',
        'bash visual_evidence_probe/run_localization.sh worker --gpu 1 --indices 1,3,5,7,9,11',
        'bash visual_evidence_probe/run_localization.sh report', '```', '',
        '[完整数据](summary.json)；[固定方案](../../experiments/localization/PROTOCOL.md)。', '']
    for i in range(12):
        rec=read(folder(i)/'observations.json')
        lines.append(f"- {i:03d} {rec['uid']}: [seed42图](objects/{i:03d}/figures/42.jpg)、[seed43图](objects/{i:03d}/figures/43.jpg)、[OBJ与拟合](objects/{i:03d}/fitted/)、[3D热图/体素](objects/{i:03d}/heat_meshes/)。")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json',dict(time=now(),stage='complete_pending_review',natural_records=24,control_records=16))
    print(OUT/'REPORT.md',flush=True)
