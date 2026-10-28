"""Review complete frozen experiment outputs without selecting new score rules."""
from pathlib import Path
import os

import numpy as np
from PIL import Image,ImageDraw

from visual_evidence_probe.common import read,write,verify,binding,now,ROOT
from visual_evidence_probe.experiments.localization.config import OUT,folder,check


def latent_coordinates(index,key,score_root=None,suffix='latent_grid'):
    """Undo global fitting before attaching evidence to the original SLat support."""
    from scipy.spatial import cKDTree
    from visual_evidence_probe.experiments.localization.visuals import voxelize
    prior=OUT.parent/'input_consistency_baselines_20261025'
    if index<4:
        letter='a' if key=='42' else 'b'
        transform=np.array(read(prior/'objects'/f'{index:03d}'/f'refined/{letter}.json')['chosen']['transform'])
        ss_path=OUT.parent/'correlated_views_cropped_control_20261025/objects'/f'{index:03d}'/f'results/real2/{key}_ss.npz'
    else:
        transform=np.array(read(folder(index)/f'fitted/{key}.json')['chosen']['transform'])
        ss_path=folder(index)/f'generated/{key}_ss.npz'
    inverse=np.linalg.inv(transform)
    maps_path=(folder(index) if score_root is None else score_root)/'scores'/f'{key}.npz'
    with np.load(maps_path) as a:
        xyz=a['pred_xyz'];scores=a['fused_dino'];valid=a['common']
    points=xyz@inverse[:3,:3].T+inverse[:3,3]
    cells,values,counts,outside=voxelize(points,scores,valid)
    with np.load(ss_path) as a:coords=a['coords']
    tree=cKDTree(coords[:,1:])
    dist,nearest=tree.query(cells,distance_upper_bound=2.)
    matched=np.isfinite(dist)
    token_counts=np.bincount(nearest[matched],weights=counts[matched],minlength=len(coords))
    token_sums=np.bincount(nearest[matched],weights=values[matched]*counts[matched],minlength=len(coords))
    observed=token_counts>0
    token_scores=np.full(len(coords),np.nan,np.float32)
    token_scores[observed]=token_sums[observed]/token_counts[observed]
    p=folder(index)/'heat_meshes'/f'{key}_{suffix}.npz'
    np.savez_compressed(p,coords=coords,scores=token_scores,observed=observed,counts=token_counts,
                        world_to_original_generation=inverse,observed_voxels=cells,voxel_scores=values)
    write(p.with_suffix('.json'),dict(uses_gt=False,coordinates='original pre-fit SS/SLat canonical grid64',
        inverse_fit_applied=True,active_ss_support=binding(ss_path),observed_tokens=int(observed.sum()),
        source_scores=binding(maps_path),
        total_tokens=len(coords),observed_token_fraction=float(observed.mean()),
        nearest_support_max_distance_voxels=2.,out_of_cube_samples=outside,
        unknown='NaN with observed=False, never low error by default'))


def zoom(index,key):
    source=folder(index)/'figures'/f'{key}.jpg'
    target=source.with_name(f'{key}_zoom.jpg')
    with np.load(folder(index)/'scores'/f'{key}.npz') as a:
        union=a['pred_mask']|a['observed_mask']
    sheet=Image.open(source).convert('RGB')
    canvas=sheet.copy();draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,canvas.width,31),fill='white')
    draw.text((5,5),'Display crop from observed/predicted masks, same for every method; per-panel 5-95% heat scale; gray = unknown.',fill='black')
    for view,mask in enumerate(union):
        ys,xs=np.where(mask)
        if not len(xs):continue
        side=mask.shape[0];left,right=int(xs.min()),int(xs.max()+1);top,bottom=int(ys.min()),int(ys.max()+1)
        pad=max(2,int(.12*max(right-left,bottom-top)))
        box=np.array([max(0,left-pad),max(0,top-pad),min(side,right+pad),min(side,bottom+pad)])
        box=np.round(box*160/side).astype(int)
        y=32+view*188+28
        for col in range(10):
            panel=sheet.crop((col*160,y,(col+1)*160,y+160))
            enlarged=panel.crop(tuple(box)).resize((160,160),Image.Resampling.NEAREST if col>=2 else Image.Resampling.LANCZOS)
            canvas.paste(enlarged,(col*160,y))
    canvas.save(target,quality=94)


def main():
    check()
    amendment=read(OUT/'buffer_amendment.json')
    for b in amendment['sources']+[amendment['audit'],amendment['base_protocol']]:verify(b)
    summary=read(OUT/'summary.json');p=read(OUT/'protocol.json')
    if not summary['all_successful'] or summary['natural_records']!=24 or summary['control_records']!=16:
        raise ValueError('incomplete result')
    all_records=[];metadata=[]
    for index in range(12):
        obs=read(folder(index)/'observations.json')
        keys=['42','43']+(['perfect_geometry','local_defect'] if index>=4 else [])
        for key in keys:
            r=read(folder(index)/'metrics'/f'{key}.json')
            if not r['success'] or r['uses_gt_for_scores']:raise ValueError('invalid record')
            # The frozen report has already verified every full mesh SHA once.
            # Do not reread gigabytes merely to repeat that verification.
            for name in ('maps','labels'):verify(r[name])
            meta=read(folder(index)/'scores'/f'{key}.json')
            if not meta['input_only'] or meta['projection_error_pixels']>2.e-3:raise ValueError('input geometry audit failed')
            heat=read(folder(index)/'heat_meshes'/f'{key}_input_scores.json')
            if heat['uses_gt']:raise ValueError('heat map used GT')
            all_records.append(r);metadata.append(meta)
            zoom(index,key)
            if key.isdecimal():latent_coordinates(index,key)
        if not read(folder(index)/'camera_audit.json')['passed']:raise ValueError('camera gate failed')
    records=[r for r in all_records if not r['control'] and r['split']=='omni_test']
    scores=[read(folder(r['index'])/'scores'/f"{r['key']}.json") for r in records]
    fits=[read(folder(r['index'])/'fitted'/f"{r['key']}.json") for r in records]
    model2=read(OUT/'runtime/2_ss.json');model3=read(OUT/'runtime/3_ss.json')
    slat2=read(OUT/'runtime/2_slat.json');slat3=read(OUT/'runtime/3_slat.json')
    if model2['checkpoint']!=model3['checkpoint'] or slat2['sha256']!=slat3['sha256']:
        raise ValueError('model binding mismatch')
    timings=dict(score_mean_seconds=float(np.mean([r['elapsed_seconds'] for r in scores])),
                 score_median_seconds=float(np.median([r['elapsed_seconds'] for r in scores])),
                 fit_mean_seconds=float(np.mean([r['elapsed_seconds'] for r in fits])))
    crop_root=OUT/'input_crop_resolution_control'
    crop=read(crop_root/'summary.json')
    cp=read(crop_root/'protocol.json')
    for b in (cp['source'],cp['base_protocol'],cp['buffer_fix']):verify(b)
    if not crop['all_successful'] or crop['records']!=16:raise ValueError('crop control incomplete')
    cv=read(crop_root/'raster_validation_amendment.json')
    verify(cv['source']);verify(cv['base'])
    if len(cv['validations'])!=8 or not all(v['passed'] for v in cv['validations']):raise ValueError('crop native validation incomplete')
    aliasing_views=[dict(index=v['index'],view=j,iou=x) for v in cv['validations']
                   for j,x in enumerate(v['cropped_mesh_raster_ious']) if x<.90]
    for index in range(4,12):
        for seed in ('42','43'):
            latent_coordinates(index,seed,crop_root/'objects'/f'{index:03d}',suffix='crop_latent_grid')
    receipt=dict(time=now(),all_passed=True,records=40,natural_records=24,control_records=16,
        original_sources_unchanged=True,buffer_fix_validated=True,all_camera_checks_passed=True,
        max_projection_error_pixels=max(r['projection_error_pixels'] for r in metadata),
        model_bindings_identical=True,gt_used_only_for_evaluation_and_explicit_controls=True,
        heatmap_records=40,display_crops_use_only_input_and_prediction=True,timings=timings,
        large_meshes='verified by frozen report before relocation; legacy copies migrating in background',
        full_mesh_sha_verification='completed once by the frozen report before storage relocation',
        reviewer_source=binding(__file__))
    receipt['crop_control_records']=16
    receipt['cropped_geometry_raster_low_iou_views']=aliasing_views
    receipt['native_and_identically_resampled_mask_threshold']=.90
    write(OUT/'validation_receipt.json',receipt)
    table=summary['summaries']['omni_new_objects']['0.02']
    from visual_evidence_probe.experiments.localization.report import paired
    extra=dict(raw={f'{c}_minus_{b}':paired(summary['summaries']['omni_new_objects'],c,b)
                    for c in ('cross_rgb','direct_rgb') for b in ('boundary','shuffled_dino')},
               crop={f'{c}_minus_{b}':paired(crop['summaries'],c,b)
                     for c in ('cross_rgb','direct_rgb') for b in ('boundary','shuffled_dino')})
    write(OUT/'photometric_baseline_comparisons.json',dict(time=now(),comparisons=extra,
          scope='paired analysis of predeclared baseline methods; no new score or fitted parameter'))
    f=lambda x:'NA' if x is None else f'{x:.4f}'
    lines=['# 第三次验证结论：可见几何错误定位', '',f"复核时间：{receipt['time']}", '',
        '本轮完成8个新Omni物体×2seed的自然重建与定位；旧4个HSSD对象另作开发诊断。',
        '另有8个正确几何和8个人工局部形变控制，共40条记录，全部通过完整性检查。',
        '新输入为扫描对象的渲染图，Omni200曾用于原论文；不称为新的实拍场景测试。未训练模型。', '',
        '## 共同支持的内部区域：主阈值0.02', '',
        '|分数|AP↑|AUROC↑|AP/误差率↑|Top20% precision↑|深度误差Spearman↑|有效物体|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in table.items():
        lines.append(f"|{name}|{f(v['ap'])}|{f(v['auroc'])}|{f(v['ap_lift'])}|{f(v['precision_top20'])}|{f(v['spearman'])}|{v['defined_objects']}/8|")
    coverage=table['cross_dino']['mean_coverage']
    lines+=['',f"共同支持覆盖率：{coverage:.1%}。覆盖率分母为两种前景交集且离边界>2像素的内部区域，非整个物体表面。", '',
        '无正/负例的二分类指标为NA，保留所有对象并注明有效数量。按seed后按物体聚合，',
        'AP/误差率是物体级比值的均值；不能把像素或两个seed当独立样本。', '',
        '## 全可见图像区域（包含轮廓缺失/多余，未知跨图处回退直接DINO）', '',
        '|分数|AP↑|AUROC↑|AP/误差率↑|', '|---|---:|---:|---:|']
    for name,v in summary['omni_full_visible_summary'].items():
        lines.append(f"|{name}|{f(v['ap'])}|{f(v['auroc'])}|{f(v['ap_lift'])}|")
    lines+=['', '## 对空间打乱和边界基线的比较', '']
    for name,v in summary['paired_comparisons'].items():lines.append(f'- {name}: {v}')
    lines+=['', '## 输入裁剪下的事后分辨率控制：全部八个新物体', '',
        '完整图128会使细物体内部像素几乎消失；此控制不剔除物体，不改原主结果。',
        '只改变残差/评估的栅格：按已有输入mask裁剪到模型使用的相机，原mesh、特征、方法、阈值保持一致。',
        '不是GT裁剪，不重新生成或拟合；应单独解释为事后分辨率控制。', '',
        '额外限制：风筝一个侧视图的直接裁剪重渲染mask IoU=0.8526，原始相机IoU=0.9790。',
        '该失败保留；补充验证在原始尺度及同样裁剪/插值/阈值的mask上维持0.90门槛，另验裁剪仿射。',
        '验证规则的这一事后修订显式登记，裁剪控制含重采样边界误差，不能取代原始尺度主结果。', '',
        '|分数|AP↑|AUROC↑|AP/误差率↑|Top20% precision↑|Spearman↑|有效物体|覆盖率|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in crop['summaries']['0.02'].items():
        lines.append(f"|{name}|{f(v['ap'])}|{f(v['auroc'])}|{f(v['ap_lift'])}|{f(v['precision_top20'])}|{f(v['spearman'])}|{v['defined_objects']}/8|{v['mean_coverage']:.3f}|")
    lines+=['', '裁剪控制的物体级配对比较：', '']
    for name,v in crop['paired_comparisons'].items():lines.append(f'- {name}: {v}')
    lines+=['', '预先列入比较的RGB简单基线的补充物体级统计（未新增分数或调参）：', '']
    for domain,comparisons in extra.items():
        for name,v in comparisons.items():lines.append(f'- {domain} / {name}: {v}')
    lines+=['', '## 逐物体支持量：保留所有对象，不能只看覆盖率百分比', '',
        '|索引|UID|原图共同支持像素/seed|裁剪共同支持像素/seed|原图内部误差比例/seed|',
        '|---|---|---|---|---|']
    for i in range(4,12):
        obs=read(folder(i)/'observations.json')
        raw=[read(folder(i)/f'metrics/{s}.json')['metrics']['0.02'] for s in ('42','43')]
        cr=[read(crop_root/'objects'/f'{i:03d}'/f'metrics/{s}.json')['metrics']['0.02'] for s in ('42','43')]
        lines.append(f"|{i}|{obs['uid']}|{[r['common_pixels'] for r in raw]}|{[r['common_pixels'] for r in cr]}|{[r['interior_error_rate'] for r in raw]}|")
    lines+=['', '## 对照与成本', '',
        '正确GT几何的可见深度错误为零。它仍可能产生非零RGB/DINO差异，因此外观残差不等于几何真值。',
        '人工局部形变结果和旧HSSD开发对象独立保存在summary.json，不混入上述自然结果。',
        f"4视图输入分数计算平均 {timings['score_mean_seconds']:.2f} 秒，中位数 {timings['score_median_seconds']:.2f} 秒；", 
        f"输入轮廓全局拟合平均 {timings['fit_mean_seconds']:.2f} 秒。计时使用已缓存的原图DINO特征，包含渲染图编码、残差和分数缓存写入，", 
        '不包含初次模型加载、GT评分和图像/三维文件导出。', '',
        '三维热图除拟合后mesh坐标外，另导出`*_latent_grid.npz`：先逆转全局拟合变换，',
        '再对应原始SS/SLat网格，距离不超过2格才附着到已有稀疏token；无支持token为NaN/unknown。',
        '这只是定位证据的坐标适配，不是已经完成latent修复，也没有预测缺失几何的唯一深度。', '',
        '## 代码与复现', '',
        '首版缓冲的世界XYZ插值在小面片出现亚像素回投偏差；保留原源码和失败，使用独立反投影补丁。',
        '[补丁绑定](outputs/visible_error_localization_20261026/buffer_amendment.json)记录全部更改。', '',
        '```bash',
        '/home/zjr/anaconda3/envs/reconviagen/bin/python -m visual_evidence_probe.analysis_scripts.localization_corrected worker --gpu 0 --indices 0',
        '/home/zjr/anaconda3/envs/reconviagen/bin/python -m visual_evidence_probe.analysis_scripts.localization_corrected worker --gpu 2 --indices 2,4,6,8,10',
        '/home/zjr/anaconda3/envs/reconviagen/bin/python -m visual_evidence_probe.analysis_scripts.localization_corrected worker --gpu 3 --indices 1,3,5,7,9,11',
        '/home/zjr/anaconda3/envs/reconviagen/bin/python -m visual_evidence_probe.analysis_scripts.localization_corrected report', '```', '',
        '裁剪分辨率控制另运行`python -m visual_evidence_probe.analysis_scripts.localization_crop_control --gpu 0`。', '',
        '[完整数值](outputs/visible_error_localization_20261026/summary.json)、',
        '[复核回执](outputs/visible_error_localization_20261026/validation_receipt.json)、',
        '[协议](experiments/localization/PROTOCOL.md)。', '', '## 图像与三维热图', '']
    for i in range(12):
        obs=read(folder(i)/'observations.json');base=f'outputs/visible_error_localization_20261026/objects/{i:03d}'
        lines.append(f"- {i:03d} {obs['uid']}：[seed42放大图]({base}/figures/42_zoom.jpg)、[seed43放大图]({base}/figures/43_zoom.jpg)、[三维热图/体素]({base}/heat_meshes/)、[输出mesh]({base}/fitted/)。")
        if i>=4:
            lines.append(f"  裁剪控制：[seed42图](outputs/visible_error_localization_20261026/input_crop_resolution_control/objects/{i:03d}/figures/42.jpg)、[seed43图](outputs/visible_error_localization_20261026/input_crop_resolution_control/objects/{i:03d}/figures/43.jpg)。")
    reports=Path(os.environ.get('VISUAL_EVIDENCE_REPORT_ROOT',str(ROOT)))
    (reports/'第三次验证结论.md').write_text('\n'.join(lines)+'\n')
    generated=OUT/'REPORT.md'
    text=generated.read_text()
    start=text.find('```bash');end=text.find('```',start+7)
    if start>=0 and end>=0:
        text=text[:start]+'实际运行使用独立反投影补丁；复现/恢复命令见[第三次验证结论](../../第三次验证结论.md)。'+text[end+3:]
        generated.write_text(text)
    write(OUT/'status.json',dict(time=now(),stage='complete_reviewed',natural_records=24,control_records=16))
    print(receipt,flush=True)


if __name__=='__main__':main()
