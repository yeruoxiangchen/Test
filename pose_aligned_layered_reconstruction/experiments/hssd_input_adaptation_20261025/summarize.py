"""Paired per-object report, native-frame galleries and fixed-placement scenes."""
import numpy as np
from PIL import Image,ImageDraw
from .common import *

METRICS=['chamfer_l1','fscore_0p02','ss_iou','visible_recall','occluded_recall','free_space_violation']

def metric_values(mesh,ss):
    part=mesh['partitioned']
    return {'chamfer_l1':mesh['surface']['chamfer_l1'],'fscore_0p02':mesh['surface']['fscore_0p02'],
        'ss_iou':ss['occupancy']['iou'],'visible_recall':part['visible']['recall_0p02'],
        'occluded_recall':part['externally_occluded']['recall_0p02'],
        'free_space_violation':part['predicted_surface_free_space_violation_fraction']}

def averages(rows):
    return {k:float(np.mean([r[k] for r in rows if r[k] is not None]))
            if any(r[k] is not None for r in rows) else None for k in METRICS}

def paired(rows,arm,before,after,uid=None):
    left={(r['uid'],r['seed']):r['metrics'] for r in rows if r['arm']==arm and r['variant']==before and (uid is None or r['uid']==uid)}
    right={(r['uid'],r['seed']):r['metrics'] for r in rows if r['arm']==arm and r['variant']==after and (uid is None or r['uid']==uid)}
    if left.keys()!=right.keys() or not left:raise ValueError('Unpaired comparison')
    delta={k:[right[i][k]-left[i][k] for i in left if right[i][k] is not None and left[i][k] is not None] for k in METRICS}
    return {'objects':len({key[0] for key in left}),'seed_pairs':len(left),
        'mean_after_minus_before':{k:float(np.mean(v)) if v else None for k,v in delta.items()},
        'cd_wins':sum(x<-1.e-12 for x in delta['chamfer_l1']),
        'fscore_wins':sum(x>1.e-12 for x in delta['fscore_0p02']),
        'exact_input_aliases':sum(r['reused_as_alias'] for r in rows if r['arm']==arm and r['variant']==after and (uid is None or r['uid']==uid))}

def figures(rows):
    from ..round2_v2.render_mesh_audit import render_file
    records={};p=plan()['plan'];clipping=[]
    def archived(prefix):
        record=read(SHARED/'visuals/records'/f'{prefix}.json')
        verify(record['mesh'])
        if record['matrix_original_to_rendered']!=np.eye(4).tolist():raise ValueError('Unexpected archived render transform')
        for v in record['views']:verify({'path':v['png'],'sha256':v['png_sha256']})
        return record
    for obj in p['objects']:
        uid=obj['uid'];records[uid,'gt',None,None]=archived(uid+'_gt')
        for row in rows:
            if row['uid']!=uid:continue
            v,a,s=row['variant'],row['arm'],row['seed']
            event('render_mesh_gallery',uid=uid,variant=v,arm=a,seed=s)
            if row['owner_variant']=='full_all':record=archived(f'{uid}_{a}_{s}')
            else:record=render_file(verify(row['mesh']['mesh']),f'{uid}_{v}_{a}_{s}',output=OUT/'visuals',
                expected_sha256=row['mesh']['mesh']['sha256'])
            records[uid,v,a,s]=record
            for view in record['views']:
                if view['foreground_touches_image_border'] or view['vertices_outside_xy'] or view['vertices_outside_depth']:
                    clipping.append({'uid':uid,'variant':v,'arm':a,'seed':s,'view':view['view']})
    out=OUT/'figures';out.mkdir(exist_ok=True);links=[]
    for obj in p['objects']:
        uid=obj['uid']
        for arm in ARMS:
            for seed in SEEDS:
                side,header=320,28
                sheet=Image.new('RGB',(3*side,header+5*(side+header)),'white');draw=ImageDraw.Draw(sheet)
                draw.text((8,6),f'{uid} / {arm} / seed {seed} / fixed native cameras, no fitting',fill='black')
                for row,variant in enumerate(['gt',*VARIANTS]):
                    record=records[uid,'gt',None,None] if variant=='gt' else records[uid,variant,arm,seed]
                    label=variant
                    if variant=='full_all':label+=' (archived)'
                    elif variant=='full_visible' and owner_variant(obj,variant)=='full_all':label+=' (identical input alias)'
                    for col,view in enumerate(record['views']):
                        x,y=col*side,header+row*(side+header)
                        draw.text((x+4,y+6),f'{label} / view {col}',fill='black')
                        with Image.open(view['png']) as image:sheet.paste(image.resize((side,side)),(x,y+header))
                dest=out/f'{uid}_{arm}_{seed}.jpg';sheet.save(dest,quality=95);links.append(binding(dest))
    write(OUT/'visual_audit.json',{'figures':links,'clipping':clipping,'review_required':True,
        'camera_policy':'three fixed native orthographic cameras, unchanged material/scale, no fitted transform'})
    return links

def main():
    import trimesh
    p=plan()['plan'];ident=freeze_runtime()
    smoke=read(OUT/'smoke.json')
    if not smoke['passed'] or smoke['identity']!=ident:raise ValueError('Full baseline replay/smoke required')
    if not read(OUT/'hires_audit.json')['passed']:raise ValueError('High-resolution audit required')
    references={(r['arm'],r['uid'],r['seed']):r for r in p['reference_records']}
    old_ident=digest(read(verify(p['source_evaluation_identity'])));rows=[]
    for variant,arm,obj,seed in cases():
        owner=owner_variant(obj,variant)
        mp,mesh=get_result(variant,arm,obj,seed);sp,ss=get_result(variant,arm,obj,seed,'ss')
        expected=old_ident if owner=='full_all' else ident
        if mesh['identity']!=expected or ss['identity']!=expected:raise ValueError('Mixed result identity')
        if not mesh['success'] or not ss['success'] or mesh['fitted_transform']:raise ValueError('Incomplete/unfair result')
        if owner=='full_all':
            ref=references[arm,obj['uid'],seed];verify(ref['ss']);verify(ref['mesh'])
        for b in [mesh['mesh'],mesh['world_mesh'],mesh['gt'],mesh['ss'],ss['coords']]:verify(b)
        rows.append({'variant':variant,'owner_variant':owner,'arm':arm,'uid':obj['uid'],'seed':seed,
            'reused_from_original':owner=='full_all','reused_as_alias':owner!=variant,
            'mesh_record':binding(mp),'ss_record':binding(sp),'mesh':mesh,'metrics':metric_values(mesh,ss)})
    object_means={};means={};contrasts={}
    for arm in ARMS:
        object_means[arm]={v:{o['uid']:averages([r['metrics'] for r in rows if (r['arm'],r['variant'],r['uid'])==(arm,v,o['uid'])])
            for o in p['objects']} for v in VARIANTS}
        means[arm]={v:averages(list(object_means[arm][v].values())) for v in VARIANTS}
        contrasts[arm]={f'{a}_to_{b}':{'macro':paired(rows,arm,a,b),
            'per_object':{o['uid']:paired(rows,arm,a,b,o['uid']) for o in p['objects']}} for a,b in p['comparisons']}
    world=OUT/'world';world.mkdir(exist_ok=True);worlds=[]
    for variant in VARIANTS:
        for arm in ARMS:
            for seed in SEEDS:
                scene=trimesh.Scene()
                for r in rows:
                    if (r['variant'],r['arm'],r['seed'])==(variant,arm,seed):
                        scene.add_geometry(trimesh.load(verify(r['mesh']['world_mesh']),process=False),node_name=r['uid'])
                dest=world/f'{variant}_{arm}_{seed}.glb';scene.export(dest);worlds.append(binding(dest))
    links=figures(rows)
    result={'time':now(),'identity':ident,'logical_cases':len(rows),'new_samples':54,'original_references':24,'aliases':18,
        'objects':4,'scenes':1,'means':means,'per_object':object_means,'contrasts':contrasts,'records':rows,
        'figures':links,'worlds':worlds,'scope':p['scope'],'review_required':True}
    write(OUT/'summary.json',result)
    lines=['# SharedScene4 输入适配对照结果','','54 个新增完整采样已完成。96 个逻辑比较位置包含旧引用 24 个、相同输入别名 18 个；独立物体仍为 4 个、场景 1 个。',
        '',p['scope'],'','## 各模型内的输入比较','','| 模型 | 输入 | CD ↓ | F@0.02 ↑ | SS IoU ↑ | 可见 recall ↑ | 外遮挡 recall ↑ |',
        '|---|---|---:|---:|---:|---:|---:|']
    fmt=lambda x:'null' if x is None else f'{x:.6f}'
    for arm in ARMS:
        for v in VARIANTS:
            m=means[arm][v];lines.append(f'| {arm} | {v} | '+ ' | '.join(fmt(m[k]) for k in METRICS[:5])+' |')
    lines+=['','## 成对差值','','下表 CD/F 为后者减前者。CD 负值、F 正值表示改善；不是跨模型的容量匹配消融。',
        '','| 模型 | 对照 | ΔCD ↓ | ΔF（百分点）↑ | CD/F 胜数（12 个种子对） |','|---|---|---:|---:|---:|']
    for arm in ARMS:
        for name,item in contrasts[arm].items():
            r=item['macro'];d=r['mean_after_minus_before']
            lines.append(f'| {arm} | {name} | {d["chamfer_l1"]:.7f} | {100*d["fscore_0p02"]:+.3f} | {r["cd_wins"]}/{r["fscore_wins"]} |')
    lines+=['','## 空帧处理的实际受影响子集','','只有 stool（i065）从四帧变为两帧。其余三个物体与旧输入完全相同，不能称为额外重复。','']
    stool=next(o['uid'] for o in p['objects'] if len(o['visible_indices'])<4)
    for arm in ARMS:
        r=contrasts[arm]['full_all_to_full_visible']['per_object'][stool];d=r['mean_after_minus_before']
        lines.append(f'- {arm}，1 个物体 × 3 种子：ΔCD={d["chamfer_l1"]:.7f}，ΔF={100*d["fscore_0p02"]:+.3f} 个百分点。')
    lines+=['','## 验收','','原坐标评估，无 ICP/旋转/平移/缩放拟合；GT 表面分区固定为旧四帧。2072 重渲染还改变抗锯齿和采样噪声；不能把所有差异归因于新增细节。',
        '','逐物体结果与完整哈希见 [summary.json](summary.json)。请检查 [24 张三视角对比图](figures/)，关注重复椅子、错误支撑、额外平面和 bench/stool；不能只看平均分。',
        '','[低分辨率裁剪输入](crop518_input_sheet.jpg)；[高分辨率裁剪输入](crop2072_input_sheet.jpg)；[固定已知位置的场景 GLB](world/)。场景位置来自已有 canonical frame，不是模型估计。',
        '','本结果需人工审阅后决定是否改动模型。它不是 LaRI 有效性证据，也不是未见场景的泛化结论。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    event('summary_complete',new_meshes=54,logical_cases=96,figures=len(links))

if __name__=='__main__':main()
