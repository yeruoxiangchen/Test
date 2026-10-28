"""Matched depth x confidence factorial, with the appearance-only background."""
import argparse
import json
import hashlib
import math
from itertools import product
from pathlib import Path
from statistics import fmean

from .common import (ARMS, ALL_ARMS, STEPS, OUT, VARIANT, CALLS,
    binding, verify, read, write, plan, logical_cases, get_result,
    freeze_runtime, check_smoke, event, now, train_dir, physical)

APPEARANCE='appearance_control'
PRED_PRED='geometry_observed'
GT_PRED='geometry_gt_depth_pred_confidence'
PRED_UNIT='geometry_pred_depth_unit_confidence'
GT_UNIT='geometry_oracle_visible'
METHODS = (APPEARANCE,PRED_PRED,GT_PRED,PRED_UNIT,GT_UNIT)
FACTORIAL_METHODS=(PRED_PRED,GT_PRED,PRED_UNIT,GT_UNIT)
SEEDS = (42,43,44)
METRICS = ('chamfer_l1','fscore_0p01','fscore_0p02','normal_consistency',
    'ss_iou','ss_precision','ss_recall','visible_recall','occluded_recall','free_space_violation')
DIRECTIONS = {m:'lower' if m in ('chamfer_l1','free_space_violation') else 'higher' for m in METRICS}
CONTRASTS = {
    'depth_at_predicted_confidence':(GT_PRED,PRED_PRED),
    'depth_at_unit_confidence':(GT_UNIT,PRED_UNIT),
    'confidence_at_predicted_depth':(PRED_UNIT,PRED_PRED),
    'confidence_at_gt_depth':(GT_UNIT,GT_PRED),
    'joint_gt_unit_minus_predicted':(GT_UNIT,PRED_PRED),
    **{arm+'_minus_appearance':(arm,APPEARANCE) for arm in FACTORIAL_METHODS},
}
INTERACTION='depth_confidence_interaction'
ALL_CONTRASTS=(*CONTRASTS,INTERACTION)
PARTITIONS = ('visible','externally_occluded','other_unobserved')
EXPECTED_PARTITIONS = {
    'hssd_102344049_i076':dict(zip(PARTITIONS,(9225,9349,1426))),
    'hssd_102344049_i075':dict(zip(PARTITIONS,(5247,9477,5276))),
    'hssd_102344049_i080':dict(zip(PARTITIONS,(862,19138,0))),
    'hssd_102344049_i065':dict(zip(PARTITIONS,(5691,14309,0))),
}
SCOPE = ('四个旧训练对象、同一HSSD共享场景的深度×置信度2×2机制诊断。新增GT深度+预测置信度、'
    '预测深度+有效置信度1两支各200步训练；复用已完成的预测深度/置信度与GT可见深度/置信度1两格，'
    '另复列匹配外观背景。固定crop2072_visible照片、初始化、预算和配对噪声。GT仅为全场景可见首交点，'
    '结合目标可见mask；不读取隐藏目标表面。每个深度分支只使用自身有效性，单位置信度不等于正确性保证。'
    'GT深度组是特权诊断，不是可部署RGB成绩。三个seed为对象内采样重复，不是独立训练或场景。'
    '四对象机制定位不能证明泛化、原LaRI贡献或完整过拟合；相机/mask/canonical frame仍有GT辅助。')


def metric_values(mesh,ss):
    p=mesh['partitioned'];s=mesh['surface'];o=ss['occupancy']
    values={'chamfer_l1':s['chamfer_l1'],'fscore_0p01':s['fscore_0p01'],'fscore_0p02':s['fscore_0p02'],
        'normal_consistency':s['normal_consistency'],'ss_iou':o['iou'],'ss_precision':o['precision'],
        'ss_recall':o['recall'],'visible_recall':p['visible']['recall_0p02'],
        'occluded_recall':p['externally_occluded']['recall_0p02'],
        'free_space_violation':p['predicted_surface_free_space_violation_fraction']}
    for k,v in values.items():
        if v is None and k in ('visible_recall','occluded_recall'):continue
        if v is None or not math.isfinite(v):raise ValueError('Missing/nonfinite metric: '+k)
    return values


def average(rows):
    rows=list(rows)
    if not rows:raise ValueError('Empty statistic')
    return {m:None if any(r[m] is None for r in rows) else fmean(r[m] for r in rows) for m in METRICS}


def delta(a,b):
    return {m:None if a[m] is None or b[m] is None else a[m]-b[m] for m in METRICS}


def contrasts(cells):
    result={name:delta(cells[a],cells[b]) for name,(a,b) in CONTRASTS.items()}
    result[INTERACTION]=delta(result['depth_at_unit_confidence'],result['depth_at_predicted_confidence'])
    return result


def signs(rows):
    rows=list(rows)
    return {m:None if any(row[m] is None for row in rows) else {
        'positive':sum(row[m]>1e-12 for row in rows),
        'zero':sum(abs(row[m])<=1e-12 for row in rows),
        'negative':sum(row[m]<-1e-12 for row in rows),'pairs':len(rows)} for m in METRICS}


def ranges(rows):
    rows=list(rows)
    return {m:None if any(r[m] is None for r in rows) else [min(r[m] for r in rows),max(r[m] for r in rows)] for m in METRICS}


def wins(rows):
    rows=list(rows);output={}
    for m in METRICS:
        if any(r[m] is None for r in rows):output[m]=None;continue
        values=[r[m]*(1 if DIRECTIONS[m]=='higher' else -1) for r in rows]
        output[m]={'better':sum(x>1e-12 for x in values),'equal':sum(abs(x)<=1e-12 for x in values),
            'worse':sum(x< -1e-12 for x in values),'pairs':len(values)}
    return output


def summarize_statistics(rows,object_ids=None):
    rows=list(rows);uids=list(object_ids) if object_ids is not None else sorted({r['uid'] for r in rows})
    if len(uids)!=4 or len(set(uids))!=4:raise ValueError('Require four distinct objects')
    indexed={}
    for r in rows:
        key=r['uid'],r['seed'],r['arm']
        if key in indexed:raise ValueError('Duplicate result')
        indexed[key]=r
    if set(indexed)!=set(product(uids,SEEDS,METHODS)):
        raise ValueError('Require four objects x three seeds x five methods (48 factorial + 12 background)')
    per_seed={};per_object={}
    for uid in uids:
        per_seed[uid]={}
        for seed in SEEDS:
            cells={arm:indexed[uid,seed,arm]['metrics'] for arm in METHODS}
            per_seed[uid][str(seed)]={'cells':cells,'contrasts':contrasts(cells)}
        paired=list(per_seed[uid].values())
        per_object[uid]={'sampling_seeds':list(SEEDS),
            'cells':{arm:average(r['cells'][arm] for r in paired) for arm in METHODS},
            'contrasts':{name:average(r['contrasts'][name] for r in paired) for name in ALL_CONTRASTS},
            'delta_seed_range':{name:ranges(r['contrasts'][name] for r in paired) for name in ALL_CONTRASTS},
            'seed_win_counts':{name:wins(r['contrasts'][name] for r in paired) for name in CONTRASTS},
            'interaction_seed_sign_counts':signs(r['contrasts'][INTERACTION] for r in paired)}
    macro={'objects':4,'scenes':1,'sampling_seeds_per_object':3,
        'weighting':'equal mean of four object means; paired sampling seeds within each object',
        'cells':{arm:average(r['cells'][arm] for r in per_object.values()) for arm in METHODS},
        'contrasts':{name:average(r['contrasts'][name] for r in per_object.values()) for name in ALL_CONTRASTS},
        'object_win_counts':{name:wins(r['contrasts'][name] for r in per_object.values()) for name in CONTRASTS},
        'interaction_object_sign_counts':signs(r['contrasts'][INTERACTION] for r in per_object.values())}
    return {'per_seed':per_seed,'per_object':per_object,'macro':macro,'metric_directions':DIRECTIONS,
        'pair_win_counts':{name:wins(per_seed[uid][str(seed)]['contrasts'][name] for uid in uids for seed in SEEDS)
                           for name in CONTRASTS},
        'factorial_cells':{'predicted_depth_predicted_confidence':PRED_PRED,'gt_depth_predicted_confidence':GT_PRED,
            'predicted_depth_unit_confidence':PRED_UNIT,'gt_depth_unit_confidence':GT_UNIT},
        'factorial_cases':48,'appearance_background_cases':12,
        'contrast_convention':'raw later-minus-reference difference as named; CD/free-space lower is better',
        'interaction_convention':'(GT depth/unit conf - predicted depth/unit conf) - (GT depth/pred conf - predicted depth/pred conf); raw difference of differences, not a method win',
        'range_convention':'observed three-seed min/max, not a confidence interval',
        'null_policy':'any missing stratum propagates null; no silent denominator change'}


def self_check_statistics():
    rows=[]
    for i,uid in enumerate(('a','b','c','d')):
        for seed in SEEDS:
            for arm in METHODS:
                depth=int(arm in (GT_PRED,GT_UNIT));unit=int(arm in (PRED_UNIT,GT_UNIT))
                value=i+seed/10+depth*(2*i-1)+2*unit+.5*depth*unit
                values={m:float(value) for m in METRICS}
                rows.append({'uid':uid,'seed':seed,'arm':arm,'metrics':values})
    r=summarize_statistics(rows)
    if not math.isclose(r['macro']['contrasts']['depth_at_predicted_confidence']['chamfer_l1'],2.,rel_tol=0.,abs_tol=1e-12):
        raise AssertionError('Wrong paired object weighting')
    counts=r['macro']['object_win_counts']['depth_at_predicted_confidence']
    if counts['chamfer_l1']['better']!=1 or counts['fscore_0p02']['better']!=3:
        raise AssertionError('Wrong metric directions')
    if not math.isclose(r['macro']['contrasts'][INTERACTION]['chamfer_l1'],.5,abs_tol=1e-12):
        raise AssertionError('Wrong factorial difference-of-differences')
    if r['macro']['interaction_object_sign_counts']['chamfer_l1']['positive']!=4:
        raise AssertionError('Interaction signs must not be labeled quality wins')
    for bad in (rows[:-1],rows+[rows[0]]):
        try:summarize_statistics(bad)
        except ValueError:pass
        else:raise AssertionError('Accepted missing/duplicate results')
    null=[dict(r,metrics=dict(r['metrics'])) for r in rows];null[2]['metrics']['visible_recall']=None
    if summarize_statistics(null)['macro']['contrasts'][INTERACTION]['visible_recall'] is not None:
        raise AssertionError('Unavailable stratum changed denominator')
    return {'passed':True,'pairing_and_object_weighting':True,'missing_duplicate_rejected':True,
            'factorial_interaction':True,'interaction_sign_not_quality_win':True,'null_policy':True}


def compare_training_records(runs, histories):
    """Compare only intentionally shared numerical/input fields across protocols."""
    if set(runs) != set(METHODS) or set(histories) != set(METHODS):
        raise ValueError('Require all five matched training arms')
    paired_fields = ('uid','index','original_index','epoch','offset','diffusion_time',
        'noise_sha','learning_rate','object_group_exposures')
    policy_fields = ('steps','seed','lr','warmup','betas','weight_decay','batch','p_uncond',
        'weights','amp','residual_channels','projection_bias','parent_checkpoint',
        'observation_variant','flow_loss','view_selection')
    base = runs[METHODS[0]]
    for arm in METHODS:
        run = runs[arm]
        if [r['step'] for r in histories[arm]] != list(range(1,STEPS+1)):
            raise ValueError('Incomplete/duplicate training history: '+arm)
        for key in ('initial_residual_sha','trainable_parameter_count','trainable_parameter_names','bindings'):
            if run[key] != base[key]:
                raise ValueError('Unequal matched training field: '+key)
        for key in policy_fields:
            if run['contract'][key] != base['contract'][key]:
                raise ValueError('Unequal training policy: '+key)
        if run['contract']['steps'] != STEPS or run['trainable_parameter_count'] != 3164160:
            raise ValueError('Training budget or adapter parameter count changed')
        # The evidence binding necessarily differs; source RGB/masks and SS
        # latent target must not. Keep the original logical bindings intact.
        keep = ('uid','original_index','inputs','target_source','target_tensor_sha')
        data = [{k:r[k] for k in keep} for r in run['data_identity']]
        reference = [{k:r[k] for k in keep} for r in base['data_identity']]
        if data != reference or len(data) != 4:
            raise ValueError('Training image/latent target identity differs')
        for row, reference_row in zip(histories[arm], histories[METHODS[0]]):
            for key in paired_fields:
                if row[key] != reference_row[key]:
                    raise ValueError('Training schedule/noise mismatch: '+key)
    return {'passed':True,'steps_per_arm':STEPS,'same_initial_adapter':True,
        'same_parameter_count':True,'trainable_parameter_count':base['trainable_parameter_count'],
        'same_frozen_parent':True,'same_inputs_and_latent_targets':True,
        'same_training_order_noise_and_lr':True,'paired_history_fields':list(paired_fields),
        'shared_policy_fields':list(policy_fields),
        'intentional_difference':'two crossed visible-depth/confidence evidence factors; protocol identity/arm differ'}


def training_fairness(p):
    from .evaluate import training_binding
    runs, histories, sources = {}, {}, {}
    for arm in METHODS:
        if arm in ARMS:
            folder = train_dir(arm)
            source = {'run':binding(folder/'run.json'), 'complete':binding(folder/'complete.json'),
                'history':binding(folder/'history.jsonl'), 'checkpoint':binding(folder/'last.pt'),
                'training':training_binding(arm)}
        else:
            source = p['reference_training'][arm]
        sources[arm] = source
        runs[arm] = read(verify(source['run']))
        complete = read(verify(source['complete']))
        if (complete['step'] != STEPS or complete['contract'] != runs[arm]['contract']
                or complete['data_identity'] != runs[arm]['data_identity']
                or complete['checkpoint'] != source['checkpoint']):
            raise ValueError('Training completion binding differs: '+arm)
        verify(source['checkpoint'])
        histories[arm] = [json.loads(line) for line in verify(source['history']).read_text().splitlines()]
    result = compare_training_records(runs,histories)
    result['sources'] = sources
    return result


def diagonal_replay():
    from .evaluate import verify_nested
    path=OUT/'evidence/diagonal_check.json';row=read(path)
    required={(uid,arm) for uid in EXPECTED_PARTITIONS for arm in (PRED_PRED,GT_UNIT)}
    if (row.get('schema')!='depth_confidence_diagonal_exact_v1' or row.get('passed') is not True
            or row.get('diagonal_evidence_replay_exact') is not True
            or row.get('plan_identity')!=plan()['identity'] or len(row.get('checks',[]))!=8
            or {(check['uid'],check['arm']) for check in row['checks']}!=required
            or any(check.get('all_tensors_exact') is not True for check in row['checks'])):
        raise ValueError('Two old factorial diagonals were not reproduced exactly')
    verify_nested(row)
    return {'passed':True,'comparisons':8,'record':binding(path)}


def collect_rows(p,identity):
    from .evaluate import validate_ss_record, validate_mesh_record
    cases=logical_cases();objects={o['uid']:o for o in p['objects']}
    if cases!=p['cases'] or set(objects)!=set(EXPECTED_PARTITIONS):raise ValueError('Case/object set changed')
    rows=[];conditions={}
    for case in cases:
        uid,seed,arm=case['uid'],case['seed'],case['arm'];obj=objects[uid]
        if arm not in METHODS or case['is_reference']!=(arm not in ARMS):raise ValueError('Invalid method/reference label')
        mp,mesh,sp,ss=get_result(case)
        condition=read(verify(case['condition_record']))
        if (not condition['success'] or condition['object_uid']!=uid or condition['seed']!=seed
                or condition['variant']!=VARIANT or condition['identity']!=p['base_identity']):
            raise ValueError('Condition source differs')
        if conditions.setdefault((uid,seed),condition['inputs'])!=condition['inputs']:
            raise ValueError('Methods use different image conditions')
        if mesh['inputs']!=condition['inputs'] or ss['inputs']!=condition['inputs']:
            raise ValueError('Actual images differ between SS/SLat and paired reference')
        if not mesh['success'] or not ss['success'] or mesh['object_uid']!=uid or mesh['seed']!=seed:
            raise ValueError('Incomplete/wrong mesh')
        if case['is_reference']:
            if (binding(mp)!=case['reference_mesh'] or binding(sp)!=case['reference_ss']
                    or mesh['identity']!=case['expected_identity'] or ss['identity']!=case['expected_identity']
                    or mesh['arm']!=arm or ss['arm']!=arm):
                raise ValueError('Archived factorial/background reference changed')
        else:
            validate_ss_record(ss,case,identity)
            validate_mesh_record(mesh,case,identity,ss,binding(sp))
        if mesh['ss']!=binding(sp) or mesh['slat_binding']!=p['slat_binding'] or mesh['calls']!=CALLS:
            raise ValueError('Wrong support, SLat model or incomplete sampling')
        if mesh['fitted_transform'] or mesh['canonical_from_world']!=obj['canonical_from_world'] or mesh['gt']!=obj['mesh']:
            raise ValueError('Changed target/native coordinate transform')
        counts={key:mesh['partitioned'][key]['gt_samples'] for key in PARTITIONS}
        if counts!=EXPECTED_PARTITIONS[uid]:raise ValueError('Fixed GT partition changed')
        for key in ['mesh','world_mesh','gt']:verify(mesh[key])
        rows.append({'uid':uid,'seed':seed,'arm':arm,'is_reference':case['is_reference'],
            'mesh_record':binding(mp),'ss_record':binding(sp),'condition_record':case['condition_record'],
            'mesh':mesh,'metrics':metric_values(mesh,ss),'support_voxels':ss['occupancy']['predicted_voxels'],
            'target_voxels':ss['occupancy']['target_voxels'],'support_count_ratio':ss['occupancy']['count_ratio']})
    if len(rows)!=60 or sum(r['is_reference'] for r in rows)!=36:raise ValueError('Require 24 new + 36 archived results')
    summarize_statistics(rows,list(objects))
    return rows,{'passed':True,'logical_cases':60,'factorial_cases':48,'appearance_background_cases':12,
        'new_meshes':24,'references':36,
        'same_images_and_frozen_slat':True,'all_objects_seeds_retained':True,'native_frame':True}


def _verify_render(record, expected_mesh):
    from ..round2_v2.render_mesh_audit import parameters
    if (physical(record['mesh']['path']).resolve() != physical(expected_mesh['path']).resolve()
            or record['mesh']['sha256'] != expected_mesh['sha256']):
        raise ValueError('Render source mesh differs')
    verify(record['mesh'])
    transform = [[float(i==j) for j in range(4)] for i in range(4)]
    if record['matrix_original_to_rendered'] != transform:
        raise ValueError('Unexpected per-result render transform')
    expected = {'mesh_sha256':expected_mesh['sha256'],'parameters':parameters(),'transform':transform}
    if record['binding_hash'] != hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest():
        raise ValueError('Render cameras/material/scale/versions differ')
    if [v['view'] for v in record['views']] != [0,1,2]:
        raise ValueError('Require the same three fixed render cameras')
    for view in record['views']:
        verify({'path':view['png'],'sha256':view['png_sha256']})


def figures(rows,p,identity):
    from PIL import Image,ImageDraw
    from ..round2_v2.render_mesh_audit import render_file
    archived = {};identities=set()
    for archive_binding in p['reference_visual_audits']:
        archive=read(verify(archive_binding));identities.add(archive['identity'])
        if archive['identity'] not in (p['previous_identity'],p['sharedss_identity']):
            raise ValueError('Archived visual audit identity changed')
        for source in archive['render_records']:
            if source['arm'] not in ('mesh_gt',APPEARANCE,PRED_PRED,GT_UNIT):continue
            key=source['uid'],source['arm'],source.get('seed')
            if key in archived and archived[key]!=source['record']:
                raise ValueError('Conflicting archived render record')
            archived[key]=source['record']
    if identities!={p['previous_identity'],p['sharedss_identity']}:
        raise ValueError('Require both completed SharedSS and visible-oracle visual audits')
    records,sources,clipping = {},[],[]
    for obj in p['objects']:
        uid = obj['uid']; source = archived[uid,'mesh_gt',None]
        path = verify(source); record = read(path)
        _verify_render(record,obj['mesh']); records[uid,'mesh_gt',None] = record
        sources.append({'uid':uid,'arm':'mesh_gt','reused':True,'record':source})
    for row in rows:
        uid,seed,arm = row['uid'],row['seed'],row['arm']
        if row['is_reference']:
            source = archived[uid,arm,seed]
            path = verify(source); record = read(path)
        else:
            prefix = f'{uid}_{arm}_{seed}'; event('render_gallery',uid=uid,seed=seed,arm=arm)
            record = render_file(verify(row['mesh']['mesh']),prefix,output=OUT/'visuals',
                expected_sha256=row['mesh']['mesh']['sha256'],
                identity={'evaluation_identity':identity,'uid':uid,'seed':seed,'arm':arm})
            path = OUT/'visuals/records'/f'{prefix}.json'; source = binding(path)
        _verify_render(record,row['mesh']['mesh']); records[uid,arm,seed] = record
        sources.append({'uid':uid,'seed':seed,'arm':arm,'reused':row['is_reference'],'record':source})
    for (uid,arm,seed),record in records.items():
        for view in record['views']:
            if view['foreground_touches_image_border'] or view['vertices_outside_xy'] or view['vertices_outside_depth']:
                clipping.append({'uid':uid,'seed':seed,'arm':arm,'view':view['view']})
    destination = OUT/'figures'; destination.mkdir(parents=True,exist_ok=True); links=[]
    labels={'mesh_gt':'GT mesh','appearance_control':'Appearance control (archived)',
        PRED_PRED:'Predicted depth / predicted confidence (old)',
        GT_PRED:'GT visible depth / predicted confidence',
        PRED_UNIT:'Predicted depth / valid confidence=1',
        GT_UNIT:'GT visible depth / valid confidence=1 (old)'}
    for obj in p['objects']:
        uid = obj['uid']
        for seed in SEEDS:
            side,header = 320,32
            sheet = Image.new('RGB',(3*side,header+6*(side+header)),'white'); draw=ImageDraw.Draw(sheet)
            draw.text((8,8),f'{uid} / seed {seed} / same images / native frame / no fitting',fill='black')
            for row_index,arm in enumerate(('mesh_gt',*METHODS)):
                record = records[uid,arm,None if arm=='mesh_gt' else seed]
                for column,view in enumerate(record['views']):
                    x,y = column*side,header+row_index*(side+header)
                    draw.text((x+4,y+8),f"{labels[arm]} / view {view['view']}",fill='black')
                    image_path = verify({'path':view['png'],'sha256':view['png_sha256']})
                    with Image.open(image_path) as im:
                        sheet.paste(im.convert('RGB').resize((side,side),Image.Resampling.LANCZOS),(x,y+header))
            path = destination/f'{uid}_seed{seed}_depth_confidence.jpg'; sheet.save(path,quality=95)
            links.append({'uid':uid,'seed':seed,**binding(path)})
    return {'identity':identity,'figures':links,'render_records':sources,'clipping':clipping,
        'archived_gt_render_count':4,'archived_method_render_count':36,'new_mesh_render_count':24,
        'review_required':True,'human_review_completed':False,
        'camera_policy':'same original three cameras/material/scale; no per-result fitting'}


def export_worlds(rows,identity):
    import trimesh
    dest=OUT/'world';dest.mkdir(parents=True,exist_ok=True);worlds=[]
    for arm,seed in product(METHODS,SEEDS):
        selected=[r for r in rows if r['arm']==arm and r['seed']==seed]
        if len(selected)!=4:raise ValueError('Incomplete four-object world')
        scene=trimesh.Scene();sources=[]
        for r in selected:
            mesh=trimesh.load(verify(r['mesh']['world_mesh']),process=False,force='mesh')
            scene.add_geometry(mesh,node_name=r['uid'],geom_name=r['uid'])
            sources.append({'uid':r['uid'],'world_mesh':r['mesh']['world_mesh'],'canonical_from_world':r['mesh']['canonical_from_world']})
        scene.metadata.update({'identity':identity,'arm':arm,'seed':seed,'layout':'archived known placement, not prediction'})
        path=dest/f'{arm}_seed{seed}_known_placements.glb';scene.export(path)
        worlds.append({'arm':arm,'seed':seed,'objects':4,**binding(path),'sources':sources})
    return worlds


def report_text(result):
    fmt=lambda x:'null' if x is None else f'{x:.7f}'
    lines=['# 可见深度 × 置信度：完整2×2诊断','',SCOPE,'',result['plan_scope'],'',
        '24个新增SS/mesh + 36个旧结果引用 = 60个逻辑结果。主2×2为48个，匹配外观为12个背景结果；五臂各200步，三个seed仅是配对采样。',
        '', '## 四格及主要对照','',
        '| 深度 | 预测置信度 | 有效位置置信度1 |',
        '|---|---|---|',
        f'| 预测深度 | {PRED_PRED}（旧） | {PRED_UNIT}（新） |',
        f'| GT可见首表面深度 | {GT_PRED}（新） | {GT_UNIT}（旧） |',
        '', 'depth_at_predicted_confidence / depth_at_unit_confidence：固定置信度后GT深度减预测深度；confidence_at_predicted_depth / confidence_at_gt_depth：固定深度后单位置信度减预测置信度。',
        '', 'depth_confidence_interaction = (GT深度/单位置信度 − 预测深度/单位置信度) − (GT深度/预测置信度 − 预测深度/预测置信度)。这是差上差，不能当作某个模型的性能胜负。正负方向按原指标单位解释。',
        '', '## 完整逐对象/seed结果','',
        '| 对象 | seed | 方法 | 支持数 | CD↓ | F@.01↑ | F@.02↑ | normal↑ | SS IoU↑ | SS precision↑ | SS recall↑ | 可见R↑ | 遮挡R↑ | 自由空间违规↓ |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['records']:
        lines.append(f"| {r['uid']} | {r['seed']} | {r['arm']} | {r['support_voxels']} | "+' | '.join(fmt(r['metrics'][m]) for m in METRICS)+' |')
    lines+=['','## 对象均值与宏平均','',
        '| 对象/汇总 | 方法 | CD↓ | F@.01↑ | F@.02↑ | normal↑ | SS IoU↑ | SS precision↑ | SS recall↑ | 可见R↑ | 遮挡R↑ | 自由空间违规↓ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    aggregates=list(result['statistics']['per_object'].items())+[('四对象宏平均',result['statistics']['macro'])]
    for label,item in aggregates:
        for arm in METHODS:lines.append(f'| {label} | {arm} | '+' | '.join(fmt(item['cells'][arm][m]) for m in METRICS)+' |')
    lines+=['','## 配对差值','', '原始差值，F和recall为[0,1]单位；未乘100转为百分点。CD/违规负值、F/recall正值表示改善。', '',
        '| 对象/汇总 | 对照 | ΔCD | ΔF@.01 | ΔF@.02 | Δnormal | ΔSS IoU | ΔSS precision | ΔSS recall | Δ可见R | Δ遮挡R | Δ违规 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,item in aggregates:
        for name in ALL_CONTRASTS:lines.append(f'| {label} | {name} | '+' | '.join(fmt(item['contrasts'][name][m]) for m in METRICS)+' |')
    lines+=['','## 公平性与范围','',
        f"五组各{STEPS}步，同新adapter初始化与参数量（{result['training_fairness']['trainable_parameter_count']}），冻结父模型相同；每步对象、顺序、时间t、噪声hash及学习率配对核验通过。",
        '', '相同SLat权重、25/19调用、同图像条件和原GT分区。无逐结果旋转/平移/缩放/ICP；世界GLB使用已知摆放位置。保留全部对象和seed，包括长凳失败。',
        '', '既有两格的完整features/weights/valid在CPU逐值重放通过后才允许复用旧结果；新GT分支只用已剥离的全场景可见首交点depth/valid，预测分支的有效性不由GT筛选。单位置信度只取消预测置信度加权/门槛，不保证预测深度正确。无隐藏目标GT条件、额外损失或新latent编码。',
        '', '这是限定的四对象机制定位。若两种置信度下的深度效应不同，应报告交互，不能把联合oracle全部增益归于单一因素。不依据四对象结果宣称跨场景泛化或无限调阈值。',
        '', '[完整数据与来源](summary.json)，[渲染核验](visual_audit.json)。下列12图依次展示GT/外观/预测深度与置信度/GT深度与预测置信度/预测深度与单位置信度/GT深度与单位置信度，每项三个固定视角：','']
    for f in result['figures']:lines.append(f"- [{f['uid']} / seed {f['seed']}]({Path(f['path']).relative_to(OUT)})")
    lines+=['',f"[15个已知位置场景GLB](world/)；检测到{len(result['visual_clipping'])}条裁剪记录。自动完成不等于质量/视觉审阅通过。",'']
    return '\n'.join(lines)


def main():
    p=plan()['plan'];identity=freeze_runtime();smoke=check_smoke()
    if tuple(ALL_ARMS)!=METHODS:raise ValueError('Frozen method order differs from summary')
    diagonals=diagonal_replay();fair=training_fairness(p);rows,acceptance=collect_rows(p,identity)
    stats=summarize_statistics(rows,[o['uid'] for o in p['objects']])
    event('summary_inputs_verified',logical_cases=60,new_meshes=24)
    worlds=export_worlds(rows,identity);visuals=figures(rows,p,identity);visuals['worlds']=worlds
    write(OUT/'visual_audit.json',visuals)
    result={'time':now(),'identity':identity,'plan':binding(OUT/'plan.json'),'smoke':smoke,
        'logical_cases':60,'factorial_cases':48,'appearance_background_cases':12,
        'new_meshes':24,'new_ss_samples':24,'original_references':36,'objects':4,'scenes':1,
        'sampling_seeds':list(SEEDS),'training_steps_per_arm':STEPS,'training_fairness':fair,
        'diagonal_evidence_replay':diagonals,
        'statistics':stats,'acceptance':acceptance,'records':rows,'figures':visuals['figures'],'worlds':worlds,
        'visual_clipping':visuals['clipping'],'visual_audit':binding(OUT/'visual_audit.json'),
        'scope':SCOPE,'plan_scope':p['scope'],'review_required':True,'human_review_completed':False}
    if 'previous_summary' in p:
        verify(p['previous_summary']);result['previous_visible_oracle_reference']=p['previous_summary']
    write(OUT/'summary.json',result);(OUT/'REPORT.md').write_text(report_text(result),encoding='utf-8')
    event('summary_complete',logical_cases=60,new_meshes=24,figures=12,worlds=15)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:print(self_check_statistics())
    else:main()
