"""Independent aggregation and matched-training audit; CPU, read-only by default.

Uses original SS/mesh JSON, not the production aggregation functions. It checks
those reported scalar metrics, not a new mesh surface sampling run. All legacy
payload paths resolve through the frozen /data storage manifest.
"""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
from statistics import fmean
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_oracle_stat_common',HERE.parent/'common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
ARMS=('appearance_control','geometry_observed','geometry_oracle_visible')
SEEDS=(42,43,44)
CONTRASTS={'oracle_minus_predicted':(ARMS[2],ARMS[1]),
    'oracle_minus_appearance':(ARMS[2],ARMS[0]),'predicted_minus_appearance':(ARMS[1],ARMS[0])}
METRICS=('chamfer_l1','fscore_0p01','fscore_0p02','normal_consistency','ss_iou',
    'ss_precision','ss_recall','visible_recall','occluded_recall','free_space_violation')
LOWER={'chamfer_l1','free_space_violation'}


def metrics(mesh,ss):
    s=mesh['surface'];p=mesh['partitioned'];o=ss['occupancy']
    row={'chamfer_l1':s['chamfer_l1'],'fscore_0p01':s['fscore_0p01'],
        'fscore_0p02':s['fscore_0p02'],'normal_consistency':s['normal_consistency'],
        'ss_iou':o['iou'],'ss_precision':o['precision'],'ss_recall':o['recall'],
        'visible_recall':p['visible']['recall_0p02'],
        'occluded_recall':p['externally_occluded']['recall_0p02'],
        'free_space_violation':p['predicted_surface_free_space_violation_fraction']}
    if any(v is None or not math.isfinite(v) for v in row.values()):
        raise ValueError('This complete four-object cohort requires finite metric strata')
    return row


def average(rows):
    rows=list(rows)
    if not rows:raise ValueError('Empty denominator')
    return {key:fmean(r[key] for r in rows) for key in METRICS}


def difference(a,b):return {key:a[key]-b[key] for key in METRICS}


def wins(deltas):
    deltas=list(deltas);out={}
    for key in METRICS:
        values=[r[key]*(-1 if key in LOWER else 1) for r in deltas]
        out[key]={'better':sum(v>1e-12 for v in values),'equal':sum(abs(v)<=1e-12 for v in values),
            'worse':sum(v< -1e-12 for v in values),'pairs':len(values)}
    return out


def run():
    wrapper=c.plan();p=wrapper['plan'];summary=c.read(c.OUT/'summary.json')
    identity=c.digest(c.read(c.OUT/'evaluation_identity.json'))
    if summary['identity']!=identity:raise ValueError('Summary runtime mismatch')
    objects=[o['uid'] for o in p['objects']]
    if objects!=c.UIDS or len(p['cases'])!=36:raise ValueError('Wrong cohort')
    indexed={};records=[];sources=[];check_count=0;maximum_error=0.

    def compare(a,b,label):
        nonlocal check_count,maximum_error
        if isinstance(a,dict):
            if set(a)!=set(b):raise ValueError('Different keys: '+label)
            for k in a:compare(a[k],b[k],label+'/'+k)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            err=abs(a-b);maximum_error=max(maximum_error,err);check_count+=1
            if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Metric mismatch: '+label)
        else:
            check_count+=1
            if a!=b:raise ValueError('Value mismatch: '+label)

    for case in p['cases']:
        uid,seed,arm=case['uid'],case['seed'],case['arm']
        key=(uid,seed,arm)
        if key in indexed or arm not in ARMS or seed not in SEEDS:raise ValueError('Invalid/duplicate case')
        if case['is_reference']:
            mb,sb=case['reference_mesh'],case['reference_ss']
        else:
            directory=c.case_dir(case);mb=c.binding(directory/'mesh.json');sb=c.binding(directory/'ss.json')
        mesh=c.read(c.verify(mb));ss=c.read(c.verify(sb));sources.extend([mb,sb])
        if (not mesh['success'] or not ss['success'] or mesh['object_uid']!=uid or ss['object_uid']!=uid
                or mesh['seed']!=seed or ss['seed']!=seed or mesh['arm']!=arm or ss['arm']!=arm
                or mesh['ss']!=sb or mesh['inputs']!=ss['inputs']):raise ValueError('Source record mismatch')
        raw=metrics(mesh,ss)
        matches=[r for r in summary['records'] if (r['uid'],r['seed'],r['arm'])==key]
        if len(matches)!=1:raise ValueError('Summary omitted/duplicated case')
        saved=matches[0]
        if saved['mesh_record']!=mb or saved['ss_record']!=sb or saved['mesh']!=mesh:
            raise ValueError('Summary source/embedded raw mesh differs')
        compare(raw,saved['metrics'],'case/'+str(key))
        row={'uid':uid,'seed':seed,'arm':arm,'metrics':raw,'inputs':ss['inputs'],
            'support_voxels':ss['occupancy']['predicted_voxels'],
            'target_voxels':ss['occupancy']['target_voxels'],'support_count_ratio':ss['occupancy']['count_ratio'],
            'mesh_record':mb,'ss_record':sb}
        records.append(row);indexed[key]=row
    if len(indexed)!=36:raise ValueError('Incomplete denominator')

    per_object={};per_seed={}
    for uid in objects:
        per_seed[uid]={}
        for seed in SEEDS:
            cells={arm:indexed[uid,seed,arm]['metrics'] for arm in ARMS}
            if any(indexed[uid,seed,arm]['inputs']!=indexed[uid,seed,ARMS[0]]['inputs'] for arm in ARMS):
                raise ValueError('Unmatched observations')
            deltas={k:difference(cells[a],cells[b]) for k,(a,b) in CONTRASTS.items()}
            per_seed[uid][str(seed)]={'cells':cells,'contrasts':deltas}
            compare(per_seed[uid][str(seed)],summary['statistics']['per_seed'][uid][str(seed)],'per_seed')
        cells={arm:average(per_seed[uid][str(seed)]['cells'][arm] for seed in SEEDS) for arm in ARMS}
        deltas={k:difference(cells[a],cells[b]) for k,(a,b) in CONTRASTS.items()}
        wc={k:wins(per_seed[uid][str(seed)]['contrasts'][k] for seed in SEEDS) for k in CONTRASTS}
        per_object[uid]={'cells':cells,'contrasts':deltas,'seed_win_counts':wc}
        old=summary['statistics']['per_object'][uid]
        for field in per_object[uid]:compare(per_object[uid][field],old[field],'per_object/'+field)
    macro={arm:average(per_object[uid]['cells'][arm] for uid in objects) for arm in ARMS}
    macro_delta={k:difference(macro[a],macro[b]) for k,(a,b) in CONTRASTS.items()}
    object_wins={k:wins(per_object[uid]['contrasts'][k] for uid in objects) for k in CONTRASTS}
    for name,values in [('cells',macro),('contrasts',macro_delta),('object_win_counts',object_wins)]:
        compare(values,summary['statistics']['macro'][name],'macro/'+name)
    pair_wins={k:wins(per_seed[uid][str(seed)]['contrasts'][k] for uid in objects for seed in SEEDS) for k in CONTRASTS}
    seed_macro={};leave_one_seed={}
    for seed in SEEDS:
        cells={arm:average(indexed[uid,seed,arm]['metrics'] for uid in objects) for arm in ARMS}
        seed_macro[str(seed)]={'cells':cells,'contrasts':{k:difference(cells[a],cells[b]) for k,(a,b) in CONTRASTS.items()}}
        keep=[s for s in SEEDS if s!=seed]
        cells={arm:average(average(indexed[uid,s,arm]['metrics'] for s in keep) for uid in objects) for arm in ARMS}
        leave_one_seed[str(seed)]={'retained_seeds':keep,'cells':cells,
            'contrasts':{k:difference(cells[a],cells[b]) for k,(a,b) in CONTRASTS.items()}}
    concentration={}
    for contrast in CONTRASTS:
        concentration[contrast]={}
        for metric in ('fscore_0p02','chamfer_l1','occluded_recall'):
            net=sum(per_object[uid]['contrasts'][contrast][metric] for uid in objects)
            rows=[{'uid':uid,'raw_delta':per_object[uid]['contrasts'][contrast][metric],
                'share_of_signed_net':None if abs(net)<1e-15 else per_object[uid]['contrasts'][contrast][metric]/net}
                for uid in objects]
            concentration[contrast][metric]={'object_contributions':rows,
                'definition':'Object delta / sum of all four object deltas; signed shares, not independent evidence.'}
    statistic_checks=check_count
    fairness=training_audit(p,compare)
    joint_wins={}
    for contrast in CONTRASTS:
        values=[per_seed[uid][str(seed)]['contrasts'][contrast] for uid in objects for seed in SEEDS]
        joint_wins[contrast]={'cd_and_f02_improve':sum(r['chamfer_l1']< -1e-12 and r['fscore_0p02']>1e-12 for r in values),
            'cd_f02_and_occluded_recall_improve':sum(r['chamfer_l1']< -1e-12 and r['fscore_0p02']>1e-12
                and r['occluded_recall']>1e-12 for r in values),'pairs':len(values)}
    output={'time':c.now(),'identity':identity,'scope':'independent aggregation of original scalar metric records; no GPU generation or mesh resampling',
        'summary_source':c.binding(c.OUT/'summary.json'),'plan_source':c.binding(c.OUT/'plan.json'),
        'verification':{'passed':True,'raw_cases':36,'record_bindings_verified':len(sources),
            'summary_statistic_checks':statistic_checks,'training_field_checks':check_count-statistic_checks,
            'total_scalar_or_exact_value_checks':check_count,'max_abs_error':maximum_error},
        'metric_directions':{k:'lower' if k in LOWER else 'higher' for k in METRICS},
        'per_seed':per_seed,'per_object':per_object,'macro':{'cells':macro,'contrasts':macro_delta,
            'object_win_counts':object_wins},'pair_win_counts':pair_wins,'seed_macro':seed_macro,
        'leave_one_seed_out':leave_one_seed,'concentration':concentration,'joint_pair_wins':joint_wins,
        'relative_macro_cd_change_percent':{key:100*macro_delta[key]['chamfer_l1']/macro[b]['chamfer_l1']
            for key,(_,b) in CONTRASTS.items()},'training_fairness':fairness,
        'records':records,'source_bindings':sources,
        'limitations':['Four old training objects in one scene; no held-out split.',
            'Three seeds are sampling repeats, not independent model-training replicates.',
            'Oracle replaces both observed depth and confidence; not pure depth ablation.',
            '200 steps may not saturate adaptation; a failed object is not proof of architectural impossibility.',
            'Existing cameras/masks/canonical coordinates are privileged; oracle is not deployable RGB performance.']}
    return output


def training_audit(p,compare):
    policy_fields=('steps','seed','lr','warmup','betas','weight_decay','batch','p_uncond','weights','amp',
        'residual_channels','projection_bias','parent_checkpoint','observation_variant','flow_loss','view_selection')
    paired_fields=('uid','index','original_index','epoch','offset','diffusion_time','noise_sha','learning_rate','object_group_exposures')
    identity_fields=('initial_residual_sha','trainable_parameter_count','trainable_parameter_names','bindings','fixed_parent')
    data_fields=('uid','original_index','inputs','target_source','target_tensor_sha')
    sources={};runs={};histories={};completions={}
    for arm in ARMS:
        if arm==ARMS[2]:
            directory=c.train_dir(arm)
            source={k:c.binding(directory/n) for k,n in [('run','run.json'),('history','history.jsonl'),('complete','complete.json')]}
        else:source={k:p['reference_training'][arm][k] for k in ('run','history','complete')}
        sources[arm]=source
        run=c.read(c.verify(source['run']));complete=c.read(c.verify(source['complete']))
        history=[json.loads(line) for line in c.verify(source['history']).read_text().splitlines()]
        if (complete['step']!=200 or [r['step'] for r in history]!=list(range(1,201))
                or complete['contract']!=run['contract'] or complete['data_identity']!=run['data_identity']
                or run['trainable_parameter_count']!=3164160
                or run['fixed_parent']!='complete old occluded appearance_control including its trained residual'):
            raise ValueError('Incomplete or inconsistent training '+arm)
        for row in history:
            if any(not math.isfinite(row[k]) for k in ('flow_loss','gradient_norm','learning_rate','diffusion_time')):
                raise ValueError('Nonfinite training history')
        runs[arm]=run;histories[arm]=history;completions[arm]=complete
    base=runs[ARMS[0]]
    for arm in ARMS:
        run=runs[arm]
        for key in policy_fields:compare(run['contract'][key],base['contract'][key],'training_policy/'+key)
        for key in identity_fields:compare(run[key],base[key],'training_initialization/'+key)
        data=[{k:r[k] for k in data_fields} for r in run['data_identity']]
        expected=[{k:r[k] for k in data_fields} for r in base['data_identity']]
        compare(data,expected,'training_inputs_targets')
        for row,ref in zip(histories[arm],histories[ARMS[0]]):
            for key in paired_fields:compare(row[key],ref[key],'training_step/'+str(row['step'])+'/'+key)
    descriptions={}
    for arm in ARMS:
        hist=histories[arm];counts=dict(Counter(r['uid'] for r in hist))
        if counts!={uid:50 for uid in c.UIDS}:raise ValueError('Exposure mismatch')
        descriptions[arm]={'steps':200,'exposures_per_object':counts,
            'last_step_loss':hist[-1]['flow_loss'],'first50_mean_loss':fmean(r['flow_loss'] for r in hist[:50]),
            'last50_mean_loss':fmean(r['flow_loss'] for r in hist[-50:]),
            'final_residual_sha':completions[arm]['final_residual_sha']}
    return {'passed':True,'sources':sources,'policy_fields_checked':list(policy_fields),
        'paired_history_fields_checked':list(paired_fields),'steps_compared_per_arm':200,
        'initial_residual_sha':base['initial_residual_sha'],'trainable_parameter_count':base['trainable_parameter_count'],
        'same_frozen_parent':True,'same_inputs_targets':True,'same_initialization':True,
        'same_sampling_order_noise_lr':True,'runs':descriptions,
        'scope':'Verified run/complete/history JSON identities; does not reload tensors from model checkpoints.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='Write analysis.json under /data; otherwise stdout only')
    args=parser.parse_args();result=run()
    if args.write:c.write(c.OUT/'review_20261026/analysis.json',result)
    compact={'verification':result['verification'],'macro':result['macro']['cells'],
        'macro_contrasts':result['macro']['contrasts'],
        'oracle_vs_predicted_pair_wins':result['pair_win_counts']['oracle_minus_predicted'],
        'relative_macro_cd_change_percent':result['relative_macro_cd_change_percent'],
        'leave_one_seed_out_oracle_vs_predicted':{seed:row['contrasts']['oracle_minus_predicted']
            for seed,row in result['leave_one_seed_out'].items()},
        'training_fairness_passed':result['training_fairness']['passed'],
        'output':str(c.OUT/'review_20261026/analysis.json') if args.write else None}
    print(json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
