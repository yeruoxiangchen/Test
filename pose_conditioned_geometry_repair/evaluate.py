"""Controlled synthetic repair evaluation; raw-mesh and codec metrics separated."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import common as c
from .data import RepairDataset, intervention
from .model import build, conditional_sample, features
from .flow import diffuse,target_velocity,weighted_flow_loss,sample_flow,bridge_state,sample_bridge
from .vendor_pose.checkpoint import load_checkpoint
from .corruptions import roi_mask, remap_reference


def iou(a,b):
    a,b=a.bool(),b.bool()
    return float((a&b).sum()/(a|b).sum().clamp_min(1))


def load_model(stage,arm,checkpoint):
    model=build(stage,arm,checkpointing=False)
    if checkpoint:
        ck=load_checkpoint(checkpoint,model=model,stage=stage,weights='raw')
        if ck['model_identity']['arm']!=arm:
            raise ValueError('Evaluation checkpoint arm differs')
        model.repair_objective=ck['args'].get('objective','flow')
    return model.eval()


@torch.no_grad()
def flow_diagnostic(model,sample,pair,stage,objective='flow'):
    context=sample['dino_features'][:1].cuda()
    views=torch.arange(len(sample['dino_features']))
    pos=conditional_sample(sample,pair)
    records=[]
    for k,t in enumerate([.25,.5,1.] if objective=='bridge' else [.25,.5,.75]):
        f=features(pair['clean'])
        gen=torch.Generator(device='cuda').manual_seed(100003+k)
        n=torch.randn(f.shape,device='cuda',generator=gen)
        noise=n if stage=='ss' else pair['clean'].replace(n)
        x=bridge_state(pair['clean'],pair['bad'],noise,t) if objective=='bridge' else diffuse(pair['clean'],noise,t)
        target=features(pair['bad'])-features(pair['clean']) if objective=='bridge' else target_velocity(pair['clean'],noise)
        native_t=(650 if objective=='bridge' else 1000)*t
        with torch.autocast('cuda',dtype=torch.bfloat16):
            stock=model.stock_prediction(x,torch.tensor([native_t],device='cuda'),context)
            pred,_=model.adapted_prediction(x,torch.tensor([native_t],device='cuda'),context,pos,
                view_indices=views,stock_velocity=stock,compare_stock=False)
        if objective=='bridge':
            pred=pred.float()-stock.float() if stage=='ss' else pred.replace(pred.feats.float()-stock.feats.float())
            stock=torch.zeros_like(stock) if stage=='ss' else stock.replace(torch.zeros_like(stock.feats))
        _,si,so=weighted_flow_loss(stock,target,pair['mask'],stage,'repair')
        _,pi,po=weighted_flow_loss(pred,target,pair['mask'],stage,'repair')
        records.append({'t':t,'stock_edit_loss':float(si),'candidate_edit_loss':float(pi),
                        'stock_outside_loss':float(so),'candidate_outside_loss':float(po)})
    return records


def canonical_gt(row):
    import trimesh
    source=trimesh.load(row['source_glb'],force='scene',process=False,maintain_order=True)
    parts=[]
    for node in source.graph.nodes_geometry:
        transform,key=source.graph[node]
        mesh=source.geometry[key]
        if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):
            continue
        mesh=mesh.copy()
        mesh.apply_transform(transform)
        parts.append(mesh)
    mesh=trimesh.util.concatenate(parts)
    b=row['canonical']
    mesh.vertices=((np.asarray(mesh.vertices)-np.asarray(b['center']))/b['scale']*b['margin']).astype(np.float32)
    if np.abs(mesh.vertices).max()>.5001:
        raise ValueError('Raw GT normalization exceeded the native AABB')
    return mesh


def mesh_metrics(mesh,gt,seed=20261029):
    import trimesh
    from scipy.spatial import cKDTree
    if not len(mesh.faces) or not len(mesh.vertices) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Cannot score an empty or nonfinite mesh')
    state=np.random.get_state()
    try:
        np.random.seed(seed)
        a=gt.sample(20000)
        np.random.seed(seed+1)
        b=mesh.sample(20000)
    finally:
        np.random.set_state(state)
    da=cKDTree(b).query(a,workers=4)[0]
    db=cKDTree(a).query(b,workers=4)[0]
    result={'cd_l1':float((da.mean()+db.mean())/2),
            'cd_l2_squared':float((np.mean(da**2)+np.mean(db**2))/2),
            'vertices':len(mesh.vertices),'faces':len(mesh.faces),
            'components':int(len(trimesh.graph.connected_components(mesh.face_adjacency,nodes=np.arange(len(mesh.faces)))))}
    for t in [.01,.02]:
        p,r=float((db<t).mean()),float((da<t).mean())
        result[f'f@{t}']=2*p*r/max(p+r,1e-12)
    return result


def shape_mesh(latent,decoder,path):
    from .vendor_pose.runtime import decode_shape_latent,export_geometry_mesh
    import trimesh
    with torch.no_grad():
        mesh=decode_shape_latent(latent,decoder)
    export_geometry_mesh(mesh,path)
    out=trimesh.Trimesh(mesh.vertices.float().cpu().numpy(),mesh.faces.cpu().numpy(),process=False)
    del mesh
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def ss_predictions(args,ds,out):
    from trellis2 import models
    model=load_model('ss',args.arm,args.checkpoint if args.stage=='ss' else args.ss_checkpoint)
    decoder=models.from_pretrained(c.MODEL_PATHS['ss_decoder']).eval().cuda()
    rows=[]
    for idx in range(min(args.limit,len(ds))):
        sample=ds[idx]
        folder=out/'objects'/sample['uid']
        folder.mkdir(parents=True,exist_ok=True)
        for variant in args.variants:
            pair=intervention(sample,'ss',variant,'cuda')
            pack=pair['pair']
            ref=pack['bad_occ64'].cuda()
            target=sample['pair_pack']['clean_occ64'].cuda()
            region=pack['region64'].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                repaired=(sample_bridge(model,sample,pair,seed=args.seed) if args.objective=='bridge' and args.stage=='ss'
                          else sample_flow(model,sample,pair,seed=args.seed,arm=args.arm))
                stock=sample_flow(model,sample,pair,seed=args.seed,arm='repair',stock=True)
            raw=(decoder(repaired)>0)[0,0]
            stock_raw=(decoder(stock)>0)[0,0]
            pred=torch.where(region,raw,ref) if args.arm=='repair' else raw
            stock_pred=torch.where(region,stock_raw,ref)
            if args.arm=='repair' and not torch.equal(pred[~region],ref[~region]):
                raise ValueError('SS edited outside the requested region')
            np.savez_compressed(folder/f'ss_{variant}.npz',coords=torch.argwhere(pred).cpu().numpy(),
                stock_coords=torch.argwhere(stock_pred).cpu().numpy(),latent=repaired[0].float().cpu().numpy())
            loss=flow_diagnostic(model,sample,pair,'ss',args.objective if args.stage=='ss' else 'flow')
            rec={'uid':sample['uid'],'variant':variant,'kind':pair['kind'],
                 'damaged_iou':iou(ref,target),'stock_iou':iou(stock_pred,target),
                 'candidate_iou':iou(pred,target),'candidate_raw_iou':iou(raw,target),
                 'outside_occupancy_changes':int((pred[~region]!=ref[~region]).sum()),
                 'flow_diagnostic':loss,'source_checkpoint':args.checkpoint if args.stage=='ss' else args.ss_checkpoint}
            c.write(folder/f'ss_metrics_{variant}.json',rec)
            rows.append(rec)
            c.write(out/'progress.json',{'time':c.now(),'stage':'ss','records':len(rows),
                'expected':min(args.limit,len(ds))*len(args.variants),'uid':sample['uid']})
            print(json.dumps({k:v for k,v in rec.items() if k!='flow_diagnostic'}),flush=True)
    del model,decoder
    gc.collect()
    torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def shape_predictions(args,ds,out):
    from trellis2 import models
    from trellis2.modules import sparse as sp
    model=load_model('shape',args.arm,args.checkpoint)
    decoder=models.from_pretrained(c.MODEL_PATHS['shape_decoder']).eval().cuda()
    decoder.set_resolution(512)
    decoder.low_vram=True
    rows=[]
    from .figures import mesh_tiles,one,gallery
    import nvdiffrast.torch as dr
    figure_context=dr.RasterizeCudaContext(device='cuda:0')
    figures=[]
    planrows={r['object_uid']:r for r in c.sample_rows()}
    for idx in range(min(args.limit,len(ds))):
        sample=ds[idx]
        folder=out/'objects'/sample['uid']
        folder.mkdir(parents=True,exist_ok=True)
        rawgt=canonical_gt(planrows[sample['uid']])
        rawgt.export(folder/'gt.obj')
        for j,tile in enumerate(mesh_tiles(rawgt,figure_context)):
            tile.save(folder/f'tile_gt_{j}.png')
        for variant in args.variants:
            pair=intervention(sample,'shape',variant,'cuda')
            diag=flow_diagnostic(model,sample,pair,'shape',args.objective)
            sampling_seconds={}
            torch.cuda.synchronize()
            sampling_start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                candidate=(sample_bridge(model,sample,pair,seed=args.seed) if args.objective=='bridge'
                           else sample_flow(model,sample,pair,seed=args.seed,arm=args.arm))
            torch.cuda.synchronize()
            sampling_seconds['candidate_teacher_support']=time.perf_counter()-sampling_start
            sampling_start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                stock=sample_flow(model,sample,pair,seed=args.seed,arm='repair',stock=True)
            torch.cuda.synchronize()
            sampling_seconds['stock_local']=time.perf_counter()-sampling_start
            arms={'damaged':pair['bad'],'stock_local':stock,'candidate_teacher_support':candidate,
                  'native_codec_gt':pair['clean']}
            present=pair['present']
            if present.any():
                arms['damaged_input']=sp.SparseTensor(pair['bad'].feats[present],pair['bad'].coords[present])
            else:
                raise ValueError('Synthetic reference contains no actual Shape support')
            if args.ss_checkpoint:
                pack=np.load(folder/f'ss_{variant}.npz')
                q=np.unique(pack['coords']//2,axis=0)
                if not 0<len(q)<=8192:
                    raise ValueError('Predicted SS support outside native512 token contract')
                q=torch.tensor(q,device='cuda',dtype=torch.int32)
                q=torch.cat([torch.zeros_like(q[:,:1]),q],-1)
                valid=pair['present']
                # SS occupancy64 and native Shape32 differ slightly even for
                # clean GT. Preserve the input Shape support outside the ROI;
                # never let this codec mismatch create exterior holes.
                refcoords=pair['bad'].coords[valid]
                outside=~roi_mask(refcoords,pair['bounds64'],32)
                if args.arm=='repair':
                    q=torch.unique(torch.cat([q[roi_mask(q,pair['bounds64'],32)],refcoords[outside]]),dim=0)
                if not 0<len(q)<=8192:
                    raise ValueError('Composed predicted support exceeds native512 contract')
                bf,present=remap_reference(pair['bad'].coords[valid],pair['bad'].feats[valid],q)
                bad=sp.SparseTensor(bf,q)
                rollout={**pair,'bad':bad,'mask':roi_mask(q,pair['bounds64'],32),'present':present}
                if not present[~rollout['mask']].all():
                    # Record unsupported exterior rather than silently filling
                    # from GT features. This exposes SS/Shape support mismatch.
                    exterior_missing=int((~present & ~rollout['mask']).sum())
                else:
                    exterior_missing=0
                torch.cuda.synchronize()
                sampling_start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    arms['candidate_predicted_ss']=(sample_bridge(model,sample,rollout,seed=args.seed)
                        if args.objective=='bridge' else sample_flow(model,sample,rollout,seed=args.seed,arm=args.arm))
                torch.cuda.synchronize()
                sampling_seconds['candidate_predicted_ss']=time.perf_counter()-sampling_start
                # The unmodified prior must receive its own predicted SS,
                # rather than the GT support used by stock_local diagnostics.
                qstock=np.unique(pack['stock_coords']//2,axis=0)
                qstock=torch.tensor(qstock,device='cuda',dtype=torch.int32)
                qstock=torch.cat([torch.zeros_like(qstock[:,:1]),qstock],-1)
                qstock=torch.unique(torch.cat([
                    qstock[roi_mask(qstock,pair['bounds64'],32)],refcoords[outside]]),dim=0)
                if not 0<len(qstock)<=8192:
                    raise ValueError('Stock SS support exceeds the native512 contract')
                stock_feats,stock_present=remap_reference(pair['bad'].coords[valid],pair['bad'].feats[valid],qstock)
                stock_ref=sp.SparseTensor(stock_feats,qstock)
                stock_pair={**pair,'bad':stock_ref,'mask':roi_mask(qstock,pair['bounds64'],32),'present':stock_present}
                torch.cuda.synchronize()
                sampling_start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    arms['stock_predicted_ss']=sample_flow(model,sample,stock_pair,seed=args.seed,arm='repair',stock=True)
                torch.cuda.synchronize()
                sampling_seconds['stock_predicted_ss']=time.perf_counter()-sampling_start
            else:
                exterior_missing=None
            metrics={}
            for name,z in arms.items():
                mesh=shape_mesh(z,decoder,folder/f'{name}_{variant}.obj')
                metrics[name]=mesh_metrics(mesh,rawgt)
                for j,tile in enumerate(mesh_tiles(mesh,figure_context)):
                    tile.save(folder/f'tile_{name}_{variant}_{j}.png')
                np.savez_compressed(folder/f'{name}_{variant}.npz',coords=z.coords[:,1:].cpu().numpy(),
                                    normalized_feats=z.feats.float().cpu().numpy())
            rec={'uid':sample['uid'],'variant':variant,'kind':pair['kind'],'metrics':metrics,
                'sampling_seconds':sampling_seconds,
                'sampling_timing_scope':'Includes condition projection and flow calls; excludes SS, model load, mesh decode/export and metrics',
                'flow_diagnostic':diag,'outside_features_max_abs':float((candidate.feats[~pair['mask']]-pair['bad'].feats[~pair['mask']]).abs().max())
                    if (~pair['mask']).any() else 0.,
                'predicted_ss_exterior_without_reference':exterior_missing,
                'alignment':'shared native canonical frame; no ICP, per-method scaling or GT fit',
                'localization':'known synthetic intervention ROI; not automatic error discovery'}
            c.write(folder/f'shape_metrics_{variant}.json',rec)
            figures.append(one(folder,variant,figure_context))
            rows.append(rec)
            c.write(out/'progress.json',{'time':c.now(),'stage':'shape','records':len(rows),
                'expected':min(args.limit,len(ds))*len(args.variants),'uid':sample['uid']})
            print(json.dumps({'uid':sample['uid'],'variant':variant,'metrics':metrics}),flush=True)
    gallery(out,figures)
    return rows


def summarize(args,rows,out):
    key='ss' if args.stage=='ss' else 'shape'
    result={'time':c.now(),'stage':key,'records':len(rows),'objects':min(args.limit,len(RepairDataset(c.DATA/f'{args.subset}.json',key))),
            'checkpoint':args.checkpoint,'ss_checkpoint':args.ss_checkpoint,'arm':args.arm,
            'variants':args.variants,'seed':args.seed,'objective':args.objective,
            'synthetic_known_roi':True,'source':c.source_identity()['identity']}
    lossrows=[l for r in rows for l in r['flow_diagnostic']]
    stock_loss=float(np.mean([r['stock_edit_loss'] for r in lossrows]))
    candidate_loss=float(np.mean([r['candidate_edit_loss'] for r in lossrows]))
    result['edit_flow_loss']={'reference':stock_loss,'candidate':candidate_loss,
        'reference_type':'unchanged input / zero residual' if args.objective=='bridge' else 'official stock denoising flow',
        'ratio':candidate_loss/max(1e-12,stock_loss)}
    if key=='ss':
        for name in ['damaged_iou','stock_iou','candidate_iou','candidate_raw_iou','outside_occupancy_changes']:
            result[name]=float(np.mean([r[name] for r in rows]))
        result['overfit_gate_passed']=candidate_loss<.95*stock_loss and result['candidate_iou']>result['damaged_iou']+.01
    else:
        result['means']={arm:{m:float(np.mean([r['metrics'][arm][m] for r in rows]))
                    for m in ['cd_l1','f@0.01','f@0.02','components']}
                    for arm in rows[0]['metrics']}
        before=result['means']['damaged']['cd_l1']
        after=result['means']['candidate_teacher_support']['cd_l1']
        result['mesh_cd_gain_vs_damaged']=1-after/before
        result['outside_features_max_abs']=max(r['outside_features_max_abs'] for r in rows)
        result['mean_sampling_seconds']={k:float(np.mean([r['sampling_seconds'][k] for r in rows]))
            for k in rows[0]['sampling_seconds']}
        result['overfit_gate_passed']=candidate_loss<.95*stock_loss and after<before and result['outside_features_max_abs']==0
        if 'candidate_predicted_ss' in result['means']:
            actual_before=result['means'].get('damaged_input',result['means']['damaged'])['cd_l1']
            result['predicted_ss_mesh_cd_gain_vs_damaged']=1-result['means']['candidate_predicted_ss']['cd_l1']/actual_before
            result['overfit_gate_passed'] &= result['predicted_ss_mesh_cd_gain_vs_damaged']>0
            if 'stock_predicted_ss' in result['means']:
                result['predicted_ss_cd_gain_vs_stock_pipeline']=(1-result['means']['candidate_predicted_ss']['cd_l1'] /
                    result['means']['stock_predicted_ss']['cd_l1'])
        result['by_corruption']={}
        for kind in sorted({r['kind'] for r in rows}):
            group=[r for r in rows if r['kind']==kind]
            result['by_corruption'][kind]={
                'records':len(group),
                'means':{arm:{m:float(np.mean([r['metrics'][arm][m] for r in group]))
                    for m in ['cd_l1','f@0.01','f@0.02']} for arm in group[0]['metrics']}}
    result['complete']=len(rows)==result['objects']*len(args.variants)
    c.write(out/'summary.json',result)
    lines=['# Native TRELLIS.2 repair evaluation','',result['time'],'',
        f'Objects: {result["objects"]}; records: {result["records"]}; stage: {key}; objective: {args.objective}.',
        'Controlled synthetic ROI; no automatic error localization or real-generation generalization claim.','',
        '## Fixed-state latent diagnostic','',str(result['edit_flow_loss']),'']
    if key=='shape':
        lines+=['| Mesh / support | CD-L1 | F@.01 | F@.02 |','|---|---:|---:|---:|']
        for arm,m in result['means'].items():
            lines.append(f'| {arm} | {m["cd_l1"]:.8f} | {m["f@0.01"]:.6f} | {m["f@0.02"]:.6f} |')
        lines+=['','`damaged` retains GT support with corrupted features; `damaged_input` is the actual sparse reference.',
                'Compare predicted-SS results with actual damaged input, and teacher-support results with the same support control.','',
                'Corruption-wise results, including identity preservation, are in summary.json.','']
    else:
        lines+=[f'Damaged IoU: {result["damaged_iou"]:.6f}; candidate IoU: {result["candidate_iou"]:.6f}.','']
    lines+=[f'Overfit gate: {result["overfit_gate_passed"]}. This is not a Dev64 success claim.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['ss','shape'],required=True)
    p.add_argument('--subset',choices=['overfit','dev'],default='overfit')
    p.add_argument('--arm',choices=['repair','full'],default='repair')
    p.add_argument('--checkpoint',default='')
    p.add_argument('--ss-checkpoint',default='')
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=32)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--objective',choices=['flow','bridge'],default='flow')
    p.add_argument('--variants',type=int,nargs='+',default=[0])
    args=p.parse_args()
    out=Path(args.output).resolve()
    if not out.is_relative_to(c.ROOT/'outputs'):
        raise ValueError('Evaluation output outside project')
    if not all(0<=v<4 for v in args.variants) or args.limit<1:
        raise ValueError('Invalid evaluation selection')
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    c.write(out/'config.json',{'args':vars(args),'time':c.now(),'source':c.source_identity()})
    if args.stage=='ss':
        rows=ss_predictions(args,RepairDataset(c.DATA/f'{args.subset}.json','ss'),out)
    else:
        if args.ss_checkpoint:
            ss_predictions(args,RepairDataset(c.DATA/f'{args.subset}.json','ss'),out)
        rows=shape_predictions(args,RepairDataset(c.DATA/f'{args.subset}.json','shape'),out)
    summarize(args,rows,out)


if __name__=='__main__':
    main()
