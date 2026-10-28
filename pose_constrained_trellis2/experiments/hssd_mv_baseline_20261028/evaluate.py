"""Held-out inference with explicit training provenance and no target inputs."""
import argparse
import gc
import math
import os
from pathlib import Path
from . import common as c


def validate_checkpoint(config, training_identity, expected_step):
    from pose_constrained_trellis2.infer import _validate_manifest_binding
    _validate_manifest_binding(config,training_identity,training_identity['stage'])
    if config['step'] != expected_step: raise ValueError('Unexpected checkpoint step')
    if config['context_mode'] != c.CONFIG['context_mode']:raise ValueError('Context contract changed')


def run(arm, uid, seed, smoke=False):
    os.environ.update(c.environment())
    import numpy as np
    import torch
    import trimesh
    from scipy.spatial import cKDTree
    from pose_constrained_trellis2.data import PoseTrellis2Dataset
    from pose_constrained_trellis2.conditioning import build_2d_context,sample_to_device
    from pose_constrained_trellis2.models import load_trainable_state_dict
    from pose_constrained_trellis2.infer import _checkpoint_configuration,_sample_stage
    from pose_constrained_trellis2 import runtime as rt
    p=c.plan(); source=next(r for r in p['objects'] if r['uid']==uid)
    expected_step=2 if smoke else c.CONFIG['steps']
    dest=c.OUT/('smoke/evaluation' if smoke else 'evaluations')/arm/uid/f'seed_{seed}'
    dest.mkdir(parents=True,exist_ok=True)
    dataset=PoseTrellis2Dataset(c.DATA/'all.json',stage='inference')
    index=next(i for i,r in enumerate(dataset.rows) if r['uid']==uid)
    input_row=dataset.rows[index]
    assert c.sha(input_row['pose_file'])==input_row['pose_file_sha256'],'Changed evaluation input'
    sample=sample_to_device(dataset[index],torch.device('cuda'))
    # Inference loader does not deserialize ss.npz/shape.npz or mesh targets.
    assert not any(k in sample for k in ('target','target_coords','target_feats'))
    views=torch.arange(len(sample['dino_features']),device='cuda')
    results={}; checkpoints={}; models=c.models()
    rt.SS_DECODER_MODEL=models['ss_decoder'];rt.SHAPE_DECODER_MODEL=models['shape_decoder']

    def release():gc.collect();torch.cuda.empty_cache()

    def stage_model(stage):
        if arm=='official_reference':
            return rt.load_official_flow(stage,model_path=models[stage+'_flow']).cuda().eval()
        folder=c.OUT/('smoke/training' if smoke else 'training')/stage
        path=folder/'checkpoints'/f'step_{expected_step:06d}.pt'
        config=_checkpoint_configuration(path,stage)
        if config['model_path']!=models[stage+'_flow']:raise ValueError('Wrong official parent checkpoint')
        training=PoseTrellis2Dataset(c.DATA/'train.json',stage=stage,
                       indices=str(p['smoke_train_index']) if smoke else 'all').identity()
        validate_checkpoint(config,training,expected_step)
        model=rt.load_pose_conditioned_flow(stage,device='cuda',model_path=models[stage+'_flow'],
                    lora_rank=config['lora_rank'],lora_alpha=config['lora_alpha'],gradient_checkpointing=False).eval()
        load_trainable_state_dict(model,config['payload']['model_trainable_state'])
        checkpoints[stage]={'path':str(path),'step':expected_step,'training_identity':training,
                            'evaluation_manifest_sha256':c.sha(c.DATA/'all.json')}
        return model

    def sample_stage(stage,model,noise):
        if arm=='mv_independent':
            return _sample_stage(stage,model,noise,sample=sample,view_indices=views,
                 context_mode=c.CONFIG['context_mode'],steps_override=12,quiet=True)
        positive=build_2d_context(sample['dino_features'],views,mode='first')
        sampler,parameters=rt.make_sampler(stage)
        prediction=sampler.sample(model,noise,cond=positive,neg_cond=torch.zeros_like(positive),
                                   **parameters,verbose=False).samples
        return prediction,{'sampler':parameters,'conditioning':'native first image only; no adapters or GT shape inputs'}

    failure=None
    with torch.no_grad():
        ss=stage_model('ss')
        # Reset AFTER loading the model: adapter initialization cannot shift noise.
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        noise=torch.randn((1,8,16,16,16),device='cuda')
        ss_z,results['ss']=sample_stage('ss',ss,noise)
        np.savez_compressed(dest/'ss.npz',z=ss_z.float().cpu().numpy())
        del ss,noise;release()
        decoder=rt.load_ss_decoder(device='cuda')
        try:coords=rt.ss_latent_to_shape_coords(ss_z,decoder,shape_resolution=32)
        except RuntimeError as e:
            if 'empty support' not in str(e) and 'contract permits' not in str(e):raise
            failure=str(e)
        del decoder,ss_z;release()
        if failure is None:
            shape=stage_model('shape')
            torch.manual_seed(seed+100000);torch.cuda.manual_seed_all(seed+100000)
            noise=rt.make_shape_noise(coords,device='cuda')
            shape_z,results['shape']=sample_stage('shape',shape,noise)
            assert torch.isfinite(shape_z.feats).all()
            np.savez_compressed(dest/'shape.npz',coords=shape_z.coords[:,1:].cpu().numpy(),
                                normalized_feats=shape_z.feats.float().cpu().numpy())
            del shape,noise;release()
            decoder=rt.load_shape_decoder(device='cuda');decoder.low_vram=True
            m=rt.decode_shape_latent(shape_z,decoder)
            v=m.vertices.float().cpu().numpy();f=m.faces.cpu().numpy()
            if not len(v) or not len(f):failure='empty decoded mesh'
            else:
                assert np.isfinite(v).all() and f.min()>=0 and f.max()<len(v)
                mesh=trimesh.Trimesh(v,f,process=False);mesh.export(dest/'mesh.ply')
                world=mesh.copy();world.apply_transform(sample['object_to_world'].cpu().numpy());world.export(dest/'mesh_world.ply')
            del m,decoder,shape_z;release()
    # Ground truth first read AFTER the predicted mesh is finalized.
    stats={'cd_l1':math.sqrt(3.),'f@0.01':0.,'f@0.02':0.}
    if failure is None:
        gt=trimesh.load(c.native.DATA/'objects'/uid/'target_t2.ply',force='mesh',process=False)
        np.random.seed(20261028);a=gt.sample(100000)
        np.random.seed(20261029);b=mesh.sample(100000)
        da=cKDTree(b).query(a)[0];db=cKDTree(a).query(b)[0]
        stats={'cd_l1':float((da.mean()+db.mean())/2)}
        for threshold in (.01,.02):
            precision,recall=float((db<threshold).mean()),float((da<threshold).mean())
            stats[f'f@{threshold}']=2*precision*recall/max(precision+recall,1e-12)
        T=sample['object_to_world'].cpu().numpy()
        scales=np.linalg.svd(T[:3,:3],compute_uv=False)
        assert np.ptp(scales)<1e-5, 'World transform must be a similarity'
        stats['world_cd_l1']=stats['cd_l1']*float(scales.mean())
    c.write(dest/'result.json',{'time':c.now(),'arm':arm,'uid':uid,'split':source['split'],'seed':seed,
        'success':failure is None,'failure':failure,'metrics':stats,'checkpoints':checkpoints,
        'samplers':results,'gt_pose_controlled':True,'gt_mesh_read_after_generation':True,
        'views':list(map(int,views.cpu())),'source_frame_ids':sample['frame_ids'],
        'sampled_points':100000,'fitted_alignment':False,'smoke_only':smoke,
        'mesh':str(dest/'mesh.ply') if failure is None else None,
        'model_source':models,'runtime_identity':c.read(c.OUT/'runtime.json')['identity']})
    print(f'CASE_COMPLETE {arm} {uid} seed={seed} success={failure is None}',flush=True)
    if smoke and failure:raise RuntimeError('Smoke generation failed: '+failure)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=c.CONFIG['arms'],required=True)
    p.add_argument('--uid',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.arm,a.uid,a.seed,a.smoke)
