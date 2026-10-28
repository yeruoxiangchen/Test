"""GENA3D source audit and explicit input exchange. Never fabricate completion."""
import argparse
import importlib.util
import shutil
import subprocess
import sys
import traceback
from .common import *

FILES = ['amodal_completion.png','sd_img_cut.png','visibility_instance_mask.png','occ_mask0.png']


def audit():
    weights=Path(read(OUT/'external_weights.json')['root']) if (OUT/'external_weights.json').exists() else OUT/'external_weights'
    result={'time':now(),'gena_repository':str(GENA),
        'gena_revision':subprocess.check_output(['git','-C',str(GENA),'rev-parse','HEAD'],text=True).strip(),
        'pi3_revision':subprocess.check_output(['git','-C',str(GENA/'dependency/Pi3'),'rev-parse','HEAD'],text=True).strip(),
        'requirements':{name:importlib.util.find_spec(name) is not None for name in
            ['torch','numpy','cv2','rembg','open3d','spconv','nvdiffrast','utils3d','safetensors','huggingface_hub']},
        'gena_weights_present':(weights/'GENA3D/pipeline.json').exists(),
        'pi3_weights_present':(weights/'Pi3/model.safetensors').exists(),
        'input_contract':FILES,
        'frontend_required':'predicted 2D amodal completion and occlusion mask; official README recommends OAAD',
        'no_shortcuts':'GT clean RGB/amodal mask is not substituted for predicted completion; no raw/copy placeholder is called GENA3D',
        'coordinate_limit':'official generator estimates its own normalized object frame; native CD against our canonical GT requires an audited frame bridge, not unlabelled ICP',
        'baseline_status':'not evaluated on shared capture; separate official-demo compatibility smoke available',
        'sources':['https://github.com/Colezwhy/GENAmodal3D','https://huggingface.co/Colezwhy/GENA3D',
                   'https://github.com/yyfz/Pi3','https://github.com/saraao/amodal']}
    write(OUT/'external_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def importcheck():
    """Import the actual inference classes; package discovery alone is insufficient."""
    result={'time':now(),'passed':False,'scope':'imports only; no weights or inference'}
    import faulthandler
    faulthandler.dump_traceback_later(120)
    write(OUT/'external_import_check.json',{**result,'status':'running'})
    print('Importing GENA3D and Pi3 inference classes (no weights)',flush=True)
    try:
        sys.path.insert(0,str(GENA))
        from gena3d.pipelines import SparseImageTo3D
        from dependency.Pi3.pi3.models.pi3 import Pi3
        import inspect
        result.update(passed=True,gena_class=inspect.getfile(SparseImageTo3D),pi3_class=inspect.getfile(Pi3))
    except Exception:
        result['traceback']=traceback.format_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        write(OUT/'external_import_check.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def export():
    from PIL import Image
    frozen=plan()
    entries=[]
    for row in frozen['plan']['objects']:
        for i in FRAMES:
            target=OUT/'gena_inputs'/row['uid']/f'{i:03d}'
            target.mkdir(parents=True,exist_ok=True)
            source=OUT/'rgb'/f'{i:03d}.png'
            mask=OUT/'masks'/row['uid']/f'{i:03d}.png'
            for src,name in [(source,'sd_img_cut.png'),(mask,'visibility_instance_mask.png')]:
                dest=target/name
                if dest.exists() and sha(src)!=sha(dest):raise ValueError('Changed GENA input export')
                if not dest.exists():shutil.copyfile(src,dest)
            entries.append({'uid':row['uid'],'frame':i,'input_dir':str(target),
                'rgb':binding(target/'sd_img_cut.png'),'visible_mask':binding(target/'visibility_instance_mask.png'),
                'required_predicted_files':['amodal_completion.png','occ_mask0.png'],
                'completed':all((target/n).exists() for n in FILES)})
    write(OUT/'gena_input_manifest.json',{'plan_identity':frozen['identity'],'entries':entries,
        'status':'awaiting_predicted_frontend','gt_completion_used':False,
        'full_frame':True,'input_frames':FRAMES})
    event('gena_inputs_exported',views=len(entries),frontend='required')


def download():
    # A long, explicit user-run command. Download no datasets or unrelated models.
    from huggingface_hub import HfApi,snapshot_download,hf_hub_download
    api=HfApi()
    models=[]
    for repo in ['Colezwhy/GENA3D','yyfz233/Pi3']:
        info=api.model_info(repo,files_metadata=True)
        files=[f for f in info.siblings if f.rfilename.endswith(('.json','.safetensors'))]
        models.append({'repo':repo,'revision':info.sha,'size_bytes':sum(f.size or 0 for f in files),
                       'files':[f.rfilename for f in files]})
    total=sum(m['size_bytes'] for m in models)
    # Leave space for an SS encoder and the upstream u2net preprocessor.
    if total+500_000_000>10_000_000_000:
        destination=Path('/data/zjr/scene_shared_baselines_20261025')
    else: destination=OUT/'external_weights'
    write(OUT/'external_download_plan.json',{'time':now(),'models':models,'known_bytes':total,
        'destination':str(destination),'additional':'SS encoder from TRELLIS + u2net; DINO reused from existing cache'})
    for model in models:
        event('download_external',repo=model['repo'],bytes=model['size_bytes'])
        snapshot_download(model['repo'],revision=model['revision'],allow_patterns=model['files'],
                          local_dir=destination/model['repo'].split('/')[-1],max_workers=2)
    for suffix in ['json','safetensors']:
        hf_hub_download('microsoft/TRELLIS-image-large','ckpts/ss_enc_conv3d_16l8_fp16.'+suffix)
    import os
    os.environ['U2NET_HOME']=str(destination/'u2net')
    from rembg.sessions.u2net import U2netSession
    U2netSession.download_models()
    write(OUT/'external_weights.json',{'root':str(destination),'models':models,'time':now()})
    event('external_download_complete',root=str(destination))


def smoke():
    """One upstream example, not a claimed HSSD baseline result."""
    import sys
    import torch
    import trimesh
    from PIL import Image
    from torchvision import transforms
    weights=read(OUT/'external_weights.json')
    root=Path(weights['root'])
    import os
    os.environ['U2NET_HOME']=str(root/'u2net')
    sys.path.insert(0,str(GENA))
    from gena3d.pipelines import SparseImageTo3D
    from dependency.Pi3.pi3.models.pi3 import Pi3
    # Resolve only paths; architecture/checkpoints are unchanged.
    def stereo(self):
        self.models['stereo_model']=Pi3.from_pretrained(str(root/'Pi3')).to(self.device).eval()
    def dino(self,name):
        hub=Path(torch.hub.get_dir())/'facebookresearch_dinov2_main'
        if not hub.exists():raise FileNotFoundError('Existing local DINO repository not found: '+str(hub))
        self.models['image_cond_model']=torch.hub.load(str(hub),name,source='local',pretrained=True).eval()
        self.image_cond_model_transform=transforms.Compose([transforms.Normalize(
            mean=[.485,.456,.406],std=[.229,.224,.225])])
    SparseImageTo3D._init_stereo_model=stereo
    SparseImageTo3D._init_image_cond_model=dino
    original=SparseImageTo3D.sample_sparse_structure
    def skip_visualization(self,*args,**kwargs):
        kwargs['get_voxel_vis']=False
        return original(self,*args,**kwargs),None
    SparseImageTo3D.sample_sparse_structure=skip_visualization
    demos=sorted({p.parent for p in (GENA/'examples').rglob('amodal_completion.png')
                  if all((p.parent/n).exists() for n in FILES)})
    if not demos:raise ValueError('No complete upstream demo')
    demo=demos[0]
    output=OUT/'gena_official_smoke';output.mkdir(exist_ok=True)
    event('gena_official_smoke',input=str(demo))
    pipeline=SparseImageTo3D.from_pretrained_with_custom(str(root/'GENA3D'))
    pipeline.cuda()
    images=[Image.open(demo/n) for n in FILES]
    result=pipeline.run([images[0]],[images[1]],mask=[images[2]],occ_mask=[images[3]],seed=42,
        sparse_structure_sampler_params={'steps':12,'cfg_strength':7.5},
        slat_sampler_params={'steps':12,'cfg_strength':3.},formats=['mesh'],get_voxel_vis=False)
    mesh=result['mesh'][0]
    converted=trimesh.Trimesh(mesh.vertices.detach().cpu().numpy(),mesh.faces.detach().cpu().numpy(),process=False)
    if not len(converted.faces):raise ValueError('GENA official-demo mesh is empty')
    converted.export(output/'mesh.ply')
    write(output/'receipt.json',{'time':now(),'mesh':binding(output/'mesh.ply'),
        'input_files':[binding(demo/n) for n in FILES],'weights':weights,
        'scope':'single upstream demo compatibility only; not shared HSSD evaluation',
        'runtime_changes':['local Pi3 and cached DINO paths','skip debug voxel rendering, preserve coordinate tensor']})
    event('gena_official_smoke_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['audit','importcheck','export','download','smoke'])
    a=p.parse_args()
    globals()[a.action]()
