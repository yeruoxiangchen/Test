import argparse
import numpy as np
import torch
from .config import OUT,PRIOR,PARAMS,ARMS,folder,read,write,setup,now
from visual_evidence_probe.experiments.dense_evidence.evaluation import render_depth
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj
from visual_evidence_probe.experiments.editor_support.figures import _images,_grid,_plotting


@torch.no_grad()
def one(index,seed,context,device):
    root=folder(index,seed)
    if not (root/'complete.json').exists():return None
    camera=read(PRIOR/'data'/f'{index:03d}'/'novel_cameras.json')
    k=torch.tensor(camera['intrinsics'],device=device,dtype=torch.float32);w=torch.tensor(camera['extrinsics'],device=device,dtype=torch.float32)
    gt=torch.load(PRIOR/'data'/f'{index:03d}'/'evaluation_targets.pt',map_location='cpu',weights_only=False)['novel']
    with np.load(PRIOR/'objects'/f'{index:03d}'/str(seed)/'novel_region.npz') as data:region=data['region']
    views=np.argsort(-region.sum((1,2)),kind='stable')[:3].tolist()
    rows=[dict(label='GT scan\nEvaluation only',pred=gt,images=_images(gt,views))]
    paths=[('Frozen30K\nOriginal',root/'original.obj')]
    uid=''
    for endpoint in ('candidate','accepted'):
        for arm in ARMS:
            result=read(root/arm/'result.json');uid=result['uid'];step=result['candidate_step'] if endpoint=='candidate' else result['selected_step']
            paths.append((f'{arm}\n{endpoint}, step {step}',root/arm/f'{endpoint}.obj'))
    for label,path in paths:
        mesh=load_obj(path);pred=render_depth(mesh,k,w,context)
        cpu={key:value.cpu() for key,value in pred.items()}
        rows.append(dict(label=label,pred=cpu,images=_images(cpu,views)))
    boxes=[]
    for view in views:
        hit=(rows[0]['pred']['mask'][view]|rows[1]['pred']['mask'][view]).numpy();yy,xx=np.nonzero(hit)
        if len(yy):
            width=min(256,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*1.15)+4)
            x=max(0,min(256-width,int((xx.min()+xx.max()-width)/2)));y=max(0,min(256-width,int((yy.min()+yy.max()-width)/2)))
            boxes.append((x,y,x+width,y+width))
        else:boxes.append((0,0,256,256))
    for row in rows:row['images']=[im[y:y1,x:x1] for im,(x,y,x1,y1) in zip(row['images'],boxes)]
    dest=OUT/'figures/meshes'/f'{index:03d}_{seed}.jpg'
    _grid(rows,views,f'{uid}, seed {seed}',
        'Round8 | identical cameras and unchanged frame | no GT used in editing or acceptance',dest,
        '80 attempted updates; anchored editors may roll back. Candidate and actual accepted outputs are separate.',
        view_labels=[f'New view {view}\nEvaluation only' for view in views])
    with np.load(root/'evidence_sealed.npz') as data:weights,train=data['weights'],data['train_views'].tolist()
    plt=_plotting();fig,axes=plt.subplots(1,len(train),figsize=(3*len(train),3),squeeze=False)
    for ax,view in zip(axes[0],train):
        ax.imshow(weights[view],cmap='viridis',vmin=0,vmax=1);ax.set_title(f'Train view {view}');ax.axis('off')
    fig.suptitle('Frozen evidence weights: NCC anchors=1; rejected teacher rays=0',fontsize=10)
    evidence=OUT/'figures/weights'/dest.name;evidence.parent.mkdir(parents=True,exist_ok=True);fig.savefig(evidence,dpi=120);plt.close(fig)
    return dict(index=index,seed=seed,meshes=str(dest),weights=str(evidence),same_cameras_and_frame=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='1');p.add_argument('--indices',default='4,5,6,7,8,9,10,11');p.add_argument('--seeds',default='42,43');p.add_argument('--overwrite',action='store_true');a=p.parse_args();setup(a.gpu)
    import nvdiffrast.torch as dr
    device=torch.device('cuda:0');context=dr.RasterizeCudaContext(device=device);path=OUT/'figures/manifest.json';manifest=read(path) if path.exists() else dict(cases={})
    for index in map(int,a.indices.split(',')):
        for seed in map(int,a.seeds.split(',')):
            key=f'{index:03d}_{seed}'
            if key in manifest['cases'] and not a.overwrite:continue
            value=one(index,seed,context,device)
            if value:manifest['cases'][key]=value;write(path,manifest);print(dict(figure=key),flush=True)
    manifest.update(time=now(),complete=len(manifest['cases'])==16,expected=16);write(path,manifest)


if __name__=='__main__':main()
