import argparse
import numpy as np
import torch
from .config import OUT,ROUND7,PARAMS,ARMS,folder,read,write,setup,now
from visual_evidence_probe.experiments.dense_evidence.evaluation import render_depth
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj
from visual_evidence_probe.experiments.editor_support.figures import _images,_grid,_plotting


@torch.no_grad()
def one(index,seed,context,device):
    root=folder(index,seed)
    if not (root/'complete.json').exists():return None
    camera=read(ROUND7/'data'/f'{index:03d}'/'novel_cameras.json');k=torch.tensor(camera['intrinsics'],device=device,dtype=torch.float32);w=torch.tensor(camera['extrinsics'],device=device,dtype=torch.float32)
    gt=torch.load(ROUND7/'data'/f'{index:03d}'/'evaluation_targets.pt',map_location='cpu',weights_only=False)['novel']
    with np.load(root/'novel_region.npz') as data:region=data['region']
    views=np.argsort(-region.sum((1,2)),kind='stable')[:3].tolist();outputs={}
    rendered={}
    for endpoint in ('candidate','proposal'):
        rows=[dict(label='GT scan\nEvaluation only',pred=gt)]
        sources=[('Frozen30K\nOriginal',root/'original.obj')]
        for arm in ARMS:
            r=read(root/arm/'result.json');step=r['candidate_step'] if endpoint=='candidate' else PARAMS['steps']
            sources.append((f'{arm}\n{endpoint}, step {step}',root/arm/f'{endpoint}.obj'))
        for label,path in sources:
            resolved=path.resolve()
            if resolved not in rendered:
                pred=render_depth(load_obj(path),k,w,context);rendered[resolved]={key:value.cpu() for key,value in pred.items()}
            rows.append(dict(label=label,pred=rendered[resolved]))
        boxes=[]
        for view in views:
            hit=(gt['mask'][view]|rows[1]['pred']['mask'][view]).numpy();yy,xx=np.nonzero(hit)
            if len(yy):
                width=min(256,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*1.15)+4);x=max(0,min(256-width,int((xx.min()+xx.max()-width)/2)));y=max(0,min(256-width,int((yy.min()+yy.max()-width)/2)))
                boxes.append((x,y,x+width,y+width))
            else:boxes.append((0,0,256,256))
        for row in rows:row['images']=[im[y:y1,x:x1] for im,(x,y,x1,y1) in zip(_images(row['pred'],views),boxes)]
        dest=OUT/'figures'/endpoint/f'{index:03d}_{seed}.jpg'
        _grid(rows,views,f'Round9 matched-support DIAGNOSTIC: {index:03d}, seed {seed}',
            'All arms use GT-pruned support | oracle depth only in training input cameras',dest,
            'Same frame/cameras. Oracle is privileged; estimated arms also use GT-pruned support. Not deployment results.',
            view_labels=[f'New evaluation view {v}' for v in views]);outputs[endpoint]=str(dest)
    with np.load(root/'estimated_target.npz') as data:estimated,valid=data['depth'],data['valid']
    with np.load(root/'oracle_target.npz') as data:truth=data['depth']
    meta=read(root/'targets_complete.json');train=meta['train_views'];plt=_plotting();fig,axes=plt.subplots(3,len(train),figsize=(len(train)*2.8,8),squeeze=False)
    selected=truth[valid];lo,hi=np.quantile(selected,[.02,.98]) if len(selected) else (0,1)
    depth_cmap=plt.get_cmap('viridis').copy();depth_cmap.set_bad('white');error_cmap=plt.get_cmap('magma').copy();error_cmap.set_bad('white')
    for col,view in enumerate(train):
        for row,image in enumerate((estimated[view],truth[view],np.abs(estimated[view]-truth[view]))):
            ax=axes[row,col];ax.imshow(np.ma.masked_where(~valid[view],image),cmap=depth_cmap if row<2 else error_cmap,vmin=lo if row<2 else 0,vmax=hi if row<2 else .1)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Train view {view}')
            if col==0:ax.set_ylabel(('Estimate','GT diagnostic','Absolute error')[row])
    fig.suptitle('Identical supervised pixels | all4 arms privileged by GT intersection mask',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94));dest=OUT/'figures/targets'/f'{index:03d}_{seed}.jpg';dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=125);plt.close(fig);outputs['targets']=str(dest)
    return dict(index=index,seed=seed,**outputs,all_arms_privileged=True,same_cameras_and_frame=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--indices',default='5,6,10,11');p.add_argument('--seeds',default='42,43');a=p.parse_args();setup(a.gpu)
    import nvdiffrast.torch as dr
    device=torch.device('cuda:0');context=dr.RasterizeCudaContext(device=device);path=OUT/'figures/manifest.json';manifest=read(path) if path.exists() else dict(cases={})
    for index in map(int,a.indices.split(',')):
        for seed in map(int,a.seeds.split(',')):
            key=f'{index:03d}_{seed}'
            if key in manifest['cases']:continue
            value=one(index,seed,context,device)
            if value:manifest['cases'][key]=value;write(path,manifest);print(dict(figure=key),flush=True)
    manifest.update(time=now(),complete=len(manifest['cases'])==8,expected=8);write(path,manifest)


if __name__=='__main__':main()
