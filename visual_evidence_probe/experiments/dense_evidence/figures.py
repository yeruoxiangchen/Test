"""Same-frame scientific renders, input regions, and provider coverage."""
import argparse
import gc

import numpy as np
import torch

from .config import OUT,PARAMS,folder,data_folder,registry,setup,read,write,now
from .data import observations
from .evidence import observed_domain
from .evaluation import render_depth
from visual_evidence_probe.experiments.geometry_evidence.metrics import fixed_depth_error
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj
from visual_evidence_probe.experiments.editor_support.figures import _plotting,_images,_grid


def input_panel(index,seed,images,region,train,heldout,suffix=''):
    from PIL import Image,ImageDraw
    columns=min(len(images),4);rows=(len(images)+columns-1)//columns
    panel=Image.new('RGB',(columns*260,rows*286+28),'white');draw=ImageDraw.Draw(panel)
    for view,image in enumerate(images):
        rgb=np.array(image.resize((256,256)));selected=region[view]>.5
        rgb[selected]=(.75*rgb[selected]+.25*np.array([250,185,35])).astype(np.uint8)
        x,y=(view%columns)*260,(view//columns)*286
        panel.paste(Image.fromarray(rgb),(x,y+24))
        draw.text((x+3,y+5),f'View {view}: '+('HELD OUT' if view in heldout else 'TRAIN'),fill='black')
    draw.text((4,rows*286+5),'Yellow: sealed input-only local edit region. Heldout RGB excluded from DA3 and editing.',fill='black')
    dest=OUT/'figures/inputs'/f'{index:03d}_{seed}{suffix}.jpg';dest.parent.mkdir(parents=True,exist_ok=True);panel.save(dest,quality=93)
    return str(dest)


def depth_panel(index,seed,region,train,device):
    root=folder(index,seed);provider=read(root/'dense_provider.json')
    with np.load(provider['path']+'/prediction.npz') as a:depth,valid=a['depth'],a['valid']
    with np.load(root/'sparse_sealed.npz') as a:sparse,svalid=a['depth'][train],a['valid'][train]
    plt=_plotting();cols=len(train);fig,axes=plt.subplots(3,cols,figsize=(max(7,cols*2.2),7.2),squeeze=False)
    foreground=depth[valid];lo,hi=np.quantile(foreground,[.02,.98]) if len(foreground) else (0,1)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('white')
    for col,view in enumerate(train):
        for row,array,mask in ((0,depth[col],np.isfinite(depth[col])&(depth[col]>0)),
                (1,depth[col],valid[col]),(2,sparse[col],svalid[col])):
            ax=axes[row,col];ax.imshow(np.ma.masked_where(~mask,array),vmin=lo,vmax=hi,cmap=cmap)
            if region[view].any() and (region[view]<.5).any():ax.contour(region[view],levels=[.5],colors=['red'],linewidths=.6)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Train view {view}',fontsize=9)
            if col==0:ax.set_ylabel(('DA3 raw','DA3 consistent','NCC anchors')[row],fontsize=9)
    fig.suptitle(f'{index:03d}, seed {seed}: camera-z evidence (not GT)\nDense editing keeps NCC anchors and fills additional reliable DA3 rays',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.92));dest=OUT/'figures/evidence'/f'{index:03d}_{seed}.jpg';dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=125,pil_kwargs={'quality':92});plt.close(fig)
    return str(dest)


@torch.no_grad()
def case(index,seed,context,device):
    root=folder(index,seed)
    if not (root/'complete.json').exists():return None
    complete=read(root/'complete.json');camera=read(data_folder(index)/'novel_cameras.json')
    k=torch.tensor(camera['intrinsics'],device=device,dtype=torch.float32);w=torch.tensor(camera['extrinsics'],device=device,dtype=torch.float32)
    cache=torch.load(data_folder(index)/'evaluation_targets.pt',map_location=device,weights_only=False)
    gt=cache['novel']
    with np.load(root/'novel_region.npz') as a:region=a['region']
    with np.load(root/'region_sealed.npz') as a:input_region,train,heldout=a['region'],a['train_views'].tolist(),a['heldout_views'].tolist()
    views=np.argsort(-region.sum((1,2)),kind='stable')[:3].tolist()
    rows=[dict(label='GT scan\nEvaluation only',pred={key:gt[key].cpu() for key in ('depth','mask','xyz')})]
    sources=[('Frozen30K\nOriginal',root/'original.obj')]
    for arm,endpoint in (('sparse_observed','candidate'),('dense_observed','candidate'),
            ('sparse_observed','accepted'),('dense_observed','accepted')):
        result=read(root/arm/'result.json');step=result['candidate_step'] if endpoint=='candidate' else result['selected_step']
        before=result['original_metrics']['novel']['local_target_depth_mae'];after=result['metrics'][endpoint]['novel']['local_target_depth_mae']
        gain=f'{(before-after)/before:+.1%}' if before and after is not None else 'NA'
        evidence='NCC' if arm.startswith('sparse') else 'NCC + DA3'
        label=f'{evidence}: {endpoint}\nStep {step}; local depth {gain}'+('\nNO EDIT' if step==0 else '')
        sources.append((label,root/arm/f'{endpoint}.obj'))
    receipts=[]
    for label,path in sources:
        mesh=load_obj(path);pred=render_depth(mesh,k,w,context)
        rows.append(dict(label=label,pred={key:pred[key].cpu() for key in ('depth','mask','xyz')}))
        receipts.append(dict(path=str(path),resolved_path=str(path.resolve()),vertices=len(mesh.vertices),faces=len(mesh.faces)))
        del mesh,pred
    boxes=[]
    for view in views:
        mask=(rows[0]['pred']['mask'][view]|rows[1]['pred']['mask'][view]).numpy();yy,xx=np.nonzero(mask)
        if len(yy):
            width=min(256,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*1.15)+4)
            x0=max(0,min(256-width,int((xx.min()+xx.max()-width)/2)));y0=max(0,min(256-width,int((yy.min()+yy.max()-width)/2)))
            boxes.append((x0,y0,x0+width,y0+width))
        else:boxes.append((0,0,256,256))
    for row in rows:row['images']=[im[y0:y1,x0:x1] for im,(x0,y0,x1,y1) in zip(_images(row['pred'],views),boxes)]
    dest=OUT/'figures/meshes'/f'{index:03d}_{seed}.jpg'
    _grid(rows,views,f'{complete["uid"]}, seed {seed}',
        'Same new evaluation cameras and unchanged frame | candidates and accepted outputs shown separately',dest,
        'Accepted = revised input-only policy; unchanged outputs remain. GT never used to select edits.',
        view_labels=[f'New view {v}' for v in views])
    plt=_plotting();fig,axes=plt.subplots(5,3,figsize=(10.8,11.2),squeeze=False)
    domain=(region>.5)&rows[0]['pred']['mask'].numpy();base_error=fixed_depth_error(rows[1]['pred'],rows[0]['pred']).numpy()
    cap=max(float(np.quantile(base_error[domain],.95)),.005) if domain.any() else .02
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('white')
    for ri,row in enumerate(rows[1:]):
        error=fixed_depth_error(row['pred'],rows[0]['pred']).numpy()
        for ci,view in enumerate(views):
            ax=axes[ri,ci];im=ax.imshow(np.ma.masked_where(~rows[0]['pred']['mask'][view].numpy(),error[view]),cmap=cmap,vmin=0,vmax=cap)
            if region[view].any() and (region[view]<.5).any():ax.contour(region[view],levels=[.5],colors=['#33ee88'],linewidths=.5)
            x0,y0,x1,y1=boxes[ci];ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5);ax.set_xticks([]);ax.set_yticks([])
            if ri==0:ax.set_title(f'New view {view}',fontsize=9)
            if ci==0:ax.text(-.06,.5,row['label'],transform=ax.transAxes,ha='right',va='center',fontsize=8)
    fig.subplots_adjust(left=.28,right=.91,bottom=.045,top=.94,hspace=.03,wspace=.03)
    cax=fig.add_axes([.94,.25,.012,.48]);fig.colorbar(im,cax=cax,label='Absolute camera-z error')
    fig.suptitle(f'{complete["uid"]}, seed {seed}: fixed-target errors\nGreen = input-derived local region; every row shares the same scale',fontsize=10)
    fig.text(.5,.012,f'Missing rays pay 0.1; saturates above {cap:.4f}. No GT alignment.',ha='center',fontsize=8)
    errors=OUT/'figures/errors'/dest.name;errors.parent.mkdir(parents=True,exist_ok=True);fig.savefig(errors,dpi=125,pil_kwargs={'quality':93});plt.close(fig)
    rec,_,_,images,_=observations(index,device)
    inputs=input_panel(index,seed,images,input_region,train,heldout)
    raw_inputs=None
    if rec['kind']=='real_scene' and rec.get('raw_images'):
        from PIL import Image
        raw=[Image.open(item['path']).convert('RGB') for item in rec['raw_images']]
        raw_inputs=input_panel(index,seed,raw,input_region,train,heldout,suffix='_raw_scene')
    evidence=depth_panel(index,seed,input_region,train,device)
    del cache,gt,rows;gc.collect();torch.cuda.empty_cache()
    return dict(index=index,seed=seed,meshes=str(dest),errors=str(errors),inputs=inputs,raw_inputs=raw_inputs,evidence=evidence,
        same_cameras_and_frame=True,views=views,sources=receipts,color_scale_max=cap)


def summary():
    source=read(OUT/'summary.json');plt=_plotting();fig,axes=plt.subplots(2,3,figsize=(12,7.2))
    colors=['#a9b9c6','#4f7c9c','#dfba8d','#c66b2d']
    for row,split in enumerate(('development','validation')):
        for ax,(key,title,scale) in zip(axes[row],(
                ('input.local_depth_relative_reduction','Input local depth reduction (%)',100),
                ('novel.local_depth_relative_reduction','New-view local depth reduction (%)',100),
                ('surface.chamfer_l1','Global Chamfer improvement',1))):
            cases=source['groups'][split]['dense_observed']['cases']
            if not cases:
                ax.text(.5,.5,'No evaluated cases yet',ha='center',va='center',transform=ax.transAxes)
                ax.set_title(f'{split}: {title}',fontsize=9);ax.set_xticks([]);ax.set_yticks([]);continue
            stats=[source['groups'][split][arm]['endpoints']['accepted'][key] for arm in ('sparse_legacy','sparse_observed','dense_legacy','dense_observed')]
            values=np.array([s['mean'] or 0 for s in stats])*scale
            low=np.array([(s['ci95'] or [0,0])[0] for s in stats])*scale;high=np.array([(s['ci95'] or [0,0])[1] for s in stats])*scale
            ax.bar(range(4),values,color=colors,yerr=np.maximum(np.stack((values-low,high-values)),0),capsize=3)
            ax.axhline(0,color='#555',lw=.7);ax.set_xticks(range(4),['NCC\nold','NCC\nrevised','+DA3\nold','+DA3\nrevised']);ax.set_title(f'{split}: {title}',fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'Seventh probe: accepted edits, frozen30K | {source["states"]}/{source["expected_states"]} records\nNo-ops retained; object bootstrap95% CI',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));dest=OUT/'figures/summary.jpg';dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=160,pil_kwargs={'quality':94});plt.close(fig);return str(dest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--indices');p.add_argument('--seeds',default='42,43');p.add_argument('--overwrite',action='store_true');a=p.parse_args();setup(a.gpu)
    import nvdiffrast.torch as dr
    device=torch.device('cuda:0');context=dr.RasterizeCudaContext(device=device)
    dest=OUT/'figures/manifest.json';manifest=read(dest) if dest.exists() else dict(cases={})
    indices=list(map(int,a.indices.split(','))) if a.indices else [r['index'] for r in registry()['rows']]
    for index in indices:
        for seed in map(int,a.seeds.split(',')):
            key=f'{index:03d}_{seed}'
            if key in manifest['cases'] and not a.overwrite:continue
            result=case(index,seed,context,device)
            if result:manifest['cases'][key]=result;write(dest,manifest);print(dict(stage='figure_complete',case=key),flush=True)
    if (OUT/'summary.json').exists():manifest['summary']=summary()
    manifest.update(time=now(),complete=len(manifest['cases'])==len(registry()['rows'])*2,expected=len(registry()['rows'])*2);write(dest,manifest)


if __name__=='__main__':main()
