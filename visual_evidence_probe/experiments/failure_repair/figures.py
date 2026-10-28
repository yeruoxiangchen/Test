import argparse
import gc

from .config import OUT, PARAMS, folder, data_folder, bind_engine, setup, read, write, now


def case(index, seed, context, device):
    import numpy as np
    import torch
    from visual_evidence_probe.experiments.dense_evidence.evaluation import render_depth
    from visual_evidence_probe.experiments.dense_evidence.data import observations
    from visual_evidence_probe.experiments.dense_evidence.figures import input_panel
    from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj
    from visual_evidence_probe.experiments.geometry_evidence.metrics import fixed_depth_error
    from visual_evidence_probe.experiments.editor_support.figures import _images, _grid, _plotting
    root = folder(index,seed); result = read(root / 'full/result.json')
    camera = read(data_folder(index) / 'novel_cameras.json')
    k = torch.tensor(camera['intrinsics'],device=device,dtype=torch.float32)
    w = torch.tensor(camera['extrinsics'],device=device,dtype=torch.float32)
    cache = torch.load(data_folder(index) / 'evaluation_targets.pt',map_location=device,weights_only=False)
    gt = cache['novel']
    with np.load(root / 'novel_region.npz') as a:
        region = a['region']
    views = np.argsort(-region.sum((1,2)),kind='stable')[:3].tolist()
    rows = [dict(label='GT scan\nEvaluation only',pred={q:gt[q].cpu() for q in ('depth','mask','xyz')})]
    for label,path in [('Raw30K\nBefore input alignment',root/'native_original.obj'),
                       ('Aligned30K\nShared repair base',root/'original.obj'),
                       ('Full procedure\nDelivered result',root/'full/accepted.obj'),
                       ('Training-selected\nCandidate only',root/'full/candidate.obj')]:
        mesh = load_obj(path); pred = render_depth(mesh,k,w,context)
        rows.append(dict(label=label,pred={q:pred[q].cpu() for q in ('depth','mask','xyz')}))
    boxes=[]
    for view in views:
        mask=np.logical_or.reduce([row['pred']['mask'][view].numpy() for row in rows])
        yy,xx=np.nonzero(mask)
        if len(yy):
            width=min(256,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*1.15)+4)
            x0=max(0,min(256-width,int((xx.min()+xx.max()-width)/2)))
            y0=max(0,min(256-width,int((yy.min()+yy.max()-width)/2)))
            boxes.append((x0,y0,x0+width,y0+width))
        else:boxes.append((0,0,256,256))
    # One crop per camera shared by all rows; no per-method scale or recentering.
    for row in rows:
        row['images']=[im[y0:y1,x0:x1] for im,(x0,y0,x1,y1) in zip(_images(row['pred'],views),boxes)]
    dest = OUT / 'figures/meshes' / f'{index:03d}_{seed}.jpg'
    _grid(rows,views,f'{result["uid"]}, seed {seed}',
          'Same novel cameras | raw, aligned base, and delivered correction',dest,
          'Candidate is diagnostic; unchanged delivered outputs are retained. GT used only for evaluation.',
          view_labels=[f'New view {v}' for v in views])
    base_error = fixed_depth_error(rows[2]['pred'],rows[0]['pred']).numpy()
    full_error = fixed_depth_error(rows[3]['pred'],rows[0]['pred']).numpy()
    domain = (rows[2]['pred']['mask'] | rows[0]['pred']['mask']).numpy()
    cap = max(float(np.quantile(base_error[domain],.95)),.005) if domain.any() else .02
    plt = _plotting(); fig,axes = plt.subplots(3,3,figsize=(9,8.4),squeeze=False)
    for ri, (label, array, cmap, lo, hi) in enumerate([
        ('Base error',base_error,'magma',0,cap), ('Delivered error',full_error,'magma',0,cap),
        ('Improvement (+)',base_error-full_error,'RdBu',-cap,cap)]):
        for ci,view in enumerate(views):
            ax=axes[ri,ci]; im=ax.imshow(np.ma.masked_where(~domain[view],array[view]),cmap=cmap,vmin=lo,vmax=hi)
            x0,y0,x1,y1=boxes[ci];ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
            ax.set_xticks([]);ax.set_yticks([])
            if ci==0:ax.set_ylabel(label)
            if ri==0:ax.set_title(f'New view {view}')
    fig.suptitle(f'{result["uid"]} / {seed}: fixed-domain depth errors')
    fig.tight_layout(); error_path = OUT/'figures/errors'/dest.name; error_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(error_path,dpi=140);plt.close(fig)
    _,_,_,images,_ = observations(index,device)
    with np.load(root/'region_sealed.npz') as a:
        inputs = input_panel(index,seed,images,a['region'],a['train_views'].tolist(),a['heldout_views'].tolist())
    return dict(index=index,seed=seed,meshes=str(dest),errors=str(error_path),inputs=inputs,
                figure_version=2,shared_crop_per_camera=boxes)


def scatter():
    from collections import defaultdict
    import numpy as np
    from visual_evidence_probe.experiments.editor_support.figures import _plotting
    from .report import paired_gain
    if not (OUT/'summary.json').exists():return
    summary=read(OUT/'summary.json'); grouped=defaultdict(list)
    for row in summary['records']:grouped[row['uid']].append(row)
    if not grouped:return
    plt=_plotting();fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for uid,rows in sorted(grouped.items()):
        x=float(np.mean([r['original_metrics']['surface']['chamfer_l1'] for r in rows]))
        cd=float(np.mean([paired_gain(r,'surface.chamfer_l1',relative=True) for r in rows]))*100
        suspicion=float(np.mean([r['input_score']['score'] for r in rows]))
        axes[0].scatter(x,cd);axes[0].annotate(str(rows[0]['index']),(x,cd),fontsize=7)
        axes[1].scatter(suspicion,cd);axes[1].annotate(str(rows[0]['index']),(suspicion,cd),fontsize=7)
    for ax in axes:ax.axhline(0,color='gray',lw=.8);ax.set_ylabel('Delivered CD reduction (%)');ax.grid(alpha=.2)
    axes[0].set_xlabel('Aligned-base CD (GT evaluation only)');axes[1].set_xlabel('Pre-edit input suspicion (no GT)')
    fig.suptitle('Round10: object means over two seeds; all outputs retained')
    fig.tight_layout();dest=OUT/'figures/initial_error_vs_gain.png';dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--indices',required=True);a=p.parse_args()
    bind_engine();setup(a.gpu)
    import torch
    import nvdiffrast.torch as dr
    device=torch.device('cuda:0');context=dr.RasterizeCudaContext(device=device)
    path=OUT/'figures/manifest.json';manifest=read(path) if path.exists() else dict(cases={})
    with torch.no_grad():
        for index in map(int,a.indices.split(',')):
            for seed in PARAMS['seeds']:
                key=f'{index:03d}_{seed}'
                if key in manifest['cases'] and manifest['cases'][key].get('figure_version')==2:continue
                manifest['cases'][key]=case(index,seed,context,device)
                write(path,manifest);print(dict(stage='figure_complete',case=key),flush=True)
                gc.collect();torch.cuda.empty_cache()
    scatter();manifest.update(time=now(),complete=len(manifest['cases'])==32);write(path,manifest)


if __name__ == '__main__':
    main()
