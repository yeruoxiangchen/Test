import numpy as np
from PIL import Image,ImageDraw

from .config import folder


def colors(score,valid):
    import matplotlib
    cmap=matplotlib.colormaps['turbo']
    valid=np.asarray(valid,bool)
    vals=score[valid]
    low,high=np.quantile(vals,[.05,.95]) if len(vals) else (0.,1.)
    norm=np.clip((score-low)/max(high-low,1.e-6),0,1)
    rgb=(cmap(norm)[...,:3]*255).astype(np.uint8)
    rgb[~valid]=235
    return rgb


def visualize(index,key,maps,gt_depth,gt_mask,error,images,mesh,k,w,context):
    from visual_evidence_probe.render import render
    renders,_=render(mesh,k,w,side=128,context=context)
    columns=('input','render','GT error >.02','boundary','direct_rgb','direct_dino','cross_dino','shuffled_dino','fused_dino','support')
    size,header=160,28
    sheet=Image.new('RGB',(size*len(columns),(size+header)*len(images)+32),'white')
    draw=ImageDraw.Draw(sheet)
    draw.text((5,5),'Heatmap colors: each panel 5-95% input-score range; gray = outside support; GT shown only for evaluation.',fill='black')
    for i,source in enumerate(images):
        union=maps['pred_mask'][i]|maps['observed_mask'][i]
        err=np.full((*union.shape,3),235,np.uint8)
        err[union]=[62,158,91];err[union&error[i]]=[220,45,45]
        panels=[source,renders[i],Image.fromarray(err)]
        for name in columns[3:-1]:
            domain=maps['common'][i] if name in ('cross_dino','shuffled_dino','fused_dino') else union
            panels.append(Image.fromarray(colors(maps[name][i],domain)))
        panels.append(Image.fromarray(colors(maps['support'][i],union)))
        y=32+i*(size+header)
        for j,(name,image) in enumerate(zip(columns,panels)):
            draw.text((j*size+4,y+4),name,fill='black')
            sheet.paste(image.resize((size,size),Image.Resampling.NEAREST if j>=2 else Image.Resampling.LANCZOS),(j*size,y+header))
    path=folder(index)/'figures'/f'{key}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(path,quality=93)


def voxelize(xyz,score,valid):
    points=np.asarray(xyz)[valid];values=np.asarray(score)[valid]
    cells=np.floor((points+.5)*64).astype(np.int32)
    inside=((cells>=0)&(cells<64)).all(1)
    cells,values=cells[inside],values[inside]
    if not len(cells):return np.empty((0,3),np.int32),np.empty(0,np.float32),np.empty(0,np.int32),int((~inside).sum())
    unique,inverse,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
    sums=np.bincount(inverse,weights=values,minlength=len(unique))
    return unique,(sums/counts).astype(np.float32),counts.astype(np.int32),int((~inside).sum())


def paint_mesh(mesh,maps,path):
    import trimesh
    from .config import write
    indices,values,counts,outside=voxelize(maps['pred_xyz'],maps['fused_dino'],maps['common'])
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),indices=indices,scores=values,counts=counts)
    cells=np.floor((np.asarray(mesh.vertices)+.5)*64).astype(np.int32)
    in_bounds=((cells>=0)&(cells<64)).all(1)
    volume=np.full((64,64,64),np.nan,np.float32)
    if len(indices):volume[tuple(indices.T)]=values
    vertex_scores=np.zeros(len(cells),np.float32)
    vertex_scores[in_bounds]=volume[tuple(cells[in_bounds].T)]
    observed=in_bounds&np.isfinite(vertex_scores)
    rgba=np.full((len(cells),4),[170,170,170,255],np.uint8)
    if observed.any():rgba[observed,:3]=colors(vertex_scores,observed)[observed]
    result=mesh.copy();result.visual=trimesh.visual.ColorVisuals(mesh=result,vertex_colors=rgba);result.export(path)
    write(path.with_suffix('.json'),dict(uses_gt=False,grid_resolution=64,grid_transform='identity',
        aggregation='mean of input-only fused evidence over observed predicted surface voxels',
        unknown='gray, no hidden-surface assertion',observed_voxels=len(indices),
        colored_vertices=int(observed.sum()),total_vertices=len(cells),out_of_cube_samples=outside))
