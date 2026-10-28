"""Diagnose subpixel interpolation in the new localization raster buffers."""
import os
from pathlib import Path
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('CUDA_HOME','/usr/local/cuda-12.6')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/home/zjr/Tracker/visual_evidence_probe/outputs/torch_extensions')
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')

import numpy as np
import torch

from visual_evidence_probe.runtime import setup
from visual_evidence_probe.experiments.localization.config import folder,observation,write,now,read,verify
from visual_evidence_probe.experiments.localization.geometry import buffers
from visual_evidence_probe.experiments.localization.scores import project


def main():
    import trimesh
    import nvdiffrast.torch as dr
    from scipy.ndimage import binary_erosion
    setup();context=dr.RasterizeCudaContext(device=torch.device('cuda:0'))
    _,k,w,_,_=observation(0)
    m=trimesh.load(verify(read(folder(0)/'fitted/42.json')['mesh']),force='mesh',process=False)
    b=buffers(m,k,w,context,128)
    ax=(torch.arange(128,device='cuda:0')+.5)*518/128-.5
    yy,xx=torch.meshgrid(ax,ax,indexing='ij');expected=torch.stack((xx,yy),-1)
    rows=[]
    for i in range(len(k)):
        uv,_=project(b['xyz'][i],k[i].cuda(),w[i].cuda())
        diff=(uv-expected).cpu().numpy()
        mask=b['mask'][i].cpu().numpy()
        inner=binary_erosion(mask,iterations=2)
        for name,valid in [('all',mask),('interior',inner)]:
            values=np.abs(diff[valid])
            rows.append(dict(view=i,domain=name,count=int(valid.sum()),signed_mean=diff[valid].mean(0).tolist(),
                abs_quantiles=np.quantile(values,[0,.5,.9,.99,.999,1]).tolist()))
    write(folder(0)/'projection_audit_v1.json',dict(time=now(),rows=rows))
    print(rows,flush=True)


if __name__=='__main__':main()
