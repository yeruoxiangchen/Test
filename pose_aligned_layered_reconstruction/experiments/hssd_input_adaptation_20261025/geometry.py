"""Pixel-center-correct crop/resize; no fitted object transform or GT surface crop."""
import itertools
import math
import numpy as np
from PIL import Image

SIDE=518

def resize_affine(scale):
    return np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1.]],dtype=np.float64)

def crop_affine(box,output_side=SIDE):
    left,top,right,bottom=map(int,box)
    if right-left<=0 or right-left!=bottom-top:raise ValueError('Need a positive square crop')
    scale=output_side/(right-left)
    # PIL resize maps centers using (u + .5)*scale - .5.
    return np.array([[scale,0,scale*(.5-left)-.5],
                     [0,scale,scale*(.5-top)-.5],[0,0,1.]],dtype=np.float64)

def projected_cube_box(k,w,margin=1.1):
    corners=np.array(list(itertools.product([-.5,.5],repeat=3)),dtype=float)
    camera=np.c_[corners,np.ones(8)]@np.asarray(w).T
    if camera[:,2].min()<=.001:raise ValueError('Canonical cube crosses camera plane')
    uv=camera[:,:3]@np.asarray(k).T;uv=uv[:,:2]/uv[:,2:]
    low,high=uv.min(0),uv.max(0)
    side=min(SIDE,int(math.ceil(margin*float(np.max(high-low+1)))))
    center=(low+high)/2
    left,top=np.floor(center+.5-side/2).astype(int)
    # Stay inside the actual photo. Otherwise padded pixels would incorrectly
    # be considered observed by the frozen model's frustum-only validity test.
    left,top=np.clip([left,top],0,SIDE-side).astype(int)
    return [int(left),int(top),int(left+side),int(top+side)]

def crop_image(image,box,*,mask=False):
    out=image.crop(tuple(box))
    method=Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
    return out.resize((SIDE,SIDE),method)

def projection_check(k,w,box):
    points=np.array([[0,0,0],[.2,-.1,.1],[-.25,.15,-.2]])
    camera=np.c_[points,np.ones(3)]@np.asarray(w).T
    uv=camera[:,:3]@np.asarray(k).T;uv=uv[:,:2]/uv[:,2:]
    affine=crop_affine(box)
    expected=np.c_[uv,np.ones(3)]@affine.T
    other=camera[:,:3]@(affine@k).T;other=other[:,:2]/other[:,2:]
    return float(np.abs(expected[:,:2]-other).max())
