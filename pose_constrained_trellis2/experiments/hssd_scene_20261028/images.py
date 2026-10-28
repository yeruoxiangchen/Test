"""Pixel-centre camera contract for Pillow crop/resize of HSSD photographs."""
from pose_constrained_trellis2.prepare_pose_cache import _prepare_view

def prepare_view(*args,**kwargs):
    image,K,metadata=_prepare_view(*args,**kwargs)
    affine=K.new_tensor(metadata['source_to_feature'])
    scale=float(affine[0,0]);offset=(scale-1.)*.5
    # Pillow's mapping: dst=(src-left+0.5)*scale-0.5.
    K=K.clone();K[0,2]+=offset;K[1,2]+=offset
    affine=affine.clone();affine[0,2]+=offset;affine[1,2]+=offset
    metadata={**metadata,'source_to_feature':affine.tolist(),
              'pixel_centres':'dst=(src-crop_origin+0.5)*scale-0.5'}
    return image,K,metadata
