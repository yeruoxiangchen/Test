"""Compare actual repaired extraction to the installed encoder public forward."""
import os
from pose_constrained_trellis2.experiments.hssd_scene_20261028 import common as c
os.environ.update(c.environment())
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pose_constrained_trellis2.experiments.hssd_scene_20261028.dino import NativeDinoV3FeatureExtractor

with torch.no_grad():
    row=c.plan()['plan']['objects'][0];p=c.DATA/'objects'/row['uid']/f"input_{row['views'][0]['frame']:03d}.png"
    image=Image.open(p).convert('RGB');x=torch.tensor(np.asarray(image)/255.,dtype=torch.float32).permute(2,0,1)[None].cuda()
    ext=NativeDinoV3FeatureExtractor(c.models()['dino'],image_size=512);ext.to('cuda')
    x=ext.transform(x);got=ext.extract_features(x)
    h=ext.model.embeddings(x,bool_masked_pos=None);positions=ext.model.rope_embeddings(x)
    raw=ext.model.model(h,position_embeddings=positions).last_hidden_state
    expected=F.layer_norm(raw,raw.shape[-1:])
    torch.testing.assert_close(got,expected,rtol=0,atol=0)
    assert got.shape==(1,1029,1024) and torch.isfinite(got).all()
    c.write(c.OUT/'dino_compatibility.json',{'time':c.now(),'passed':True,'shape':list(got.shape),
        'maximum_difference':float((got-expected).abs().max()),'normalization':'functional LayerNorm on prenorm tokens; no affine final norm'})
    print('DINO real-image GPU compatibility PASS',list(got.shape))
