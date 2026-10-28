import numpy as np
from PIL import Image
import torch

from .config import data_folder, folder, read


def observations(index, device):
    record = read(data_folder(index)/'observations.json')
    images = [Image.open(r['path']).convert('RGB') for r in record['images']]
    masks = np.stack([np.asarray(Image.open(r['path']).convert('L')) > 127 for r in record['masks']])
    return record, torch.tensor(record['intrinsics'],device=device,dtype=torch.float32), \
        torch.tensor(record['extrinsics'],device=device,dtype=torch.float32), images, masks


def load_latent(index, seed, device):
    from trellis.modules.sparse import SparseTensor
    saved = torch.load(folder(index,seed)/'original_latent.pt',map_location=device,weights_only=False)
    return SparseTensor(feats=saved['features'],coords=saved['coords'])


def fit_transform(index,seed,device):
    return torch.tensor(read(folder(index,seed)/'fit.json')['matrix'],device=device,dtype=torch.float32)


def model_input(index):
    from pose_aligned_layered_reconstruction.dino_context import build_masked_dino_contexts
    rec = read(data_folder(index)/'observations.json')
    cache = torch.load(data_folder(index)/'features.pt',map_location='cpu',weights_only=False)
    features = cache['features'].float()
    return dict(intrinsics=torch.tensor(rec['intrinsics'],dtype=torch.float32),
        extrinsics=torch.tensor(rec['extrinsics'],dtype=torch.float32),
        image_size=(518,518),grid_transform='identity',extrinsics_type='w2c',camera_forward_sign=1.,
        **build_masked_dino_contexts(features),visual_patch_features=features)
