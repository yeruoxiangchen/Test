"""Freeze object-disjoint splits and build native targets + posed DINO caches.

No old flow checkpoint or old SLat features are consumed. The older SS cache
supplies GT occupancy coordinates and the rendering transform, never its z.
SS is freshly encoded with the T2 pipeline's native SS encoder.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import zipfile
import numpy as np
from . import common as c


def native_header(path):
    with zipfile.ZipFile(path) as z:
        with z.open('coords.npy') as f:
            ver = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format._read_array_header(f, ver)
    return shape


def voxel_projection_radius(camera,intrinsic,resolution=64):
    """Conservative pixel uncertainty for projecting an occupied voxel centre.

    A real surface point lies within half a voxel's diagonal. Propagate this
    bound through perspective division, then add one rasterization pixel.
    This validates geometry without rejecting thin parts merely because
    centre pixels fall outside their narrow silhouettes.
    """
    xyz=np.asarray(camera,dtype=np.float64)
    k=np.asarray(intrinsic,dtype=np.float64)
    delta=np.sqrt(3.)/(2*resolution)
    z=xyz[:,2]
    valid=z>max(1e-4,delta*2)
    if not valid.any():
        raise ValueError('No positive-depth points for camera validation')
    xyz,z=xyz[valid],z[valid]
    bx=abs(k[0,0])*delta*(z+np.abs(xyz[:,0]))/(z*(z-delta))
    by=abs(k[1,1])*delta*(z+np.abs(xyz[:,1]))/(z*(z-delta))
    radius=max(1,int(np.ceil(max(bx.max(),by.max())))+1)
    if radius>32:
        raise ValueError('Camera lies outside this fixed-renderer projection-validation range')
    return radius


def freeze():
    if (c.DATA / 'plan.json').exists():
        return c.plan()
    splits, rejected = {}, []
    for split in ['train', 'val']:
        payload = c.read(c.SOURCE / f'{split}.json')
        if payload['shape_encoder'] != 'microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16':
            raise ValueError('Source is not native TRELLIS.2 Shape')
        unique = {}
        for row in payload['samples']:
            # Select the first frozen trajectory, never the best model result.
            unique.setdefault(row['object_uid'], row)
        good = []
        for uid, row in unique.items():
            shape = native_header(row['shape_latent'])
            if not (len(shape) == 2 and shape[1] == 3 and 0 < shape[0] <= 8192):
                rejected.append({'uid': uid, 'reason': 'native512_token_limit', 'shape': shape})
                continue
            if not all(Path(row[k]).is_file() for k in ['shape_latent', 'ss_latent', 'source_glb']):
                rejected.append({'uid': uid, 'reason': 'missing_local_source'})
                continue
            good.append({**row, 'native_shape_tokens': shape[0]})
        splits[split] = sorted(good, key=lambda r: c.digest(['repair_round1', r['object_uid']]))
    assert len(splits['train']) >= 2000 and len(splits['val']) == 64
    rows = [{**r, 'split': 'train', 'overfit32': i < 32} for i, r in enumerate(splits['train'][:2000])]
    rows += [{**r, 'split': 'dev', 'overfit32': False} for r in splits['val']]
    assert not ({r['object_uid'] for r in rows if r['split'] == 'train'} &
                {r['object_uid'] for r in rows if r['split'] == 'dev'})
    value = {'version': 'pose_conditioned_geometry_repair.native_objaverse.v1',
             'seed': 20261029, 'train_objects': 2000, 'dev_objects': 64,
             'overfit_objects': 32, 'samples': rows, 'rejected': rejected,
             'source_manifests': {k: {'path': str(c.SOURCE / f'{k}.json'),
                                     'sha256': c.sha(c.SOURCE / f'{k}.json')} for k in ['train', 'val']},
             'native_shape_contract': '32 channels; 32^3; <=8192; cached native T2 encoder posterior sample',
             'ss_contract': 'fresh T2 SS encoding of source GT occupancy64; NOT old cache z',
             'independence': 'UID-disjoint adaptation Dev64; pretrained base training overlap is unknown',
             'selection': 'frozen hash order; no model, GT metric, or manual visual selection'}
    record = {'identity': c.digest(value), 'plan': value}
    c.write(c.DATA / 'plan.json', record, immutable=True)
    c.write(c.OUT / 'models.json', c.MODEL_PATHS, immutable=True)
    c.write(c.OUT / 'source_at_setup.json', c.source_identity())
    print({'train': 2000, 'dev': 64, 'overfit': 32, 'rejected': len(rejected)}, flush=True)
    return record


def formal_candidates():
    path=c.DATA/'formal_candidates.json'
    if path.exists():
        value=c.read(path)
        assert c.digest(value['plan'])==value['identity']
        return value
    parent=c.plan()
    overfit={r['object_uid'] for r in parent['plan']['samples'] if r['overfit32']}
    samples=[]
    for source_split in ['train','val']:
        original=c.SOURCE/f'{source_split}.json'
        assert c.sha(original)==parent['plan']['source_manifests'][source_split]['sha256']
        unique={}
        for row in c.read(original)['samples']:
            unique.setdefault(row['object_uid'],row)
        for uid,row in unique.items():
            if not all(Path(row[k]).is_file() for k in ['shape_latent','ss_latent','source_glb']):
                continue
            shape=native_header(row['shape_latent'])
            if len(shape)==2 and shape[1]==3 and 0<shape[0]<=8192:
                samples.append({**row,'split':'train' if source_split=='train' else 'dev',
                    'source_split':source_split,'overfit32':uid in overfit,'native_shape_tokens':shape[0]})
    samples.sort(key=lambda r:(r['split']!='train',c.digest(['repair_round1',r['object_uid']])))
    payload={'version':'native_objaverse.formal_validity_pool.v1','parent_plan_identity':parent['identity'],
        'samples':samples,'selection_rule':'Fixed hash order; first 2000 valid source-train; prefer original valid Dev64, fill from unused valid reserves if necessary',
        'validity':'Native schemas and mean voxel-tolerant mask support >=0.8; no trained-model score',
        'no_metric_selection':True,'old_results_and_parent_plan_preserved':True}
    value={'identity':c.digest(payload),'plan':payload}
    c.write(path,value,immutable=True)
    return value


def choose_formal_uids(valid_train,valid_dev,overfit,train_count=2000,dev_count=64):
    if len(valid_train)<train_count:
        raise ValueError('Insufficient valid native training objects')
    train=list(valid_train[:train_count])
    dev=list(valid_dev[:dev_count])
    reserves=[uid for uid in valid_train[train_count:] if uid not in set(overfit)]
    dev+=reserves[:dev_count-len(dev)]
    if len(dev)!=dev_count or set(train)&set(dev) or set(dev)&set(overfit):
        raise ValueError('Cannot construct a disjoint valid formal split')
    return train,dev


def cached_visibility(row,folder,meta):
    if 'projection_tolerant_support' in meta:
        return float(np.mean(meta['projection_tolerant_support']))
    if float(np.mean(meta['projection_mask_support']))>=.8:
        # Max-filtered support is bounded below by the measured raw support.
        return float(np.mean(meta['projection_mask_support']))
    auxiliary=folder/'visibility_validation_v2.json'
    if auxiliary.exists():
        return c.read(auxiliary)['mean_tolerant_support']
    from PIL import Image
    from scipy.ndimage import maximum_filter
    with np.load(row['ss_latent'],allow_pickle=False) as payload:
        xyz=np.asarray(payload['target_coords'],dtype=np.float64)
    points=np.c_[(xyz+.5)/64-.5,np.ones(len(xyz))]
    scores=[]
    for view in row['views'][:8]:
        camera=(points@np.asarray(view['extrinsic_w2c']).T)[:,:3]
        k=np.asarray(view['intrinsic_normalized'],dtype=float);k[:2]*=512
        uv=camera[:,:2]/camera[:,2:3]*np.array([k[0,0],k[1,1]])+k[:2,2]
        valid=(camera[:,2]>1e-5)&(uv[:,0]>=0)&(uv[:,0]<=511)&(uv[:,1]>=0)&(uv[:,1]<=511)
        q=np.rint(uv[valid]).astype(int)
        radius=voxel_projection_radius(camera[valid],k)
        mask=np.asarray(Image.open(view['mask']).convert('L'))/255.
        dilated=maximum_filter(mask,size=radius*2+1,mode='constant')
        scores.append(float(dilated[q[:,1],q[:,0]].mean()))
    value=float(np.mean(scores))
    c.write(auxiliary,{'time':c.now(),'mean_tolerant_support':value,'support':scores,
        'existing_pose_targets_and_hashes_unchanged':True},immutable=True)
    return value


def freeze_formal_selection():
    path=c.DATA/'formal_plan.json'
    if path.exists():
        value=c.read(path)
        assert c.digest(value['plan'])==value['identity']
        return value
    pool=formal_candidates()
    valid={'train':[],'dev':[]}
    rejected=[]
    for row in pool['plan']['samples']:
        folder=c.DATA/'objects'/row['object_uid']
        if (folder/'rejected.json').exists():
            rec=c.read(folder/'rejected.json')
            rejected.append({'uid':row['object_uid'],'reason':rec['reason'],
                             'mean_tolerant_support':rec['mean_tolerant_support']})
        elif (folder/'complete.json').exists():
            meta=c.read(folder/'complete.json')
            coverage=cached_visibility(row,folder,meta)
            if coverage>=.8:
                valid[row['split']].append(row['object_uid'])
            else:
                rejected.append({'uid':row['object_uid'],'reason':'Legacy cache failed the same formal visibility check',
                                 'mean_tolerant_support':coverage})
        else:
            raise ValueError('Candidate validation incomplete: '+row['object_uid'])
    overfit={r['object_uid'] for r in c.plan()['plan']['samples'] if r['overfit32']}
    train,dev=choose_formal_uids(valid['train'],valid['dev'],overfit)
    payload={'version':'native_objaverse.formal_validity_selection.v1',
        'parent_plan_identity':c.plan()['identity'],'candidate_pool_identity':pool['identity'],
        'splits':{'train':train,'dev':dev},'counts':{'train':2000,'dev':64},
        'excluded':rejected,'dev_reserves_used':max(0,64-len(valid['dev'])),
        'selection':'Data-validity screening before formal training, fixed hash order; no model metric selection',
        'overfit_uids_in_dev':0,'formal_initialization':'fresh official T2; no overfit checkpoint initialization'}
    value={'identity':c.digest(payload),'plan':payload}
    c.write(path,value,immutable=True)
    return value


class DinoCompatibility:
    @staticmethod
    def load():
        import torch.nn.functional as F
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

        class Extractor(DinoV3FeatureExtractor):
            def extract_features(self, image):
                image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
                h = self.model.embeddings(image, bool_masked_pos=None)
                positions = self.model.rope_embeddings(image)
                layers = getattr(self.model, 'layer', None)
                if layers is None:
                    layers = self.model.model.layer
                for layer in layers:
                    h = layer(h, position_embeddings=positions)
                return F.layer_norm(h, h.shape[-1:])

        extractor = Extractor(c.MODEL_PATHS['dino'], image_size=512)
        extractor.to('cuda')
        return extractor


def atomic_torch_save(value, path):
    import torch
    path = Path(path).resolve()
    if not path.is_relative_to(c.ROOT):
        raise ValueError('External output path')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def prepare_objects(subset, rank=0, world=1):
    import torch
    from PIL import Image
    from trellis2 import models
    from .corruptions import KINDS, corrupt_occupancy, pair_seed
    from .vendor_pose.constants import SAMPLE_FORMAT
    from .vendor_pose.prepare_pose_cache import prepare_rgba_view
    from .vendor_pose.geometry import projection_geometry

    rows=(formal_candidates()['plan']['samples'] if subset=='all' else c.plan()['plan']['samples'])
    if subset == 'overfit':
        rows = [r for r in rows if r['overfit32']]
    elif subset == 'dev':
        rows = [r for r in rows if r['split'] == 'dev']
    elif subset == 'train':
        rows = [r for r in rows if r['split'] == 'train']
    elif subset != 'all':
        raise ValueError(subset)
    rows = rows[rank::world]
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    encoder = models.from_pretrained(c.MODEL_PATHS['ss_encoder']).eval().cuda()
    decoder = models.from_pretrained(c.MODEL_PATHS['ss_decoder']).eval().cuda()
    dino = None
    for i, row in enumerate(rows):
        uid = row['object_uid']
        folder = c.DATA / 'objects' / uid
        folder.mkdir(parents=True, exist_ok=True)
        done = folder / 'complete.json'
        if (folder/'rejected.json').exists():
            continue
        if done.exists():
            if c.read(done)['plan_identity'] != c.plan()['identity']:
                raise ValueError('Object belongs to a different frozen plan')
            continue
        print(c.now(), 'prepare', subset, rank, i+1, len(rows), uid, flush=True)
        # Only occupancy and canonical metadata are read from the old SS cache.
        # Its old z is deliberately never accessed.
        with np.load(row['ss_latent'], allow_pickle=False) as old:
            occ_coords = np.asarray(old['target_coords'], dtype=np.int64)
            old_center = np.asarray(old['normalize_center'])
            old_scale = float(old['normalize_scale'])
        if occ_coords.ndim != 2 or occ_coords.shape[1] != 3 or not len(occ_coords) or occ_coords.min() < 0 or occ_coords.max() >= 64:
            raise ValueError('Invalid GT occupancy64')
        with np.load(row['shape_latent'], allow_pickle=False) as shape:
            xyz = np.asarray(shape['coords'], dtype=np.int64)
            feats = np.asarray(shape['feats'])
            if feats.shape != (len(xyz), 32) or not np.isfinite(feats).all():
                raise ValueError('Invalid native Shape target')
            assert np.allclose(shape['normalize_center'], old_center, atol=1e-5)
            assert abs(float(shape['normalize_scale']) - old_scale) <= max(1e-5, abs(old_scale)*2e-5)
        clean = torch.zeros((1, 1, 64, 64, 64), device='cuda')
        q = torch.as_tensor(occ_coords, device='cuda').long()
        clean[0, 0, q[:, 0], q[:, 1], q[:, 2]] = 1
        z = encoder(clean, sample_posterior=False)
        rec = decoder(z) > 0
        codec_iou = float(((rec & clean.bool()).sum() / (rec | clean.bool()).sum().clamp_min(1)).item())
        if not torch.isfinite(z).all() or not rec.any():
            raise ValueError('Invalid SS roundtrip')
        np.savez_compressed(folder / 'ss.npz', z=z[0].float().cpu().numpy())
        np.savez_compressed(folder / 'occupancy.npz', coords=occ_coords.astype(np.uint8))
        pairs = []
        for variant, kind in enumerate(KINDS):
            seed = pair_seed(uid, variant)
            bad, bounds, region, changed = corrupt_occupancy(occ_coords, seed, kind)
            zb = encoder(torch.from_numpy(bad.astype(np.float32))[None, None].cuda(), sample_posterior=False)
            assert torch.isfinite(zb).all()
            pairs.append({'kind': kind, 'seed': seed, 'bounds64': torch.from_numpy(bounds),
                          'bad_ss': zb[0].float().cpu(), 'bad_occ64': torch.from_numpy(bad),
                          'region64': torch.from_numpy(region), 'changed_cells': changed})
        atomic_torch_save({'pairs': pairs, 'clean_ss': z[0].float().cpu(),
                           'clean_occ64': clean[0, 0].bool().cpu()}, folder / 'pairs.pt')
        if dino is None:
            dino = DinoCompatibility.load()
        all_features,intrinsics,extrinsics,prep_meta,mask_support=[],[],[],[],[]
        tolerant_support,projection_radii=[],[]
        unusable_view=None
        for view_idx, view in enumerate(row['views'][:8]):
            image = np.asarray(Image.open(view['image']).convert('RGBA'))
            mask = np.asarray(Image.open(view['mask']).convert('L')) / 255.
            K = torch.tensor(view['intrinsic_normalized'], dtype=torch.float32)
            K[:2] *= 512
            try:
                prepared,k_crop,meta=prepare_rgba_view(image,K,mask=mask,margin=.1,threshold=.8,require_foreground=True)
            except ValueError as error:
                if str(error)!='RGBA/mask contains no foreground above threshold':
                    raise
                unusable_view={'index':view_idx,'image':view['image'],'mask':view['mask'],'error':str(error)}
                break
            prepared.save(folder / f'input_{view_idx:02d}.png')
            features = dino([prepared]).detach().half().cpu()
            if tuple(features.shape) != (1, 1029, 1024) or not torch.isfinite(features).all():
                raise ValueError('DINOv3 token contract failed')
            all_features.append(features)
            intrinsics.append(k_crop)
            extrinsics.append(view['extrinsic_w2c'])
            prep_meta.append(meta)
            geo = projection_geometry(coords=torch.tensor(occ_coords).int(), resolution=64,
                intrinsics=K[None], extrinsics=torch.tensor(view['extrinsic_w2c'])[None], image_height=512, image_width=512)
            cam = geo['camera'][0]
            uv = torch.stack([K[0,0]*cam[:,0]/cam[:,2]+K[0,2], K[1,1]*cam[:,1]/cam[:,2]+K[1,2]], -1).round().long()
            valid = geo['valid'][0]
            uv = uv[valid]
            support = float(mask[uv[:,1].numpy(),uv[:,0].numpy()].mean()) if len(uv) else 0.
            mask_support.append(support)
            from scipy.ndimage import maximum_filter
            radius=voxel_projection_radius(cam[valid].numpy(),K.numpy())
            dilated=maximum_filter(mask,size=2*radius+1,mode='constant')
            tolerant_support.append(float(dilated[uv[:,1].numpy(),uv[:,0].numpy()].mean()) if len(uv) else 0.)
            projection_radii.append(radius)
        if unusable_view is not None:
            c.write(folder/'rejected.json',{'time':c.now(),'uid':uid,'parent_plan_identity':c.plan()['identity'],
                'reason':'At least one selected input view has no foreground at the fixed mask threshold .8',
                'mean_tolerant_support':0.,'view_failure':unusable_view,'source_row':row,'cache_retained':True})
            print(c.now(),'data_validity_exclusion',uid,'empty_foreground',unusable_view['index'],flush=True)
            continue
        payload = {'format': SAMPLE_FORMAT, 'uid': uid, 'dino_features': torch.cat(all_features),
                   'intrinsics': torch.stack(intrinsics), 'extrinsics': torch.tensor(extrinsics, dtype=torch.float32),
                   'extrinsics_type': 'w2c', 'camera_forward_sign': 1., 'camera_image_y_sign': 1.,
                   'grid_transform': 'identity', 'image_size': [512,512],
                   'source_views': row['views'][:8], 'preprocessing': prep_meta}
        atomic_torch_save(payload, folder / 'pose.pt')
        native_support = {tuple(t) for t in xyz}
        ss_support = {tuple(t) for t in np.unique(occ_coords//2, axis=0)}
        support_iou = len(native_support & ss_support) / max(1, len(native_support | ss_support))
        meta = {'time': c.now(), 'plan_identity': c.plan()['identity'], 'uid': uid,
                'pose_file': str(folder/'pose.pt'), 'ss_target_file': str(folder/'ss.npz'),
                'shape_target_file': row['shape_latent'], 'pair_file': str(folder/'pairs.pt'),
                'ss_codec_iou64': codec_iou, 'native_shape_tokens': len(xyz),
                'ss32_native_shape32_iou': support_iou, 'projection_mask_support': mask_support,
                'projection_tolerant_support':tolerant_support,'projection_radius_pixels':projection_radii,
                'shape_label_origin': 'existing native T2 Shape512, frozen posterior sample',
                'ss_label_origin': 'fresh native encoder, GT surface occupancy64 from rendering cache',
                'old_z_loaded': False, 'old_flow_checkpoint_loaded': False,
                'source_row': row,
                'files': {name: c.sha(folder/name) for name in ['pose.pt','ss.npz','pairs.pt','occupancy.npz']},
                'shape_sha256': c.sha(row['shape_latent'])}
        if np.mean(tolerant_support)<.8:
            c.write(folder/'rejected.json',{'time':c.now(),'uid':uid,
                'parent_plan_identity':c.plan()['identity'],
                'reason':'Insufficient geometric observation support; transparent/hidden geometry or rendering mismatch, not an automatic pose correction',
                'mean_tolerant_support':float(np.mean(tolerant_support)),
                'tolerant_support':tolerant_support,'raw_support':mask_support,
                'source_row':row,'cache_retained':True})
            print(c.now(),'data_validity_exclusion',uid,float(np.mean(tolerant_support)),flush=True)
            continue
        c.write(done, meta)
        del clean, z, rec, pairs
        gc.collect()
        torch.cuda.empty_cache()
    print(c.now(), 'prepare_complete', subset, rank, len(rows), flush=True)


def publish(subset):
    from .vendor_pose.constants import CACHE_FORMAT
    rows=c.sample_rows()
    formal=None if subset=='overfit' else freeze_formal_selection()
    lookup={r['object_uid']:r for r in rows}
    selected=([r for r in rows if r['overfit32']] if subset=='overfit' else
              [lookup[uid] for uid in formal['plan']['splits'][subset]])
    records = []
    for row in selected:
        meta = c.read(c.DATA/'objects'/row['object_uid']/'complete.json')
        assert meta['plan_identity'] == c.plan()['identity']
        records.append({**{k:meta[k] for k in ['uid','pose_file','ss_target_file','shape_target_file','pair_file']},
            'pose_file_sha256':meta['files']['pose.pt'],
            'ss_target_file_sha256':meta['files']['ss.npz'],
            'shape_target_file_sha256':meta['shape_sha256'],
            'pair_file_sha256':meta['files']['pairs.pt']})
    assert len(records) == {'overfit':32,'train':2000,'dev':64}[subset]
    manifest=c.DATA/f'{subset}.json'
    effective_identity=c.plan()['identity'] if formal is None else formal['identity']
    payload={'format':CACHE_FORMAT,'root':str(c.DATA),
             'plan_identity':effective_identity,'samples':records}
    if manifest.exists() and c.read(manifest)!=payload:
        # The initial Overfit32 manifest was sealed before optional SHA fields
        # were added. Preserve its identity for exact resume; bind hashes in
        # a sidecar after verifying every existing core field is unchanged.
        old=c.read(manifest)
        assert old['plan_identity']==payload['plan_identity']
        assert len(old['samples'])==len(records)
        for previous,current in zip(old['samples'],records):
            assert all(current[k]==v for k,v in previous.items())
        c.write(c.DATA/f'{subset}_hash_bindings.json',{'manifest_sha256':c.sha(manifest),
            'samples':records},immutable=True)
    else:
        c.write(manifest,payload,immutable=True)
    print(c.now(), 'published', subset, len(records), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['freeze','objects','publish'])
    p.add_argument('--subset', default='overfit', choices=['overfit','train','dev','all'])
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world', type=int, default=1)
    args = p.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'objects':
        prepare_objects(args.subset, args.rank, args.world)
    else:
        publish(args.subset)


if __name__ == '__main__':
    main()
