"""V1/V2 manifests, object-safe splits, shared view selection and latent binding."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..paired_dataset import PairedLayerDataset, canonical_json_sha256, sha256_file
from ..dino_context import build_masked_dino_contexts
from ..contracts import LAYERED_SAMPLE_MANIFEST_FORMAT
from ..contracts import LAYERED_SAMPLE_FORMAT
from .contracts import MANIFEST_FORMAT, adapt_v1, compatibility_view, validate_sample


def write_manifest(path, rows, config, root):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rows = list(rows)
    if not rows or len({r['uid'] for r in rows}) != len(rows):
        raise ValueError('empty/duplicate sample UIDs')
    payload = {'format': MANIFEST_FORMAT, 'root': str(Path(root).resolve()),
               'samples': rows, 'config': config, 'config_hash': canonical_json_sha256(config),
               'sample_count': len(rows), 'object_count': len({r['object_uid'] for r in rows})}
    payload['manifest_identity'] = canonical_json_sha256({k: payload[k] for k in ('format', 'config_hash', 'samples')})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


class SceneDataset(Dataset):
    def __init__(self, manifest, require_labels=True, require_ss_target=False, verify_hashes=True):
        self.path = Path(manifest).resolve(strict=True)
        payload = json.loads(self.path.read_text())
        self.base = None
        self.labels, self.ss, self.verify = require_labels, require_ss_target, verify_hashes
        if payload['format'] == LAYERED_SAMPLE_MANIFEST_FORMAT:
            self.base = PairedLayerDataset(self.path, require_labels=require_labels,
                require_ss_target=require_ss_target, verify_hashes=False, build_trellis_contexts=False)
            self.rows, self.config = self.base.rows, self.base.config
        elif payload['format'] == MANIFEST_FORMAT:
            self.rows, self.config = payload['samples'], payload['config']
            self.root = Path(payload['root']).resolve()
            if payload['config_hash'] != canonical_json_sha256(self.config):
                raise ValueError('V2 manifest config hash mismatch')
            identity = canonical_json_sha256({k: payload[k] for k in ('format', 'config_hash', 'samples')})
            if payload['manifest_identity'] != identity:
                raise ValueError('V2 manifest identity mismatch')
            if payload['sample_count'] != len(self.rows) or len({r['uid'] for r in self.rows}) != len(self.rows):
                raise ValueError('V2 manifest count or UID mismatch')
            if payload['object_count'] != len({r['object_uid'] for r in self.rows}):
                raise ValueError('V2 object count mismatch')
        else:
            raise ValueError('unsupported manifest format')
        self.verified = set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.base is not None:
            if self.verify and index not in self.verified:
                if sha256_file(self.base._resolve(row['sample'])) != row['sha256']:
                    raise ValueError('changed legacy sample')
                self.verified.add(index)
            sample = adapt_v1(self.base[index])
        else:
            path = (self.root / row['sample']).resolve()
            if self.verify and index not in self.verified:
                if sha256_file(path) != row['sha256']:
                    raise ValueError(f'changed V2 sample: {path}')
                self.verified.add(index)
            sample = torch.load(path, map_location='cpu', weights_only=True)
            if sample['uid'] != row['uid'] or sample['object_uid'] != row['object_uid']:
                raise ValueError('sample identity mismatch')
            if row.get('source_sample_format') == LAYERED_SAMPLE_FORMAT:
                if sample.get('format') != LAYERED_SAMPLE_FORMAT:
                    raise ValueError('mixed-manifest V1 source format mismatch')
                sample = adapt_v1(sample)
        validate_sample(sample, self.labels, self.ss)
        return sample

    def identity(self):
        return {'manifest': str(self.path), 'sha256': sha256_file(self.path),
                'sample_count': len(self), 'object_uids': sorted({r.get('object_uid', r['uid']) for r in self.rows}),
                'config': self.config}

    def assert_disjoint(self, other):
        overlap = set(self.identity()['object_uids']) & set(other.identity()['object_uids'])
        if overlap:
            raise ValueError(f'train/validation object overlap: {sorted(overlap)[:5]}')


def select_views(sample, indices):
    """All visual contexts and geometry see exactly the same selected views."""
    views = len(sample['intrinsics'])
    keys = ('raw_visual_patch_features', 'masked_visual_patch_features', 'visible_target_mask',
            'intrinsics', 'extrinsics', 'layer_depths_gt', 'layer_target_gt', 'ray_stop_index_gt',
            'ray_overflow_gt', 'supervision_valid_mask', 'amodal_target_mask_gt', 'view_ids',
            'source_to_feature_affines')
    result = dict(sample)
    for key in keys:
        if key in sample and torch.is_tensor(sample[key]):
            if len(sample[key]) != views:
                raise ValueError(f'view count mismatch: {key}')
            result[key] = sample[key].index_select(0, indices.to(sample[key].device))
    result.update(build_masked_dino_contexts(result['masked_visual_patch_features']))
    return compatibility_view(result)


def slat_target(sample, device, normalization):
    """Original latent order is preserved; normalization is checkpoint-bound."""
    from .._bootstrap import bootstrap_external_paths
    bootstrap_external_paths()
    from trellis.modules import sparse as sp
    binding = sample.get('slat_binding', sample.get('source_binding', {}))
    path = binding.get('slat', binding.get('path'))
    if not path:
        raise ValueError('sample has no SLat target; build labels before stage slat')
    expected = binding.get('slat_sha256', binding.get('sha256'))
    if not expected or sha256_file(path) != expected:
        raise ValueError('SLat target hash missing or mismatched')
    with np.load(path, allow_pickle=False) as data:
        coords = np.asarray(data['coords'], dtype=np.int32)
        feats = np.asarray(data['feats'], dtype=np.float32)
    if coords.shape[1] == 3:
        coords = np.concatenate((np.zeros((len(coords), 1), np.int32), coords), 1)
    if len(coords) != len(feats) or feats.shape[1] != 8 or np.any(coords[:, 0] != 0):
        raise ValueError('invalid single-object SLat target')
    if np.any(coords[:, 1:] < 0) or np.any(coords[:, 1:] >= 64) or len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError('invalid SLat coordinates')
    if 'target_coords' in sample:
        original = sample['target_coords'].cpu().numpy()[:, -3:]
        if not np.array_equal(np.unique(original, axis=0), np.unique(coords[:, 1:], axis=0)):
            raise ValueError('SS/SLat support mismatch')
    features = torch.from_numpy(feats).to(device)
    mean = features.new_tensor(normalization['mean'])
    std = features.new_tensor(normalization['std'])
    if (std <= 0).any() or not torch.isfinite(features).all():
        raise ValueError('invalid SLat normalization/features')
    return sp.SparseTensor(feats=(features - mean) / std, coords=torch.from_numpy(coords).to(device))
