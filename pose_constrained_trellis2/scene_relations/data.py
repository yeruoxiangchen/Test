"""Group manifests expose target-free inference and separately loaded labels."""
from functools import lru_cache
from pathlib import Path
from ..data import load_pose_payload, load_ss_target
from .shape_support import load_shape_target, NATIVE_SHAPE_MAX_TOKENS


class GroupDataset:
    def __init__(self, manifest, stage, *, max_tokens=NATIVE_SHAPE_MAX_TOKENS):
        import json
        self.path = Path(manifest)
        self.document = json.loads(self.path.read_text())
        self.groups = self.document['groups']
        self.stage = stage
        self.max_tokens = max_tokens
        assert stage in ('ss', 'shape', 'inference')
        assert len({g['id'] for g in self.groups}) == len(self.groups)
        for group in self.groups:
            assert 1 <= len(group['members']) <= 2
            assert len({r['uid'] for r in group['members']}) == len(group['members'])
            assert all(r['split'] == group['split'] and r['scene'] == group['scene'] for r in group['members'])

    def __len__(self): return len(self.groups)

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        group = self.groups[index]
        samples = []
        for row in group['members']:
            sample = load_pose_payload(Path(row['pose_file']), row['uid'])
            sample.update(uid=row['uid'], row=row)
            if self.stage == 'ss':
                sample['target'] = load_ss_target(Path(row['ss_target_file']), row['uid'])
            elif self.stage == 'shape':
                sample['target_coords'], sample['target_feats'] = load_shape_target(Path(row['shape_target_file']), row['uid'], max_tokens=self.max_tokens)
            samples.append(sample)
        return samples
