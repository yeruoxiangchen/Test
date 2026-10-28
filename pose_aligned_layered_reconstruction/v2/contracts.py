"""Versioned V2 records; an explicit compatibility view serves v1 utilities."""
from ..contracts import LAYERED_SAMPLE_FORMAT, validate_paired_sample
from ..contracts import LayeredPrediction
import torch

SAMPLE_FORMAT = 'pose_aligned_layered_reconstruction.paired_sample.v2'
MANIFEST_FORMAT = 'pose_aligned_layered_reconstruction.paired_manifest.v2'
CHECKPOINT_FORMAT = 'pose_aligned_layered_reconstruction.joint_checkpoint.v2'


class LayeredPredictionV2(LayeredPrediction):
    """Same tensors, deterministic suffix sums, final count class >= L."""
    def valid_probabilities(self):
        self.validate()
        probabilities = torch.softmax(self.stop_logits.float(), dim=1)
        suffix, running = [], torch.zeros_like(probabilities[:, 0])
        for probability in reversed(probabilities.unbind(1)):
            running = running + probability
            suffix.append(running)
        return torch.stack(list(reversed(suffix)), 1)[:, 1:]

    def to(self, device, *, dtype=None):
        value = super().to(device, dtype=dtype)
        return type(self)(value.depths, value.stop_logits, value.target_logits, value.confidence_logits)

    def select_views(self, indices):
        value = super().select_views(indices)
        return type(self)(value.depths, value.stop_logits, value.target_logits, value.confidence_logits)


def validate_sample(sample, require_labels=False, require_ss_target=False):
    if sample.get('format') != SAMPLE_FORMAT:
        raise ValueError('not a V2 sample')
    for key in ('uid', 'object_uid', 'scene_uid', 'trajectory_uid', 'data_domain'):
        if not isinstance(sample.get(key), str) or not sample[key]:
            raise ValueError(f'missing V2 identity: {key}')
    if sample.get('depth_kind') != 'camera_forward_z':
        raise ValueError('V2 depths must be camera-forward z')
    validate_paired_sample({**sample, 'format': LAYERED_SAMPLE_FORMAT}, require_labels=require_labels,
                          require_ss_target=require_ss_target)
    if require_labels:
        overflow = sample.get('ray_overflow_gt')
        if overflow is None or overflow.shape != sample['ray_stop_index_gt'].shape:
            raise ValueError('V2 labels require an explicit overflow/censoring mask')


def adapt_v1(sample):
    result = {**sample, 'format': SAMPLE_FORMAT,
        'scene_uid': sample['uid'] + ':legacy', 'trajectory_uid': sample['uid'] + ':original',
        'data_domain': 'legacy_clean', 'depth_kind': 'camera_forward_z',
        'label_geometry_kind': 'target_vae_proxy', 'legacy_overflow_unknown': True}
    if 'ray_stop_index_gt' in sample:
        # Old records discarded overflow. Conservatively mark all capped rays
        # censored rather than fabricate knowledge of exact termination.
        result['ray_overflow_gt'] = sample['ray_stop_index_gt'] >= sample['layer_depths_gt'].shape[1]
    return result


def compatibility_view(sample):
    """Only the tag changes: geometry and all observations remain untouched."""
    return {**sample, 'format': LAYERED_SAMPLE_FORMAT}
