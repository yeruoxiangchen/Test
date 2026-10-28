"""Shared geometry lookup for dense SS and active sparse SLat coordinates."""
import torch

from ..layer_matching import match_queries_to_layers
from ..projection import resize_sampling_grid, project_masked_dino, sample_maps
from .ray_geometry import CameraBundle
from .contracts import LayeredPredictionV2


CONDITION_MODES = ('full', 'single_layer', 'no_layer')


def first_surface_prediction(prediction):
    """Retain the first ordered scene surface and its probability of existing.

    The first surface may be an occluder.  We do not search the target ownership
    labels for a later target surface.  Count classes 1..L become one >=1 class.
    """
    prediction.validate()
    logits = prediction.stop_logits
    return LayeredPredictionV2(prediction.depths[:, :1],
        torch.cat((logits[:, :1], torch.logsumexp(logits[:, 1:], dim=1, keepdim=True)), dim=1),
        prediction.target_logits[:, :1], prediction.confidence_logits[:, :1]).validate()


def projected_appearance_inputs(sample, projection, masked, *, device):
    """Same observed raw/masked images, visible cue and camera projection; no layers.

    The returned channels retain the complete DINO observations before the
    learned control projection.  No predicted/ground-truth depth or ownership
    enters this path, and support is only camera-frustum validity.
    """
    raw = project_masked_dino({**sample,
        'masked_visual_patch_features': sample['raw_visual_patch_features']}, projection, device=device)
    visible = sample['visible_target_mask'].to(device=device, dtype=torch.float32)
    if visible.ndim == 3:
        visible = visible[:, None]
    visible = sample_maps(visible, projection['image_grid'])[:, 0, :, None].clamp(0., 1.)
    visible = visible * projection['valid'][..., None]
    return torch.cat((raw, masked, visible), dim=-1)


def query_geometry(sample, prediction, points, *, absolute_tolerance=.02, relative_tolerance=0.):
    cameras = CameraBundle.from_sample(sample, device=points.device)
    projection = cameras.project(points, prediction.spatial_size[1])
    return match_queries_to_layers(prediction, image_grid=projection['image_grid'],
        layer_grid=resize_sampling_grid(projection['image_grid'], image_size=cameras.image_size,
                                       map_size=prediction.spatial_size),
        camera_depth=projection['camera_depth'], frustum_valid=projection['valid'],
        visible_target_mask=sample['visible_target_mask'].to(points.device),
        absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance)


def active_points(coords, grid_transform, side=32):
    if coords.ndim != 2 or coords.shape[1] != 4 or (coords[:, 0] != 0).any():
        raise ValueError('active coordinates must be single-batch [N,4]')
    if ((coords[:, 1:] < 0) | (coords[:, 1:] >= side)).any():
        raise ValueError('active coordinates out of bounds')
    points = (coords[:, 1:].float() + .5) / side - .5
    if grid_transform == 'pixal3d_rotation':
        rotation = points.new_tensor(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
        points = points @ rotation.T
    elif grid_transform != 'identity':
        raise ValueError('unknown grid transform')
    return points
