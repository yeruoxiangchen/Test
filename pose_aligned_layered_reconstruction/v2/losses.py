"""Masked layer supervision and GT-anchored cross-view point consistency."""
import torch
import torch.nn.functional as F

from ..projection import resize_sampling_grid, sample_maps
from .ray_geometry import make_patch_rays


def resized_labels(sample, prediction):
    device, size = prediction.depths.device, prediction.spatial_size
    def resize(key, channel=False):
        value = sample[key].to(device=device).float()
        value = value[:, None] if channel else value
        if tuple(value.shape[-2:]) != size:
            value = F.interpolate(value, size=size, mode='nearest')
        return value[:, 0] if channel else value
    depth = resize('layer_depths_gt')
    target = resize('layer_target_gt')
    stop = resize('ray_stop_index_gt', True).long()
    if depth.shape != prediction.depths.shape:
        raise ValueError('configured layers must match supervision; explicitly rebuild labels to change L')
    mask = torch.ones_like(depth, dtype=torch.bool)
    if 'supervision_valid_mask' in sample:
        source = sample['supervision_valid_mask'].to(device).float()
        if source.ndim == 3:
            source = source[:, None]
        mask = F.interpolate(source, size=size, mode='nearest').bool().expand_as(depth)
    count = stop.clamp(0, prediction.layer_count)
    valid = torch.arange(prediction.layer_count, device=device)[None, :, None, None] < count[:, None]
    return depth, target, count, valid & mask, mask


def layer_loss(prediction, sample, confidence_scale=.05):
    depth, target, count, valid, mask = resized_labels(sample, prediction)
    weight = valid.float()
    denom = weight.sum().clamp_min(1)
    depth_loss = (F.smooth_l1_loss(prediction.depths.float(), depth, reduction='none') * weight).sum() / denom
    ownership = (F.binary_cross_entropy_with_logits(prediction.target_logits.float(), target, reduction='none') * weight).sum() / denom
    # The final count class is >= L, including rays censored by the layer cap.
    count_mask = mask.all(1).float()
    stop = (F.cross_entropy(prediction.stop_logits.float(), count, reduction='none') * count_mask).sum() / count_mask.sum().clamp_min(1)
    confidence_target = torch.exp(-(prediction.depths.detach().float() - depth).abs() / confidence_scale)
    confidence = (F.binary_cross_entropy_with_logits(prediction.confidence_logits.float(), confidence_target, reduction='none') * weight).sum() / denom
    total = depth_loss + .25 * stop + .25 * ownership + .1 * confidence
    return total, {'depth': depth_loss.detach(), 'stop': stop.detach(), 'ownership': ownership.detach(),
                   'confidence': confidence.detach(), 'depth_mae': ((prediction.depths - depth).abs() * weight).sum().detach() / denom}


def multiview_point_loss(prediction, sample, cameras, max_anchors=64, correspondence_tolerance=.04):
    with torch.autocast(prediction.depths.device.type, enabled=False):
        return _multiview_point_loss(prediction, sample, cameras, max_anchors, correspondence_tolerance)


def _multiview_point_loss(prediction, sample, cameras, max_anchors=64, correspondence_tolerance=.04):
    """Compare predicted 3D points at GT correspondences, never layer k to k.

    All valid target intersections, including hidden ones, may be anchors.
    Correspondence selection uses labels, not learned confidence/ownership.
    """
    gt, owner, count, valid, supervision = resized_labels(sample, prediction)
    views, layers, height, width = gt.shape
    rays = make_patch_rays(cameras, (height, width))
    valid_probabilities = prediction.valid_probabilities()
    total = prediction.depths.sum() * 0.
    pairs = 0
    for q in range(views):
        indices = torch.nonzero((valid[q] & (owner[q] > .5)).flatten(), as_tuple=False).flatten()
        if not len(indices):
            continue
        if len(indices) > max_anchors:
            indices = indices[torch.linspace(0, len(indices) - 1, max_anchors, device=indices.device).long()]
        pixel = indices % (height * width)
        gt_depth = gt[q].flatten()[indices]
        pred_depth = prediction.depths[q].flatten()[indices]
        xyz = rays.origins[q, pixel] + gt_depth[:, None] * rays.axial_directions[q, pixel]
        pred_xyz = rays.origins[q, pixel] + pred_depth[:, None] * rays.axial_directions[q, pixel]
        for s in range(views):
            if s == q:
                continue
            camera = cameras.select(slice(s, s + 1))
            projection = camera.project(xyz, width)
            grid = resize_sampling_grid(projection['image_grid'], image_size=cameras.image_size, map_size=(height, width))
            def nearest(value):
                return F.grid_sample(value[s:s + 1].float(), grid[:, :, None], mode='nearest', align_corners=True)[0, :, :, 0].T
            depths = nearest(gt)
            eligible = nearest((valid & (owner > .5)).float()) > .5
            distance = (depths - projection['camera_depth'][0, :, None]).abs()
            distance = distance.masked_fill(~eligible, torch.inf)
            error, layer = distance.min(-1)
            matches = (error < correspondence_tolerance) & projection['valid'][0]
            if not matches.any():
                continue
            # Interpolation is validity-weighted so invalid suffixes cannot pull
            # real surfaces towards arbitrary placeholder depths at boundaries.
            probabilities = valid_probabilities[s:s + 1]
            numerator = sample_maps(prediction.depths[s:s + 1] * probabilities, grid)[0].T
            denominator = sample_maps(probabilities, grid)[0].T.clamp_min(1.e-8)
            selected = (numerator / denominator).gather(1, layer[:, None])[:, 0]
            image_grid = projection['image_grid'][0]
            ih, iw = cameras.image_size
            u = (image_grid[:, 0] + 1) * (iw - 1) / 2
            v = (image_grid[:, 1] + 1) * (ih - 1) / 2
            k = camera.intrinsics[0]
            direction = torch.stack(((u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1],
                                     torch.full_like(u, cameras.forward_sign)), -1)
            direction = direction @ camera.w2c[0, :3, :3]
            other_xyz = camera.centers[0] + selected[:, None] * direction
            total = total + F.smooth_l1_loss(pred_xyz[matches], other_xyz[matches], reduction='sum')
            pairs += int(matches.sum())
    return total / max(pairs * 3, 1), {'cross_view_pairs': pairs}
