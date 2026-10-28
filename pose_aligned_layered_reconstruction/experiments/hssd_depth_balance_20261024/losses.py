"""Change only the differentiable depth reduction, preserving the released heads."""
import torch
import torch.nn.functional as F
from ...v2.losses import resized_labels, layer_loss


def depth_groups(valid, target):
    other = valid & (target <= .5)
    target = valid & (target > .5)
    # A valid earlier non-target hit occludes any later target hit on that ray.
    preceding_other = torch.cat((torch.zeros_like(other[:, :1]), other[:, :-1]), 1).long().cumsum(1) > 0
    return {'non_target':other, 'exposed_target':target & ~preceding_other,
            'hidden_target':target & preceding_other}


def supervised_loss(prediction, sample, arm):
    if arm not in ('uniform', 'balanced'):
        raise ValueError(arm)
    original, metrics = layer_loss(prediction, sample)
    depth, target, _, valid, _ = resized_labels(sample, prediction)
    errors = F.smooth_l1_loss(prediction.depths.float(), depth, reduction='none')
    uniform = (errors * valid.float()).sum() / valid.sum().clamp_min(1)
    groups = depth_groups(valid, target)
    group_losses = []
    for name, mask in groups.items():
        count = int(mask.sum())
        metrics[name + '_count'] = count
        metrics[name + '_mae'] = (prediction.depths.float()-depth).abs()[mask].mean().detach() if count else None
        if count:
            group_losses.append(errors[mask].mean())
    balanced = torch.stack(group_losses).mean() if group_losses else prediction.depths.sum()*0.
    selected = uniform if arm == 'uniform' else balanced
    metrics.update(depth=selected.detach(), uniform_depth=uniform.detach(), balanced_depth=balanced.detach(),
                   active_depth_groups=len(group_losses))
    # Subtract a differentiable uniform loss, NOT metrics['depth'] (detached).
    total = original if arm == 'uniform' else original + (balanced - uniform)
    return total, metrics
