"""Policies consume observed-mask scores only, never evaluation metrics."""
import math

THRESHOLDS = (.80, .85, .90)


def choose_source(scores, threshold=None):
    """Both observed views must pass; rank eligible donors by mean, tie A first."""
    eligible = []
    for key in sorted(scores):
        values = scores[key]
        if len(values) != 2 or not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError('expected exactly two finite observed-view IoUs')
        if threshold is None or min(values) >= threshold:
            eligible.append(key)
    return max(eligible, key=lambda k: sum(scores[k])) if eligible else None


def select_checkpoint(snapshots, tolerance=.005):
    """Identity always available; no observed view may regress beyond tolerance."""
    initial = snapshots[0]
    if initial['step'] != 0:
        raise ValueError('identity checkpoint must come first')
    eligible = [s for s in snapshots if all(
        new >= old - tolerance for new, old in zip(s['ious'], initial['ious']))]
    return max(eligible, key=lambda s: (sum(s['ious']), -s['step']))
