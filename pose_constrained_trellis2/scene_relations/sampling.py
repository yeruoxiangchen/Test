"""Synchronous scene-group Euler integration using the official CFG math."""
import numpy as np
import torch
from ..conditioning import build_2d_context
from ..runtime import make_sampler
from .model import bind_neighbors, summarize_state


class BoundObject(torch.nn.Module):
    def __init__(self, model, sample, positive, views, neighbors):
        super().__init__()
        self.model, self.sample = model, sample
        self.positive, self.views, self.neighbors = positive, views, neighbors
        self.cached_pose = None

    def forward(self, state, t, context):
        positive = context is self.positive
        sample = self.sample if positive else None
        if positive and self.cached_pose is None:
            self.cached_pose = self.model._pose_condition(state, sample, self.views)
        with bind_neighbors(self.model, state, sample, self.neighbors if positive else []):
            result, _ = self.model.adapted_prediction(
                state, t, context, sample, view_indices=self.views if positive else None,
                precomputed_condition=self.cached_pose if positive else None, compare_stock=False)
        return result


def synchronous_step(states, predict):
    """predict must see the entire unchanged old state, never an updated neighbor."""
    previous = tuple(states)
    return [predict(i, previous) for i in range(len(previous))]


@torch.no_grad()
def sample_group(model, stage, states, samples, *, steps=12):
    sampler, parameters = make_sampler(stage)
    parameters = dict(parameters)
    parameters.pop('steps')
    rescale = parameters.pop('rescale_t')
    sequence = np.linspace(1, 0, steps + 1)
    sequence = rescale * sequence / (1 + (rescale - 1) * sequence)
    views = [torch.arange(len(s['dino_features']), device=states[0].device) for s in samples]
    positives = [build_2d_context(s['dino_features'], v, mode='first') for s, v in zip(samples, views)]
    bounds = [BoundObject(model, s, p, v, []) for s, p, v in zip(samples, positives, views)]
    for t, previous_t in zip(sequence[:-1], sequence[1:]):
        summaries = [summarize_state(x, s, stage, v) for x, s, v in zip(states, samples, views)]
        def predict(index, old):
            bounds[index].neighbors = [r for j, r in enumerate(summaries) if j != index]
            return sampler.sample_once(bounds[index], old[index], float(t), float(previous_t),
                                       cond=positives[index], neg_cond=torch.zeros_like(positives[index]),
                                       **parameters).pred_x_prev
        states = synchronous_step(states, predict)
    return states
