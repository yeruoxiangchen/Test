"""Rollback without sharing live Adam tensors with the saved checkpoint."""
import copy
import torch


@torch.no_grad()
def restore(delta,optimizer,saved_delta,saved_state,minimum_lr):
    previous_lr=optimizer.param_groups[0]['lr']
    delta.copy_(saved_delta)
    # Optimizer.load_state_dict may retain same-device tensor references.
    # Loading the saved dictionary directly corrupts it on the next Adam step.
    optimizer.load_state_dict(copy.deepcopy(saved_state))
    optimizer.param_groups[0]['lr']=max(minimum_lr,previous_lr*.5)
