"""Synchronous data-parallel training for the gated first experiment.

All ranks have an identical frozen parent. Only trainable gradients are
averaged (one coalesced all-reduce); no old-project implementation is imported.
The 32-object gate must pass before a non-smoke run is allowed.
"""
from __future__ import annotations

import json
from datetime import timedelta
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoints import (load_parent_ss30k, load_target_lari_checkpoint,
                          save_layered_ss_checkpoint, save_target_lari_checkpoint)
from .native_ss import build_layered_ss_components
from .distributed_data import verify_dataset_shard, validate_verification_receipts
from .paired_dataset import PairedLayerDataset, collate_one
from .prepare_official_experiment import atomic_json, read_json
from .target_lari import TargetConditionedLaRI, target_lari_loss
from .train_layered_ss import make_parser, _sample_t, _sample_views, _zero_init_audit
from .train_target_lari import _model_forward


def average_gradients(parameters, world_size):
    gradients = [p.grad if p.grad is not None else torch.zeros_like(p) for p in parameters]
    flat = torch.cat([g.reshape(-1) for g in gradients])
    dist.all_reduce(flat)
    flat.div_(world_size)
    offset = 0
    for parameter in parameters:
        n = parameter.numel()
        if parameter.grad is None:
            parameter.grad = torch.empty_like(parameter)
        parameter.grad.copy_(flat[offset:offset + n].view_as(parameter))
        offset += n


def main():
    parser = make_parser()
    parser.add_argument("--stage", choices=("lari", "ss"), required=True)
    parser.add_argument("--overfit_gate", default="")
    parser.add_argument("--distributed_smoke", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=100)
    args = parser.parse_args()
    if args.grad_accum != 1 or args.amp_dtype == "fp16":
        raise ValueError("distributed first-round trainer supports one sample/rank and bf16/none")
    if args.distributed_smoke:
        if args.max_steps > 3:
            raise ValueError("distributed smoke is limited to three optimizer steps")
    elif not args.overfit_gate or not read_json(args.overfit_gate).get("formal_training_allowed"):
        raise ValueError("formal training requires a passed 32-object overfit gate")
    rank, local_rank, world = map(int, (os.environ["RANK"], os.environ["LOCAL_RANK"], os.environ["WORLD_SIZE"]))
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "run_config.json", {**vars(args), "world_size": world, "effective_batch": world})
    dist.barrier()
    dataset = PairedLayerDataset(args.manifest, require_labels=args.stage == "lari" or args.layer_source == "oracle",
                                require_ss_target=args.stage == "ss", verify_hashes=False,
                                build_trellis_contexts=args.stage == "ss")
    if not args.distributed_smoke and len(dataset) != 2000:
        raise ValueError("first formal experiment must contain exactly 2000 objects")
    # Previously rank0 alone re-read ~94GB while other ranks waited inside a
    # 600-second NCCL barrier. Verify every byte, but partition the work first.
    receipt = verify_dataset_shard(dataset, rank, world, output)
    receipts = [None] * world
    dist.all_gather_object(receipts, receipt)
    verification = validate_verification_receipts(dataset, receipts)
    atomic_json(output / f"startup_rank{rank}.json", {"phase": "model_initialization", "rank": rank,
        "world_size": world, "verified_objects": receipt["verified_objects"],
        "total_objects": receipt["verified_objects"], "updated_epoch": time.time()})
    sampler_data = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, drop_last=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler_data, num_workers=args.num_workers,
                        collate_fn=collate_one, pin_memory=True, persistent_workers=args.num_workers > 0)
    parent = lari = initial_audit = None
    if args.stage == "lari":
        model = TargetConditionedLaRI(hidden_channels=args.lari_hidden_channels, layers=args.lari_layers,
                                      decoder_blocks=args.lari_decoder_blocks).to(device)
    else:
        sampler, model, _ = build_layered_ss_components(pretrained=args.pretrained, device=device,
            layer_channels=args.layer_channels, lari_hidden_channels=args.lari_hidden_channels,
            lari_layers=args.lari_layers, lari_decoder_blocks=args.lari_decoder_blocks,
            gradient_checkpointing=args.gradient_checkpointing)
        parent = load_parent_ss30k(model, args.parent_ss30k, weights=args.parent_weights)
        if args.layer_source != "predicted" or not args.target_lari_checkpoint:
            raise ValueError("formal SS run must use the trained predicted layer branch")
        lari = load_target_lari_checkpoint(model.target_lari, args.target_lari_checkpoint)
        model.target_lari.requires_grad_(False)
        initial_audit = _zero_init_audit(model=model, sampler=sampler, sample=dataset[0], device=device,
                                        layer_source="predicted", amp_dtype=torch.bfloat16,
                                        amp_enabled=args.amp_dtype == "bf16")
    parameters = [p for p in model.parameters() if p.requires_grad]
    # Verify identical initialization instead of assuming random states match.
    initial = torch.cat([p.detach().flatten() for p in parameters])
    reference = initial.clone()
    dist.broadcast(reference, src=0)
    if not torch.equal(reference, initial):
        raise RuntimeError("ranks have different trainable initialization")
    del initial, reference
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(.9, .95))
    identity = {**dataset.identity(), **verification, "world_size": world, "effective_batch": world,
                "optimizer_synchronization": "coalesced_mean_of_trainable_gradients", "stage": args.stage,
                "parent": parent, "target_lari": lari, "layer_source": args.layer_source,
                "overfit_gate": args.overfit_gate, "distributed_smoke": args.distributed_smoke}
    # Per-rank noise differs only after all model initialization is verified.
    torch.manual_seed(args.seed + rank * 100003)
    torch.cuda.manual_seed_all(args.seed + rank * 100003)
    step, epoch, history = 0, 0, []
    start = time.time()
    model.train()
    if args.stage == "ss":
        for module in (model.flow, model.aggregator, model.block_condition, model.target_lari):
            module.eval()
    while step < args.max_steps:
        sampler_data.set_epoch(epoch)
        for sample in loader:
            if step >= args.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            lr = args.learning_rate * min(1., (step + 1) / max(args.warmup_steps, 1))
            for group in optimizer.param_groups:
                group["lr"] = lr
            if args.stage == "lari":
                # Match the single-GPU overfit head's FP32 supervision path.
                prediction = _model_forward(model, sample, device)
                loss, values = target_lari_loss(prediction, sample)
                metrics = {k: float(v) for k, v in values.items()}
            else:
                target = sample["target"].to(device).float()[None]
                condition = sample["stock_condition"].to(device)
                indices = _sample_views(sample, args, device)
                t_value = _sample_t(device)
                xt, velocity = sampler._get_model_gt(target, t_value, torch.randn_like(target))
                timestep = torch.full((1,), 1000. * t_value, device=device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp_dtype == "bf16"):
                    parent_prediction = model.parent_prediction(xt, timestep, condition, sample, view_indices=indices)
                    parent_loss = F.mse_loss(parent_prediction.float(), velocity.float())
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp_dtype == "bf16"):
                    prediction, stats = model.adapted_prediction(xt, timestep, condition, sample,
                        view_indices=indices, use_layer_branch=True, layer_source="predicted")
                    loss = F.mse_loss(prediction.float(), velocity.float())
                metrics = {"flow_loss": float(loss.detach()), "parent_loss": float(parent_loss),
                           "gain": float(parent_loss - loss.detach()), "target_support": float(stats["target_support_mean"]),
                           "view_count": float(len(indices))}
            loss.backward()
            average_gradients(parameters, world)
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in parameters):
                raise RuntimeError("non-finite parameters after distributed step")
            values = torch.tensor(list(metrics.values()), device=device)
            dist.all_reduce(values)
            values /= world
            metrics = dict(zip(metrics, values.cpu().tolist()))
            step += 1
            row = {"step": step, "object_exposures": step * world, "learning_rate": lr, **metrics,
                   "gradient_norm_before_clip": float(norm), "elapsed_seconds": time.time() - start}
            if rank == 0:
                history.append(row)
                if step == 1:
                    atomic_json(output / "training_started.json", {"step": step, "world_size": world,
                        "updated_epoch": time.time(), "all_sample_hashes_verified_before_training": True})
                with (output / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                if step == 1 or step % args.log_every == 0:
                    print(json.dumps(row), flush=True)
                if step % args.save_every == 0 or step == args.max_steps:
                    for name in (f"step_{step:06d}.pt", "last.pt"):
                        if args.stage == "lari":
                            save_target_lari_checkpoint(output / "checkpoints" / name, model=model, step=step,
                                optimizer=optimizer, data_identity=identity, history=history)
                        else:
                            save_layered_ss_checkpoint(output / "checkpoints" / name, model=model, step=step,
                                optimizer=optimizer, data_identity=identity, history=history, parent_binding=parent)
        epoch += 1
    # Check ranks ended at the same trainable weights, including optimizer sync.
    state = torch.cat([p.detach().flatten() for p in parameters])
    reference = state.clone()
    dist.broadcast(reference, src=0)
    difference = (state - reference).abs().max()
    dist.all_reduce(difference, op=dist.ReduceOp.MAX)
    if float(difference) != 0:
        raise RuntimeError("distributed parameters diverged")
    if rank == 0:
        atomic_json(output / "report.json", {"passed": True, "step": step, "epoch": epoch,
            "data_identity": identity, "model": model.metadata(), "last": history[-1],
            "rank_parameter_max_abs_difference": float(difference), "initial_zero_audit": initial_audit})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
