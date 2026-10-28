"""Train pose-constrained TRELLIS.2 SS or Shape adapters from a clean start."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import (
    ema_ramp_decay,
    ema_to_device,
    initialize_ema,
    load_checkpoint,
    restore_rng_state,
    rng_state,
    save_checkpoint,
    update_checkpoint_alias,
    update_ema,
)
from .conditioning import build_2d_context, select_view_indices
from .constants import (
    DINO_FULL_TOKENS,
    SHAPE_FLOW_MODEL,
    SIGMA_MIN,
    SS_FLOW_MODEL,
)
from .data import PoseTrellis2Dataset, collate_one
from .dependency import import_sparse_module, trellis2_root
from .models import optimizer_parameter_groups, validate_trainable_surface
from .runtime import load_pose_conditioned_flow


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ss", "shape"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--new-lr", type=float, default=1.0e-4)
    parser.add_argument("--lora-lr", type=float, default=3.0e-5)
    parser.add_argument("--new-weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=-1,
        help="-1 selects 600 for SS and 3000 for Shape",
    )
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--p-uncond", type=float, default=0.1)
    parser.add_argument("--min-condition-views", type=int, default=1)
    parser.add_argument("--max-condition-views", type=int, default=8)
    parser.add_argument(
        "--context-mode",
        choices=("first", "concat"),
        default="first",
        help="2D cross-attention uses the first selected view by default",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verify-cache-hashes", action="store_true")
    parser.add_argument("--skip-initial-stock-audit", action="store_true")
    parser.add_argument(
        "--stock-eval-every",
        type=int,
        default=0,
        help="optional diagnostic interval; 0 avoids duplicate stock forwards",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "max_steps",
        "save_every",
        "log_every",
        "grad_accum",
        "lora_rank",
        "lora_alpha",
        "new_lr",
        "lora_lr",
        "grad_clip",
        "min_condition_views",
        "max_condition_views",
    )
    invalid = [name for name in positive if float(getattr(args, name)) <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if int(args.min_condition_views) > int(args.max_condition_views):
        raise ValueError("min-condition-views exceeds max-condition-views")
    if not 0.0 <= float(args.p_uncond) < 1.0:
        raise ValueError("p-uncond must be in [0,1)")
    if not 0.0 < float(args.ema_decay) < 1.0:
        raise ValueError("ema-decay must be in (0,1)")
    if (
        int(args.warmup_steps) < -1
        or int(args.max_objects) < 0
        or int(args.stock_eval_every) < 0
    ):
        raise ValueError("invalid warmup/max-objects value")


class TrainingForward(nn.Module):
    """Route the adapted forward through DDP, including the unconditional branch."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Any,
        t: torch.Tensor,
        context: torch.Tensor,
        sample: dict[str, Any] | None,
        *,
        view_indices: torch.Tensor | None,
        stock_velocity: Any,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        prediction, stats = self.model.adapted_prediction(
            x,
            t,
            context,
            sample,
            view_indices=view_indices,
            stock_velocity=stock_velocity,
            compare_stock=False,
        )
        if sample is None:
            reference = prediction if torch.is_tensor(prediction) else prediction.feats
            anchor = reference.new_zeros((), dtype=torch.float32)
            for parameter in self.model.aggregator.parameters():
                anchor = anchor + parameter.reshape(-1)[0].float() * 0.0
            if torch.is_tensor(prediction):
                prediction = prediction + anchor.to(prediction.dtype)
            else:
                prediction = prediction.replace(
                    prediction.feats + anchor.to(prediction.dtype)
                )
        return prediction, stats


def _shape_tensor(sample: dict[str, Any], device: torch.device) -> Any:
    sparse = import_sparse_module()
    coords = sample["target_coords"].to(device=device, dtype=torch.int32)
    coords = torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=1)
    feats = sample["target_feats"].to(device=device, dtype=torch.float32)
    return sparse.SparseTensor(feats=feats, coords=coords)


def target_tensor(sample: dict[str, Any], stage: str, device: torch.device) -> Any:
    if stage == "ss":
        return sample["target"].to(device=device, dtype=torch.float32)[None]
    return _shape_tensor(sample, device)


def diffuse(target: Any, t: float) -> tuple[Any, Any]:
    scale = SIGMA_MIN + (1.0 - SIGMA_MIN) * float(t)
    if torch.is_tensor(target):
        noise = torch.randn_like(target)
        x_t = (1.0 - float(t)) * target + scale * noise
        velocity = (1.0 - SIGMA_MIN) * noise - target
        return x_t, velocity
    noise_feats = torch.randn_like(target.feats)
    x_t = target.replace((1.0 - float(t)) * target.feats + scale * noise_feats)
    velocity = target.replace((1.0 - SIGMA_MIN) * noise_feats - target.feats)
    return x_t, velocity


def flow_mse(prediction: Any, target: Any) -> torch.Tensor:
    if torch.is_tensor(prediction):
        return F.mse_loss(prediction.float(), target.float())
    if not torch.equal(prediction.coords, target.coords):
        raise RuntimeError("sparse prediction/target coordinates differ")
    return F.mse_loss(prediction.feats.float(), target.feats.float())


def sample_t(stage: str, device: torch.device) -> float:
    if stage == "ss":
        value = torch.sigmoid(torch.randn((), device=device) + 1.0)
    else:
        value = torch.rand((), device=device)
    return float(value.item())


def warmup_steps(args: argparse.Namespace) -> int:
    value = int(args.warmup_steps)
    if value < 0:
        value = 600 if args.stage == "ss" else 3_000
    if value > int(args.max_steps):
        raise ValueError("warmup exceeds max steps")
    return value


def warmup_factor(step: int, count: int) -> float:
    return 1.0 if int(count) <= 0 else min(1.0, float(step) / float(count))


def all_finite(parameters: list[nn.Parameter]) -> bool:
    return bool(parameters) and all(
        bool(torch.isfinite(parameter.detach()).all().item())
        for parameter in parameters
    )


def distributed_bool(value: bool, device: torch.device, world_size: int) -> bool:
    tensor = torch.tensor(int(value), device=device)
    if int(world_size) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def mean_across_ranks(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if int(world_size) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(float(world_size))
    return float(tensor.item())


@torch.no_grad()
def initial_stock_audit(
    model: nn.Module,
    sample: dict[str, Any],
    *,
    stage: str,
    context_mode: str,
    min_views: int,
    max_views: int,
    device: torch.device,
) -> dict[str, Any]:
    target = target_tensor(sample, stage, device)
    x_t, _ = diffuse(target, 0.5)
    indices = select_view_indices(
        int(sample["dino_features"].shape[0]),
        minimum=min_views,
        maximum=max_views,
        device="cpu",
    )
    context = build_2d_context(
        sample["dino_features"], indices, mode=context_mode  # type: ignore[arg-type]
    ).to(device)
    t = torch.full((1,), 500.0, device=device)
    stock = model.stock_prediction(x_t, t, context)
    condition, stats = model._pose_condition(x_t, sample, indices)
    raw = model._adapted_core_forward(x_t, t, context, condition)
    if torch.is_tensor(raw):
        raw_difference = float((raw.float() - stock.float()).abs().amax().item())
    else:
        raw_difference = float(
            (raw.feats.float() - stock.feats.float()).abs().amax().item()
        )
    prediction, public_stats = model.adapted_prediction(
        x_t, t, context, sample, view_indices=indices, stock_velocity=stock
    )
    if torch.is_tensor(prediction):
        anchored_difference = float(
            (prediction.float() - stock.float()).abs().amax().item()
        )
    else:
        anchored_difference = float(
            (prediction.feats.float() - stock.feats.float()).abs().amax().item()
        )
    report = {
        "stage": stage,
        "raw_zero_adapter_max_abs": raw_difference,
        "anchored_zero_adapter_max_abs": anchored_difference,
        "supported_fraction": float(stats["supported_fraction"].item()),
        "zero_init_stock_anchor": float(public_stats["zero_init_stock_anchor"].item()),
        "selected_views": int(len(indices)),
    }
    report["passed"] = bool(
        math.isfinite(raw_difference)
        and raw_difference <= 2.0e-2
        and anchored_difference == 0.0
        and report["supported_fraction"] > 0.0
        and report["zero_init_stock_anchor"] == 1.0
    )
    if not report["passed"]:
        raise RuntimeError(f"initial stock-preservation audit failed: {report}")
    return report


def _model_identity(
    args: argparse.Namespace, model: nn.Module, audit: dict[str, Any] | None
) -> dict[str, Any]:
    source = args.model_path or (
        SS_FLOW_MODEL if args.stage == "ss" else SHAPE_FLOW_MODEL
    )
    root = trellis2_root()
    return {
        "stage": args.stage,
        "base_model": source,
        "base_initialization": "official_trellis2_checkpoint",
        "old_30k_checkpoint_loaded": False,
        "distillation": False,
        "trellis2_root": str(root),
        "dino_full_token_count": DINO_FULL_TOKENS,
        "context_mode": args.context_mode,
        "pose_condition_views": "all_selected",
        "cross_attention_reference": "first_selected"
        if args.context_mode == "first"
        else "concatenated_selected",
        "trainable_surface": validate_trainable_surface(model),
        "initial_stock_audit": audit,
    }


def main() -> None:
    args = make_parser().parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 adapter training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(hours=12)
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    process_seed = int(args.seed) + rank * 100_003
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32 - 1))
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)

    dataset = PoseTrellis2Dataset(
        args.manifest,
        stage=args.stage,
        indices=args.indices,
        verify_hashes=bool(args.verify_cache_hashes),
    )
    if int(args.max_objects) > 0:
        dataset.rows = dataset.rows[: int(args.max_objects)]
    if len(dataset) == 0:
        raise RuntimeError("training selected no objects")
    data_identity = dataset.identity()
    distributed_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(args.seed),
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=distributed_sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_one,
        # Keep iterator/worker seeding off the global CPU RNG.  This is
        # required for an exact mid-epoch resume because view-subset sampling
        # also uses the global CPU generator.
        generator=torch.Generator(),
    )

    model = load_pose_conditioned_flow(
        args.stage,
        device=device,
        model_path=args.model_path or None,
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    audit = None
    if not args.resume and not args.skip_initial_stock_audit:
        audit = initial_stock_audit(
            model,
            dataset[0],
            stage=args.stage,
            context_mode=args.context_mode,
            min_views=int(args.min_condition_views),
            max_views=int(args.max_condition_views),
            device=device,
        )
    model_identity = _model_identity(args, model, audit)
    training_forward = TrainingForward(model)
    wrapped: nn.Module = training_forward
    if world_size > 1:
        wrapped = DistributedDataParallel(
            training_forward,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model,
            new_lr=float(args.new_lr),
            lora_lr=float(args.lora_lr),
            new_weight_decay=float(args.new_weight_decay),
        ),
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = float(group["lr"])
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")
    ema = initialize_ema(model)
    step = 0
    micro_step = 0
    history: list[dict[str, Any]] = []
    output = Path(args.output_dir).expanduser().resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume, model=model, stage=args.stage, weights="raw"
        )
        if checkpoint.get("data_identity") != data_identity:
            raise RuntimeError("resume data identity differs")
        saved_identity = checkpoint.get("model_identity", {})
        comparable = (
            "stage",
            "base_model",
            "context_mode",
            "dino_full_token_count",
        )
        mismatch = {
            key: (saved_identity.get(key), model_identity.get(key))
            for key in comparable
            if saved_identity.get(key) != model_identity.get(key)
        }
        if mismatch:
            raise RuntimeError(f"resume model identity differs={mismatch}")
        saved_args = checkpoint.get("args", {})
        resume_contract = (
            "seed",
            "lora_rank",
            "lora_alpha",
            "new_lr",
            "lora_lr",
            "new_weight_decay",
            "grad_clip",
            "warmup_steps",
            "ema_decay",
            "amp_dtype",
            "p_uncond",
            "min_condition_views",
            "max_condition_views",
            "context_mode",
            "gradient_checkpointing",
            "grad_accum",
        )
        argument_mismatch = {
            key: (saved_args.get(key), getattr(args, key))
            for key in resume_contract
            if saved_args.get(key) != getattr(args, key)
        }
        if argument_mismatch:
            raise RuntimeError(
                f"resume training contract differs={argument_mismatch}"
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        ema = ema_to_device(model, checkpoint["ema_trainable_state"])
        step = int(checkpoint["step"])
        micro_step = int(checkpoint.get("micro_step", step * int(args.grad_accum)))
        history = list(checkpoint.get("history", []))
        saved_rng = checkpoint.get("rng_by_rank")
        if saved_rng is not None:
            if not isinstance(saved_rng, list) or len(saved_rng) != world_size:
                raise RuntimeError(
                    "resume RNG state count differs from the current world size"
                )
            restore_rng_state(saved_rng[rank])
        else:
            if world_size > 1:
                raise RuntimeError(
                    "distributed resume checkpoint lacks per-rank RNG states"
                )
            restore_rng_state(checkpoint.get("rng", {}))

    resolved_warmup = warmup_steps(args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    model.train()
    batches_per_epoch = len(loader)
    if batches_per_epoch <= 0:
        raise RuntimeError("training loader has no per-rank batches")
    epoch = micro_step // batches_per_epoch
    resume_batch_offset = micro_step % batches_per_epoch
    started = time.time()
    accumulation: list[dict[str, Any]] = []
    while step < int(args.max_steps):
        distributed_sampler.set_epoch(epoch)
        assert loader.generator is not None
        loader.generator.manual_seed(int(args.seed) + 1_000_003 * epoch)
        for batch_index, sample in enumerate(loader):
            if batch_index < resume_batch_offset:
                continue
            if step >= int(args.max_steps):
                break
            view_indices = select_view_indices(
                int(sample["dino_features"].shape[0]),
                minimum=int(args.min_condition_views),
                maximum=int(args.max_condition_views),
                device="cpu",
            )
            positive_context = build_2d_context(
                sample["dino_features"],
                view_indices,
                mode=args.context_mode,
            ).to(device=device)
            unconditional = random.random() < float(args.p_uncond)
            context = (
                torch.zeros_like(positive_context)
                if unconditional
                else positive_context
            )
            pose_sample = None if unconditional else sample
            used_views = None if unconditional else view_indices
            target = target_tensor(sample, args.stage, device)
            t_value = sample_t(args.stage, device)
            x_t, target_velocity = diffuse(target, t_value)
            t = torch.full((1,), 1000.0 * t_value, device=device)
            stock = None
            stock_loss = None
            if int(args.stock_eval_every) > 0 and (
                (step + 1) % int(args.stock_eval_every) == 0
            ):
                with torch.no_grad():
                    stock = model.stock_prediction(x_t, t, context)
                    stock_loss = flow_mse(stock, target_velocity)
            synchronized = (micro_step + 1) % int(args.grad_accum) == 0
            sync_context = (
                wrapped.no_sync()
                if world_size > 1 and not synchronized
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    prediction, stats = wrapped(
                        x_t,
                        t,
                        context,
                        pose_sample,
                        view_indices=used_views,
                        stock_velocity=stock,
                    )
                    loss_value = flow_mse(prediction, target_velocity)
                    loss = loss_value / float(args.grad_accum)
                scaler.scale(loss).backward()
            micro_step += 1
            accumulation.append(
                {
                    "loss": float(loss_value.detach().item()),
                    "stock_loss": (
                        float(stock_loss.detach().item())
                        if stock_loss is not None
                        else None
                    ),
                    "t": t_value,
                    "views": float(0 if unconditional else len(view_indices)),
                    "supported_fraction": float(
                        stats["supported_fraction"].detach().item()
                    ),
                }
            )
            if not synchronized:
                continue
            scaler.unscale_(optimizer)
            gradients_finite = all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in trainable
            )
            if not distributed_bool(gradients_finite, device, world_size):
                raise RuntimeError("non-finite adapter gradients")
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip)).item()
            )
            next_step = step + 1
            lr_scale = warmup_factor(next_step, resolved_warmup)
            for group in optimizer.param_groups:
                group["lr"] = float(group["base_lr"]) * lr_scale
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not distributed_bool(all_finite(trainable), device, world_size):
                raise RuntimeError("optimizer produced non-finite adapter weights")
            decay = ema_ramp_decay(float(args.ema_decay), next_step)
            update_ema(model, ema, decay=decay)
            step = next_step
            mean_loss = mean_across_ranks(
                float(np.mean([row["loss"] for row in accumulation])),
                device,
                world_size,
            )
            measured_stock = [
                float(row["stock_loss"])
                for row in accumulation
                if row["stock_loss"] is not None
            ]
            mean_stock = (
                mean_across_ranks(float(np.mean(measured_stock)), device, world_size)
                if measured_stock
                else None
            )
            mean_support = mean_across_ranks(
                float(np.mean([row["supported_fraction"] for row in accumulation])),
                device,
                world_size,
            )
            if rank == 0 and (step == 1 or step % int(args.log_every) == 0):
                record = {
                    "step": step,
                    "loss": mean_loss,
                    "stock_loss": mean_stock,
                    "gain": mean_stock - mean_loss if mean_stock is not None else None,
                    "supported_fraction": mean_support,
                    "grad_norm": grad_norm,
                    "lr_scale": lr_scale,
                    "ema_decay": decay,
                    "elapsed_seconds": time.time() - started,
                }
                history.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            accumulation.clear()
            checkpoint_due = (
                step % int(args.save_every) == 0 or step == int(args.max_steps)
            )
            if checkpoint_due:
                local_rng = rng_state()
                if world_size > 1:
                    rng_by_rank: list[dict[str, Any] | None] = [None] * world_size
                    dist.all_gather_object(rng_by_rank, local_rng)
                    if any(value is None for value in rng_by_rank):
                        raise RuntimeError("failed to gather per-rank RNG states")
                    gathered_rng = [
                        value for value in rng_by_rank if value is not None
                    ]
                else:
                    gathered_rng = [local_rng]
            if rank == 0 and checkpoint_due:
                checkpoint_path = output / "checkpoints" / f"step_{step:06d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    ema=ema,
                    stage=args.stage,
                    step=step,
                    micro_step=micro_step,
                    args=vars(args),
                    data_identity=data_identity,
                    model_identity=model_identity,
                    history=history,
                    rng_by_rank=gathered_rng,
                )
                update_checkpoint_alias(
                    checkpoint_path,
                    output / "checkpoints" / "last.pt",
                )
        resume_batch_offset = 0
        epoch += 1
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
