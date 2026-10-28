#!/usr/bin/env python3
"""Evaluate ray depth, stopping, and target ownership, including hidden layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoints import load_target_lari_checkpoint
from .paired_dataset import PairedLayerDataset
from .target_lari import TargetConditionedLaRI, _resize_supervision
from .train_target_lari import _model_forward


def _mean_selected(value: torch.Tensor, mask: torch.Tensor) -> float | None:
    return float(value[mask].float().mean().item()) if bool(mask.any()) else None


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require_disjoint_from_training", action="store_true")
    parser.add_argument("--raw_view_ablation", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = PairedLayerDataset(args.manifest, build_trellis_contexts=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["model_metadata"]
    model = TargetConditionedLaRI(
        input_channels=int(metadata["input_channels"]),
        hidden_channels=int(metadata["hidden_channels"]),
        layers=int(metadata["layers"]),
        decoder_blocks=int(metadata["decoder_blocks"]),
        minimum_depth_gap=float(metadata["minimum_depth_gap"]),
    ).to(device).eval()
    binding = load_target_lari_checkpoint(model, args.checkpoint)
    training_identity = checkpoint.get("data_identity", {})
    training_objects = training_identity.get("object_uids")
    evaluation_objects = dataset.identity()["object_uids"]
    if training_objects is None:
        overlap = evaluation_objects if training_identity.get("object_uid_hash") == dataset.identity()["object_uid_hash"] else None
    else:
        overlap = sorted(set(training_objects) & set(evaluation_objects))
    if args.require_disjoint_from_training and (overlap is None or len(overlap) > 0):
        raise ValueError("evaluation objects overlap training, or their disjointness cannot be verified")
    modes = ("correct", "raw_view_cyclic1") if args.raw_view_ablation else ("correct",)
    rows = []
    for sample in dataset:
        for mode in modes:
            current = dict(sample)
            if mode == "raw_view_cyclic1":
                current["raw_visual_patch_features"] = torch.roll(current["raw_visual_patch_features"], 1, 0)
            prediction = _model_forward(model, current, device)
            depths, stop, target = _resize_supervision(sample, size=prediction.spatial_size)
            depths, stop, target = depths.to(device), stop.to(device), target.to(device)
            layers = torch.arange(prediction.layer_count, device=device)[None, :, None, None]
            valid = layers < stop[:, None]
            target_gt = target.bool() & valid
            non_target_before = ((~target.bool() & valid).long().cumsum(1) - (~target.bool() & valid).long()) > 0
            hidden_target = target_gt & non_target_before
            predicted_stop = prediction.stop_logits.argmax(1)
            predicted_target = (prediction.target_logits > 0) & (layers < predicted_stop[:, None])
            intersection = int((target_gt & predicted_target).sum())
            union = int((target_gt | predicted_target).sum())
            error = (prediction.depths.float() - depths).abs()
            rows.append({
                "uid": sample["uid"],
                "mode": mode,
                "valid_depth_mae": _mean_selected(error, valid),
                "target_depth_mae": _mean_selected(error, target_gt),
                "occluded_target_depth_mae": _mean_selected(error, hidden_target),
                "stop_accuracy": float((predicted_stop == stop).float().mean()),
                "target_ownership_iou_on_layer_slots": intersection / union if union else None,
                "valid_intersections": int(valid.sum()),
                "occluded_target_intersections": int(hidden_target.sum()),
            })
    report = {
        "passed": True,
        "scope": "layer-prediction metrics; not mesh reconstruction accuracy",
        "checkpoint": binding,
        "evaluation_data": dataset.identity(),
        "training_overlap_object_uids": overlap,
        "heldout_verified": overlap == [],
        "records": rows,
    }
    destination = Path(args.output_report).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
