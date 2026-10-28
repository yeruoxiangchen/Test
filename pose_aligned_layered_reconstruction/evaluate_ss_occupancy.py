"""Paired, complete-sampling SS occupancy evaluation; no SLat/mesh required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoints import load_parent_ss30k, load_target_lari_checkpoint, load_layered_ss_checkpoint
from .native_ss import LayeredSSCFGFlow, build_layered_ss_components
from .paired_dataset import PairedLayerDataset, canonical_json_sha256
from .pretrained import DEFAULT_PRETRAINED, load_trellis_model
from .prepare_official_experiment import atomic_json, read_json


def occupancy_metrics(pred, target):
    def encoded(coords):
        coords = np.asarray(coords, dtype=np.int64)[:, -3:]
        if len(coords) and (coords.min() < 0 or coords.max() >= 64):
            raise ValueError("occupancy coordinates leave 64-cube")
        return np.unique(coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2])
    p, g = encoded(pred), encoded(target)
    if not len(g):
        raise ValueError("empty occupancy GT")
    tp = len(np.intersect1d(p, g))
    precision, recall = tp / max(len(p), 1), tp / len(g)
    return {"iou": tp / max(len(p) + len(g) - tp, 1), "precision": precision,
            "recall": recall, "f1": 2 * tp / max(len(p) + len(g), 1),
            "predicted_voxels": len(p), "target_voxels": len(g),
            "count_ratio": len(p) / len(g)}


def aggregate_records(records):
    keys = ("iou", "precision", "recall", "f1", "count_ratio")
    modes = sorted({r["mode"] for r in records})
    summary = {mode: {key: float(np.mean([r[key] for r in records if r["mode"] == mode])) for key in keys} for mode in modes}
    by_key = {(r["uid"], r["seed"], r["mode"]): r for r in records}
    pairs = []
    for uid, seed, mode in by_key:
        if mode == "candidate" and (uid, seed, "parent") in by_key:
            candidate, parent = by_key[(uid, seed, mode)], by_key[(uid, seed, "parent")]
            pairs.append({"uid": uid, "seed": seed, **{k + "_gain": candidate[k] - parent[k] for k in keys}})
    if pairs:
        objects = sorted({r["uid"] for r in pairs})
        object_gains = [float(np.mean([r["iou_gain"] for r in pairs if r["uid"] == uid])) for uid in objects]
        rng = np.random.default_rng(20260909)
        boot = np.asarray(object_gains)[rng.integers(0, len(objects), (2000, len(objects)))].mean(1)
        summary["paired"] = {"object_count": len(objects), "sample_pairs": len(pairs),
                             **{key + "_gain": float(np.mean([r[key + "_gain"] for r in pairs])) for key in keys},
                             "object_iou_win_rate": float(np.mean(np.asarray(object_gains) > 0)),
                             "iou_gain_object_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist()}
    return summary


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("parent", "candidate", "paired"), default="paired")
    parser.add_argument("--layered_ss_checkpoint", default="")
    parser.add_argument("--target_lari_checkpoint", default="")
    parser.add_argument("--layer_source", choices=("predicted", "oracle"), default="predicted")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=5.)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require_disjoint", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output_dir).resolve()
    if args.aggregate_only:
        results = [read_json(p) for p in sorted(output.glob("records/*.json"))]
        records = [r for result in results for r in result["records"]]
        summary = aggregate_records(records)
        atomic_json(output / "aggregate.json", {"summary": summary, "records": records})
        print(json.dumps(summary), flush=True)
        return
    dataset = PairedLayerDataset(args.manifest, require_labels=args.layer_source == "oracle", require_ss_target=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sampler, model, _ = build_layered_ss_components(device=device)
    parent = load_parent_ss30k(model)
    lari = load_target_lari_checkpoint(model.target_lari, args.target_lari_checkpoint) if args.target_lari_checkpoint else None
    candidate = None
    if args.mode != "parent":
        if not args.layered_ss_checkpoint:
            raise ValueError("candidate evaluation requires an adapter checkpoint")
        candidate = load_layered_ss_checkpoint(model, args.layered_ss_checkpoint, parent_binding=parent,
                                               target_lari_binding=lari, layer_source=args.layer_source)
        if args.require_disjoint:
            evaluation = set(dataset.identity()["object_uids"])
            for identity in [candidate["data_identity"], (lari or {}).get("data_identity")]:
                if identity is not None and ("object_uids" not in identity or evaluation & set(identity["object_uids"])):
                    raise RuntimeError("evaluation overlaps adapter/layer predictor training")
    decoder, decoder_binding = load_trellis_model(DEFAULT_PRETRAINED, "ss_dec_conv3d_16l8_fp16")
    decoder.to(device).eval().requires_grad_(False)
    model.eval().requires_grad_(False)
    identity = {"manifest_sha256": dataset.identity()["manifest_sha256"], "parent": parent,
                "candidate": candidate, "lari": lari, "decoder": decoder_binding,
                "mode": args.mode, "seeds": args.seeds, "steps": args.steps, "cfg": args.cfg,
                "layer_source": args.layer_source, "amp": "bf16", "threshold_logit": 0.,
                "ground_truth_removed_from_predicted_condition": True}
    identity_hash = canonical_json_sha256(identity)
    records = []
    modes = ("parent", "candidate") if args.mode == "paired" else (args.mode,)
    end = args.end or len(dataset)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"identity_{args.start}_{end}.json", identity)
    for index in range(args.start, end):
        sample = dataset[index]
        uid = str(sample["uid"])
        gt = sample["target_coords"].numpy()
        # Labels are allowed only for the explicit privileged oracle control.
        condition_sample = {k: v for k, v in sample.items() if k not in ("target", "target_coords")
                            and (args.layer_source == "oracle" or k not in ("layer_depths_gt", "ray_stop_index_gt", "layer_target_gt"))}
        positive = sample["stock_condition"].to(device)
        negative = torch.zeros_like(positive)
        for seed in map(int, args.seeds.split(",")):
            for mode in modes:
                destination = output / "records" / f"{index:05d}_{seed}_{mode}.json"
                if destination.is_file():
                    saved = read_json(destination)
                    if saved["identity_hash"] != identity_hash:
                        raise RuntimeError("evaluation resume identity mismatch")
                    records.extend(saved["records"])
                    continue
                generator = torch.Generator(device=device).manual_seed(seed * 1_000_003 + int(uid[:8], 16))
                initial = torch.randn((1, 8, 16, 16, 16), device=device, generator=generator)
                flow = LayeredSSCFGFlow(model, positive, condition_sample, layer_source=args.layer_source,
                                        use_layer_branch=mode != "parent")
                begin = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latent = sampler.sample(flow, initial, cond=positive, neg_cond=negative, steps=args.steps,
                                            cfg_strength=args.cfg, cfg_interval=[.5, 1.], rescale_t=3., verbose=False).samples
                    logits = decoder(latent.to(next(decoder.parameters()).dtype)).float()
                if not bool(torch.isfinite(latent).all()) or not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("non-finite occupancy inference")
                coords = torch.argwhere(logits > 0)[:, [0, 2, 3, 4]].cpu().numpy().astype(np.int32)
                row = {"index": index, "uid": uid, "seed": seed, "mode": mode,
                       **occupancy_metrics(coords, gt), "elapsed_seconds": time.time() - begin}
                coord_dir = output / "coordinates"
                coord_dir.mkdir(exist_ok=True)
                np.savez_compressed(coord_dir / f"{index:05d}_{seed}_{mode}.npz", coords=coords, uid=uid)
                atomic_json(destination, {"identity_hash": identity_hash, "records": [row]})
                records.append(row)
                print(json.dumps(row), flush=True)
    summary = aggregate_records(records)
    atomic_json(output / f"report_{args.start}_{end}.json", {"passed": True, "identity": identity,
                "scope": "SS_occupancy_not_final_mesh", "summary": summary, "records": records})
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
