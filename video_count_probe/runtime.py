from __future__ import annotations

import contextlib
import csv
import faulthandler
import importlib.metadata
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .core import (MODEL_ID, atomic_json, build_plan, checked_plan, digest, file_sha, is_complete,
                   read_json, source_hash, token_groups, writer_lock)

LOCK_NAME = "video_count_model.json"


def versions():
    packages = ["torch", "diffusers", "transformers", "accelerate", "huggingface-hub", "safetensors", "tokenizers"]
    return {p: importlib.metadata.version(p) for p in packages}


def download_model(model_dir, tokenizer_only=False, attempts=5):
    """Explicit long-running operation. All weights/cache stay under the chosen directory."""
    from huggingface_hub import HfApi, hf_hub_download
    from .downloads import retry_hub_call

    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    with writer_lock(model_dir / ".download.lock"):
        status_path = model_dir / "download_status.json"
        receipt_path = model_dir / LOCK_NAME
        pin_path = model_dir / "download_revision.json"
        old = read_json(receipt_path) if receipt_path.exists() else (read_json(pin_path) if pin_path.exists() else None)
        info = retry_hub_call(lambda: HfApi().model_info(MODEL_ID, revision=old["revision"] if old else None,
                                                        files_metadata=True, timeout=30),
                              "repository metadata", status_path, attempts=attempts)
        paths = [s for s in info.siblings if s.rfilename == "model_index.json" or
                 s.rfilename.startswith(("scheduler/", "tokenizer/", "text_encoder/", "transformer/", "vae/"))]
        paths = [s for s in paths if s.rfilename.endswith((".json", ".safetensors", ".model", ".txt"))
                 and not any(x in s.rfilename for x in (".fp16.", ".bf16.", ".fp32."))]
        if not paths or not any(s.rfilename.startswith("transformer/") and s.rfilename.endswith(".safetensors") for s in paths):
            raise RuntimeError("Unexpected model repository layout")
        full_bytes = sum(s.size or 0 for s in paths)
        if tokenizer_only:
            paths = [s for s in paths if s.rfilename.startswith("tokenizer/") or
                     s.rfilename.endswith("config.json") or s.rfilename == "model_index.json"]
        print(f"Full model: {full_bytes/1e9:.2f} GB. Downloading selected files {sum(s.size or 0 for s in paths)/1e9:.3f} GB from {MODEL_ID}@{info.sha} to {model_dir}", flush=True)
        # Pin even an interrupted download so a retry cannot mix revisions.
        atomic_json(pin_path, {"model_id": MODEL_ID, "revision": info.sha})
        entries = []
        for index, s in enumerate(paths, 1):
            print(f"[{index}/{len(paths)}] downloading/resuming {s.rfilename} ({(s.size or 0)/1e9:.3f} GB)", flush=True)
            retry_hub_call(lambda: hf_hub_download(MODEL_ID, filename=s.rfilename, revision=info.sha,
                                                   local_dir=str(model_dir), force_download=False, etag_timeout=30),
                           s.rfilename, status_path, attempts=attempts)
            path = model_dir / s.rfilename
            if s.size is not None and path.stat().st_size != s.size:
                raise RuntimeError(f"Wrong file size: {path}")
            print(f"Verifying {s.rfilename}", flush=True)
            atomic_json(status_path, {"state": "running", "stage": "verify", "item": s.rfilename,
                                      "files_verified": index - 1, "files_total": len(paths)})
            actual = file_sha(path)
            lfs = s.lfs
            expected = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            if expected and expected != actual:
                raise RuntimeError(f"Checksum mismatch: {path}")
            entries.append({"path": s.rfilename, "size": path.stat().st_size,
                            "mtime_ns": path.stat().st_mtime_ns, "sha256": actual})
        receipt = {"model_id": MODEL_ID, "revision": info.sha, "files": entries}
        if tokenizer_only:
            atomic_json(model_dir / "tokenizer_files.json", receipt)
            print("Tokenizer/configuration staged; full model weights are still required.", flush=True)
        else:
            atomic_json(receipt_path, receipt)
            print(f"Verified model ready: {receipt_path}", flush=True)
        audit = audit_tokenizer(model_dir)
        audit["revision"] = info.sha
        audit["full_model_bytes"] = full_bytes
        atomic_json(model_dir / "tokenizer_audit.json", audit)
        atomic_json(status_path, {"state": "complete", "stage": "tokenizer_only" if tokenizer_only else "full_model",
                                  "revision": info.sha, "files_verified": len(paths)})
        print(f"Tokenizer audit passed: {audit['prompts_checked']} prompts; longest={audit['max_tokens']}/512 tokens", flush=True)


def audit_tokenizer(model_dir):
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained(model_dir, subfolder="tokenizer", local_files_only=True)
    checked, max_tokens = 0, 0
    for case in build_plan("pilot")["cases"]:
        for field in ("count_prompt", "timed_prompt", "relay_prompt"):
            text = case[field]
            encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
            length = len(encoded.input_ids)
            if length > 512:
                raise RuntimeError(f"Prompt too long: {case['id']}/{field}={length}")
            if field == "relay_prompt":
                token_groups(encoded.offset_mapping, case["segments"])
            max_tokens = max(max_tokens, length)
            checked += 1
    return {"prompts_checked": checked, "max_tokens": max_tokens, "is_fast": tokenizer.is_fast}


def model_receipt(model_dir):
    model_dir = Path(model_dir)
    receipt_path = model_dir / LOCK_NAME
    if not receipt_path.exists():
        raise RuntimeError(f"Model is not staged/verified: {receipt_path}. Run run.sh download first.")
    receipt = read_json(receipt_path)
    if receipt["model_id"] != MODEL_ID:
        raise RuntimeError("Unexpected model id")
    for item in receipt["files"]:
        path = model_dir / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"] or path.stat().st_mtime_ns != item["mtime_ns"]:
            raise RuntimeError(f"Model file changed: {path}; rerun download to verify")
    return receipt


def environment(model_dir, require_model=True):
    print("Importing torch / Diffusers (slow /home I/O can delay this) ...", flush=True)
    faulthandler.dump_traceback_later(60, repeat=False)
    try:
        import torch
        from diffusers import WanPipeline, WanTransformer3DModel
        from transformers import T5TokenizerFast
        import imageio_ffmpeg
        data = {"python": sys.executable, "packages": versions(), "cuda_available": torch.cuda.is_available(),
                "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(), "model_dir": str(model_dir),
                "source_hash": source_hash()}
        if data["packages"]["diffusers"] != "0.35.1":
            raise RuntimeError("This adapter is tested against diffusers==0.35.1 only")
        if require_model:
            if not data["cuda_available"]:
                raise RuntimeError("CUDA is not available in this process")
            data["revision"] = model_receipt(model_dir)["revision"]
        return data
    finally:
        faulthandler.cancel_dump_traceback_later()


def check_gpu(allow_busy=False, min_free_mib=16000):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one physical GPU index or UUID")
    result = subprocess.run(["nvidia-smi", "-i", visible,
                             "--query-gpu=uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    row = next(csv.reader(result.stdout.splitlines()))
    uuid, free, used, utilization = row[0].strip(), int(row[1]), int(row[2]), int(row[3])
    if free < min_free_mib:
        raise RuntimeError(f"{visible}: only {free} MiB free, require {min_free_mib}; select another GPU")
    if not allow_busy and (used > 1000 or utilization > 10):
        raise RuntimeError(f"GPU {visible} is busy ({used} MiB used, {utilization}% utilization). Select an idle GPU; --allow-busy is an explicit sharing override.")
    return {"uuid": uuid, "physical_device": visible, "free_mib_before_load": free,
            "used_mib_before_load": used, "allow_busy": allow_busy}


def encode_cached(pipe, prompt, cache_dir, cache_identity):
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    key = digest({"prompt": prompt, "identity": cache_identity})
    path = Path(cache_dir) / (key + ".pt")
    if path.exists():
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value.get("key") != key or value["embedding"].shape != (1, 512, pipe.transformer.config.text_dim):
            raise RuntimeError(f"Invalid text cache: {path}")
        return value["embedding"].to("cuda:0")
    tokenized = pipe.tokenizer(prompt_clean(prompt), add_special_tokens=True, truncation=False)
    if len(tokenized.input_ids) > 512:
        raise ValueError(f"Prompt exceeds 512 tokens ({len(tokenized.input_ids)}); cannot truncate routing spans")
    with torch.no_grad():
        embedding = pipe._get_t5_prompt_embeds(prompt, max_sequence_length=512, device=torch.device("cuda:0"), dtype=torch.bfloat16)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.partial")
    torch.save({"key": key, "embedding": embedding.detach().cpu()}, temp)
    os.replace(temp, path)
    return embedding


def load_pipe(model_dir, config):
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler
    from transformers import T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(model_dir, subfolder="tokenizer", local_files_only=True)
    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(model_dir, vae=vae, tokenizer=tokenizer,
                                       torch_dtype=torch.bfloat16, local_files_only=True)
    if tuple(pipe.transformer.config.patch_size) != (1, 2, 2) or pipe.transformer.config.added_kv_proj_dim is not None:
        raise RuntimeError("Unexpected Wan transformer layout")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=config["flow_shift"])
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload(gpu_id=0)
    return pipe


def run_generation(root, mode, model_dir, allow_busy=False, limit=None, retry_failures=False):
    import torch
    from diffusers.utils import export_to_video
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from .routing import RelayRouting, temporal_bias

    run_dir = Path(root) / mode
    plan = checked_plan(run_dir / "plan.json")
    config = plan["config"]
    receipt = model_receipt(model_dir)
    package_versions = versions()
    if package_versions["diffusers"] != "0.35.1":
        raise RuntimeError("Expected diffusers==0.35.1")
    identity = {"plan_hash": plan["plan_hash"], "source_hash": source_hash(),
                "model_receipt_hash": digest(receipt), "packages": package_versions}
    with writer_lock(run_dir / ".writer.lock"):
        identity_path = run_dir / "identity.json"
        if identity_path.exists() and read_json(identity_path) != identity:
            raise RuntimeError("Code/model/environment changed since this run; use a fresh --root")
        atomic_json(identity_path, identity)
        pending = [j for j in plan["jobs"] if not is_complete(run_dir, j, identity)]
        if limit is not None:
            if limit < 1:
                raise ValueError("--limit must be positive")
            pending = pending[:limit]
        if not pending:
            print("All videos already verified complete", flush=True)
            return
        for job in pending:
            err = run_dir / "errors" / (job["id"] + ".json")
            if err.exists() and not retry_failures:
                raise RuntimeError(f"Previous failure {err}; inspect it then use --retry-failures")
        gpu = check_gpu(allow_busy)
        # A machine-wide per-GPU lock prevents this probe's smoke and pilot overlapping.
        with writer_lock(Path("/tmp") / ("video_count_probe_" + gpu["uuid"] + ".lock")):
            print(f"Loading {MODEL_ID}; {len(pending)} pending jobs, output={run_dir}", flush=True)
            pipe = load_pipe(model_dir, config)
            cases = {c["id"]: c for c in plan["cases"]}
            cache_identity = {"revision": receipt["revision"], "packages": package_versions, "source": identity["source_hash"]}
            cache = Path(root) / "text_cache"
            negative = encode_cached(pipe, config["negative_prompt"], cache, cache_identity)
            (run_dir / "videos").mkdir(parents=True, exist_ok=True)
            for index, job in enumerate(pending):
                started = time.perf_counter()
                path = run_dir / "videos" / (job["id"] + ".mp4")
                temp = path.with_name(job["id"] + ".partial.mp4")
                print(f"[{index+1}/{len(pending)}] {job['case_id']} seed={job['seed']} {job['method']}", flush=True)
                try:
                    case = cases[job["case_id"]]
                    prompt = case[job["method"]] if job["method"] in ("count_prompt", "timed_prompt") else case["relay_prompt"]
                    if prompt_clean(prompt) != prompt:
                        raise ValueError("Prompt cleaning changed character offsets")
                    embeds = encode_cached(pipe, prompt, cache, cache_identity)
                    route = None
                    manager = contextlib.nullcontext()
                    if job["method"].startswith("relay"):
                        offsets = pipe.tokenizer(prompt, return_offsets_mapping=True, padding="max_length",
                                                 max_length=512, truncation=False).offset_mapping
                        if len(offsets) != 512:
                            raise ValueError("Routing tokenizer length differs from encoded context")
                        groups = token_groups(offsets, case["segments"])
                        bias = temporal_bias(case["latent_frames"], 512, groups, config["relay_sigma"])
                        manager = RelayRouting(pipe.transformer, bias, (config["height"] // 16) * (config["width"] // 16),
                                               strength=1.0 if job["method"] == "relay" else 0.0)
                        route = {"groups": groups, "bias_sha256": __import__("hashlib").sha256(bias.numpy().tobytes()).hexdigest()}
                    generator = torch.Generator(device="cpu").manual_seed(job["seed"])
                    # Construct identical initial noise explicitly for every paired method.
                    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels, config["height"], config["width"],
                                                   config["frames"], torch.float32, torch.device("cpu"), generator)
                    latent_hash = __import__("hashlib").sha256(latents.numpy().tobytes()).hexdigest()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    generation_start = time.perf_counter()
                    with manager as routing:
                        output = pipe(prompt_embeds=embeds, negative_prompt_embeds=negative,
                                      height=config["height"], width=config["width"], num_frames=config["frames"],
                                      num_inference_steps=config["steps"], guidance_scale=config["guidance_scale"],
                                      latents=latents, generator=generator, output_type="np").frames[0]
                        routed_calls = routing.routed_calls if routing else 0
                    torch.cuda.synchronize()
                    generation_seconds = time.perf_counter() - generation_start
                    if job["method"] == "relay" and routed_calls != config["steps"] * len(pipe.transformer.blocks):
                        raise RuntimeError(f"Routing did not run on exactly the conditional passes: {routed_calls}")
                    if output.shape[0] != config["frames"] or not __import__("numpy").isfinite(output).all():
                        raise RuntimeError("Wrong frame count or nonfinite decoded frames")
                    export_to_video(output, str(temp), fps=config["fps"], macro_block_size=1)
                    os.replace(temp, path)
                    record = {"status": "ok", "job": job, "identity": identity, "prompt": prompt, "routing": route,
                              "initial_latent_sha256": latent_hash, "routed_calls": routed_calls,
                              "video": str(path.relative_to(run_dir)), "video_sha256": file_sha(path),
                              "generation_seconds": generation_seconds, "total_seconds": time.perf_counter() - started,
                              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                              "gpu": gpu, "frames": config["frames"], "fps": config["fps"]}
                    atomic_json(run_dir / "records" / (job["id"] + ".json"), record)
                    (run_dir / "errors" / (job["id"] + ".json")).unlink(missing_ok=True)
                    print(f"ok {job['id']} generation={generation_seconds:.1f}s peak={record['peak_allocated_mib']:.0f}MiB", flush=True)
                except BaseException as exc:
                    temp.unlink(missing_ok=True)
                    atomic_json(run_dir / "errors" / (job["id"] + ".json"),
                                {"job": job, "identity": identity, "exception": repr(exc), "traceback": traceback.format_exc()})
                    raise

