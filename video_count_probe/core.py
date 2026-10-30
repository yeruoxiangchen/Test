"""Experiment definitions; this module intentionally needs only the stdlib."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path

DEFAULT_ROOT = "/data/zjr/video_count_probe_20261025"
DEFAULT_MODEL = "/data/zjr/models/Wan2.1-T2V-1.3B-Diffusers"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
METHODS = ("count_prompt", "timed_prompt", "relay_unrouted", "relay")
SOURCE_ROOT = Path(__file__).resolve().parent
NEGATIVE = (
    "overexposed, blurred details, subtitles, text, watermark, paintings, low quality, "
    "JPEG artifacts, deformed anatomy, extra limbs, fused fingers, camera cuts, "
    "multiple people, camera movement"
)
ACTIONS = {
    "clap": {
        "scene": "A single adult wearing a blue shirt stands facing the camera in a well-lit plain room. Both hands and the upper body remain clearly visible.",
        "name": "hand clap",
        "cycle": "brings both hands together until the palms touch, then separates them again",
        "rest": "keeps both hands apart and still",
        "criterion": "One clap = one visible palm contact followed by separation. Do not count approach and separation as two claps.",
    },
    "jumping_jack": {
        "scene": "A single adult wearing a green sports shirt stands facing the camera in an empty gym. Their entire body, including both feet and hands, remains visible.",
        "name": "jumping jack",
        "cycle": "jumps from feet together and arms down to feet apart and arms overhead, then jumps back to feet together and arms down",
        "rest": "stands still with feet together and arms down",
        "criterion": "One jumping jack = closed stance -> open stance -> closed stance. Half cycles do not count as completed cycles.",
    },
    "squat": {
        "scene": "A single adult wearing a red sports shirt stands in an empty gym, seen from a fixed three-quarter view. Their entire body and feet remain clearly visible.",
        "name": "bodyweight squat",
        "cycle": "bends both knees and lowers the hips into a squat, then rises back to a fully standing posture",
        "rest": "stands still in an upright posture",
        "criterion": "One squat = upright -> visibly lowered hips and bent knees -> upright. Incomplete descents or rises do not count as completed cycles.",
    },
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash():
    return digest({str(p.relative_to(SOURCE_ROOT)): file_sha(p) for p in sorted(SOURCE_ROOT.glob("*.py"))})


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        with open(temp, "w") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextlib.contextmanager
def writer_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer is using {path}; do not start duplicate jobs.") from exc
        try:
            f.seek(0)
            f.truncate()
            f.write(str(os.getpid()))
            f.flush()
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def split_length(total, count):
    q, r = divmod(total, count)
    return [q + (i < r) for i in range(count)]


def make_case(action, count, schedule, frames=81, fps=16):
    if action not in ACTIONS or count not in (1, 2, 3):
        raise ValueError("Unknown action/count")
    if frames < 81 or frames % 4 != 1 or fps <= 0:
        raise ValueError("Count experiment needs >=81 frames of the form 4n+1 and positive fps")
    if schedule not in ("regular", "pause_after_two") or (schedule != "regular" and count != 3):
        raise ValueError("The pause condition has exactly three repetitions")
    spec = ACTIONS[action]
    latent_frames = (frames - 1) // 4 + 1
    pause = 5 if schedule == "pause_after_two" else 0
    lengths = split_length(latent_frames - pause, count)
    segments, cursor = [], 0
    for i, length in enumerate(lengths):
        if length < 4:
            raise ValueError("Prompt Relay's official window needs >=4 latent frames per segment")
        segments.append({"kind": "action", "ordinal": i + 1, "start": cursor, "end": cursor + length,
                         "text": f"During action {i + 1}, the person {spec['cycle']} exactly once, completing one full {spec['name']}, then {spec['rest']}."})
        cursor += length
        if pause and i == 1:
            segments.append({"kind": "pause", "ordinal": None, "start": cursor, "end": cursor + pause,
                             "text": f"During the pause after action 2, the person {spec['rest']} without performing any action."})
            cursor += pause
    global_prompt = (
        spec["scene"] + " A single continuous realistic shot with a fixed camera. "
        f"The person performs exactly {count} complete repetitions of a {spec['name']} in the whole video."
    )
    count_prompt = global_prompt + f" Each complete repetition means the person {spec['cycle']}."
    if pause:
        count_prompt += " After the second repetition the person pauses visibly before the third."
    full_prompt = global_prompt
    for segment in segments:
        full_prompt += " "
        segment["char_start"] = len(full_prompt)
        full_prompt += segment["text"]
        segment["char_end"] = len(full_prompt)
        segment["start_seconds"] = segment["start"] * 4 / fps
        segment["end_seconds"] = min((frames - 1) / fps, segment["end"] * 4 / fps)
    timed = global_prompt + " " + " ".join(
        f"From {s['start_seconds']:.2f} to {s['end_seconds']:.2f} seconds: {s['text']}" for s in segments)
    return {"id": f"{action}_n{count}_{schedule}", "action": action, "target_count": count,
            "schedule": schedule, "criterion": spec["criterion"], "segments": segments,
            "latent_frames": latent_frames, "count_prompt": count_prompt,
            "timed_prompt": timed, "relay_prompt": full_prompt}


def build_plan(mode):
    if mode not in ("smoke", "pilot"):
        raise ValueError(mode)
    cases = [make_case(a, n, "regular") for a in ACTIONS for n in (1, 2, 3)]
    cases += [make_case(a, 3, "pause_after_two") for a in ACTIONS]
    seeds = [42, 123]
    if mode == "smoke":
        cases = [make_case("clap", 3, "regular")]
        seeds = [42]
    config = {"model_id": MODEL_ID, "height": 480 if mode == "pilot" else 240,
              "width": 832 if mode == "pilot" else 416, "frames": 81, "fps": 16,
              "steps": 50 if mode == "pilot" else 4, "guidance_scale": 6.0,
              "flow_shift": 8.0, "negative_prompt": NEGATIVE, "max_sequence_length": 512,
              "relay_sigma": 0.1448, "relay_window": "segment_length//2-2",
              "dtype": "bfloat16", "vae_dtype": "float32", "scheduler": "UniPCMultistepScheduler"}
    jobs = []
    for case in cases:
        for seed in seeds:
            methods = list(METHODS)
            random.Random(case["id"] + str(seed)).shuffle(methods)
            for method in methods:
                job = {"case_id": case["id"], "seed": seed, "method": method}
                job["id"] = digest(job)[:16]
                jobs.append(job)
    plan = {"schema": 1, "mode": mode, "scientific_evaluation": mode == "pilot",
            "config": config, "cases": cases, "jobs": jobs, "methods": list(METHODS),
            "limitations": ["Wan2.1-1.3B only; no conclusion about all video generators.",
                             "Prompt Relay formula port to Diffusers/Wan2.1; not official Wan2.2 reproduction.",
                             "TempoControl not run: official code lists ~110GB single-GPU memory.",
                             "No learned method or fine-tuning in this initial baseline screen.",
                             "Human event annotations are required; generated video count is not a quality metric."]}
    plan["plan_hash"] = digest(plan)
    return plan


def prepare(root, mode):
    path = Path(root) / mode / "plan.json"
    plan = build_plan(mode)
    with writer_lock(path.parent / ".writer.lock"):
        if path.exists() and read_json(path) != plan:
            raise RuntimeError(f"Existing plan differs: {path}. Use another --root.")
        atomic_json(path, plan)
    return plan


def checked_plan(path):
    plan = read_json(path)
    content = {k: v for k, v in plan.items() if k != "plan_hash"}
    if digest(content) != plan["plan_hash"]:
        raise ValueError("Plan fingerprint mismatch")
    return plan


def token_groups(offsets, segments):
    """Offsets avoid the first-occurrence bug with repeated identical action text."""
    groups, used = [], set()
    for s in segments:
        ids = [i for i, (a, b) in enumerate(offsets) if b > a and a < s["char_end"] and b > s["char_start"]]
        if not ids or used.intersection(ids):
            raise ValueError("Empty/overlapping local token span; check tokenizer offsets and truncation")
        used.update(ids)
        groups.append({"start": s["start"], "end": s["end"], "tokens": ids})
    return groups


def is_complete(run_dir, job, identity):
    path = Path(run_dir) / "records" / (job["id"] + ".json")
    if not path.exists():
        return False
    record = read_json(path)
    if record.get("status") != "ok":
        return False
    if record.get("identity") != identity or record.get("job") != job:
        raise RuntimeError(f"Result identity differs: {path}; use a new output root")
    video = Path(run_dir) / record["video"]
    return video.is_file() and file_sha(video) == record["video_sha256"]

