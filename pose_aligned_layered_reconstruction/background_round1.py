"""Durable, fail-closed supervisor for the fixed 2026-09-09 first experiment.

Run under systemd --user, not an interactive tool session. No training resumes
are inferred: preparation/evaluation can reuse verified records, while an
interrupted formal optimizer run requires an explicit recovery decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
ROOT = PACKAGE / "outputs/round1_20260909"
DATA = ROOT / "data_v2"
TRAIN32_SHA = "9bc6b0396eeae2322bf038152a1849f6ab2c703377d71573d9e7695468b92dc7"
PARENT_SHA = "042a1b5467b05975584aeb571dec6ffaed5096edcc6abe4aa88600c9c9506b7f"
PREFIX = "pose_aligned_layered_reconstruction."
EXISTING_PREP = [
    (0, 400, "1", "prepare2000_0_400_qa_v2"),
    (400, 800, "3", "prepare2000_400_800_qa_v2"),
    (800, 1200, "4", "prepare2000_800_1200_qa_v2"),
    (1200, 1600, "5", "prepare2000_1200_1600_qa_v3"),
    (1600, 2000, "6", "prepare2000_1600_2000_qa_v3"),
]
MODULES = {"run_experiment_job", "prepare_official_experiment", "train_target_lari",
           "train_layered_ss", "train_distributed_round", "evaluate_ss_occupancy",
           "evaluate_target_lari", "assess_overfit"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def process_signature(pid):
    """PID + kernel start time + exact argv; zombies and PID reuse are not live."""
    try:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        argv = (path / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        return (int(pid), fields[19], tuple(argv))
    except (OSError, ValueError, IndexError):
        return None


def in_scope(signature):
    if signature is None:
        return False
    argv = signature[2]
    return any(PREFIX + module in argv for module in MODULES) and any(
        "pose_aligned_layered_reconstruction/outputs/round1_20260909/" in arg for arg in argv)


def validate_records(directory, manifest, expected_modes, expected_steps):
    """Refuse incomplete, duplicate or mixed-identity shards (including hashes)."""
    from .paired_dataset import canonical_json_sha256

    identities = [read(p) for p in sorted(Path(directory).glob("identity_*.json"))]
    if not identities or any(item != identities[0] for item in identities):
        raise ValueError("missing/mixed evaluation identities")
    identity = identities[0]
    if (identity["manifest_sha256"] != sha(manifest) or identity["steps"] != 25
            or identity["cfg"] != 5 or identity["seeds"] != "42,43"
            or identity["parent"]["sha256"] != PARENT_SHA
            or identity["candidate"]["step"] != expected_steps
            or identity["layer_source"] != "predicted"
            or not identity["ground_truth_removed_from_predicted_condition"]):
        raise ValueError("final evaluation differs from fixed data/model/protocol")
    rows = read(manifest)["samples"]
    expected = {(r["uid"], seed, mode) for r in rows for seed in (42, 43) for mode in expected_modes}
    index_uids = {index: r["uid"] for index, r in enumerate(rows)}
    records = []
    for path in sorted(Path(directory).glob("records/*.json")):
        payload = read(path)
        if payload["identity_hash"] != canonical_json_sha256(identity):
            raise ValueError(f"record identity mismatch: {path}")
        records.extend(payload["records"])
    keys = [(r["uid"], r["seed"], r["mode"]) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("incomplete or duplicated final evaluation")
    for record in records:
        if index_uids.get(record["index"]) != record["uid"]:
            raise ValueError("evaluation index/UID mismatch")
        if not all(math.isfinite(record[k]) for k in ("iou", "precision", "recall", "f1", "count_ratio")):
            raise ValueError("non-finite final metrics")
    return identity, records


class StopRequested(Exception):
    pass


def require_unstarted_formal_run(directory):
    """This recovery is only for initialization failure, never optimizer resume."""
    directory = Path(directory)
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if files != {"run_config.json"}:
        raise RuntimeError("old formal run contains training/artifact files; startup-only recovery forbidden")
    config = read(directory / "run_config.json")
    if config.get("stage") != "lari" or config.get("max_steps") != 4000 or config.get("world_size") != 6:
        raise RuntimeError("unexpected failed formal run configuration")


class Supervisor:
    def __init__(self, *, recover_formal_startup=False):
        from .prepare_official_experiment import source_rows

        self.rows = source_rows("train2000")
        self.children = {}
        self.tracked = {}
        self.jobs = set()
        self.prep = [{"start": a, "end": b, "gpu": gpu, "name": name, "retried": False}
                     for a, b, gpu, name in EXISTING_PREP]
        self.recover_formal_startup = recover_formal_startup
        self.prepare_enabled = not recover_formal_startup
        self.formal_a_name = "formal2000_lari_v2" if recover_formal_startup else "formal2000_lari_v1"
        self.formal_b_name = "formal2000_ss_v1"
        self.job_suffix = "_retry2" if recover_formal_startup else ""
        self.prep_errors = {}
        self.stop_requested = False
        self.stage = "adopting_existing_work"
        self.detail = "Keep existing overfit; formal training requires the final step2000 gate."
        self.last_heartbeat = 0.
        self.started = time.time()
        self.code_hashes = {p.name: sha(p) for p in sorted(PACKAGE.glob("*.py"))}
        self.log = (ROOT / "background.log").open("a", buffering=1)
        self.capture_processes()
        self.jobs.update(item[3] for item in EXISTING_PREP)
        self.jobs.add("overfit1000_eval_0_32")
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))

    def emit(self, message):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} [{self.stage}] {message}"
        print(line, flush=True)
        self.log.write(line + "\n")

    def capture_processes(self):
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                sig = process_signature(int(path.name))
                if in_scope(sig):
                    self.tracked[sig[0]] = sig

    def job_status(self, name):
        path = ROOT / "jobs" / (name + ".json")
        status = read(path) if path.exists() else {"name": name, "state": "missing"}
        if status["state"] in ("starting", "running"):
            sig = process_signature(status.get("pid", -1))
            if in_scope(sig):
                self.tracked[sig[0]] = sig
            elif name not in self.children or self.children[name].poll() is not None:
                status = {**status, "state": "interrupted"}
        if name in self.children:
            result = self.children[name].poll()
            if result is None and status["state"] == "missing":
                status = {**status, "state": "starting"}
            elif result is not None and status["state"] in ("missing", "running", "starting"):
                status = {**status, "state": "failed", "return_code": result}
        return status

    def launch(self, name, gpus, module, arguments, *, distributed=False):
        if self.stop_requested:
            raise StopRequested()
        if any(not (PACKAGE / p).exists() or sha(PACKAGE / p) != digest
               for p, digest in self.code_hashes.items()):
            raise RuntimeError("source code changed after background handoff; pause for review")
        for suffix in (".json", ".log"):
            if (ROOT / "jobs" / (name + suffix)).exists():
                raise FileExistsError(f"job name already used: {name}; do not infer optimizer resume")
        command = [sys.executable, "-u", "-m", PREFIX + "run_experiment_job", "--name", name,
                   "--gpus", gpus, "--module", module]
        if distributed:
            command.append("--distributed")
        command += ["--", *map(str, arguments)]
        self.emit("launch " + name)
        self.children[name] = subprocess.Popen(command, cwd=WORKSPACE, stdout=self.log, stderr=subprocess.STDOUT)
        self.jobs.add(name)
        self.capture_processes()
        return name

    def prepare_tick(self):
        if not self.prepare_enabled:
            return
        for item in self.prep:
            status = self.job_status(item["name"])
            if status["state"] in ("starting", "running", "complete"):
                continue
            if item["retried"]:
                self.prep_errors[item["name"]] = status
                continue
            # Avoid overlapping preparation and the inherited diagnostic on GPU5.
            if item["gpu"] == "5" and self.job_status("overfit1000_eval_0_32")["state"] in ("starting", "running"):
                continue
            name = f"background_prepare2000_{item['start']}_{item['end']}_v4"
            item["name"] = self.launch(name, item["gpu"], "prepare_official_experiment", [
                "--output_dir", DATA, "--split", "train2000", "--start", item["start"], "--end", item["end"]])
            item["retried"] = True

    def progress(self):
        result = {"prepared_train2000_objects": sum((DATA / "objects" / r["uid"] / "record.json").is_file()
                                                      for r in self.rows)}
        for name in ("overfit32_ss_predicted_v1", self.formal_a_name, self.formal_b_name):
            path = ROOT / name / "history.jsonl"
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 8192))
                    lines = stream.read().splitlines()
                try:
                    result[name] = json.loads(lines[-1])
                except (ValueError, IndexError):
                    pass  # Writer may be completing its current JSON line.
        for label, name in (("formal_lari", self.formal_a_name), ("formal_ss", self.formal_b_name)):
            startup = [read(path) for path in (ROOT / name).glob("startup_rank*.json")]
            result[label] = {"run_dir": name, "step": result.get(name, {}).get("step", 0),
                             "startup": startup,
                             "verified_objects": sum(row.get("verified_objects", 0) for row in startup)}
        result["dev64_manifest_ready"] = (DATA / "dev64.json").is_file()
        for name in ("overfit32_candidate2000", "formal_dev64_eval"):
            result[name + "_records"] = sum(1 for _ in (ROOT / name / "records").glob("*.json"))
        return result

    def status(self, state="running", error=None):
        value = {"state": state, "stage": self.stage, "detail": self.detail, "supervisor_pid": os.getpid(),
                 "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "elapsed_seconds": time.time() - self.started, "progress": self.progress(),
                 "formal_runs": {"lari": self.formal_a_name, "ss": self.formal_b_name},
                 "jobs": {name: {k: v for k, v in self.job_status(name).items()
                                  if k in ("state", "pid", "gpus", "log", "return_code")}
                          for name in sorted(self.jobs)}, "preparation_errors": self.prep_errors,
                 "error": error}
        atomic(ROOT / "background_status.json", value)
        return value

    def tick(self):
        if self.stop_requested or (ROOT / "STOP").exists():
            raise StopRequested()
        self.prepare_tick()
        if time.time() - self.last_heartbeat >= 30:
            value = self.status()
            self.emit("heartbeat " + json.dumps(value["progress"]))
            self.last_heartbeat = time.time()

    def set_stage(self, stage, detail):
        self.stage, self.detail = stage, detail
        self.emit(detail)
        self.status()

    def wait_job(self, name):
        start = time.time()
        while True:
            self.tick()
            status = self.job_status(name)
            if status["state"] == "complete":
                return
            if status["state"] not in ("starting", "running"):
                raise RuntimeError(f"job {name} {status['state']}; inspect {ROOT / 'jobs' / (name + '.log')}")
            if time.time() - start > 24 * 3600:
                raise TimeoutError(f"job {name} exceeds 24h safety limit")
            time.sleep(5)

    def wait_overfit(self):
        self.set_stage("overfit_training", "Waiting for the existing 32-object SS run to finish step2000.")
        path = ROOT / "overfit32_ss_predicted_v1/report.json"
        while True:
            self.tick()
            if path.exists():
                try:
                    report = read(path)
                except json.JSONDecodeError:
                    time.sleep(2)
                    continue
                if (not report["passed"] or report["last"]["step"] != 2000
                        or report["data_identity"]["manifest_sha256"] != TRAIN32_SHA
                        or not report["initial_zero_audit"]["passed"]):
                    raise RuntimeError("overfit training report failed fixed identity/step checks")
                if not any(process_signature(pid) == sig and PREFIX + "train_layered_ss" in sig[2]
                           for pid, sig in self.tracked.items()):
                    return
            elif not any(process_signature(pid) == sig and PREFIX + "train_layered_ss" in sig[2]
                         for pid, sig in self.tracked.items()):
                raise RuntimeError("overfit process ended without a complete report; no inferred training resume")
            time.sleep(5)

    def shutdown_children(self):
        # Adopted interactive jobs are outside the systemd cgroup. Signal only
        # exact, still-live PID/start-time/argv identities inside this experiment.
        self.capture_processes()
        for pid, sig in self.tracked.items():
            if process_signature(pid) == sig:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in self.children.values():
            if child.poll() is None:
                child.terminate()

    def run(self):
        if sha(DATA / "train32.json") != TRAIN32_SHA:
            raise RuntimeError("immutable train32 manifest changed")
        atomic(ROOT / "background_plan.json", {"source_code_sha256": self.code_hashes,
            "unit": "tracker-palr-round1-20260909.service", "gpus": [1, 2, 3, 4, 5, 6],
            "formal_lari_steps": 4000, "formal_ss_steps": 3000, "formal_objects": 2000,
            "overfit_final_steps": 2000, "dev_objects": 64, "seeds": [42, 43],
            "no_automatic_formal_optimizer_resume": True, "stop_on_gate_failure": True})
        self.wait_overfit()
        self.set_stage("overfit_final_evaluation", "Evaluate all 32 objects / two seeds at the fixed final step2000.")
        of_ss = ROOT / "overfit32_ss_predicted_v1/checkpoints/step_002000.pt"
        of_lari = ROOT / "overfit32_lari_v1/checkpoints/step_002000.pt"
        candidate_dir = ROOT / "overfit32_candidate2000"
        name = self.launch("background_overfit2000_eval", "2", "evaluate_ss_occupancy", [
            "--manifest", DATA / "train32.json", "--output_dir", candidate_dir, "--mode", "candidate",
            "--layered_ss_checkpoint", of_ss, "--target_lari_checkpoint", of_lari, "--seeds", "42,43"])
        self.wait_job(name)
        validate_records(candidate_dir, DATA / "train32.json", ("candidate",), 2000)
        gate = ROOT / "overfit32_gate_final.json"
        self.wait_job(self.launch("background_overfit_final_gate", "2", "assess_overfit", [
            "--manifest", DATA / "train32.json", "--baseline_dir", ROOT / "overfit32_parent",
            "--candidate_dir", candidate_dir,
            "--lari_initial_report", ROOT / "overfit32_lari_v1/eval_initial.json",
            "--lari_final_report", ROOT / "overfit32_lari_v1/eval_final.json", "--output_report", gate]))
        if not read(gate).get("formal_training_allowed"):
            raise RuntimeError("final overfit gate failed; formal training forbidden")
        self.set_stage("prepare_formal_data", "Final overfit gate passed; finish all 2000 objects without filtering.")
        while True:
            self.tick()
            if self.prep_errors:
                raise RuntimeError("formal data preparation failed after one verified-record retry; see preparation_errors")
            if all(self.job_status(item["name"])["state"] == "complete" for item in self.prep):
                break
            time.sleep(5)
        self.prepare_enabled = False
        self.wait_job(self.launch("background_finalize_train2000", "2", "prepare_official_experiment", [
            "--output_dir", DATA, "--split", "train2000", "--finalize_only"]))
        self.set_stage("prepare_dev64", "Materialize reserved Dev64 with strict split/camera/target checks.")
        self.wait_job(self.launch("background_prepare_dev64", "2", "prepare_official_experiment", [
            "--output_dir", DATA, "--split", "dev64"]))
        if self.job_status("overfit1000_eval_0_32")["state"] in ("running", "starting"):
            self.wait_job("overfit1000_eval_0_32")
        self.train_formal(gate)

    def run_formal_startup_recovery(self):
        from .paired_dataset import PairedLayerDataset
        from .prepare_official_experiment import source_rows

        old_status = read(ROOT / "background_status.json")
        if old_status["state"] != "blocked" or old_status["stage"] != "formal_lari_training":
            raise RuntimeError("startup recovery requires the known failed formal LaRI stage")
        require_unstarted_formal_run(ROOT / "formal2000_lari_v1")
        if sha(DATA / "train32.json") != TRAIN32_SHA:
            raise RuntimeError("overfit manifest changed")
        gate = ROOT / "overfit32_gate_final.json"
        gated = read(gate)
        if not gated.get("passed") or not gated.get("formal_training_allowed"):
            raise RuntimeError("cannot recover into formal training without a passed final gate")
        identity, _ = validate_records(ROOT / "overfit32_candidate2000", DATA / "train32.json", ("candidate",), 2000)
        if identity != gated["candidate_identity"]:
            raise RuntimeError("recovery gate and actual final overfit identity mismatch")
        data_identities = {}
        for split, expected in (("train2000", self.rows), ("dev64", source_rows("dev64"))):
            dataset = PairedLayerDataset(DATA / (split + ".json"), verify_hashes=False)
            if [row["uid"] for row in dataset.rows] != [row["uid"] for row in expected]:
                raise RuntimeError("recovery manifest changed object protocol")
            data_identities[split] = dataset.identity()
        # All sample bytes are checked by six disjoint verification shards in
        # the trainer before model initialization; no integrity gate is skipped.
        recovery_path = ROOT / "background_recovery2.json"
        if recovery_path.exists():
            raise FileExistsError("startup recovery attempt2 already recorded; inspect before retrying")
        atomic(recovery_path, {"previous_status": old_status, "source_code_sha256": self.code_hashes,
            "data_identities": data_identities, "reason": "rank0-only full cache SHA check exceeded NCCL barrier timeout",
            "previous_optimizer_updates": 0, "formal_lari_output": self.formal_a_name,
            "unchanged_budget": {"lari_steps": 4000, "ss_steps": 3000, "world_size": 6},
            "recovery": "explicit fresh startup after failure before optimizer initialization, not optimizer resume"})
        self.set_stage("formal_startup_recovery", "Keep completed data/overfit results; restart only unstarted formal training in a new directory.")
        self.train_formal(gate)

    def train_formal(self, gate):
        from .evaluate_ss_occupancy import aggregate_records

        train = DATA / "train2000.json"
        common = ["--manifest", train, "--overfit_gate", gate, "--save_every", "1000", "--log_every", "25",
                  "--num_workers", "2", "--grad_accum", "1", "--learning_rate", "0.0001", "--warmup_steps", "100",
                  "--seed", "42", "--min_condition_views", "8", "--max_condition_views", "8"]
        formal_a, formal_b = ROOT / self.formal_a_name, ROOT / self.formal_b_name
        for path in (formal_a, formal_b):
            if path.exists():
                raise FileExistsError(f"formal output exists: {path}; never restart optimizer implicitly")
        self.set_stage("formal_lari_training", "Fresh LaRI head, 2000 objects, 6 GPUs, 4000 updates.")
        self.wait_job(self.launch("background_formal_lari4000" + self.job_suffix, "1,2,3,4,5,6", "train_distributed_round",
            [*common, "--stage", "lari", "--output_dir", formal_a, "--max_steps", "4000"], distributed=True))
        a_report = read(formal_a / "report.json")
        if not a_report["passed"] or a_report["step"] != 4000 or a_report["rank_parameter_max_abs_difference"] != 0:
            raise RuntimeError("formal LaRI training integrity check failed")
        lari = formal_a / "checkpoints/step_004000.pt"
        self.set_stage("formal_ss_training", "Fresh SS30K + new zero-initialized conditions; frozen formal LaRI; 3000 updates.")
        self.wait_job(self.launch("background_formal_ss3000" + self.job_suffix, "1,2,3,4,5,6", "train_distributed_round",
            [*common, "--stage", "ss", "--output_dir", formal_b, "--max_steps", "3000",
             "--layer_source", "predicted", "--target_lari_checkpoint", lari, "--gradient_checkpointing"], distributed=True))
        b_report = read(formal_b / "report.json")
        if (not b_report["passed"] or b_report["step"] != 3000
                or b_report["rank_parameter_max_abs_difference"] != 0 or not b_report["initial_zero_audit"]["passed"]):
            raise RuntimeError("formal SS training integrity check failed")
        ss = formal_b / "checkpoints/step_003000.pt"
        self.set_stage("formal_dev64_evaluation", "Fixed final checkpoints, Dev64 parent/candidate paired occupancy; no score-based selection.")
        output = ROOT / "formal_dev64_eval"
        jobs, boundaries = [], [0, 11, 22, 33, 44, 54, 64]
        for index, gpu in enumerate(("1", "2", "3", "4", "5", "6")):
            begin, end = boundaries[index:index + 2]
            jobs.append(self.launch(f"background_dev64_{begin}_{end}" + self.job_suffix, gpu, "evaluate_ss_occupancy", [
                "--manifest", DATA / "dev64.json", "--output_dir", output, "--mode", "paired",
                "--layered_ss_checkpoint", ss, "--target_lari_checkpoint", lari, "--seeds", "42,43",
                "--steps", "25", "--cfg", "5", "--start", begin, "--end", end, "--require_disjoint"]))
        for name in jobs:
            self.wait_job(name)
        identity, records = validate_records(output, DATA / "dev64.json", ("parent", "candidate"), 3000)
        if identity["candidate"]["sha256"] != sha(ss) or identity["lari"]["sha256"] != sha(lari):
            raise RuntimeError("evaluation did not use fixed final checkpoints")
        summary = aggregate_records(records)
        report = {"experiment_complete": True, "scope": "SS occupancy; not full mesh or occlusion robustness",
                  "summary": summary, "evaluation_record_count": len(records), "evaluation_identity": identity,
                  "formal_training_manifest_sha256": sha(train), "overfit_gate": read(gate),
                  "checkpoint_selection": "fixed final steps preregistered before Dev64",
                  "improvement_observed": summary["paired"]["iou_gain"] > 0,
                  "no_failed_objects_filtered": True}
        atomic(ROOT / "round1_report.json", report)
        atomic(output / "aggregate.json", {"summary": summary, "records": records})
        self.set_stage("complete", "First experiment completed. Completion does not imply a positive gain; read round1_report.json.")
        self.status("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print_plan", action="store_true")
    parser.add_argument("--recover_formal_startup", action="store_true")
    args = parser.parse_args()
    if args.print_plan:
        print(json.dumps({"output_root": str(ROOT), "gpus": "1,2,3,4,5,6",
            "sequence": ["existing overfit2000", "final 32x2 sampling + gate", "complete train2000 / Dev64 data",
                         "fresh LaRI4000", "fresh SS3000", "Dev64x2 parent/candidate paired sampling"],
            "stop": "systemctl --user stop tracker-palr-round1-20260909.service"}, indent=2))
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "background.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor = Supervisor(recover_formal_startup=args.recover_formal_startup)
        try:
            if args.recover_formal_startup:
                supervisor.run_formal_startup_recovery()
            else:
                supervisor.run()
        except StopRequested:
            supervisor.emit("Stop requested; terminating only this experiment's registered processes.")
            supervisor.shutdown_children()
            supervisor.status("stopped", "Stopped by user; checkpoints/logs retained. No implicit optimizer resume.")
        except Exception:
            error = traceback.format_exc()
            supervisor.emit(error)
            supervisor.shutdown_children()
            supervisor.status("blocked", error)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
