"""Frozen Full dev/test selection and official-image preflight, without model calls.

Original issue text is used. This is a code-side evaluation, not a newly invented
screenshot benchmark. Images are executed rootlessly with visible protected tests;
the protocol is explicitly distinguished from the official Docker leaderboard.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from .benchmark import official_grade
from .config import RuntimeConfig
from .control import Events, atomic_json
from .official_data import install_test_patch, test_command
from .repository import IGNORED, git, validate_tree
from .runtime import create_sandbox
from .scale_compat import compatibility_runtime, prepare_project_metadata
from .scale_images import resolve_task_image


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def immutable_json(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Frozen input differs; use a new output directory: {path}")
    atomic_json(path, value)


def previously_used(root):
    ids, evidence = set(), {}
    for name in ("swebench_lite_v3", "swebench_extended_v5"):
        p = root / "datasets/generated" / name / "selection.json"
        if not p.is_file():
            raise ValueError(f"Cannot establish exclusions without {p}")
        data = read(p)
        for key in ("selected_ids", "primary_ids", "reserve_ids", "excluded_previously_seen_ids"):
            ids.update(data.get(key, []))
        evidence[str(p)] = sha256(p)
    return ids, evidence


def balanced_order(records, seed):
    """Round-robin repositories, with a seeded hash ordering fixed before inference."""
    groups = defaultdict(list)
    for row in records:
        groups[row["repo"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['instance_id']}".encode()).hexdigest())
    ordered = []
    while any(groups.values()):
        for name in sorted(groups):
            if groups[name]:
                ordered.append(groups[name].pop(0))
    return ordered


def choose_records(by_split, excluded, verified_ids, targets, reserves, seed):
    all_ids = set()
    selected = []
    for split in ("dev", "test"):
        rows = by_split[split]
        ids = [r["instance_id"] for r in rows]
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in ids):
            raise ValueError("Unsafe instance ID")
        if len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError("Duplicate instance IDs within or across official splits")
        all_ids.update(ids)
        candidates = balanced_order([r for r in rows if r["instance_id"] not in excluded], seed)
        need = targets[split] + reserves[split]
        if len(candidates) < need:
            raise ValueError(f"Insufficient {split} candidates: {len(candidates)} < {need}")
        for i, row in enumerate(candidates[:need]):
            normalized = dict(row)
            changes = []
            for key in ("patch", "test_patch"):
                if key in row and "\r\n" in row[key]:
                    normalized[key] = row[key].replace("\r\n", "\n")
                    changes.append(key)
            selected.append({**normalized, "patch_crlf_normalized": changes,
                             "sca_split": split, "source_split": split, "selection_rank": i,
                             "selection_role": "primary" if i < targets[split] else "environment_reserve",
                             "in_verified": row["instance_id"] in verified_ids})
    return selected


def prepare(root, output, *, dev_count=100, test_count=300, seed=20261024):
    import pyarrow.parquet as pq

    data = root / "datasets/swebench_full_pinned"
    source = read(data / "manifest.json")
    pins = read(root / "configs/scale_4gpu_resources.json")
    if source["resources_lock_sha256"] != sha256(root / "configs/scale_4gpu_resources.json"):
        raise ValueError("Dataset manifest and resource pins differ")
    for folder, key in [("swe-bench-tasks", "tasks_commit"), ("SWE-bench", "grader_commit")]:
        if git(root / "vendor" / folder, "rev-parse", "HEAD").stdout.decode().strip() != pins[key]:
            raise ValueError(f"Unexpected upstream checkout: {folder}")
    by_split = {}
    for name, dataset in source["datasets"].items():
        for split, item in dataset["files"].items():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("Dataset checksum mismatch")
            rows = pq.read_table(item["path"]).to_pylist()
            if name == "full":
                by_split[split] = rows
            else:
                verified_ids = {r["instance_id"] for r in rows}
    if not verified_ids.issubset({r["instance_id"] for r in by_split["test"]}):
        raise ValueError("Verified is not a subset of this Full test revision")
    excluded, evidence = previously_used(root)
    targets = {"dev": dev_count, "test": test_count}
    if min(targets.values()) < 1:
        raise ValueError("Both target counts must be positive")
    reserves = {k: max(1, (n + 3) // 4) for k, n in targets.items()}
    records = choose_records(by_split, excluded, verified_ids, targets, reserves, seed)
    definitions = {}
    for row in records:
        task = root / "vendor/swe-bench-tasks/tasks" / row["instance_id"]
        for filename, key in [("gold.patch", "patch"), ("test.patch", "test_patch")]:
            if (task / filename).read_text().strip() != row[key].strip():
                raise ValueError(f"Dataset/task definition mismatch: {row['instance_id']} {filename}")
        # Current pinned Full includes parser/image fields. Fail early on an incompatible release.
        for key in ("image", "log_parser", "eval_type"):
            if not row.get(key):
                raise ValueError(f"Pinned Full row lacks {key}: {row['instance_id']}")
        for filename in ("eval.sh", "gold.patch", "test.patch", "task.yaml"):
            definitions[str(task / filename)] = sha256(task / filename)
    selection = {"dataset": source["datasets"]["full"]["repo_id"],
                 "revision": source["datasets"]["full"]["revision"], "seed": seed,
                 "targets": targets, "reserves": reserves, "excluded_ids": sorted(excluded),
                 "exclusion_files_sha256": evidence, "source_manifest_sha256": sha256(data / "manifest.json"),
                 "upstream_files_sha256": definitions, "pins": pins,
                 "selected_ids": [r["instance_id"] for r in records],
                 "repositories": {s: dict(Counter(r["repo"] for r in records if r["sca_split"] == s))
                                  for s in targets},
                 "selection_rule": "Original dev/test splits; seeded repository round-robin; no screenshot filter. "
                                   "Ordered reserves replace only environment failures before model evaluation.",
                 "input_mode": "original_issue_text", "gpu_started": False}
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "selected_records.json", records)
    immutable_json(output / "selection.json", selection)
    return selection


def copy_image_repository(source, destination):
    """Copy buggy source/build artifacts, removing history and materializing internal links only."""
    source = source.resolve()
    links, preserved, stats = [], {}, {"files": 0, "bytes": 0}

    def visit(src, dst, ancestors):
        resolved = src.resolve()
        if not resolved.is_relative_to(source):
            raise ValueError(f"Repository link leaves /testbed: {src}")
        if src.is_symlink():
            links.append({"path": str(src.relative_to(source)), "target": str(resolved.relative_to(source))})
        if resolved.is_dir():
            if resolved in ancestors:
                if not src.is_symlink():
                    raise ValueError("Cyclic directory link")
                # SQLFluff intentionally tests a self-referencing directory link.
                # Preserve that exact internal fixture instead of recursively materializing it.
                target = os.readlink(src)
                if Path(target).is_absolute():
                    raise ValueError("Cyclic source links must be repository-relative")
                dst.symlink_to(target, target_is_directory=True)
                preserved[str(dst.relative_to(destination))] = target
                stats["files"] += 1
                return
            dst.mkdir()
            for child in sorted(resolved.iterdir()):
                if child.name not in IGNORED | {".tox"}:
                    visit(child, dst / child.name, ancestors | {resolved})
        elif resolved.is_file():
            stats["files"] += 1
            stats["bytes"] += resolved.stat().st_size
            if stats["files"] > 50000 or stats["bytes"] > 1_000_000_000:
                raise ValueError("Full task exceeds 50,000 files / 1GB prepared repository limit")
            shutil.copy2(resolved, dst)
        else:
            raise ValueError(f"Unsupported repository entry: {src}")
    visit(source, destination, set())
    return {**stats, "materialized_links": links, "preserved_links": preserved}


def tree_hash(repo):
    h = hashlib.sha256()
    links = {str(p.relative_to(repo)): os.readlink(p) for p in repo.rglob("*") if p.is_symlink()}
    for path in validate_tree(repo, max_files=50000, max_bytes=1_000_000_000, allowed_symlinks=links):
        h.update(str(path.relative_to(repo)).encode() + b"\0")
        h.update(str(path.lstat().st_mode & 0o777).encode() + b"\0")
        h.update(("symlink:" + os.readlink(path) if path.is_symlink() else sha256(path)).encode() + b"\0")
    return h.hexdigest()


def audit_one(root, output, instance, timeout=600, retry=False):
    records = {r["instance_id"]: r for r in read(output / "selected_records.json")}
    record = records[instance]
    folder = output / "tasks" / instance
    if (folder / "audit.json").exists():
        old = read(folder / "audit.json")
        if old["status"] == "validated" or not retry:
            return old
    if folder.exists():
        if not retry:
            raise ValueError(f"Incomplete attempt preserved: {folder}; use --retry-failed")
        archive = output / "preflight_attempts" / f"{instance}_{time.time_ns()}"
        archive.parent.mkdir(exist_ok=True)
        folder.rename(archive)
    folder.mkdir(parents=True)
    item = {"instance_id": instance, "split": record["sca_split"], "repo": record["repo"],
            "status": "error", "started_at": time.time()}
    try:
        upstream = root / "vendor/swe-bench-tasks/tasks" / instance
        for filename, key in [("test.patch", "test_patch"), ("gold.patch", "patch")]:
            if (upstream / filename).read_text().strip() != record[key].strip():
                raise ValueError("Upstream task definition changed")
        print(f"Resolving pinned task environment for {instance}", flush=True)
        image, image_repo, environment_source = resolve_task_image(root, output, record, folder)
        atomic_json(folder / "image.json", image)
        item["environment_source"] = environment_source
        if not environment_source["environment_adapted"]:
            base = git(image_repo, "rev-parse", record["base_commit"] + "^{tree}").stdout
            if git(image_repo, "rev-parse", "HEAD^{tree}").stdout != base:
                raise ValueError("Image checkout tree differs from the declared buggy base")
        repo = folder / "repo"
        item["export"] = copy_image_repository(image_repo, repo)
        install_test_patch(repo, record["test_patch"])
        overlay_fields, compatibility = compatibility_runtime(root, record, image)
        item["compatibility"] = compatibility
        item["preflight_protocol"] = "full_exact_source_compat_v2"
        original_command = test_command((upstream / "eval.sh").read_text())
        command = original_command
        if overlay_fields.get("python_overlay"):
            command = re.sub(r"(?<![\w./-])pytest(?=\s|$)", "python -m pytest", command)
        body = ("#!/bin/bash\nset -uo pipefail\nexport MPLBACKEND=Agg\n"
                "printf '\\n>>>>> Start Test Output\\n'\n" + command +
                "\nSCA_TEST_EXIT=$?\nprintf '\\n>>>>> End Test Output\\n>>>>> Test Exit Code: %s\\n' "
                '"$SCA_TEST_EXIT"\nexit "$SCA_TEST_EXIT"\n')
        (repo / ".sca_eval.sh").write_text(body)
        protected = sorted({".sca_eval.sh", *re.findall(r"^(?:\+\+\+ b/|--- a/)(.+)$", record["test_patch"], re.M),
                            *[str(p.relative_to(repo)) for p in repo.rglob("conftest.py")]})
        runtime = RuntimeConfig(backend="oci", rootfs=image["rootfs"], image=image["requested_image"],
                                image_digest=image["platform_digest"], max_repo_files=50000,
                                max_repo_bytes=1_000_000_000, allowed_symlinks=item["export"]["preserved_links"],
                                **overlay_fields)
        events = Events(folder / "preflight_events.jsonl")
        metadata_paths, metadata_evidence = prepare_project_metadata(
            root, record, repo, runtime, events, adapted=environment_source["environment_adapted"])
        if metadata_paths:
            protected = sorted(set(protected) | set(metadata_paths))
            compatibility["project_metadata"] = metadata_evidence
        atomic_json(folder / "benchmark.json", {"record": record, "harness_checkout": str(root / "vendor/SWE-bench"),
                    "test_command": command, "image_digest": image["platform_digest"],
                    "original_test_command": original_command,
                    "environment_source": environment_source, "compatibility": compatibility,
                    "runtime_adaptation": "Pinned OCI runtime via Bubblewrap, exact buggy source and protected tests; "
                                          "explicit dependency/donor adaptations recorded; not the official Docker protocol."})
        (folder / "reference.patch").write_text(record["patch"])
        task = {"task_id": instance, "repo": "repo", "problem": record["problem_statement"], "images": [],
                "test_command": "bash .sca_eval.sh", "protected_paths": protected,
                "split": record["sca_split"], "group_id": record["repo"],
                "provenance": "swebench_full_original_text_oci_adapted", "reference_patch": "reference.patch",
                "benchmark": "benchmark.json", "runtime": runtime.model_dump()}
        atomic_json(folder / "task.json", task)
        git(repo, "add", "-A", "-f", "--", ".")
        git(repo, "commit", "-q", "--allow-empty", "-m", "Private buggy base with protected tests")
        scores = {}
        for phase, patch in (("baseline", None), ("reference", record["patch"])):
            check = folder / (phase + "_check")
            shutil.copytree(repo, check, symlinks=True)
            if patch:
                git(check, "apply", "--binary", "-", stdin=patch.encode())
            sb = create_sandbox(check, events, protected, timeout, repo, runtime)
            try:
                probe = sb.probe()
                test = sb.run("bash .sca_eval.sh", log_path=folder / (phase + ".log"))
                score = official_grade(folder / "benchmark.json", folder / (phase + ".log"))
                scores[phase] = {"returncode": test["returncode"], "timed_out": test["timed_out"], "official": score,
                                 "filesystem_probe": probe["filesystem_probe"]}
            finally:
                sb.cancel()
        item.update(scores)
        before, after = scores["baseline"], scores["reference"]
        failures = before["official"].get("tests_status", {}).get("FAIL_TO_PASS", {}).get("failure")
        valid = (before["official"]["parsed"] and not before["official"]["resolved"] and failures
                 and after["official"]["resolved"] and not before["timed_out"] and not after["timed_out"])
        item["status"] = "validated" if valid else "environment_failed"
        item["repo_sha256"] = tree_hash(repo)
        item["task_sha256"] = sha256(folder / "task.json")
        item["benchmark_sha256"] = sha256(folder / "benchmark.json")
        item["reference_sha256"] = sha256(folder / "reference.patch")
        item["image_digest"] = image["platform_digest"]
        if valid:
            # Only disposable copies created by this preflight are removed, preserving logs and original input.
            shutil.rmtree(folder / "baseline_check")
            shutil.rmtree(folder / "reference_check")
    except Exception as error:
        item.update(error=str(error), error_type=type(error).__name__)
    item["finished_at"] = time.time()
    atomic_json(folder / "audit.json", item)
    print(json.dumps({k: v for k, v in item.items() if k not in ("baseline", "reference", "export")}), flush=True)
    return item


def finalize(output, split=None):
    if split not in (None, "dev", "test"):
        raise ValueError("Finalize split must be dev, test, or None")
    selection = read(output / "selection.json")
    records = read(output / "selected_records.json")
    audits, manifests = [], {"dev": [], "test": []}
    for record in records:
        folder = output / "tasks" / record["instance_id"]
        path = folder / "audit.json"
        audit = read(path) if path.exists() else {"instance_id": record["instance_id"], "status": "pending"}
        audits.append(audit)
        task_split = record["sca_split"]
        if audit["status"] == "validated" and len(manifests[task_split]) < selection["targets"][task_split]:
            manifests[task_split].append({"task_id": record["instance_id"], "task_file": str(folder / "task.json"),
                                    "group_id": record["repo"], "split": task_split,
                                    "provenance": "swebench_full_original_text_oci_adapted",
                                    "in_verified": record["in_verified"], "audit_sha256": sha256(path)})
    counts = {s: len(rows) for s, rows in manifests.items()}
    states = {s: Counter(a["status"] for r, a in zip(records, audits) if r["sca_split"] == s)
              for s in manifests}
    ready_by_split = {s: counts[s] == selection["targets"][s] and not states[s]["pending"] for s in manifests}
    wanted = [split] if split else ["dev", "test"]
    report = {"candidates": len(records), "preflight_states": dict(Counter(a["status"] for a in audits)),
              "targets": selection["targets"], "validated_selected": counts,
              "ready": all(ready_by_split[s] for s in wanted), "ready_by_split": ready_by_split,
              "requested_split": split or "all", "states_by_split": {s: dict(c) for s, c in states.items()},
              "audits": audits, "interpretation": "Environment/gold preflight only; no model repair accuracy."}
    if report["ready"]:
        for task_split in wanted:
            rows = manifests[task_split]
            manifest = output / f"{task_split}.jsonl"
            payload = "".join(json.dumps(r) + "\n" for r in rows)
            if manifest.exists() and manifest.read_text() != payload:
                raise ValueError("Evaluation manifest already frozen; use a new dataset version")
            manifest.write_text(payload)
    atomic_json(output / "preflight_report.json", report)
    return report
