#!/usr/bin/env python3
"""Copy Coder storage, verify it, then preserve old prefixes with reversible symlinks.

Run apply/rollback on the host, with all SCA jobs stopped. These actions never
launch a model or remove the original directories; the large copy is user-run.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import pwd
import re
import signal
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = Path("/data/zjr/sca_coder_runtime_20261026")
MODEL = ".cache/models/Qwen3-Coder-30B-A3B-Instruct"
ITEMS = [(".model-env", "model-env"), (".coder-env", "coder-env"), (MODEL, "model")]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def assert_idle(project, target, proc_root=Path("/proc")):
    # A private PID namespace can hide the very jobs this guard must find.
    if (proc_root / "1/comm").read_text().strip() != "systemd":
        raise RuntimeError("Run apply/rollback on the host; a private process namespace cannot prove SCA is idle")
    current = os.getpid()
    blockers = []
    unreadable = []
    username = pwd.getpwuid(os.getuid()).pw_name
    for folder in proc_root.iterdir():
        if not folder.name.isdecimal() or int(folder.name) == current:
            continue
        comm = "unknown"
        try:
            if folder.stat().st_uid != os.getuid():
                continue
            comm = (folder / "comm").read_text().strip()
            arguments = [part.decode(errors="replace") for part in (folder / "cmdline").read_bytes().split(b"\0")
                         if part]
            # systemd's PAM session helper is non-dumpable even for its owning
            # user. Match both its exact process name and rewritten argv before
            # attempting protected exe/cwd links; arbitrary Python remains fatal.
            if comm == "(sd-pam)" and arguments == ["(sd-pam)"]:
                continue
            if (comm == "sshd" and len(arguments) == 1
                    and re.fullmatch(re.escape("sshd: " + username + "@") + r"(?:pts/\d+|notty)",
                                     arguments[0].strip())):
                continue  # This user's exact OpenSSH session helper, also non-dumpable.
            args = " ".join(arguments)
            executable = os.readlink(folder / "exe")
            cwd = os.readlink(folder / "cwd")
        except FileNotFoundError:
            continue
        except PermissionError:
            unreadable.append({"pid": int(folder.name), "comm": comm})
            continue
        env_use = any(prefix in executable for prefix in (
            str(project / ".coder-env"), str(project / ".model-env"),
            str(target / "coder-env"), str(target / "model-env")))
        project_use = str(project) in args or cwd.startswith(str(project))
        service = any(name in args for name in (
            "run_scale_4gpu.py", "local_model_api.py", "scale_resources.py", "vllm.entrypoints"))
        if env_use or (project_use and service):
            blockers.append(int(folder.name))
    if unreadable:
        raise RuntimeError("Cannot inspect owned processes; refusing an unverified storage switch: "
                           + json.dumps({"unreadable": unreadable, "active_sca_pids": blockers}))
    if blockers:
        raise RuntimeError("SCA runtime/controller is still active; stop it before migration: " + str(blockers))


@contextlib.contextmanager
def locks(project):
    with contextlib.ExitStack() as stack:
        paths = [project / ".storage-migration.lock", *sorted((project / "outputs").glob("*/controller.lock"))]
        for path in paths:
            handle = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Storage migration or an evaluation controller is already running") from error
        yield


class Migration:
    def __init__(self, project, target):
        self.project, self.target = project.resolve(), target.absolute()
        if self.target.is_symlink() or self.target.resolve() != self.target:
            raise ValueError("Target must be a real directory path, without symlink parents")
        if self.target == self.project or self.project in self.target.parents or self.target in self.project.parents:
            raise ValueError("Target and project must be separate directory trees")
        self.marker = self.target / "migration_owner.json"
        self.state_path = self.target / "status.json"
        self.owner = None

    def plan(self):
        receipt = read(self.project / MODEL / "download_complete.json")
        return {"project": str(self.project), "target": str(self.target),
                "model_bytes": sum(row["bytes"] for row in receipt["files"].values()),
                "items": [{"source": str(self.project / source), "destination": str(self.target / name)}
                          for source, name in ITEMS],
                "copies_started": False, "originals_retained_as_backups": True,
                "note": "apply performs a long copy and checksum verification; it does not resume evaluation"}

    def owned(self, create=False):
        expected = {"protocol": "sca_storage_v1", "project": str(self.project), "target": str(self.target)}
        if self.marker.exists():
            owner = read(self.marker)
            if any(owner.get(k) != v for k, v in expected.items()) or not re.fullmatch(
                    "[a-f0-9]{32}", str(owner.get("token", ""))):
                raise ValueError("Target ownership marker belongs to another migration")
        elif create:
            if self.target.exists() and any(self.target.iterdir()):
                raise ValueError("Refusing an unrelated nonempty target directory")
            self.target.mkdir(parents=True, exist_ok=True)
            owner = {**expected, "token": uuid.uuid4().hex}
            # A marker is created before copying or renaming any source.
            with self.marker.open("x") as handle:
                json.dump(owner, handle)
        else:
            raise ValueError("No matching migration marker exists")
        self.owner = owner
        return owner

    def paths(self):
        for source, name in ITEMS:
            old = self.project / source
            backup = old.with_name(old.name + ".storage-backup-" + self.owner["token"])
            yield old, self.target / name, backup

    def update(self, stage, **details):
        state = {"stage": stage, "pid": os.getpid(), "updated_at": time.time(), **details}
        atomic(self.state_path, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    @staticmethod
    def layout(old, destination, backup):
        if old.is_symlink():
            if old.resolve() != destination or not backup.is_dir() or backup.is_symlink():
                raise ValueError("Unexpected source symlink or missing original backup: " + str(old))
            return "linked"
        if old.exists():
            if not old.is_dir() or backup.exists() or backup.is_symlink():
                raise ValueError("Source/backup conflict: " + str(old))
            return "original"
        if backup.is_dir() and not backup.is_symlink():
            return "rename_gap"
        raise ValueError("Original source and backup are both missing: " + str(old))

    def copy_verify(self, old, destination, backup):
        layout = self.layout(old, destination, backup)
        source = old if layout == "original" else backup
        if destination.is_symlink() or destination.resolve() != destination:
            raise ValueError("Destination must not redirect copying through a symlink")
        destination.mkdir(exist_ok=True)
        self.update("copying", source=str(source), destination=str(destination))
        subprocess.run(["rsync", "-aH", "--partial", "--info=progress2", "--", str(source) + "/",
                        str(destination) + "/"], check=True)
        self.update("verifying", source=str(source), destination=str(destination))
        model = old == self.project / MODEL
        # --delete is paired with --dry-run: report extras, never remove them.
        check = subprocess.run(["rsync", "-aHni", "--delete", *([] if model else ["--checksum"]),
                                "--out-format=%i %n%L", "--", str(source) + "/", str(destination) + "/"],
                               check=True, text=True, stdout=subprocess.PIPE)
        if check.stdout.strip():
            raise RuntimeError("Copied tree does not exactly match source: " + check.stdout[:2000])
        if model:
            if digest(source / "download_complete.json") != digest(destination / "download_complete.json"):
                raise ValueError("Model download receipt changed")
            receipt = read(destination / "download_complete.json")
            for index, (name, row) in enumerate(receipt["files"].items()):
                path = destination / name
                if path.resolve().parent != destination or path.is_symlink():
                    raise ValueError("Unexpected model receipt filename")
                self.update("verifying_model", file=name, completed=index, total=len(receipt["files"]))
                if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                    raise ValueError("Model checksum mismatch: " + name)

    def rollback(self):
        self.owned()
        rows = list(self.paths())
        for old, destination, backup in rows:
            self.layout(old, destination, backup)  # Validate all paths before restoring any.
        for old, destination, backup in reversed(rows):
            layout = self.layout(old, destination, backup)
            if layout == "linked":
                old.unlink()  # This is the exact expected symlink, never a directory.
            if layout in {"linked", "rename_gap"}:
                backup.rename(old)
        self.update("rolled_back", target_copies_retained=True)

    def smoke(self):
        expected = read(self.project / ".coder-env/setup_complete.json")
        code = ("import json,os,sys,torch,vllm,transformers; "
                "print('SCA_STORAGE_SMOKE='+json.dumps({'prefix':os.path.realpath(sys.prefix),"
                "'base_prefix':os.path.realpath(sys.base_prefix),'torch':torch.__version__,"
                "'vllm':vllm.__version__,'transformers':transformers.__version__,"
                "'torch_file':os.path.realpath(torch.__file__),'vllm_file':os.path.realpath(vllm.__file__)}))")
        log = self.target / "cpu_import_check.log"
        try:
            result = subprocess.run([str(self.project / ".coder-env/bin/python"), "-c", code],
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            output = getattr(error, "stdout", None) or getattr(error, "output", None) or str(error)
            log.write_text(output.decode(errors="replace") if isinstance(output, bytes) else output)
            raise RuntimeError("CPU import check failed; inspect " + str(log)) from error
        log.write_text(result.stdout)
        lines = [line for line in result.stdout.splitlines() if line.startswith("SCA_STORAGE_SMOKE=")]
        info = json.loads(lines[-1].split("=", 1)[1]) if lines else {}
        if (info.get("prefix") != str(self.target / "coder-env")
                or info.get("base_prefix") != str(self.target / "model-env")
                or info.get("vllm") != expected["vllm_version"]
                or info.get("transformers") != expected["transformers_version"]
                or not all(str(info.get(key, "")).startswith(str(self.target / "coder-env") + "/")
                           for key in ("torch_file", "vllm_file"))):
            raise RuntimeError("CPU import smoke or physical environment location differs: " + str(info)
                               + "; inspect " + str(log))
        return info

    def apply(self):
        assert_idle(self.project, self.target)
        self.owned(create=True)
        layouts = [self.layout(*row) for row in self.paths()]
        for _, destination, _ in self.paths():
            if destination.is_symlink() or destination.resolve() != destination:
                raise ValueError("Destination must not redirect copying through a symlink")
        if self.state_path.exists() and read(self.state_path).get("stage") == "completed":
            if any(layout != "linked" for layout in layouts):
                raise ValueError("Completed migration layout changed; inspect it before retrying")
            print(json.dumps({**read(self.state_path), "already_completed": True}), flush=True)
            return  # Never overwrite a used environment with its historical backup.
        try:
            for old, destination, backup in self.paths():
                self.copy_verify(old, destination, backup)
            assert_idle(self.project, self.target)
            self.update("switching")
            for old, destination, backup in self.paths():
                layout = self.layout(old, destination, backup)
                if layout == "original":
                    old.rename(backup)
                if layout != "linked":
                    old.symlink_to(destination, target_is_directory=True)
            self.update("cpu_import_check")
            info = self.smoke()
            self.update("completed", imports=info, backups=[str(row[2]) for row in self.paths()],
                        gpu_started=False, evaluation_resumed=False)
        except BaseException as error:
            try:
                self.rollback()
            except Exception as rollback_error:
                self.update("failed_needs_manual_recovery", error=repr(error), rollback_error=repr(rollback_error))
            else:
                self.update("failed_restored", error=type(error).__name__ + ": " + str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "status", "apply", "rollback"])
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    migration = Migration(args.project, args.target)
    if args.action == "plan":
        print(json.dumps(migration.plan(), ensure_ascii=False, indent=2))
    elif args.action == "status":
        print(json.dumps(read(migration.state_path) if migration.state_path.exists() else {"stage": "not_started"},
                         ensure_ascii=False, indent=2))
    else:
        def cancel(signum, frame):
            raise KeyboardInterrupt("Migration cancellation requested")
        signal.signal(signal.SIGTERM, cancel)
        with locks(migration.project):
            if args.action == "apply":
                migration.apply()
            else:
                assert_idle(migration.project, migration.target)
                migration.rollback()


if __name__ == "__main__":
    main()
