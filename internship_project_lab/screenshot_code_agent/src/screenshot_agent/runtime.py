"""Official-image runtimes sharing the existing tool and cancellation interfaces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from .config import RuntimeConfig
from .sandbox import BubblewrapSandbox, SandboxUnavailable


def verified_overlay(directory, manifest_sha256):
    if directory is None:
        return None
    path = Path(directory).resolve()
    manifest = path / "manifest.json"
    if not path.is_dir() or not manifest.is_file():
        raise SandboxUnavailable(f"Compatibility overlay receipt is missing: {path}")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_sha256:
        raise SandboxUnavailable(f"Compatibility overlay manifest changed: {path}")
    # Do not let an overlay symlink expose data outside the prepared dependency directory.
    for entry in path.rglob("*"):
        if entry.is_symlink() and not entry.resolve().is_relative_to(path):
            raise SandboxUnavailable(f"Compatibility overlay link escapes its directory: {entry}")
    return path


class VenvSandbox(BubblewrapSandbox):
    """Task dependencies from a dedicated prefix, mounted read-only at its real path.

    This is an explicitly adapted runtime, not an official Docker image.
    """

    def __init__(self, *args, runtime: RuntimeConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime = runtime
        self.prefix = Path(runtime.prefix or "").resolve()
        if not runtime.prefix or not (self.prefix / "bin/python").is_file():
            raise SandboxUnavailable("A prepared task-only Python prefix is required")

    def command(self, argv, mounts=None):
        cmd = super().command(argv, mounts)
        index = cmd.index("--chdir")
        additions = ["--ro-bind", str(self.prefix), str(self.prefix),
                     "--setenv", "PATH", str(self.prefix / "bin") + ":/usr/bin:/bin",
                     "--setenv", "PYTHONPATH", "/workspace/src:/workspace",
                     "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1"]
        cmd[index:index] = additions
        return cmd

    def description(self):
        return {**super().description(), "backend": "adapted_python_bubblewrap",
                "prefix": str(self.prefix), "official_container": False}


class OCISandbox(BubblewrapSandbox):
    """Read-only official OCI rootfs, writable snapshot, no network or GPU devices."""

    def __init__(self, *args, runtime: RuntimeConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime = runtime
        self.rootfs = Path(runtime.rootfs or "").resolve()
        if not runtime.rootfs or not (self.rootfs / runtime.python.lstrip("/")).is_file():
            raise SandboxUnavailable("OCI rootfs with the declared Python interpreter is required")
        if runtime.workdir != "/testbed":
            raise ValueError("Official OCI workdir must be /testbed")
        self.python_overlay = verified_overlay(runtime.python_overlay, runtime.python_overlay_manifest_sha256)
        self.library_overlay = verified_overlay(runtime.library_overlay, runtime.library_overlay_manifest_sha256)
        self.scratch = self.repo.parent / ("runtime_scratch_" + self.repo.name)
        self.scratch.mkdir(exist_ok=True)

    def command(self, argv: list[str], mounts: dict[str, Path] | None = None) -> list[str]:
        env_python = str(Path(self.runtime.python).parent)
        cmd = [self.executable, "--unshare-all", "--die-with-parent", "--cap-drop", "ALL",
               "--ro-bind", str(self.rootfs), "/", "--proc", "/proc", "--dev", "/dev",
               *self.tmp_mount(), "--dir", "/tmp/home", "--clearenv",
               "--setenv", "PATH", env_python + ":/opt/miniconda3/bin:/usr/bin:/bin",
               "--setenv", "HOME", "/tmp/home", "--setenv", "LANG", "C.UTF-8",
               "--setenv", "PYTHONPATH", "/testbed/src:/testbed", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
               "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
               "--setenv", "GIT_CONFIG_NOSYSTEM", "1", "--setenv", "GIT_CONFIG_GLOBAL", "/dev/null",
               "--bind", str(self.repo), "/testbed", "--bind", str(self.repo), "/workspace",
               "--ro-bind", str(Path(__file__).with_name("worker.py")), "/sca_worker.py",
               "--bind", str(self.scratch), "/root"]
        if self.python_overlay:
            cmd += ["--ro-bind", str(self.python_overlay), "/tmp/sca-python",
                    "--setenv", "PYTHONPATH", "/tmp/sca-python:/testbed/src:/testbed"]
        if self.library_overlay:
            cmd += ["--ro-bind", str(self.library_overlay), "/tmp/sca-libs",
                    "--setenv", "LD_LIBRARY_PATH", "/tmp/sca-libs"]
        display = getattr(self, 'public_display', None)
        if display is not None:
            if display.proc.poll() is not None:
                raise SandboxUnavailable('Owned public X server exited')
            cmd += ['--dir', '/tmp/.X11-unix', '--ro-bind', str(display.socket), str(display.socket),
                    '--setenv', 'DISPLAY', display.name, '--setenv', 'PYVISTA_OFF_SCREEN', 'true',
                    '--setenv', 'MPLBACKEND', 'Agg', '--setenv', 'LIBGL_ALWAYS_SOFTWARE', '1',
                    # Mesa LLVM needs GLIBCXX_3.4.30; the image's system C++ library
                    # supplies it, whereas its historical Conda library does not.
                    '--setenv', 'LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6']
        for name in self.protected:
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
                raise ValueError("Protected paths must be literal repository-relative paths")
            source = (self.protected_root / rel).resolve()
            if not source.is_relative_to(self.protected_root):
                raise ValueError("Protected path escapes repository")
            if source.exists():
                cmd += ["--ro-bind", str(source), str(Path("/testbed") / rel),
                        "--ro-bind", str(source), str(Path("/workspace") / rel)]
        # The OCI root is read-only. A fresh mount namespace needs a writable
        # parent in which bwrap can create each read-only grading mountpoint.
        # Never add placeholders to the shared rootfs or expose these to Agent runs.
        if mounts:
            cmd += ["--tmpfs", "/grader"]
        for target, source in (mounts or {}).items():
            if not target.startswith("/grader/") or ".." in Path(target).parts:
                raise ValueError("Additional mounts are limited to /grader")
            cmd += ["--ro-bind", str(source.resolve()), target]
        cmd += ["--chdir", "/testbed", "--", "/usr/bin/prlimit",
                f"--as={self.runtime.memory_gib * 2**30}:{self.runtime.memory_gib * 2**30}",
                "--nofile=1024:1024", "--fsize=33554432:33554432", f"--cpu={self.timeout}:{self.timeout}",
                "--", *argv]
        return cmd

    def probe(self):
        result = self.run([self.runtime.python, "-c", "import os; print('sandbox-ok',os.getcwd())"])
        if result["returncode"] or "sandbox-ok /testbed" not in result["output"]:
            raise SandboxUnavailable(result["output"] or "Official OCI runtime probe failed")
        result["filesystem_probe"] = self.filesystem_probe()
        return result

    def filesystem(self, operation, arguments, timeout=None, *, feedback_version=1, edit_guard=None):
        result = self.run([self.runtime.python, "/sca_worker.py"],
                          stdin=json.dumps({"operation": operation, "arguments": arguments,
                                            "feedback_version": feedback_version, "edit_guard": edit_guard}), timeout=timeout)
        return json.dumps(result, ensure_ascii=False)

    def description(self):
        return {"backend": "official_oci_bubblewrap", "image": self.runtime.image,
                "image_digest": self.runtime.image_digest, "network": "unshared", "gpu_devices": False,
                "repository_mount": "/testbed", "workspace_alias": "/workspace",
                "system_runtime": self.runtime.python, "protected_paths": self.protected,
                "persistent_tmp": self.session_tmp is not None, "writable_workspace_git": self.session_tmp is not None,
                "address_space_limit_gib": self.runtime.memory_gib, "command_timeout_seconds": self.timeout,
                "python_overlay": self.runtime.python_overlay,
                "python_overlay_manifest_sha256": self.runtime.python_overlay_manifest_sha256,
                "library_overlay": self.runtime.library_overlay,
                "library_overlay_manifest_sha256": self.runtime.library_overlay_manifest_sha256}


class DockerSandbox(BubblewrapSandbox):
    """Docker CLI execution for hosts that provide a working Docker daemon."""

    def __init__(self, *args, runtime: RuntimeConfig, **kwargs):
        # The orchestration/cancellation code is shared; bwrap is not required here.
        self.runtime = runtime
        if not runtime.image or not runtime.image_digest or not re.fullmatch(r"sha256:[a-f0-9]{64}", runtime.image_digest):
            raise ValueError("Docker execution requires an image and pinned SHA-256 digest")
        super().__init__(*args, require_bwrap=False, **kwargs)
        self.docker = shutil.which("docker")
        if not self.docker:
            raise SandboxUnavailable("Docker CLI is unavailable")
        self.container_names = set()

    def command(self, argv, mounts=None):
        name = "sca-" + uuid.uuid4().hex
        with self.lock:
            self.container_names.add(name)
        image_repo = self.runtime.image.split("@", 1)[0].rsplit(":", 1)[0]
        cmd = [self.docker, "run", "--rm", "--name", name, "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "256",
               "--memory", f"{self.runtime.memory_gib}g", "--cpus", "2", "--user", f"{os.getuid()}:{os.getgid()}",
               "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m", "--workdir", "/testbed",
               "--env", "HOME=/tmp", "--env", "PYTHONPATH=/testbed", "--env", "PYTHONDONTWRITEBYTECODE=1",
               "--env", "OMP_NUM_THREADS=1",
               "--mount", f"type=bind,src={self.repo},dst=/testbed",
               "--mount", f"type=bind,src={self.repo},dst=/workspace",
               "--mount", f"type=bind,src={Path(__file__).with_name('worker.py')},dst=/sca_worker.py,readonly"]
        for name in self.protected:
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
                raise ValueError("Invalid protected path")
            source = (self.protected_root / rel).resolve()
            if not source.is_relative_to(self.protected_root):
                raise ValueError("Protected path escapes repository")
            if source.exists():
                cmd += ["--mount", f"type=bind,src={source},dst=/testbed/{rel},readonly",
                        "--mount", f"type=bind,src={source},dst=/workspace/{rel},readonly"]
        for target, source in (mounts or {}).items():
            if not target.startswith("/grader/"):
                raise ValueError("Additional mounts are limited to /grader")
            cmd += ["--mount", f"type=bind,src={source.resolve()},dst={target},readonly"]
        cmd += ["--entrypoint", "/bin/bash", image_repo + "@" + self.runtime.image_digest, "-c",
                'export PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/bin:/bin; exec "$@"', "sca", *argv]
        return cmd

    def cancel(self):
        super().cancel()
        for name in self.container_names:
            subprocess.run([self.docker, "rm", "-f", name], capture_output=True, timeout=20)

    probe = OCISandbox.probe
    filesystem = OCISandbox.filesystem

    def description(self):
        return {**OCISandbox.description(self), "backend": "docker"}


def create_sandbox(repo, events, protected, timeout, protected_root, runtime):
    common = {"protected_root": protected_root}
    if runtime.backend == "bubblewrap":
        return BubblewrapSandbox(repo, events, protected, timeout, **common)
    if runtime.backend == "venv":
        return VenvSandbox(repo, events, protected, timeout, runtime=runtime, **common)
    cls = OCISandbox if runtime.backend == "oci" else DockerSandbox
    return cls(repo, events, protected, timeout, runtime=runtime, **common)
