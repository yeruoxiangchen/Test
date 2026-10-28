"""Bubblewrap backend: only a task repository is writable; all tools share this boundary."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .control import Events, RunCancelled


class SandboxUnavailable(RuntimeError):
    pass


class BubblewrapSandbox:
    def __init__(self, repo: Path, events: Events, protected: list[str] | None = None,
                 timeout: int = 60, protected_root: Path | None = None, require_bwrap: bool = True):
        self.repo = repo.resolve()
        self.events, self.timeout = events, timeout
        self.protected = [".git", *(protected or [])]
        self.protected_root = (protected_root or self.repo).resolve()
        self.executable = shutil.which("bwrap")
        if require_bwrap and not self.executable:
            raise SandboxUnavailable("Bubblewrap is required; no host-execution fallback is provided")
        if not Path("/usr/bin/python3").exists():
            raise SandboxUnavailable("The initial Bubblewrap runtime requires /usr/bin/python3")
        self.active: set[subprocess.Popen] = set()
        self.lock = threading.RLock()
        self.cancelled = threading.Event()

    def command(self, argv: list[str], mounts: dict[str, Path] | None = None) -> list[str]:
        cmd = [self.executable, "--unshare-all", "--die-with-parent", "--cap-drop", "ALL"]
        for path in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(path).exists():
                cmd += ["--ro-bind", path, path]
        cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/tmp/home",
                "--dir", "/etc", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "HOME", "/tmp/home", "--setenv", "LANG", "C.UTF-8",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "PYTHONPATH", "/workspace",
                "--bind", str(self.repo), "/workspace", "--dir", "/opt",
                "--ro-bind", str(Path(__file__).with_name("worker.py")), "/opt/sca_worker.py"]
        for name in self.protected:
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("protected_paths must be repository-relative literal paths")
            source = (self.protected_root / rel).resolve()
            if not source.is_relative_to(self.protected_root):
                raise ValueError("Protected path escapes its source repository")
            if source.exists():
                cmd += ["--ro-bind", str(source), str(Path("/workspace") / rel)]
        for target, source in (mounts or {}).items():
            if not target.startswith("/grader/"):
                raise ValueError("Additional mounts are limited to /grader")
            cmd += ["--ro-bind", str(source.resolve()), target]
        cmd += ["--chdir", "/workspace", "--"]
        # Hard resource limits cannot be raised by the command itself.
        cmd += ["/usr/bin/prlimit", "--as=2147483648:2147483648", "--nofile=128:128",
                "--fsize=8388608:8388608", f"--cpu={self.timeout}:{self.timeout}", "--", *argv]
        return cmd

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def cancel(self) -> None:
        self.cancelled.set()
        with self.lock:
            for proc in list(self.active):
                self._kill(proc)

    def run(self, command: str | list[str], *, stdin: str | None = None,
            timeout: float | None = None, mounts: dict[str, Path] | None = None,
            log_path: Path | None = None) -> dict:
        if self.cancelled.is_set():
            raise RunCancelled("Sandbox cancelled")
        argv = ["/bin/bash", "-c", command] if isinstance(command, str) else command
        started = time.monotonic()
        elapsed_limit = min(timeout or self.timeout, self.timeout)
        full = self.command(argv, mounts)
        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen(full, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                    env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
            with self.lock:
                self.active.add(proc)
                if self.cancelled.is_set():
                    self._kill(proc)
            timed_out = False
            try:
                proc.communicate(stdin.encode() if stdin is not None else None, timeout=elapsed_limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill(proc)
                proc.communicate()
            except BaseException:
                self._kill(proc)
                proc.communicate()
                raise
            finally:
                with self.lock:
                    self.active.discard(proc)
            size = output.tell()
            output.seek(0)
            if log_path is not None:
                with log_path.open("wb") as destination:
                    shutil.copyfileobj(output, destination)
                output.seek(0)
            raw = output.read(48000)
            if size > 64000:
                output.seek(-16000, os.SEEK_END)
                raw += b"\n... output truncated ...\n" + output.read(16000)
            elif size > 48000:
                raw += output.read()
            text = raw.decode("utf-8", errors="replace")
        result = {"returncode": proc.returncode, "output": text, "timed_out": timed_out,
                  "elapsed_seconds": round(time.monotonic() - started, 4), "output_bytes": size}
        self.events.emit("sandbox_command", command=command, **result)
        if self.cancelled.is_set():
            raise RunCancelled("Sandbox cancelled")
        return result

    def probe(self) -> dict:
        result = self.run(["/usr/bin/python3", "-c", "import os; print('sandbox-ok', os.getcwd())"])
        if result["returncode"] != 0 or "sandbox-ok /workspace" not in result["output"]:
            raise SandboxUnavailable(result["output"] or "Bubblewrap probe failed")
        result["filesystem_probe"] = self.filesystem_probe()
        return result

    def filesystem_probe(self) -> dict:
        from .tool_probe import filesystem_probe
        return filesystem_probe(self)

    def filesystem(self, operation: str, arguments: dict, timeout: float | None = None,
                   *, feedback_version: int = 1, edit_guard: dict | None = None) -> str:
        result = self.run(["/usr/bin/python3", "/opt/sca_worker.py"],
                          stdin=json.dumps({"operation": operation, "arguments": arguments,
                                            "feedback_version": feedback_version, "edit_guard": edit_guard}), timeout=timeout)
        return json.dumps(result, ensure_ascii=False)

    def description(self) -> dict:
        return {"backend": "bubblewrap", "network": "unshared", "gpu_devices": False,
                "repository_mount": "/workspace", "system_runtime": "/usr/bin/python3",
                "protected_paths": self.protected, "address_space_limit_gib": 2,
                "command_timeout_seconds": self.timeout}
