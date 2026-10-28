"""Private base snapshots, complete patches (including new files), and clean replay."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

IGNORED = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def git(repo: Path, *args: str, stdin: bytes | None = None, check: bool = True,
        timeout: float = 180) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8",
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    command = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
               "-c", "core.fileMode=true", "-c", "user.name=Screenshot Agent",
               "-c", "user.email=screenshot-agent@localhost", *args]
    return subprocess.run(command, cwd=repo, env=env, input=stdin, capture_output=True,
                          check=check, timeout=timeout)


def validate_tree(root: Path, *, max_files: int = 10000, max_bytes: int = 100_000_000,
                  allowed_symlinks: dict[str, str] | None = None) -> list[Path]:
    root = Path(root)
    resolved_root = root.resolve()
    allowed = allowed_symlinks or {}
    for name, target in allowed.items():
        rel = Path(name)
        if (not name or rel.is_absolute() or ".." in rel.parts or rel == Path(".")
                or str(rel) != name or set(rel.parts) & IGNORED):
            raise ValueError(f"Invalid allowed symbolic link path: {name}")
        if not target or Path(target).is_absolute() or "\0" in target:
            raise ValueError(f"Allowed symbolic link target must be relative: {name}")
        p = root / rel
        if not p.is_symlink() or os.readlink(p) != target:
            raise ValueError(f"Baseline symbolic link was removed or changed: {name}")
        try:
            destination = p.resolve()
        except (OSError, RuntimeError) as error:
            raise ValueError(f"Unresolvable symbolic link: {name}") from error
        if not destination.is_relative_to(resolved_root):
            raise ValueError(f"Symbolic link escapes repository: {name}")
    files, total = [], 0
    seen_links = set()

    def add(path, size):
        nonlocal total
        total += size
        files.append(path)
        if len(files) > max_files or total > max_bytes:
            raise ValueError(f"Repository snapshot exceeds limit ({max_files} files / {max_bytes} bytes)")

    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in sorted([*dirs, *names]):
            p = Path(folder) / name
            if p.is_symlink():
                rel = str(p.relative_to(root))
                if rel not in allowed:
                    raise ValueError(f"Repository snapshots reject unlisted symbolic links: {rel}")
                if os.readlink(p) != allowed[rel]:
                    raise ValueError(f"Baseline symbolic link was changed: {rel}")
                seen_links.add(rel)
                add(p, len(os.fsencode(allowed[rel])))
        dirs[:] = sorted(name for name in dirs if name not in IGNORED and not (Path(folder) / name).is_symlink())
        for name in sorted(names):
            p = Path(folder) / name
            if p.is_symlink():
                continue
            if not p.is_file():
                raise ValueError(f"Unsupported special file: {p.relative_to(root)}")
            add(p, p.stat().st_size)
    if seen_links != set(allowed):
        raise ValueError("Allowed symbolic link is missing or hidden beneath another link")
    return files


def prepare_repository(source: Path, run_dir: Path, **tree_limits) -> tuple[Path, Path]:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"Repository does not exist: {source}")
    if run_dir.resolve().is_relative_to(source):
        raise ValueError("Run output must be outside the input repository (use --output-root /tmp/sca-runs)")
    validate_tree(source, **tree_limits)
    baseline, workspace = run_dir / "baseline", run_dir / "workspace"
    shutil.copytree(source, baseline, symlinks=True, ignore=shutil.ignore_patterns(*IGNORED))
    git(baseline, "init", "-q")
    git(baseline, "add", "-A", "-f", "--", ".")
    git(baseline, "commit", "-q", "--allow-empty", "-m", "Isolated buggy base snapshot")
    shutil.copytree(baseline, workspace, symlinks=True)
    return baseline, workspace


def protected_changes(baseline: Path, workspace: Path, names: list[str]) -> list[str]:
    changed = []
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
            raise ValueError("Protected paths must be nonempty repository-relative paths")
        a, b = baseline / rel, workspace / rel
        if a.is_file():
            if not b.is_file() or b.is_symlink() or a.read_bytes() != b.read_bytes():
                changed.append(name)
        elif a.is_dir():
            aa = {str(p.relative_to(a)): p.read_bytes() for p in a.rglob("*") if p.is_file()}
            bb = {str(p.relative_to(b)): p.read_bytes() for p in b.rglob("*") if p.is_file()} if b.is_dir() else {}
            if aa != bb:
                changed.append(name)
        elif b.exists():
            changed.append(name)
    return changed


def export_patch(baseline: Path, workspace: Path, path: Path, protected: list[str], **tree_limits) -> dict:
    validate_tree(workspace, **tree_limits)
    violations = protected_changes(baseline, workspace, protected)
    if violations:
        raise ValueError(f"Protected files changed: {violations}")
    # Preserve new source files while excluding the same runtime/cache directories
    # omitted from the private initial snapshot. Tests may create these themselves.
    paths = [".", *[f":(exclude,glob)**/{name}/**" for name in sorted(IGNORED) if name != ".git"]]
    git(workspace, "add", "-A", "-f", "--", *paths)
    patch = git(workspace, "diff", "--cached", "--binary", "--no-ext-diff", "--no-textconv",
                "--no-renames", "HEAD", "--", *paths).stdout
    path.write_bytes(patch)
    changed = git(workspace, "diff", "--cached", "--name-only", "-z", "HEAD", "--", *paths).stdout
    return {"patch_bytes": len(patch), "changed_files": [p.decode() for p in changed.split(b"\0") if p]}


def replay_patch(baseline: Path, target: Path, patch: Path, **tree_limits) -> dict:
    shutil.copytree(baseline, target, symlinks=True)
    payload = patch.read_bytes()
    if not payload:
        return {"applied": False, "error": "No patch produced"}
    result = git(target, "apply", "--check", "--binary", "-", stdin=payload, check=False)
    if result.returncode:
        return {"applied": False, "error": result.stderr.decode(errors="replace")}
    git(target, "apply", "--binary", "-", stdin=payload)
    try:
        validate_tree(target, **tree_limits)
    except ValueError as error:
        return {"applied": False, "error": str(error)}
    return {"applied": True}
