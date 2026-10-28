"""Immutable, task-external compatibility packages for historical Full dev projects.

These overlays repair dependency drift, not task source or reference tests. Every
task must still reproduce its bug and pass the unchanged official gold checks.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from .control import atomic_json

PROFILES = {
    "pvlib/pvlib-python": ["numpy==1.23.5", "pandas==1.5.3", "scipy==1.10.1"],
    "pydicom/pydicom": ["pytest==7.4.4"],
    "pyvista/pyvista": ["vtk==9.2.6"],
    "sqlfluff/sqlfluff": ["pytest==7.4.4", "Jinja2==3.0.3", "click==8.0.4", "appdirs==1.4.4",
                         "chardet==5.2.0", "colorama==0.4.6", "diff-cover==7.7.0", "pathspec==0.11.2",
                         "PyYAML==6.0.1", "regex==2023.10.3", "tblib==1.7.0", "tqdm==4.66.1"],
}
GL_LIBRARIES = ("libGL.so.1", "libGLX.so.0", "libGLdispatch.so.0", "libXrender.so.1")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seal_overlay(folder, details):
    files = {str(p.relative_to(folder)): file_sha(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
    atomic_json(folder / "manifest.json", {"schema": 1, **details, "files_sha256": files})


def python_overlay(root, repo, rootfs):
    packages = PROFILES.get(repo)
    if not packages:
        return None
    versions = [p.name.removeprefix("python")
                for p in (Path(rootfs) / "opt/miniconda3/envs/testbed/lib").glob("python3.*")
                if (p / "encodings").is_dir()]
    if repo == "pydicom/pydicom" and versions in (["3.6"], ["3.7"]):
        # These old images already use legacy-compatible pytest; newer pytest cannot run on their interpreter.
        return None
    if len(versions) != 1 or versions[0] not in {"3.8", "3.9", "3.10", "3.11"}:
        raise ValueError(f"Cannot select compatible wheel ABI for {repo}: {versions}")
    version = versions[0]
    abi = "cp" + version.replace(".", "")
    details = {"repo": repo, "python": version, "requirements": packages,
               "protocol": "full_dev_dependency_compat_v1", "source": "pinned PyPI wheels; no task code"}
    key = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()[:16]
    cache = root / ".cache/compat"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{repo.split('/')[-1]}-{abi}-{key}"
    with (cache / f"{target.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (target / "manifest.json").exists():
            staging = cache / f"{target.name}.building.{time.time_ns()}"
            command = [str(root / ".model-env/bin/python"), "-m", "pip", "install", "--target", str(staging),
                       "--platform", "manylinux2014_x86_64", "--python-version", version,
                       "--implementation", "cp", "--abi", abi, "--only-binary=:all:", "--no-deps",
                       "--disable-pip-version-check", "--cache-dir", str(root / ".cache/pip-compat"),
                       *packages]
            print(f"Preparing compatibility packages for {repo}: {packages}", flush=True)
            subprocess.run(command, check=True, timeout=1800)
            seal_overlay(staging, details)
            if target.exists():
                target.rename(cache / f"{target.name}.incomplete.{time.time_ns()}")
            staging.rename(target)
        saved = json.loads((target / "manifest.json").read_text())
        if any(saved.get(k) != v for k, v in details.items()):
            raise ValueError("Compatibility package manifest differs from its fixed profile")
    return target


def gl_overlay(root):
    sources = {}
    for name in GL_LIBRARIES:
        candidates = [Path(base) / name for base in ("/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu")]
        path = next((p.resolve() for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError(f"Missing declared GL compatibility library: {name}")
        sources[name] = {"path": str(path), "sha256": file_sha(path)}
    key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    cache = root / ".cache/compat"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"gl-libraries-{key}"
    with (cache / f"{target.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (target / "manifest.json").exists():
            staging = cache / f"{target.name}.building.{time.time_ns()}"
            staging.mkdir()
            for name, item in sources.items():
                shutil.copyfile(item["path"], staging / name)
            seal_overlay(staging, {"protocol": "explicit_readonly_gl_compat_v1", "sources": sources})
            staging.rename(target)
    return target


def compatibility_runtime(root, record, image):
    fields, evidence = {}, {"protocol": "full_dev_dependency_compat_v1", "overlays": {}}
    if record.get("sca_split") != "dev":
        return fields, evidence
    python = python_overlay(root, record["repo"], image["rootfs"])
    libraries = gl_overlay(root) if record["repo"] == "pyvista/pyvista" else None
    for name, path in (("python", python), ("library", libraries)):
        if path:
            checksum = file_sha(path / "manifest.json")
            fields[f"{name}_overlay"] = str(path)
            fields[f"{name}_overlay_manifest_sha256"] = checksum
            evidence["overlays"][name] = {"path": str(path), "manifest_sha256": checksum}
    return fields, evidence


def prepare_project_metadata(root, record, repo, runtime, events, *, adapted=False):
    """Build target SQLFluff metadata in isolation; never reuse the donor's version/plugins.

    The caller must explicitly opt in for an adapted runtime. Only a new egg-info
    directory is permitted to change; all other source bytes, links and modes are
    compared before/after. Returns (protected_paths, preparation_evidence).
    """
    import os
    import stat

    from .runtime import create_sandbox

    if record.get("repo") != "sqlfluff/sqlfluff" or not adapted:
        return [], {}
    repo = Path(repo).resolve()
    relative = "src/sqlfluff.egg-info"
    metadata = repo / relative
    if metadata.exists() or metadata.is_symlink():
        raise ValueError("Target project metadata already exists; archive the attempt instead of overwriting it")
    if not (repo / "setup.py").is_file() or not (repo / "setup.cfg").is_file():
        raise ValueError("Target SQLFluff source has no setup.py/setup.cfg for metadata generation")

    def fingerprint():
        entries = {}
        for directory, dirs, files in os.walk(repo, followlinks=False):
            dirs[:] = sorted(d for d in dirs if Path(directory) / d != metadata)
            for name in sorted(dirs + files):
                path = Path(directory) / name
                if path == metadata:
                    continue
                item = path.lstat()
                rel = str(path.relative_to(repo))
                if stat.S_ISLNK(item.st_mode):
                    entries[rel] = ["symlink", os.readlink(path), item.st_mode & 0o777]
                elif stat.S_ISREG(item.st_mode):
                    entries[rel] = ["file", file_sha(path), item.st_mode & 0o777]
                elif stat.S_ISDIR(item.st_mode):
                    entries[rel] = ["directory", item.st_mode & 0o777]
                else:
                    raise ValueError(f"Unsupported source entry during metadata preparation: {rel}")
        return entries

    before = fingerprint()
    log = repo.parent / "project_metadata.log"
    probe_log = repo.parent / "project_metadata_probe.log"
    command = [runtime.python, "setup.py", "egg_info", "--egg-base", "src"]
    sandbox = create_sandbox(repo, events, [], 180, repo, runtime)
    try:
        built = sandbox.run(command, log_path=log)
        after = fingerprint()
        if after != before:
            changed = sorted(set(before) ^ set(after) | {p for p in before.keys() & after.keys() if before[p] != after[p]})
            raise ValueError(f"Metadata preparation changed task source outside egg-info: {changed[:10]}")
        if built["returncode"] or built["timed_out"]:
            raise RuntimeError(f"Target project metadata generation failed; inspect {log}")
        if not (metadata / "PKG-INFO").is_file():
            raise ValueError("Source setup did not generate the expected SQLFluff egg-info")
        probe = (
            "import sys,json; sys.path.insert(0,'/testbed/src'); "
            "import importlib.metadata as m; d=m.distribution('sqlfluff'); "
            "print(json.dumps({'name':d.metadata['Name'],'version':d.version,"
            "'metadata_directory':str(d._path),'entry_points':sorted("
            "[[e.group,e.name,e.value] for e in d.entry_points])}))"
        )
        inspected = sandbox.run([runtime.python, "-c", probe], log_path=probe_log)
        if inspected["returncode"] or inspected["timed_out"]:
            raise RuntimeError(f"Target project metadata query failed; inspect {probe_log}")
        details = json.loads(inspected["output"].strip().splitlines()[-1])
        if details["metadata_directory"].rstrip("/") != "/testbed/" + relative:
            raise ValueError("importlib.metadata resolved donor metadata instead of the target source")
        if fingerprint() != before:
            raise ValueError("Metadata inspection changed task source outside egg-info")
    finally:
        sandbox.cancel()
    files = {str(p.relative_to(repo)): file_sha(p) for p in sorted(metadata.rglob("*")) if p.is_file()}
    evidence = {"protocol": "exact_source_sqlfluff_egg_info_v1", "base_commit": record["base_commit"],
                "command": command, "network": "unshared", "source_unchanged": True,
                "source_files_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                "setup_cfg_sha256": file_sha(repo / "setup.cfg"), "setup_py_sha256": file_sha(repo / "setup.py"),
                "generated_files_sha256": files, "log": str(log), "probe_log": str(probe_log), **details}
    atomic_json(repo.parent / "project_metadata.json", evidence)
    return [relative], evidence
