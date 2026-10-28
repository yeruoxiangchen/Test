"""Stdlib-only filesystem worker. Executed INSIDE Bubblewrap, never imported by the agent."""

import ast
import difflib
import fnmatch
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/workspace")
MAX_BYTES = 1_000_000


def contained(path: Path, root: Path) -> bool:
    """Path.is_relative_to is unavailable in the pinned Python 3.6/3.8 images."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def path_in_repo(name: str) -> Path:
    path = Path(name)
    # OCI exposes the same bind-mounted workspace at both paths. Normalize only
    # a verified alias; an unrelated /testbed must never expand tool access.
    alias = Path("/testbed")
    if path.is_absolute() and contained(path, alias) and alias.exists() and os.path.samefile(str(alias), str(ROOT)):
        path = ROOT / path.relative_to(alias)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not contained(path, ROOT) or ".git" in path.relative_to(ROOT).parts:
        raise ValueError("Path must stay inside /workspace and outside .git")
    return path


def read_text(path: Path) -> str:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("File exceeds 1 MB tool limit; use a shell command for a targeted query")
    return path.read_text(encoding="utf-8")


def anchored_patch(content, patch):
    """Adapted from AnyCoder tools/apply_patch.py (MIT; third_party/AnyCoder.LICENSE).

    All anchors refer to ORIGINAL contents. Apply spans backwards so one new
    replacement cannot accidentally become a later SEARCH anchor.
    """
    lines, blocks, i = patch.split('\n'), [], 0
    while i < len(lines):
        if lines[i].strip() != '<<<<<<< SEARCH':
            i += 1
            continue
        try:
            middle = lines.index('=======', i+1)
            end = lines.index('>>>>>>> REPLACE', middle+1)
        except ValueError:
            raise ValueError('Unterminated SEARCH/REPLACE block')
        old, new = '\n'.join(lines[i+1:middle]), '\n'.join(lines[middle+1:end])
        if not old or content.count(old) != 1:
            raise ValueError('Every SEARCH block must be nonempty and match exactly once; nothing changed')
        start = content.index(old)
        blocks.append((start, start+len(old), new))
        i = end+1
    if not blocks:
        raise ValueError('No SEARCH/REPLACE blocks')
    blocks.sort()
    if any(b[0] < a[1] for a,b in zip(blocks, blocks[1:])):
        raise ValueError('SEARCH blocks overlap; nothing changed')
    for start,end,new in reversed(blocks):
        content = content[:start]+new+content[end:]
    return content


def execute(data: dict) -> dict:
    op, args = data["operation"], data.get("arguments", {})
    enhanced = data.get("feedback_version") == 2
    if op == "read_file":
        p = path_in_repo(args["file_path"])
        text = read_text(p)
        lines = text.splitlines()
        start = max(0, int(args.get("offset", 1)) - 1)
        end = start + min(500, max(1, int(args.get("limit", 200))))
        result = {"output": "\n".join(f"{i+1}\t{line}" for i, line in enumerate(lines[start:end], start)),
                  "total_lines": len(lines)}
        if enhanced:
            result.update(file_path=args["file_path"], start_line=start+1,
                          end_line=min(end, len(lines)), next_offset=end+1 if end < len(lines) else None,
                          file_sha256=hashlib.sha256(text.encode()).hexdigest(),
                          note="Line-number prefixes are not source text; do not include them in old_string.")
        return result
    if op in {"write_file", "edit_file", "apply_patch"}:
        p = path_in_repo(args["file_path"])
        old = read_text(p) if p.exists() else ""
        if op == "write_file":
            new = args["content"]
        elif op == "apply_patch":
            new = anchored_patch(old, args["patch"])
        else:
            before, after = args["old_string"], args["new_string"]
            matches = old.count(before) if before else 0
            if enhanced:
                file_hash = hashlib.sha256(old.encode()).hexdigest()
                guard = data.get("edit_guard") or {}
                repeated = guard.get("file_sha256") == file_hash and guard.get("failures", 0) >= guard.get("limit", 2)
                if repeated or not before or matches != 1:
                    lines = old.splitlines()
                    # Diagnostic only: show an actual nearby excerpt, never perform a fuzzy edit.
                    needle = next((line.strip() for line in before.splitlines() if line.strip()), "")
                    scores = [difflib.SequenceMatcher(None, needle, line.strip()).ratio() for line in lines]
                    center = max(range(len(lines)), key=lambda i: scores[i]) if lines and needle else 0
                    start, end = max(0, center-4), min(len(lines), center+8)
                    return {"ok": False, "error": "repeated_failed_edit" if repeated else "edit_match_failed",
                            "match_count": matches, "file_sha256": file_hash, "file_path": args["file_path"],
                            "hint": "Read the current file, copy the exact source without line numbers, then change the edit. "
                                    "Use actual newline characters, not literal backslash-n. No fuzzy replacement was made.",
                            "context_start_line": start+1,
                            "output": "\n".join(f"{i+1}\t{line}" for i, line in enumerate(lines[start:end], start))}
            if not before or matches != 1:
                raise ValueError("old_string must be nonempty and match exactly once; read more context")
            new = old.replace(before, after, 1)
        if len(new.encode()) > MAX_BYTES:
            raise ValueError("New file exceeds 1 MB")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new, encoding="utf-8")
        name = str(p.relative_to(ROOT))
        result = {"output": "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                       fromfile=f"a/{name}", tofile=f"b/{name}"))}
        if enhanced:
            result.update(ok=True, file_path=name, changed=old != new,
                          file_sha256=hashlib.sha256(new.encode()).hexdigest())
            if p.suffix == ".py":
                try:
                    ast.parse(new, filename=name)
                    result["python_syntax"] = {"valid": True, "interpreter": sys.version.split()[0]}
                except (SyntaxError, ValueError) as error:
                    # Keep the edit so multi-step repairs remain possible; verifier decides correctness.
                    result["python_syntax"] = {"valid": False, "error": str(error), "interpreter": sys.version.split()[0],
                                               "line": getattr(error, "lineno", None)}
        return result
    if op in {"glob", "grep"}:
        pattern = args.get("pattern", "**/*")
        if op == "glob" and (Path(pattern).is_absolute() or ".." in Path(pattern).parts):
            raise ValueError("glob pattern must be relative and cannot contain '..'")
        root = path_in_repo(args.get("path", "."))
        if not root.exists():
            raise ValueError("Search path does not exist; use a literal file/directory path, or glob to find filenames")
        def candidates():
            if root.is_file():
                yield root
                return
            for folder, directories, files in os.walk(str(root), followlinks=False):
                directories[:] = sorted(d for d in directories if d not in {
                    ".git", "__pycache__", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
                for name in sorted(files):
                    yield Path(folder) / name
        matches = []
        line_truncated = False
        regex = re.compile(pattern) if op == "grep" else None
        for p in candidates():
            relative = p.relative_to(ROOT)
            if any(part in {".git", "__pycache__", ".venv"} for part in relative.parts):
                continue
            if not p.is_file() or not contained(p.resolve(), ROOT):
                continue
            name = str(relative)
            if op == "glob":
                if fnmatch.fnmatch(name, pattern) or p.match(pattern):
                    matches.append(name)
            else:
                try:
                    text = read_text(p)
                except (UnicodeError, ValueError):
                    continue
                for number, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        line_truncated = line_truncated or len(line) > 300
                        matches.append(f"{name}:{number}:{line[:300]}")
                        if len(matches) > 200:
                            break
            if len(matches) > 200:
                break
        return {"output": "\n".join(matches[:200]) or "(no matches)",
                "returned_matches": min(200, len(matches)), "truncated": len(matches) > 200,
                "line_text_truncated": line_truncated,
                "hint": "Search was capped; narrow the directory/pattern to inspect remaining matches."
                        if len(matches) > 200 else ""}
    raise ValueError(f"Unsupported filesystem operation: {op}")


if __name__ == "__main__":
    try:
        result = execute(json.load(sys.stdin))
        print(json.dumps(result, ensure_ascii=False))
        if result.get("ok") is False:
            sys.exit(1)
    except Exception as error:
        print(json.dumps({"error": str(error), "error_type": type(error).__name__}))
        sys.exit(1)
