"""Public repository inspection and bounded prompts, without grading metadata."""
from __future__ import annotations

import ast
import collections
import math
import os
import re
from pathlib import Path

SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.hypothesis', '.sca_scratch'}


def public_test_files(root):
    """Find original tests, including package-local and unittest-style suites."""
    root = Path(root)
    found = []
    roots = {d for d in ('tests', 'test', 'testing') if (root / d).is_dir()}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(directory) / d).is_symlink())
        if Path(directory) == root and roots:
            dirs[:] = [d for d in dirs if d in roots]
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            if name.endswith('.py') and (name.startswith(('test_', 'unittest')) or name.endswith('_test.py')):
                found.append(path.relative_to(root).as_posix())
    return found


def words(text):
    return set(re.findall(r'[a-z][a-z0-9]{2,}', text.lower().replace('_', ' ')))


def select_public_tests(root, problem, max_cases=12, allowed_files=None):
    """Rank public test functions using ONLY issue text and original test source.

    This is a bounded regression sample, not full-suite or target-issue coverage.
    Parameterized functions retain all of their cases; no hidden test IDs are read.
    """
    # Issue templates append dependency inventories: these are environment context,
    # not relevance evidence for unrelated plotting/Jupyter/backend tests.
    problem = re.split(r'^#{1,6}\s+(?:System Information|Environment|Versions|Screenshots|Code of Conduct)\b',
                       problem, maxsplit=1, flags=re.M | re.I)[0]
    query = words(problem)
    title = words(problem.splitlines()[0]) if problem else set()
    identifiers = set(re.findall(r'\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b', problem))
    identifiers |= set(re.findall(r'`([A-Za-z_]\w*)`', problem))
    candidates = []
    for name in public_test_files(root):
        if allowed_files is not None and name not in allowed_files:
            continue
        text = (Path(root) / name).read_text(errors='replace')
        try:
            tree = ast.parse(text)
        except (SyntaxError, RecursionError):
            continue
        functions = [(n, '') for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and (cls.name.startswith('Test') or cls.name.endswith(('Test', 'Tests'))):
                functions += [(n, cls.name + '::') for n in cls.body
                              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node, parent in functions:
            if not node.name.startswith('test'):
                continue
            body = ast.get_source_segment(text, node) or ''
            candidates.append((name + '::' + parent + node.name, words(body), words(node.name), words(name), body))
    df = collections.Counter(token for _, terms, _, _, _ in candidates for token in terms)
    scores = []
    for name, terms, names, paths, body in candidates:
        score = sum(math.log(1 + len(candidates)/(1 + df[t])) *
                    (1 + 2*(t in names) + (t in paths)) for t in terms & query)
        score /= 1 + len(terms) / 80
        score += 6 * sum(i in body for i in identifiers)
        score += 6 * len(title & names) + 3 * len(title & paths)
        scores.append((score, name))
    return [name for _, name in sorted(scores, key=lambda x: (-x[0], x[1]))[:max_cases]]


def classify_changes(baseline, paths, reproductions=()):
    """Recognize new diagnostics without rejecting legitimate new package modules."""
    implementation, diagnostics = [], []
    for name in paths:
        path = Path(name)
        new = not (baseline / path).exists()
        stem = path.stem.lstrip('.')
        diagnostic = (name in reproductions or '.sca_scratch' in path.parts or
                      (new and (stem.startswith(('debug', 'test_', 'repro', 'exact_issue', 'focused_test',
                                                  'minimal_debug', 'parse_structure')) or
                                stem.endswith('_test') or path.name.startswith('.'))))
        (diagnostics if diagnostic else implementation).append(name)
    return {'implementation': implementation, 'diagnostics': diagnostics}


PUBLIC_PLUGIN = '''# Controller-owned regression sampling, identical on original/candidate.
SELECTED_NODEIDS = None
def pytest_collection_modifyitems(config, items):
    counts = {}
    keep, excluded = [], []
    for item in items:
        name = item.nodeid.split("[", 1)[0]
        if SELECTED_NODEIDS is not None and name not in SELECTED_NODEIDS:
            excluded.append(item)
            continue
        counts[name] = counts.get(name, 0) + 1
        (keep if counts[name] <= 3 and len(keep) < 36 else excluded).append(item)
    items[:] = keep
    if excluded:
        config.hook.pytest_deselected(items=excluded)
'''


def patch_excerpt(text, baseline, max_chars=12000, diagnostic_paths=()):
    """Give source files a fair share instead of slicing a concatenated diff."""
    sections = re.split(r'(?=^diff --git )', text, flags=re.M)
    rows = []
    for section in sections:
        if not section:
            continue
        header = section.splitlines()[0]
        match = re.match(r'diff --git a/(.*?) b/', header)
        name = match.group(1) if match else header
        diagnostic = (name in diagnostic_paths or name.startswith('.sca_scratch/')
                      or (not (Path(baseline) / name).exists()
                          and Path(name).name.startswith(('test_', 'debug_', 'repro', 'final_', 'analyz'))))
        rows.append((diagnostic, not (Path(baseline) / name).exists(), name, section))
    rows.sort(key=lambda x: x[:3])
    if not rows:
        return '(no patch)'
    # Enumerate ALL files before snippets; disclose omitted bodies explicitly.
    inventory = 'Changed files:\n' + '\n'.join(r[2] for r in rows) + '\n'
    if len(inventory) > max_chars // 3:
        inventory = inventory[:max_chars // 3] + '\n[file list truncated]\n'
    available = max(0, max_chars - len(inventory) - 100)
    primary = [r for r in rows if not r[0]] or rows
    quota = max(0, available // len(primary) - 40)
    parts = []
    for row in primary:
        section = row[3]
        parts.append(section if len(section) <= quota else section[:quota] + '\n[diff truncated; read this file]\n')
    return (inventory + ''.join(parts) + '\n[Diagnostic diff bodies omitted when source changes exist.]')[:max_chars]
