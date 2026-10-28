"""Local repository map and lexical code retrieval; no embedding service required.

Design references: Aider's repository map and CoreCoder-Agent's incremental
repository index. This implementation indexes only the isolated working copy.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import math
import os
import re
from pathlib import Path

from .control import atomic_json

SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.pytest_cache',
        '.mypy_cache', '.ruff_cache', 'dist', 'build', '.tox'}
SUFFIXES = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', '.c', '.cpp',
            '.h', '.hpp', '.rb', '.sh', '.sql', '.md', '.toml', '.yaml', '.yml', '.json'}


def terms(text):
    # Retain identifiers AND their snake_case/camelCase components.
    words = re.findall(r'[A-Za-z_][A-Za-z_0-9]*', text)
    parts = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', ' '.join(words)).replace('_', ' ')
    return [w.lower() for w in words + parts.split() if len(w) > 1]


def symbols(text, suffix):
    found = []
    if suffix == '.py':
        try:
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    found.append({'name':node.name, 'line':node.lineno,
                                  'kind':'class' if isinstance(node, ast.ClassDef) else 'function'})
        except (SyntaxError, RecursionError):
            pass  # Older-language syntax still has usable lexical chunks.
    if not found:
        for i, line in enumerate(text.splitlines(), 1):
            match = re.search(r'\b(?:class|def|function|func|fn|interface|struct)\s+([\w]+)', line)
            if match:
                found.append({'name':match[1], 'line':i, 'kind':'symbol'})
    return sorted(found, key=lambda s:s['line'])[:150]


class RepoSearch:
    def __init__(self, root, cache, *, max_files=10000, max_file_bytes=500000, max_chunks=30000):
        self.root, self.cache = Path(root).resolve(), Path(cache)
        self.max_files, self.max_file_bytes, self.max_chunks = max_files, max_file_bytes, max_chunks
        self.files = {}
        if self.cache.exists():
            data = json.loads(self.cache.read_text())
            if data.get('root') == str(self.root) and data.get('version') == 1:
                self.files = data['files']
        self.last = {}

    def refresh(self):
        live, changed, skipped, truncated, chunks = set(), 0, 0, False, 0
        for folder, dirs, names in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(folder)/d).is_symlink())
            for name in sorted(names):
                path = Path(folder)/name
                if (path.is_symlink() or not path.is_file() or path.suffix.lower() not in SUFFIXES
                        or not path.resolve().is_relative_to(self.root)):
                    continue
                if len(live) >= self.max_files or chunks >= self.max_chunks:
                    truncated = True
                    break
                stat = path.stat()
                if stat.st_size > self.max_file_bytes:
                    skipped += 1
                    continue
                rel = path.relative_to(self.root).as_posix()
                live.add(rel)
                stamp = [stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size]
                old = self.files.get(rel)
                if old and old['stamp'] == stamp:
                    chunks += len(old['chunks'])
                    continue
                text = path.read_text(encoding='utf-8', errors='replace')
                if '\x00' in text:
                    live.remove(rel)
                    continue
                lines, syms = text.splitlines(), symbols(text, path.suffix)
                rows = []
                for start in range(0, len(lines), 55):
                    if chunks + len(rows) >= self.max_chunks:
                        truncated = True
                        break
                    end = min(start+70, len(lines))
                    content = '\n'.join(lines[start:end])
                    rows.append({'start_line':start+1, 'end_line':end, 'content':content,
                                 'terms':dict(collections.Counter(terms(content)))})
                    if end == len(lines):
                        break
                self.files[rel] = {'stamp':stamp, 'sha256':hashlib.sha256(text.encode()).hexdigest(),
                                  'symbols':syms, 'chunks':rows}
                chunks += len(rows)
                changed += 1
            if truncated:
                break
        self.files = {k:v for k,v in self.files.items() if k in live}
        self.last = {'files':len(self.files), 'chunks':chunks, 'updated_files':changed,
                     'skipped_large_files':skipped, 'truncated':truncated}
        atomic_json(self.cache, {'version':1, 'root':str(self.root), 'files':self.files, 'stats':self.last})
        return self.last

    def search(self, query, top_k=8, *, refresh=True):
        if refresh:
            self.refresh()
        query_terms = set(terms(query))
        if not query_terms:
            return []
        documents = [(path, row, item) for path,item in self.files.items() for row in item['chunks']]
        df = collections.Counter(t for _,row,_ in documents for t in query_terms if t in row['terms'])
        avg = sum(sum(row['terms'].values()) for _,row,_ in documents)/max(1,len(documents))
        result = []
        for path, row, item in documents:
            length = sum(row['terms'].values())
            score = 0.0
            for token in query_terms:
                freq = row['terms'].get(token, 0)
                if freq:
                    idf = math.log(1+(len(documents)-df[token]+.5)/(df[token]+.5))
                    score += idf*freq*2.2/(freq+1.2*(.25+.75*length/max(1,avg)))
            score += 3*len(query_terms & set(terms(path)))
            matched = [s for s in item['symbols'] if row['start_line'] <= s['line'] <= row['end_line']
                       and query_terms & set(terms(s['name']))]
            score += 4*len(matched)
            if score:
                result.append({'path':path, 'start_line':row['start_line'], 'end_line':row['end_line'],
                               'score':round(score,4), 'symbols':matched, 'content':row['content'],
                               'file_sha256':item['sha256']})
        return sorted(result, key=lambda r:(-r['score'], r['path'], r['start_line']))[:max(1,min(20,top_k))]

    def repo_map(self, query='', max_chars=9000, *, refresh=True):
        if refresh:
            self.refresh()
        ranked = [row['path'] for row in self.search(query, 20, refresh=False)] if query else []
        paths = list(dict.fromkeys(ranked + sorted(self.files)))
        rows, size = [], 0
        for path in paths:
            syms = self.files[path]['symbols']
            text = path + ('\n  ' + '; '.join(f"{s['kind']} {s['name']} :{s['line']}" for s in syms[:25]) if syms else '')
            if size + len(text) + 1 > max_chars:
                break
            rows.append(text)
            size += len(text)+1
        return {'output':'\n'.join(rows), 'shown_files':len(rows), **self.last,
                'truncated':self.last.get('truncated', False) or len(rows)<len(paths),
                'method':'lexical retrieval + source symbols; not embedding/semantic retrieval'}
