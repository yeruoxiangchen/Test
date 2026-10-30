"""Track source semantics across tools, including shell edits and Git rollback."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from .control import NoProgress
from .mini_repair_control import RepairControl
from .repository import IGNORED, git, is_debug_artifact


class WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
        return node


class SourceState:
    """Read only source files; cache parses by stat and ignore agent Git HEAD."""
    SUFFIXES = {'.py', '.js', '.ts', '.tsx', '.jsx', '.c', '.cpp', '.h', '.rs', '.go', '.java'}

    def __init__(self, delivery):
        self.delivery = delivery
        self.cache = {}

    def snapshot(self):
        d = self.delivery
        names = git(d.baseline, 'ls-files', '--cached', '--others', '--exclude-standard', '-z',
                    work_tree=d.workspace).stdout.decode().split('\0')
        values, valid = {}, True
        for name in sorted(set(names)):
            rel = Path(name)
            if (not name or rel.suffix not in self.SUFFIXES or set(rel.parts) & IGNORED
                    or any(name == p or name.startswith(p.rstrip('/') + '/') for p in d.protected)
                    or any(part in {'tests', 'test'} for part in rel.parts)
                    or (not (d.baseline/rel).exists() and len(rel.parts) == 1 and is_debug_artifact(name))):
                continue
            path = d.workspace/rel
            if not path.resolve().is_relative_to(d.workspace.resolve()):
                valid = False
                continue
            if not path.exists():
                values[name] = 'deleted'
                continue
            info = path.stat()
            stamp = (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            old = self.cache.get(name)
            if not old or old[0] != stamp:
                if info.st_size > 4_000_000:
                    valid = False
                    continue
                data = path.read_bytes()
                okay = True
                if rel.suffix == '.py':
                    try:
                        data = ast.dump(WithoutDocstrings().visit(ast.parse(data)), include_attributes=False).encode()
                    except (SyntaxError, ValueError):
                        okay = False
                old = self.cache[name] = (stamp, hashlib.sha256(data).hexdigest(), okay)
            values[name] = old[1]
            valid = valid and old[2]
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(), valid


class SemanticControl(RepairControl):
    LIMIT = 12

    def __init__(self, delivery):
        super().__init__(delivery)
        self.source = SourceState(delivery)
        self.state, _ = self.source.snapshot()
        self.seen = {self.state}
        self.unchanged = 0

    def guard(self):
        if self.failed and self.unchanged >= self.LIMIT:
            raise NoProgress('Twelve actions without a new valid source state after public failure; candidate retained')

    def message(self):
        # The leading structured failure summary must survive long pytest tails.
        # RepairControl's tail-only truncation could hide the actual API defect.
        message = ({'role': 'user', 'content': 'Controller: task NOT verified. Remaining public failure:\n'
                    + self.failed[:6000]} if self.failed else None)
        if message:
            message['content'] += (f'\nActions without new source semantics: {self.unchanged}/{self.LIMIT}. '
                'Reads, rewritten self-tests, comments, and reverting to a previously seen state do not reset this. '
                'Fix the named failing contract; partial candidates are not successful submissions.')
        return message

    def before(self, action):
        self.guard()
        command = action.get('command', '')
        if isinstance(command, str) and (self.state, self.key(command)) in self.timed_out:
            return {'returncode': 2, 'output': 'This command timed out at this source state. Change the '
                    'implementation before retrying.', 'timed_out': False, 'retry_blocked': True}
        return None

    def after(self, action, output):
        state, valid = self.source.snapshot()
        novel = valid and state not in self.seen
        if novel:
            self.epoch += 1
            self.unchanged = 0
            self.seen.add(state)
        elif self.failed:
            self.unchanged += 1
        self.state = state
        if output.get('timed_out'):
            command = action.get('command')
            if isinstance(command, str):
                self.timed_out.add((state, self.key(command)))
            if not self.failed:
                self.failed = 'Command timed out. Inspect termination and the implementation.\n' + output['output'][-1000:]
        if action.get('tool') == 'check_patch' and self.delivery.last_response:
            self.feedback(self.delivery.last_response)
        self.delivery.sandbox.events.emit('semantic_progress', novel=novel, valid=valid,
            source_state=state, unchanged_actions=self.unchanged, failed=bool(self.failed),
            retry_blocked=output.get('retry_blocked', False))

    def feedback(self, checked):
        self.failed = '' if checked['passed'] else checked['output']
        if checked['passed']:
            self.unchanged = 0
