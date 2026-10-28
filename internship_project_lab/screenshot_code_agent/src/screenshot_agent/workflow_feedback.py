"""Visible-test baselines and frozen reproducers, separate from hidden grading."""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .repository import export_patch, replay_patch
from .runtime import create_sandbox


def junit_statuses(path):
    """Return stable case outcomes; missing/unreadable reports are inconclusive."""
    try:
        if path.is_symlink() or path.stat().st_size > 4_000_000:
            return None
        root = ET.fromstring(path.read_bytes())
        results = {}
        for case in root.iter('testcase'):
            key = case.get('classname', '') + '::' + case.get('name', '')
            if key in results:  # Ambiguous IDs cannot support a regression claim.
                return None
            results[key] = ('error' if case.find('error') is not None else
                            'failed' if case.find('failure') is not None else
                            'skipped' if case.find('skipped') is not None else 'passed')
        return results or None
    except (OSError, ET.ParseError, ValueError):
        return None


def compare_public(baseline, candidate):
    """No-new-regression is deliberately not a claim that the issue was fixed."""
    if candidate.get('patch_replay', {}).get('applied') is False:
        return {'status': 'inconclusive', 'reason': 'patch_replay_failed'}
    if baseline['timed_out'] or candidate['timed_out']:
        return {'status': 'inconclusive', 'reason': 'public_test_timeout'}
    old, new = baseline.get('cases'), candidate.get('cases')
    if baseline.get('structured'):
        if not old or not new or not any(v == 'passed' for v in old.values()):
            return {'status': 'inconclusive', 'reason': 'no_comparable_passing_baseline'}
        lost = sorted(k for k, v in old.items() if v == 'passed' and new.get(k) != 'passed')
        additional = sorted(k for k, v in new.items()
                            if v in {'failed', 'error'} and old.get(k) not in {'failed', 'error'})
        if lost or additional:
            return {'status': 'failed', 'lost_passing_tests': lost, 'additional_failures': additional}
        if baseline['returncode'] not in (0, 1) or candidate['returncode'] not in (0, 1):
            return {'status': 'inconclusive', 'reason': 'test_collection_or_runner_error'}
        return {'status': 'passed', 'checked_passing_tests': sum(v == 'passed' for v in old.values()),
                'preexisting_failures': sorted(k for k, v in old.items() if v in {'failed', 'error'})}
    if candidate['returncode'] == 0:
        return {'status': 'passed', 'scope': 'user_command_exit_code_only'}
    return {'status': 'failed' if baseline['returncode'] == 0 else 'inconclusive',
            'reason': 'command_newly_failed' if baseline['returncode'] == 0 else 'unstructured_existing_failure'}


class VisibleFeedback:
    def __init__(self, flow):
        self.flow = flow

    def public_command(self):
        f = self.flow
        parts = shlex.split(f.task.test_command)
        structured = len(parts) == 3 and parts[1:] == ['-m', 'pytest']
        if structured:
            # Repository-visible directory selection only; never use reference test IDs.
            roots = [name for name in ('tests', 'test', 'testing') if (f.baseline / name).is_dir()]
            if not roots:
                roots = sorted(p.name for p in f.baseline.glob('test_*.py') if p.is_file())
            if not roots:
                return None, True
            parts += roots + ['-p', 'no:cacheprovider', '--junitxml=.sca_public_results.xml']
        return shlex.join(parts) if structured else f.task.test_command, structured

    def run(self, label, command, *, patch=None, script=None, structured=False):
        f = self.flow
        if command is None and patch is None:
            return {'command': None, 'log': None, 'patch_replay': {'applied': True},
                    'structured': True, 'returncode': None, 'timed_out': False,
                    'output': 'No public test suite found in the original repository.', 'cases': None}
        f.budget.tool('visible_validation', {'label': label, 'command': command})
        folder = f.directory / 'public_checks' / f'{label}_{time.time_ns()}'
        folder.mkdir(parents=True)
        repo = folder / 'repo'
        if patch is None or not patch.stat().st_size:
            shutil.copytree(f.baseline, repo, symlinks=True)
            replay = {'applied': True, 'empty_patch': patch is not None}
        else:
            replay = replay_patch(f.baseline, repo, patch, **f.tree)
        common = {'command': command, 'log': str(folder / 'test.log'),
                  'patch_replay': replay, 'structured': structured}
        if not replay['applied']:
            return {**common, 'returncode': None, 'timed_out': False,
                    'output': replay['error'], 'cases': None}
        if command is None:
            return {**common, 'returncode': None, 'timed_out': False,
                    'output': 'No public test suite found in the original repository.', 'cases': None}
        if (repo / '.sca_public_results.xml').exists() or (repo / '.sca_public_results.xml').is_symlink():
            raise ValueError('Reserved public report path exists in repository')
        if script:
            name, content = script
            if (repo / name).exists() or (repo / name).is_symlink():
                raise ValueError('Reserved reproduction path exists in repository')
            (repo / name).write_bytes(content)
        timeout = min(f.config.limits.command_timeout_seconds,
                      f.config.workflow.reproduction_timeout_seconds if script else
                      f.config.workflow.public_test_timeout_seconds)
        sb = create_sandbox(repo, f.events, f.task.protected_paths, timeout, f.baseline, f.task.runtime)
        try:
            result = sb.run(command, timeout=min(timeout, f.budget.remaining_seconds()),
                            log_path=folder / 'test.log')
        finally:
            sb.cancel()
        return {**result, **common, 'cases': junit_statuses(repo / '.sca_public_results.xml') if structured else None}

    def baseline(self):
        f = self.flow
        command, structured = self.public_command()
        prior = f.state.get('public_baseline')
        if prior is None:
            prior = self.run('baseline', command, structured=structured)
            f.save(public_baseline=prior)
        elif prior['command'] != command:
            raise ValueError('Public command changed after baseline validation')
        return prior

    def register_reproduction(self, script_path, expected_failure):
        f = self.flow
        if len(f.state.get('reproduction_attempts', [])) >= f.config.workflow.max_reproductions:
            raise ValueError('Reproduction registration limit reached; use the recorded evidence')
        rel = Path(script_path)
        if rel.is_absolute() or '..' in rel.parts or rel.suffix != '.py':
            raise ValueError('Use a repository-relative standalone Python script')
        path = f.workspace / rel
        if path.is_symlink() or not path.resolve().is_relative_to(f.workspace.resolve()):
            raise ValueError('Reproduction script escapes the workspace')
        if path.stat().st_size > 64_000:
            raise ValueError('Reproduction script must be at most 64 KB')
        if not expected_failure.strip() or len(expected_failure) > 500:
            raise ValueError('Provide the assertion text expected for the reported issue')
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        name = f'.sca_reproduction_{sha[:16]}.py'
        frozen = f.directory / 'reproductions' / sha / 'script.py'
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(content)
        python = (f.task.runtime.python if f.task.runtime.backend in {'oci','docker'} else
                  str(Path(f.task.runtime.prefix).resolve() / 'bin/python') if f.task.runtime.prefix else '/usr/bin/python3')
        command = shlex.join([python, name])
        result = self.run('reproduction_baseline', command, script=(name, content))
        assertions = re.findall(r'^AssertionError(?::[^\n]*)?$', result['output'], re.MULTILINE)
        reproduced = (result['returncode'] == 1 and not result['timed_out']
                      and any(expected_failure in line for line in assertions))
        record = {'script_path': script_path, 'frozen_script': str(frozen), 'sha256': sha,
                  'sandbox_name': name, 'command': command, 'expected_failure': expected_failure,
                  'baseline': result, 'baseline_reproduced': reproduced,
                  'status': 'baseline_reproduced' if reproduced else 'unverified'}
        attempts = [*f.state.get('reproduction_attempts', []), record]
        f.save(reproduction=record, reproduction_attempts=attempts)
        return {'baseline_reproduced': reproduced, 'sha256': sha, 'output': result['output'],
                'next_action': 'Repair source; the controller will rerun this frozen script after edits.' if reproduced else
                'Fix the reproducer: the original code must fail with the expected assertion, not an import/runtime error.'}

    def reproduction(self, patch, candidate):
        f = self.flow
        recipe = f.state.get('reproduction')
        if recipe is None:
            # A user-supplied targeted command is already a fixed visible test.
            # A benchmark-wide pytest suite is never treated as an issue reproducer.
            baseline = self.baseline()
            if not f.task.benchmark and not baseline['structured']:
                reproduced = (baseline['returncode'] == 1 and not baseline['timed_out']
                              and 'AssertionError' in baseline['output'])
                verified = reproduced and candidate['returncode'] == 0 and not candidate['timed_out']
                return {'status': 'verified' if verified else 'unverified',
                        'kind': 'user_targeted_command', 'baseline_reproduced': reproduced}
            return {'status': 'missing', 'baseline_reproduced': False}
        if not recipe['baseline_reproduced']:
            return {'status': 'unverified', 'baseline_reproduced': False}
        content = Path(recipe['frozen_script']).read_bytes()
        if hashlib.sha256(content).hexdigest() != recipe['sha256']:
            raise ValueError('Frozen reproduction script changed')
        result = self.run('reproduction_candidate', recipe['command'], patch=patch,
                          script=(recipe['sandbox_name'], content))
        verified = result['returncode'] == 0 and not result['timed_out'] and result['patch_replay']['applied']
        return {'status': 'verified' if verified else 'failed', 'sha256': recipe['sha256'],
                'baseline_reproduced': True, 'candidate': result}

    def check(self, label):
        f = self.flow
        baseline = self.baseline()
        patch = f.directory / 'current.patch'
        exported = export_patch(f.baseline, f.workspace, patch, f.task.protected_paths, **f.tree)
        sha = hashlib.sha256(patch.read_bytes()).hexdigest()
        recipe = f.state.get('reproduction', {})
        recipe_sha = recipe.get('sha256')
        recipe_key = [recipe_sha, recipe.get('expected_failure'), recipe.get('baseline_reproduced')]
        for old in reversed(f.state['checks']):
            if old.get('patch_sha256') == sha and old.get('reproduction_key') == recipe_key:
                return old
        candidate = self.run('candidate', baseline['command'], patch=patch, structured=baseline['structured'])
        regression = compare_public(baseline, candidate)
        reproduction = (self.reproduction(patch, candidate) if candidate['patch_replay']['applied'] else
                        {'status': 'inconclusive', 'reason': 'patch_replay_failed'})
        reproducer_paths = {r['script_path'] for r in f.state.get('reproduction_attempts', [])}
        source_changes = [p for p in exported['changed_files'] if p not in reproducer_paths]
        verified = reproduction['status'] == 'verified'
        item = {**candidate, **exported, 'label': label, 'patch_sha256': sha,
                'reproduction_sha256': recipe_sha, 'reproduction_key': recipe_key,
                'regression': regression, 'reproduction': reproduction,
                'non_reproduction_changed_files': source_changes,
                'passed': bool(source_changes) and verified and regression['status'] == 'passed',
                'retry_needed': not source_changes or not verified or regression['status'] == 'failed'}
        f.state['checks'].append(item)
        f.save()
        return item
