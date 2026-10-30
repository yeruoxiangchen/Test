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
from .workflow_cases import development_case
from .workflow_support import PUBLIC_PLUGIN, classify_changes, public_test_files, select_public_tests


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
        # Runtime timestamps/object addresses can appear in pytest parameter IDs.
        # Do not strip all parameters or silently call these cases passing.
        drift = []
        for key in lost:
            if key in new or '[' not in key or not re.search(r'\d{2}:\d{2}:\d{2}|0x[0-9a-fA-F]+', key):
                continue
            prefix = key.split('[', 1)[0] + '['
            missing = [k for k in lost if k.startswith(prefix) and k not in new]
            replacements = [k for k, v in new.items() if k.startswith(prefix) and k not in old and v == 'passed']
            if len(missing) == len(replacements) and replacements:
                drift.append(key)
        definite_lost = [k for k in lost if k not in drift]
        if definite_lost or additional:
            return {'status': 'failed', 'lost_passing_tests': definite_lost, 'additional_failures': additional,
                    'unstable_case_ids': drift}
        if baseline['returncode'] not in (0, 1) or candidate['returncode'] not in (0, 1):
            return {'status': 'inconclusive', 'reason': 'test_collection_or_runner_error'}
        if drift:
            return {'status': 'inconclusive', 'reason': 'unstable_parameter_ids', 'unstable_case_ids': drift}
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
            if f.config.workflow.execution_mode == 'linear':
                roots = f.state.get('public_test_selection')
                if roots is None:
                    case = development_case(f.task.task_id) if f.config.workflow.development_casebook else None
                    roots = select_public_tests(f.baseline, f.task.problem, f.config.workflow.public_test_max_cases,
                                                allowed_files=case.get('public_modules') if case else None)
                    f.save(public_test_selection=roots, public_test_scope='issue-ranked public regression sample')
                # Conditional methods may not be collected by the runtime's Python/pytest.
                # Let pytest collect modules normally, then filter actual public node IDs.
                roots = list(dict.fromkeys(n.split('::', 1)[0] for n in roots))
            else:
                roots = [name for name in ('tests', 'test', 'testing') if (f.baseline / name).is_dir()]
                if not roots:
                    roots = public_test_files(f.baseline)
            if not roots:
                return None, True
            parts += roots + ['-p', 'no:cacheprovider', '--junitxml=.sca_public_results.xml']
            if f.config.workflow.execution_mode == 'linear':
                parts += ['-o', 'addopts=', '--tb=short', '-q', '-p', 'sca_public_plugin']
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
        if structured and f.config.workflow.execution_mode == 'linear':
            plugin = repo / 'sca_public_plugin.py'
            if plugin.exists() or plugin.is_symlink():
                raise ValueError('Reserved public sampling plugin path exists in repository')
            plugin.write_text(PUBLIC_PLUGIN.replace('SELECTED_NODEIDS = None',
                              'SELECTED_NODEIDS = ' + repr(f.state['public_test_selection'])))
        if script:
            name, content = script
            if (repo / name).exists() or (repo / name).is_symlink():
                raise ValueError('Reserved reproduction path exists in repository')
            (repo / name).write_bytes(content)
        timeout = min(f.config.limits.command_timeout_seconds,
                      f.config.workflow.reproduction_timeout_seconds if script else
                      f.config.workflow.public_test_timeout_seconds)
        if label.startswith('quick_'):
            timeout = min(timeout, 30)
        sb = create_sandbox(repo, f.events, f.task.protected_paths, timeout, f.baseline, f.task.runtime)
        sb.public_display = f.public_display
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

    def known_issue(self, patch=None):
        f = self.flow
        case = development_case(f.task.task_id) if f.config.workflow.development_casebook else None
        if case is None:
            return {'status': 'not_configured'}
        content = case['script'].encode()
        sha = hashlib.sha256(content).hexdigest()
        python = f.task.runtime.python if f.task.runtime.backend in {'oci', 'docker'} else 'python3'
        filename = '.sca_development_acceptance.py'
        command = shlex.join([python, filename])
        baseline = f.state.get('development_acceptance_baseline')
        if baseline is None:
            baseline = self.run('known_issue_baseline', command, script=(filename, content))
            baseline['contract_sha256'] = sha
            f.save(development_acceptance_baseline=baseline, development_case=case)
        if baseline['contract_sha256'] != sha:
            raise ValueError('Development acceptance changed during task')
        reproduced = (baseline['returncode'] == 1 and not baseline['timed_out'] and
                      'AssertionError: development acceptance:' in baseline['output'] and
                      'ImportError' not in baseline['output'] and 'ModuleNotFoundError' not in baseline['output'])
        if patch is None:
            return {'status': 'baseline_reproduced' if reproduced else 'inconclusive',
                    'baseline_reproduced': reproduced, 'output': baseline['output'], 'contract_sha256': sha}
        candidate = self.run('known_issue_candidate', command, patch=patch, script=(filename, content))
        status = ('inconclusive' if not reproduced or candidate['timed_out'] or
                  not candidate['patch_replay']['applied'] else
                  'passed' if candidate['returncode'] == 0 else 'failed')
        return {'status': status, 'baseline_reproduced': reproduced, 'contract_sha256': sha,
                'output': candidate['output'][-5000:], 'returncode': candidate['returncode'],
                'scope': case['scope']}

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
        active = f.state.get('reproduction', {})
        keep_active = not reproduced and active.get('baseline_reproduced', False)
        f.save(reproduction=active if keep_active else record, reproduction_attempts=attempts)
        return {'baseline_reproduced': reproduced, 'sha256': sha, 'output': result['output'],
                'previous_verified_reproduction_retained': keep_active,
                'next_action': 'Repair source; the controller will rerun the verified frozen script after edits.'
                if reproduced or keep_active else
                'Use a top-level assert with a message, e.g. assert actual == expected, "wrong result". '
                'Printing FAILED or defining an uncalled test function is not a failed assertion. '
                'You may also run targeted existing tests via bash; registration is optional in linear mode.'}

    def quick_check(self, patch, changed):
        """Compare syntax/import health on original and candidate source."""
        f = self.flow
        files = [p for p in changed if p.endswith('.py') and not p.startswith('.sca_scratch/')]
        packages = set()
        for name in files:
            parts = Path(name).parts
            start = 1 if parts[0] == 'src' else 0
            if len(parts) > start + 1 and (f.baseline / Path(*parts[:start+1]) / '__init__.py').is_file():
                if parts[start].isidentifier():
                    packages.add(parts[start])
        if not files:
            return {'status': 'not_applicable'}
        script = ('import os, importlib\n'
                  'for p in ' + repr(files) + ':\n'
                  ' if os.path.isfile(p):\n'
                  '  with open(p, "rb") as s: compile(s.read(), p, "exec")\n'
                  'for name in ' + repr(sorted(packages)) + ': importlib.import_module(name)\n'
                  'print("syntax/import check passed")\n')
        python = f.task.runtime.python if f.task.runtime.backend in {'oci', 'docker'} else 'python3'
        command = shlex.join([python, '-c', script])
        saved = f.state.get('quick_baselines', {})
        key = hashlib.sha256(command.encode()).hexdigest()
        if key not in saved:
            saved[key] = self.run('quick_baseline', command)
            f.save(quick_baselines=saved)
        before = saved[key]
        after = self.run('quick_candidate', command, patch=patch)
        status = ('inconclusive' if before['returncode'] or before['timed_out'] or after['timed_out'] else
                  'passed' if after['returncode'] == 0 else 'failed')
        return {'status': status, 'baseline_output': before['output'][-2000:],
                'candidate_output': after['output'][-3000:], 'command': command}

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
        changes = classify_changes(f.baseline, exported['changed_files'], reproducer_paths)
        source_changes = changes['implementation']
        verified = reproduction['status'] == 'verified'
        known = self.known_issue(patch)
        if known['status'] != 'not_configured':
            verified = known['status'] == 'passed'
        linear = f.config.workflow.execution_mode == 'linear'
        quick = self.quick_check(patch, source_changes) if linear and source_changes else {'status': 'not_run'}
        retry = not source_changes or not verified or regression['status'] == 'failed'
        if linear:
            # Lack of proof / unavailable environment is not evidence of a bad patch.
            retry = (not source_changes or regression['status'] == 'failed'
                     or reproduction['status'] == 'failed' or quick['status'] == 'failed')
            if f.config.workflow.require_issue_verification:
                retry = retry or not verified
            retry = retry or known['status'] == 'failed'
        item = {**candidate, **exported, 'label': label, 'patch_sha256': sha,
                'reproduction_sha256': recipe_sha, 'reproduction_key': recipe_key,
                'regression': regression, 'reproduction': reproduction, 'quick_check': quick,
                'known_issue': known, 'diagnostic_changed_files': changes['diagnostics'],
                'non_reproduction_changed_files': source_changes,
                'passed': bool(source_changes) and verified and regression['status'] == 'passed'
                          and quick['status'] not in {'failed', 'inconclusive'},
                'retry_needed': retry}
        f.state['checks'].append(item)
        f.save()
        return item
