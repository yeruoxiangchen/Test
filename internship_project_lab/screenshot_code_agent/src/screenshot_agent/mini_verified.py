"""Public regression checks on clean replays and controller-owned candidate retention."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .control import BudgetExceeded, atomic_json
from .mini_delivery import MiniDelivery
from .repository import replay_patch
from .runtime import create_sandbox
from .workflow_feedback import compare_public, junit_statuses
from .workflow_support import select_public_tests


class VerifiedDelivery(MiniDelivery):
    automatic = True

    def __init__(self, *args, runtime, problem='', **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime, self.problem = runtime, problem
        self.best = None
        self.last_digest = None
        self.last_response = None
        self.baseline_public = None
        self.modules = (self.case or {}).get('public_modules') or list(dict.fromkeys(
            name.split('::')[0] for name in select_public_tests(self.baseline, problem, max_cases=12)))

    def initialize(self):
        self.baseline_public = self._public(self.directory / 'public_baseline')
        atomic_json(self.directory / 'public_baseline.json', self.baseline_public)
        cases = self.baseline_public.get('cases') or {}
        if (self.baseline_public['timed_out'] or self.baseline_public['returncode'] not in (0, 1)
                or not any(v == 'passed' for v in cases.values())):
            raise ValueError('Public regression baseline is not comparable; inspect public_baseline.json')

    def _public(self, folder, patch=None):
        folder = Path(folder)
        folder.mkdir(parents=True)
        repo = folder / 'repo'
        if patch is None:
            shutil.copytree(self.baseline, repo, symlinks=True)
        elif not replay_patch(self.baseline, repo, patch, **self.tree)['applied']:
            raise ValueError('Candidate failed clean public-test replay')
        self.budget.tool('public_regression', {'modules': self.modules})
        sb = create_sandbox(repo, self.sandbox.events, self.protected, 90, self.baseline, self.runtime)
        sb.public_display = getattr(self.sandbox, 'public_display', None)
        # A clean copy contains original tests only; no benchmark metadata is consulted.
        report = repo / '.sca_scratch/public_result.xml'
        report.parent.mkdir(exist_ok=True)
        if report.exists() or report.is_symlink():
            raise ValueError('Reserved public-test report already exists')
        try:
            behavior = None
            if self.case and (patch is not None or getattr(self.budget.config, 'mini_recovery', False)):
                behavior = sb.run(['python3', self.case['path']],
                                  timeout=min(10 if getattr(self.budget.config, 'mini_recovery', False) else 45,
                                              self.budget.remaining_seconds()), log_path=folder/'behavior.log')
            if not self.modules:
                raise ValueError('No existing public test modules selected')
            command = ['python3', '-m', 'pytest', *self.modules, '-o', 'addopts=',
                       '-p', 'no:cacheprovider', '--tb=short', '-q',
                       '--junitxml=.sca_scratch/public_result.xml']
            result = sb.run(command, timeout=min(90, self.budget.remaining_seconds()), log_path=folder/'tests.log')
            return {**result, 'structured': True, 'cases': junit_statuses(report),
                    'modules': self.modules, 'behavior': behavior, 'reference_tests_used': False}
        finally:
            sb.cancel()

    def __call__(self):
        # Bind cached acceptance to exact exported bytes, not model claims or Git HEAD.
        probe = self.directory / 'current_candidate.diff'
        exported = self.export(probe)
        digest = hashlib.sha256(probe.read_bytes()).hexdigest()
        if digest == self.last_digest and self.last_response is not None:
            return dict(self.last_response)
        response = super().__call__()
        folder = self.directory / 'delivery_checks' / str(self.attempts)
        report = json.loads((folder/'report.json').read_text())
        if response['passed']:
            public = self._public(folder/'public', folder/'candidate.diff')
            comparison = compare_public(self.baseline_public, public)
            behavior = public['behavior']
            response['passed'] = comparison['status'] == 'passed' and (
                not self.case or (behavior is not None and behavior['returncode'] == 0 and not behavior['timed_out']))
            report.update(public_regression=public, comparison=comparison, passed=response['passed'])
            if not response['passed']:
                response['output'] = ('Public verification failed; fix these behaviors before submitting.\n'
                    + json.dumps(comparison, ensure_ascii=False) + '\n'
                    + (behavior['output'][-3000:] if behavior else '') + '\n' + public['output'][-5000:])
        if response['passed']:
            self.best = {'path': str(folder/'candidate.diff'), 'sha256': digest,
                         'export': exported, 'attempt': self.attempts}
            atomic_json(self.directory/'verified_candidate.json', self.best)
            response['output'] = ('Clean replay passed the supplied behavior checks and existing public '
                'regression comparison. Candidate saved. Review remaining issue requirements, then submit. '
                'Hidden grading has not run.')
        atomic_json(folder/'report.json', report)
        atomic_json(self.directory/'delivery_check.json', report)
        self.last_digest, self.last_response = digest, dict(response)
        return response

    def select_final(self, destination):
        latest = self.directory/'last_workspace_candidate.diff'
        exported = self.export(latest)
        digest = hashlib.sha256(latest.read_bytes()).hexdigest()
        # Final public checks use remaining task time only; hidden scoring is separate.
        if self.last_digest != digest:
            try:
                if self.budget.remaining_seconds() >= 100:
                    self()
            except BudgetExceeded:
                pass
        if self.best:
            source = Path(self.best['path'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != self.best['sha256']:
                raise ValueError('Saved controller candidate changed')
            shutil.copyfile(source, destination)
            selected = {'source': str(source), 'verified': True,
                        'used_earlier_candidate': self.best['sha256'] != digest,
                        'attempt': self.best['attempt'], 'sha256': self.best['sha256']}
            exported = self.best['export']
        else:
            shutil.copyfile(latest, destination)
            selected = {'source': str(latest), 'verified': False, 'used_earlier_candidate': False,
                        'reason': 'No candidate passed public checks; preserve failed attempt for grading',
                        'sha256': digest}
        atomic_json(self.directory/'candidate_selection.json', selected)
        return {**exported, 'candidate_selection': selected}
