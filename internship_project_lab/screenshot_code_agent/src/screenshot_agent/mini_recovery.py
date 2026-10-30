"""Visible-contract feedback and conservative retention of partially valid repairs."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .control import BudgetExceeded, atomic_json
from .mini_delivery import MiniDelivery
from .mini_verified import VerifiedDelivery
from .workflow_feedback import compare_public


def behavior_cases(result):
    if not result or result.get('timed_out'):
        return {}
    for line in reversed(result.get('output', '').splitlines()):
        if line.startswith('__SCA_BEHAVIOR__'):
            try:
                return json.loads(line[len('__SCA_BEHAVIOR__'):])
            except (ValueError, TypeError):
                return {}
    return {}


class RecoveryDelivery(VerifiedDelivery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial = None
        self.baseline_behavior = set()

    def initialize(self):
        super().initialize()
        self.baseline_behavior = {k for k, v in behavior_cases(self.baseline_public.get('behavior')).items()
                                  if v['passed']}

    def __call__(self):
        probe = self.directory/'current_candidate.diff'
        self.export(probe)
        digest = hashlib.sha256(probe.read_bytes()).hexdigest()
        if digest == self.last_digest and self.last_response is not None:
            return dict(self.last_response)
        response = MiniDelivery.__call__(self)
        folder = self.directory/'delivery_checks'/str(self.attempts)
        report = json.loads((folder/'report.json').read_text())
        checked = report['syntax_import']
        # A failed behavior must not suppress public regression comparison or
        # partial retention. Syntax/import failure still makes replay unhelpful.
        if checked['returncode'] == 0 and not checked['timed_out']:
            public = self._public(folder/'public', folder/'candidate.diff')
            comparison = compare_public(self.baseline_public, public)
            behavior = public['behavior']
            cases = behavior_cases(behavior)
            passed = {k for k, v in cases.items() if v['passed']}
            public_ok = comparison['status'] == 'passed'
            behavior_ok = bool(behavior and behavior['returncode'] == 0 and not behavior['timed_out'])
            response['passed'] = response['passed'] and public_ok and (not self.case or behavior_ok)
            report.update(public_regression=public, comparison=comparison, behavior_cases=cases,
                          passed=response['passed'])
            candidate = {'path': str(folder/'candidate.diff'), 'sha256': digest,
                         'export': report['export'], 'attempt': self.attempts,
                         'passed_behavior': sorted(passed)}
            if response['passed']:
                self.best = candidate
                atomic_json(self.directory/'verified_candidate.json', candidate)
            elif (public_ok and passed > self.baseline_behavior and
                  (not self.partial or passed > set(self.partial['passed_behavior']))):
                # Require a strict superset of known passing behavior, never
                # trade one required contract for another or use hidden scores.
                self.partial = candidate
                atomic_json(self.directory/'partial_candidate.json', candidate)
            if response['passed']:
                response['output'] = 'Clean replay and public checks passed. Candidate saved; use submit.'
            else:
                failures = {k: v.get('error', 'failed') for k, v in cases.items() if not v['passed']}
                response['output'] = ('Public verification FAILED. Fix these contracts before submitting.\n'
                    + json.dumps({'behavior_failures': failures, 'public_regression': comparison}, ensure_ascii=False)
                    + '\n' + (behavior['output'][-4000:] if behavior and not cases else '')
                    + ('\n' + public['output'][-3000:] if not public_ok else '')
                    + '\nNew root reproduction files are excluded from delivery; focus on source behavior.')
        atomic_json(folder/'report.json', report)
        atomic_json(self.directory/'delivery_check.json', report)
        self.last_digest, self.last_response = digest, dict(response)
        return response

    def select_final(self, destination):
        latest = self.directory/'last_workspace_candidate.diff'
        exported = self.export(latest)
        digest = hashlib.sha256(latest.read_bytes()).hexdigest()
        if self.last_digest != digest:
            try:
                if self.budget.remaining_seconds() >= 100:
                    self()
            except BudgetExceeded:
                pass
        candidate = self.best or self.partial
        if candidate:
            source = Path(candidate['path'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != candidate['sha256']:
                raise ValueError('Saved controller candidate changed')
            shutil.copyfile(source, destination)
            selected = {'source': str(source), 'verified': bool(self.best), 'partial': not bool(self.best),
                        'used_earlier_candidate': candidate['sha256'] != digest,
                        'attempt': candidate['attempt'], 'sha256': candidate['sha256'],
                        'passed_behavior': candidate['passed_behavior'],
                        'reason': 'Public-only retained candidate; final hidden grading remains authoritative'}
            exported = candidate['export']
        else:
            shutil.copyfile(latest, destination)
            selected = {'source': str(latest), 'verified': False, 'partial': False,
                        'used_earlier_candidate': False, 'sha256': digest,
                        'reason': 'No publicly demonstrated improvement; preserve latest attempt'}
        atomic_json(self.directory/'candidate_selection.json', selected)
        return {**exported, 'candidate_selection': selected}
