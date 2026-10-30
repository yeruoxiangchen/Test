"""Public delivery checks for a mini session; never load reference grading data."""
from __future__ import annotations

import json
from pathlib import Path

from .control import atomic_json
from .repository import export_session_patch

# Runs in the task's Python version, including historical Python 3.6 images.
CHECK_SOURCE = r'''
import ast, importlib, json, pathlib, sys
data = json.load(sys.stdin)
errors, changed_ast = [], []
for name, before in data['python'].items():
    path = pathlib.Path(name)
    if not path.exists():
        changed_ast.append(name)
        continue
    try:
        after = ast.parse(path.read_text(), filename=name)
        if before is None or ast.dump(after) != ast.dump(ast.parse(before, filename=name)):
            changed_ast.append(name)
    except (SyntaxError, UnicodeError) as error:
        errors.append(name + ': ' + str(error))
if not errors:
    for module in data['imports']:
        try:
            importlib.import_module(module)
        except Exception as error:
            errors.append('import ' + module + ': ' + type(error).__name__ + ': ' + str(error))
value = {'syntax_import_errors': errors, 'python_ast_changed': changed_ast,
         'non_python_changed': data['non_python']}
print('__SCA_DELIVERY__' + json.dumps(value))
sys.exit(1 if errors or not (changed_ast or data['non_python']) else 0)
'''


class MiniDelivery:
    def __init__(self, baseline, workspace, directory, protected, tree, sandbox, budget, case=None):
        self.baseline, self.workspace, self.directory = baseline, workspace, Path(directory)
        self.protected, self.tree = protected, tree
        self.sandbox, self.budget, self.case = sandbox, budget, case
        self.attempts = 0

    def __call__(self):
        self.attempts += 1
        folder = self.directory / 'delivery_checks' / str(self.attempts)
        folder.mkdir(parents=True, exist_ok=False)
        exported = self.export(folder / 'candidate.diff')
        files = exported['changed_files']
        suspects = [name for name in files if not (self.baseline / name).exists() and
                    len(Path(name).parts) == 1 and
                    (Path(name).name.startswith(('test_', 'repro_', 'reproduce_', 'debug_', 'temp_')) or
                     Path(name).name in {'patch.txt', 'final_verification.py'})]
        inputs = {'python': {}, 'imports': [], 'non_python': [name for name in files if not name.endswith('.py')]}
        for name in files:
            if not name.endswith('.py'):
                continue
            original = self.baseline / name
            inputs['python'][name] = original.read_text() if original.exists() else None
            parts = Path(name).parts
            parts = parts[1:] if parts[0] == 'src' else parts
            prefix = 'src/' if name.startswith('src/') else ''
            if len(parts) > 1 and (self.workspace / prefix / parts[0] / '__init__.py').is_file():
                inputs['imports'].append(parts[0])
        inputs['imports'] = sorted(set(inputs['imports']))
        checked = self.sandbox.run(['python3', '-c', CHECK_SOURCE], stdin=json.dumps(inputs),
                                   timeout=min(30, self.budget.remaining_seconds()))
        case_result = None
        if (checked['returncode'] == 0 and not checked['timed_out'] and self.case
                and (not suspects or getattr(self.budget.config, 'mini_recovery', False))):
            case_result = self.sandbox.run(['python3', self.case['path']],
                                          timeout=min(10 if self.budget.config.mini_repair_control else 60, self.budget.remaining_seconds()))
        passed = bool(files and not suspects and checked['returncode'] == 0 and not checked['timed_out']
                      and (not self.case or (case_result is not None and case_result['returncode'] == 0
                                            and not case_result['timed_out'])))
        report = {'passed': passed, 'attempt': self.attempts, 'export': exported,
                  'suspected_debug_files': suspects, 'syntax_import': checked,
                  'visible_development_case': case_result, 'reference_tests_used': False}
        atomic_json(folder / 'report.json', report)
        atomic_json(self.directory / 'delivery_check.json', report)
        if passed:
            return {'passed': True, 'output': 'Public delivery checks passed; final grading remains separate.'}
        output = ('Submission was not accepted. Fix the observed problems, then submit again.\n'
                  'A valid delivery needs a nonempty implementation change, valid Python/imports, and '
                  'the supplied visible development assertion. Comments alone do not change Python behavior.\n')
        if suspects:
            output += 'Move reproduction/debug artifacts into .sca_scratch/: ' + ', '.join(suspects) + '\n'
        output += checked['output'][-4000:]
        if case_result and case_result['timed_out']:
            output += '\nVisible behavior timed out: inspect nontermination; timeout is not a pass.\n'
        if case_result:
            output += '\nVisible developer check:\n' + case_result['output'][-4000:]
        return {'passed': False, 'output': output}

    def export(self, path):
        return export_session_patch(self.baseline, self.workspace, path, self.protected,
                                    exclude_debug=getattr(self.budget.config, 'mini_recovery', False), **self.tree)
