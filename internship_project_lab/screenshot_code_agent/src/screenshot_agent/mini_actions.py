"""Recoverable tool validation and exact, syntax-checked edits for mini."""
from __future__ import annotations

import json

EXTRA_TOOLS = [
    {'type': 'function', 'function': {'name': 'submit',
     'description': 'Submit the candidate after public checks. Failure returns actionable feedback.',
     'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'edit_file',
     'description': 'Replace exactly one source snippet. Reject ambiguous matches or invalid Python without writing.',
     'parameters': {'type': 'object', 'properties': {key: {'type': 'string'}
                    for key in ('file_path', 'old_string', 'new_string')},
                    'required': ['file_path', 'old_string', 'new_string']}}},
    {'type': 'function', 'function': {'name': 'check_patch',
     'description': 'Verify the current patch in a clean replay using public behavior and regression tests; save it if passing.',
     'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
]

# Executed with the task's Python, including historical Python 3.6, never host code.
EDIT_SOURCE = r'''
import ast, difflib, json, pathlib, sys
try:
    data = json.load(sys.stdin)
    root = pathlib.Path.cwd().resolve()
    p = root / data['file_path']
    p.resolve().relative_to(root)
    if p.is_symlink() or '.git' in p.relative_to(root).parts:
        raise ValueError('Editing symbolic links or Git metadata is not supported')
    old = p.read_text()
    before, after = data['old_string'], data['new_string']
    if not before or old.count(before) != 1:
        raise ValueError('old_string must match exactly once; read current source and include unique context')
    new = old.replace(before, after, 1)
    if len(new.encode()) > 1000000:
        raise ValueError('Edited file exceeds 1 MB')
    if p.suffix == '.py':
        ast.parse(new, filename=data['file_path'])
    p.write_text(new)
    print(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
          fromfile='a/' + data['file_path'], tofile='b/' + data['file_path'])))
except (ValueError, SyntaxError, OSError, KeyError, TypeError) as error:
    print('Edit rejected: ' + type(error).__name__ + ': ' + str(error))
    sys.exit(2)
'''


def parse_actions(calls):
    from minisweagent.exceptions import FormatError
    actions = []
    for call in calls:
        try:
            args = json.loads(call.function.arguments)
            if not isinstance(args, dict) or call.function.name not in {'bash', 'edit_file', 'check_patch', 'submit'}:
                raise ValueError('Expected bash, edit_file or check_patch with an argument object')
        except (ValueError, TypeError) as error:
            raise FormatError({'role': 'user', 'content': 'Invalid tool call: ' + str(error),
                               'extra': {'interrupt_type': 'FormatError'}}) from error
        actions.append({**args, 'tool': call.function.name, 'tool_call_id': call.id})
    return actions


def failure(message):
    return {'returncode': 2, 'output': message, 'timed_out': False,
            'exception_info': 'invalid tool input', 'input_error': True}


def execute_action(action, sandbox, budget, delivery):
    tool = action.get('tool', 'bash')
    if tool == 'submit':
        return {'returncode': 0, 'output': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n',
                'timed_out': False, 'exception_info': ''}
    if tool == 'check_patch':
        if delivery is None:
            return failure('Public verification is unavailable in this mode.')
        checked = delivery()
        return {'returncode': 0 if checked['passed'] else 1, 'output': checked['output'],
                'timed_out': False, 'exception_info': ''}
    if tool == 'edit_file':
        args = {key: action.get(key) for key in ('file_path', 'old_string', 'new_string')}
        if not all(isinstance(v, str) and '\0' not in v for v in args.values()):
            return failure('edit_file arguments must be strings without NUL bytes; use escaped source text.')
        return sandbox.run(['python3', '-c', EDIT_SOURCE], stdin=json.dumps(args),
                           timeout=min(30, budget.remaining_seconds()))
    command = action.get('command')
    if tool != 'bash' or not isinstance(command, str) or '\0' in command:
        return failure('bash command must be a string without actual NUL bytes. '
                       'Use escaped Python text such as chr(0) or bytes([0]); correct the command and retry.')
    # Do not swallow execution infrastructure errors; only invalid arguments are recoverable.
    try:
        limit = 30 if budget.config.mini_repair_control and 'pytest' not in command else sandbox.timeout
        return sandbox.run(command, timeout=min(limit, sandbox.timeout, budget.remaining_seconds()))
    except ValueError as error:
        if 'embedded null' not in str(error).lower():
            raise
        return failure('Command rejected before execution: ' + str(error))
