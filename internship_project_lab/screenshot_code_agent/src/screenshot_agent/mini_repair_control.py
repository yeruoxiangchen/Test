"""Persist public failure feedback and bound unchanged retries."""
import hashlib
import shlex

from .control import NoProgress


class RepairControl:
    def __init__(self, delivery):
        self.delivery = delivery
        self.failed = ''
        self.epoch = 0
        self.stalls = 0
        self.timed_out = set()

    @staticmethod
    def key(command):
        return hashlib.sha256(command.encode()).hexdigest()

    def message(self):
        if not self.failed:
            return None
        return {'role': 'user', 'content': 'Controller: task NOT verified. Remaining public failure:\n'
                + self.failed[-3000:] + '\nInspect and correct the implementation using edit_file. '
                'A repeated check, echo or completion summary cannot resolve this failure. '
                'Diagnose deterministic timeouts as code behavior; check loop termination. '
                'Use submit after checks pass; otherwise continue repairing.'}

    def before(self, action):
        if self.stalls >= 6:
            raise NoProgress('Six unchanged check/echo retries after public failure; candidate retained')
        command = action.get('command', '')
        if isinstance(command, str) and (self.epoch, self.key(command)) in self.timed_out:
            self.stalls += 1
            return {'returncode': 2, 'output': 'This exact command already timed out. Inspect the '
                    'implementation and use edit_file before repeating it.',
                    'timed_out': False, 'exception_info': '', 'retry_blocked': True}
        return None

    def after(self, action, output):
        tool = action.get('tool', 'bash')
        if tool == 'edit_file' and output['returncode'] == 0 and output['output'].strip():
            self.epoch += 1
            self.stalls = 0
        if output.get('timed_out'):
            command = action.get('command')
            if isinstance(command, str):
                self.timed_out.add((self.epoch, self.key(command)))
            self.failed = 'Command timed out; inspect loop termination and API use.\n' + output['output'][-1000:]
        response = self.delivery.last_response
        if response and tool in ('check_patch', 'submit'):
            self.failed = '' if response['passed'] else response['output']
            if response['passed']:
                self.stalls = 0
        if self.failed:
            try:
                tokens = shlex.split(action.get('command', ''))
            except (ValueError, TypeError):
                tokens = []
            echo_only = bool(tokens and (tokens[0] == 'echo' or
                             (tokens[0] == 'cd' and len(tokens) > 3 and tokens[2:4] == ['&&', 'echo'])))
            if tool in ('check_patch', 'submit') or echo_only:
                self.stalls += 1
        self.delivery.sandbox.events.emit('repair_control', stalls=self.stalls, epoch=self.epoch,
                                          failed=bool(self.failed), retry_blocked=output.get('retry_blocked', False))
