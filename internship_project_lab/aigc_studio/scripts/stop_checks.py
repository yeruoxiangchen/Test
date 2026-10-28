"""Stop only this project's CPU diagnostic and isolated UI preview servers."""
import os
from pathlib import Path
import signal

root = Path(os.environ['AIGC_DATA_ROOT'])
for process in Path('/proc').iterdir():
    if not process.name.isdigit():
        continue
    try:
        args = (process / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        if not args or not args[0].startswith(str(root) + '/'):
            continue
        preview = 'tests.preview:factory' in args
        cpu_check = str(root / 'vendor' / 'ComfyUI' / 'main.py') in args and '--cpu' in args and '18801' in args
        if preview or cpu_check:
            os.kill(int(process.name), signal.SIGTERM)
            print('Stopped diagnostic process', process.name)
    except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeError):
        continue
