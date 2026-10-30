"""An owned CPU-only X server for public rendering checks; never a shared display."""
from __future__ import annotations

import os
import selectors
import subprocess
from pathlib import Path


class PublicDisplay:
    def __init__(self, executable, directory):
        self.proc = None
        self.log = (Path(directory) / 'public_display.log').open('wb')
        read_fd, write_fd = os.pipe()
        try:
            self.proc = subprocess.Popen(
                [str(executable), '-displayfd', str(write_fd), '-screen', '0', '1024x768x24',
                 '-nolisten', 'tcp', '-ac'], pass_fds=(write_fd,), stdout=self.log, stderr=self.log,
                env={**os.environ, 'LIBGL_ALWAYS_SOFTWARE': '1'}, start_new_session=True)
            os.close(write_fd)
            write_fd = None
            with selectors.DefaultSelector() as selector:
                selector.register(read_fd, selectors.EVENT_READ)
                if not selector.select(timeout=15):
                    raise RuntimeError('Public Xvfb startup timed out; inspect public_display.log')
                number = os.read(read_fd, 128).decode().strip()
            if not number.isdigit() or self.proc.poll() is not None:
                raise RuntimeError('Public Xvfb failed; inspect public_display.log')
            self.name = ':' + number
            self.socket = Path('/tmp/.X11-unix') / ('X' + number)
            if not self.socket.is_socket():
                raise RuntimeError('Public Xvfb did not publish its Unix socket')
        except BaseException:
            self.close()
            raise
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
