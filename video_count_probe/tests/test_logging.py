import contextlib
import faulthandler
import io
import sys
import tempfile
import time
import unittest

from video_count_probe.cli import stage_log


class StageLoggingTests(unittest.TestCase):
    def test_native_traceback_and_normal_output_are_saved(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as root:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with stage_log(root, "check") as path:
                    print("normal stdout")
                    print("normal stderr", file=sys.stderr)
                    faulthandler.dump_traceback(file=sys.stderr)
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
            saved = path.read_text()
            self.assertIn("normal stdout", saved)
            self.assertIn("normal stderr", saved)
            self.assertIn("test_native_traceback_and_normal_output_are_saved", saved)
        self.assertIn("normal stdout", stdout.getvalue())
        self.assertIn("normal stderr", stderr.getvalue())

    def test_delayed_traceback_works_with_redirected_stderr(self):
        with tempfile.TemporaryDirectory() as root:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with stage_log(root, "check") as path:
                    try:
                        faulthandler.dump_traceback_later(0.01, repeat=False)
                        deadline = time.monotonic() + 3
                        while "Timeout" not in path.read_text() and time.monotonic() < deadline:
                            time.sleep(0.01)
                    finally:
                        # The native timer must stop before stage_log closes its file.
                        faulthandler.cancel_dump_traceback_later()
                    print("check continued after diagnostic")
                saved = path.read_text()
                self.assertIn("Timeout", saved)
                self.assertIn("test_delayed_traceback_works_with_redirected_stderr", saved)
                self.assertIn("check continued after diagnostic", saved)


if __name__ == "__main__":
    unittest.main()
