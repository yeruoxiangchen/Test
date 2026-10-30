import errno
import tempfile
import unittest
from pathlib import Path

import requests

from video_count_probe.cli import stage_log
from video_count_probe.core import read_json
from video_count_probe.downloads import is_transient, retry_hub_call


class DownloadRecoveryTests(unittest.TestCase):
    def test_interrupted_stream_retries_without_deleting_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            partial = root / "weight.incomplete"
            attempts = []

            def transfer():
                attempts.append(partial.stat().st_size if partial.exists() else 0)
                with open(partial, "ab") as f:
                    f.write(b"data")
                if len(attempts) == 1:
                    raise requests.exceptions.ChunkedEncodingError("stream ended early")
                return partial

            result = retry_hub_call(transfer, "weight", root / "status.json", sleep=lambda _: None)
            self.assertEqual(attempts, [0, 4])
            self.assertEqual(result.read_bytes(), b"datadata")

    def test_permanent_http_and_disk_errors_do_not_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            status = Path(temp) / "status.json"
            response = requests.Response()
            response.status_code = 401
            errors = [requests.HTTPError("unauthorized", response=response), OSError(errno.ENOSPC, "no space"),
                      PermissionError(errno.EACCES, "denied")]
            for error in errors:
                calls = []

                def operation():
                    calls.append(1)
                    raise error

                with self.assertRaises(type(error)):
                    retry_hub_call(operation, "weight", status, sleep=lambda _: None)
                self.assertEqual(len(calls), 1)
                self.assertEqual(read_json(status)["state"], "failed")

    def test_retry_limit_records_actual_error(self):
        with tempfile.TemporaryDirectory() as temp:
            status = Path(temp) / "status.json"
            calls = []

            def operation():
                calls.append(1)
                raise requests.ReadTimeout("temporary timeout")

            with self.assertRaises(requests.ReadTimeout):
                retry_hub_call(operation, "shard", status, attempts=2, sleep=lambda _: None)
            self.assertEqual(len(calls), 2)
            self.assertEqual(read_json(status)["error_type"], "ReadTimeout")
            self.assertEqual(read_json(status)["state"], "failed")

    def test_expired_redirect_and_short_response_are_retriable(self):
        response = requests.Response()
        response.status_code = 403
        self.assertTrue(is_transient(requests.HTTPError("credentials expired", response=response)))
        self.assertFalse(is_transient(requests.HTTPError("forbidden", response=response)))
        self.assertTrue(is_transient(OSError("Consistency check failed: file should be of size 100 but has size 50")))
        self.assertFalse(is_transient(OSError("unrelated local failure")))

    def test_download_stage_records_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "test transfer failure"):
                with stage_log(temp, "download") as path:
                    print("download started")
                    raise RuntimeError("test transfer failure")
            saved = path.read_text()
            self.assertIn("download started", saved)
            self.assertIn("Traceback", saved)
            self.assertIn("test transfer failure", saved)


if __name__ == "__main__":
    unittest.main()
