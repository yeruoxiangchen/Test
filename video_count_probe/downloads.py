"""Retries around whole Hub file requests, preserving the Hub's resume files."""
from __future__ import annotations

import re
import time

from .core import atomic_json


def brief_error(exc):
    # Public CDN URLs may contain ephemeral signed query parameters.
    return re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", str(exc))


def is_transient(exc):
    import requests
    if isinstance(exc, (requests.Timeout, requests.ConnectionError,
                        requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError)):
        return True
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status in (408, 429, 500, 502, 503, 504) or (status == 403 and "expired" in str(exc).lower())
    # Hub raises OSError on a prematurely closed, short response. Disk/permission
    # errors must not be mistaken for a network error.
    return isinstance(exc, OSError) and exc.errno is None and "Consistency check failed" in str(exc)


def retry_hub_call(operation, label, status_path, attempts=5, sleep=time.sleep):
    if attempts < 1:
        raise ValueError("Download attempts must be positive")
    for attempt in range(1, attempts + 1):
        atomic_json(status_path, {"stage": "download", "item": label, "attempt": attempt, "state": "running"})
        try:
            return operation()
        except Exception as exc:
            retry = is_transient(exc) and attempt < attempts
            atomic_json(status_path, {"stage": "download", "item": label, "attempt": attempt,
                                     "state": "retrying" if retry else "failed", "error_type": type(exc).__name__,
                                     "message": brief_error(exc)})
            if not retry:
                raise
            delay = min(2**attempt, 10)
            print(f"{label}: {type(exc).__name__}; retry {attempt+1}/{attempts} in {delay}s, retaining partial files", flush=True)
            # Re-enter hf_hub_download, obtaining a fresh redirect and resuming
            # the existing .incomplete file instead of retrying an expired URL.
            from huggingface_hub.utils import reset_sessions
            reset_sessions()
            sleep(delay)

