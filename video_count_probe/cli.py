from __future__ import annotations

import argparse
import contextlib
import datetime
import faulthandler
import json
import os
import signal
import sys
import traceback
from pathlib import Path

from .core import DEFAULT_MODEL, DEFAULT_ROOT, prepare


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return False

    def fileno(self):
        # Native diagnostics (faulthandler) bypass write(), so keep their output
        # in the persistent log even when the console stream has no descriptor.
        return self.log.fileno()


@contextlib.contextmanager
def stage_log(root, command):
    log_dir = Path(root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = log_dir / f"{command}_{stamp}_{os.getpid()}.log"
    print(f"Log: {path}", flush=True)
    with open(path, "a", buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        try:
            yield path
        except BaseException:
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description="Counted-action video baseline probe; long GPU runs are explicitly launched by the user.")
    parser.add_argument("command", choices=["prepare", "download", "check", "smoke", "pilot", "review", "report"])
    parser.add_argument("--root", default=os.environ.get("VIDEO_COUNT_ROOT", DEFAULT_ROOT))
    parser.add_argument("--model-dir", default=os.environ.get("VIDEO_COUNT_MODEL_DIR", DEFAULT_MODEL))
    parser.add_argument("--mode", choices=["smoke", "pilot"], default="pilot", help="Used by review/report")
    parser.add_argument("--allow-busy", action="store_true", help="Explicit GPU sharing override; minimum free memory still enforced")
    parser.add_argument("--limit", type=int, help="Generate at most this many remaining jobs (does not change the frozen plan)")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--annotations", type=Path, nargs="*", default=[])
    parser.add_argument("--without-model", action="store_true", help="Check Python imports without requiring downloaded weights")
    parser.add_argument("--tokenizer-only", action="store_true", help="download: stage small tokenizer/config files and audit all prompts; no weights")
    parser.add_argument("--download-attempts", type=int, default=5, help="Attempts per file for transient transfer failures")
    args = parser.parse_args()
    faulthandler.enable()

    def on_signal(signum, _frame):
        print(json.dumps({"event": "termination_signal", "signal": signum, "pid": os.getpid()}), flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if args.command == "prepare":
        for mode in ("smoke", "pilot"):
            plan = prepare(args.root, mode)
            print(f"{mode}: {len(plan['cases'])} cases, {len(plan['jobs'])} videos; {Path(args.root) / mode / 'plan.json'}")
        return
    if args.command in ("report", "review"):
        from .evaluation import make_review, report
        if args.command == "report":
            report(Path(args.root) / args.mode, args.annotations)
        else:
            make_review(Path(args.root) / args.mode)
        return
    if args.command == "download":
        # Pin interrupted downloads too. Metadata is small; no implicit weights in check/run.
        from .runtime import download_model
        with stage_log(args.root, "download"):
            download_model(args.model_dir, tokenizer_only=args.tokenizer_only, attempts=args.download_attempts)
        return
    if args.command == "check":
        from .runtime import environment
        with stage_log(args.root, "check"):
            data = environment(args.model_dir, require_model=not args.without_model)
            print(json.dumps(data, indent=2))
        return
    prepare(args.root, args.command)
    with stage_log(args.root, args.command):
        from .runtime import run_generation
        run_generation(args.root, args.command, args.model_dir, args.allow_busy, args.limit, args.retry_failures)
        from .evaluation import make_review, report
        report(Path(args.root) / args.command)
        make_review(Path(args.root) / args.command)
        print("Generation finished. Human annotations are still required; no scientific conclusion is implied.", flush=True)


if __name__ == "__main__":
    main()

