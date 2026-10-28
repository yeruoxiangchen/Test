#!/usr/bin/env python3
"""Reproducible official-source preparation; kept separate from model inference."""

import argparse
import json
from pathlib import Path

from screenshot_agent.codeocr_benchmark import build_codeocr, build_full_codeocr
from screenshot_agent.oci import pull_image
from screenshot_agent.official_data import build_official, select_records


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    pull = sub.add_parser("pull")
    pull.add_argument("--image", required=True)
    pull.add_argument("--cache", type=Path, default=Path(".cache/oci"))
    pull.add_argument("--umoci", type=Path, default=Path(".cache/umoci"))
    pull.add_argument("--registry", default="https://registry-1.docker.io")
    select = sub.add_parser("select")
    select.add_argument("--parquet", type=Path, default=Path("datasets/swebench_lite_test.parquet"))
    select.add_argument("--output", type=Path, default=Path("datasets/generated/swebench_lite_v3"))
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, default=Path("datasets/generated/swebench_lite_v3"))
    build.add_argument("--tasks", type=Path, default=Path("vendor/swe-bench-tasks"))
    build.add_argument("--harness", type=Path, default=Path("vendor/SWE-bench"))
    build.add_argument("--prefix", type=Path, required=True)
    build.add_argument("--cache", type=Path, default=Path(".cache/source_archives"))
    build.add_argument("--timeout", type=int, default=120)
    codeocr = sub.add_parser("codeocr")
    codeocr.add_argument("--source", type=Path, default=Path("vendor/CodeOCR"))
    codeocr.add_argument("--output", type=Path, default=Path("datasets/generated/codeocr_v3"))
    codeocr.add_argument("--full",action="store_true")
    codeocr.add_argument("--font",type=Path,default=Path(".cache/NotoSansMonoCJKsc-Regular.otf"))
    codeocr.add_argument("--fallback-font",type=Path,default=Path(".cache/NotoEmoji-wght.ttf"))
    args = p.parse_args()
    if args.command == "pull":
        print(json.dumps(pull_image(args.image, args.cache, args.umoci, registry=args.registry), indent=2))
    elif args.command == "select":
        rows = select_records(args.parquet, args.output)
        print(json.dumps({"selected": [r["instance_id"] for r in rows]}))
    elif args.command == "build":
        report = build_official(args.output, args.tasks, args.harness, args.prefix, args.cache, args.timeout)
        print(json.dumps({"selected": report["original_selection"], "validated": report["validated"]}))
    elif args.command == "codeocr":
        result = build_full_codeocr(args.source,args.output,args.font,args.fallback_font) if args.full else build_codeocr(args.source,args.output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
