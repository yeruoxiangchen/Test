import argparse
import json
import os
import sys
from pathlib import Path
from ..config import Settings
from .sources import DATASET, prepare, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Isolated Knowledge Desk benchmark; no GPU deployment")
    parser.add_argument("action", choices=["sources", "import", "author", "build", "validate", "freeze", "review", "dev-review", "run", "status", "compare", "judge", "score-grades"])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", default="retrieval-bm25")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument('--evaluation-profile',choices=['natural','withheld_first'],default='natural')
    parser.add_argument("--ack-frozen-test", action="store_true")
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--grades", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    root = args.dataset or Settings().root / "datasets" / DATASET
    if args.action == "dev-review":
        from .review import export_review
        if not args.output:
            parser.error("--output required")
        print(json.dumps(export_review(root, args.output, args.runs), ensure_ascii=False, indent=2))
    elif args.action == "sources":
        docs = prepare(root)
        print(json.dumps({"dataset": str(root), "documents": len(docs)}, ensure_ascii=False))
    elif args.action == "author":
        from .dataset import author
        print(json.dumps(author(root, json.load(sys.stdin)), ensure_ascii=False, indent=2))
    elif args.action == "import":
        if (root / "freeze.json").exists():
            raise SystemExit("Dataset frozen; cannot overwrite catalog")
        rows = json.load(sys.stdin)
        write_jsonl(root / "catalog.jsonl", rows)
        print(json.dumps({"imported_families": len(rows)}))
    elif args.action in {"build", "validate", "freeze", "review"}:
        from .dataset import build, validate, freeze, export_review
        result = {"build": build, "validate": validate, "freeze": freeze, "review": export_review}[args.action](root)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    elif args.action == "status":
        if not args.output:
            parser.error("--output required")
        print((args.output / "status.json").read_text())
    elif args.action == "run":
        from .runner import run
        if not args.output:
            parser.error("--output required for resumable runs")
        if args.split == "test" and not args.ack_frozen_test:
            parser.error("Frozen test requires --ack-frozen-test; tune on dev only")
        result = run(root, Settings(), args.output, args.arm, args.split, args.limit,args.evaluation_profile)
        print(json.dumps({"complete": result["complete"], "report": str(args.output / "report.md")}, ensure_ascii=False))
    elif args.action == "compare":
        from .runner import compare
        if not all((args.left, args.right, args.output)):
            parser.error("--left, --right and --output required")
        print(json.dumps(compare(args.left, args.right, args.output), ensure_ascii=False, indent=2))
    elif args.action in {"judge", "score-grades"}:
        from .grading import judge, apply_grades
        if not args.run_dir or not args.output:
            parser.error("--run-dir and --output required")
        if args.action == "judge":
            result = judge(args.run_dir, args.output, os.getenv("KB_JUDGE_BASE_URL", ""), os.getenv("KB_JUDGE_MODEL", ""), os.getenv("KB_JUDGE_API_KEY", ""), args.review_dir)
        else:
            if not args.grades:
                parser.error("--grades required")
            result = apply_grades(args.run_dir, args.grades, args.output)
        print(json.dumps({"complete": result["complete"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
