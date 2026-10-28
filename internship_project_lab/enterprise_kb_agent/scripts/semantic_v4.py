import argparse
import json
from pathlib import Path
from kb.config import Settings
from kb.benchmark.calibration import run as calibrate
from kb.benchmark.semantic import evaluate_run

p=argparse.ArgumentParser()
p.add_argument('action',choices=['calibrate','grade'])
p.add_argument('--output',type=Path,required=True)
p.add_argument('--run-dir',type=Path)
p.add_argument('--review-dir',type=Path)
args=p.parse_args()
if args.action=='calibrate':result=calibrate(Settings().root,args.output)
else:
    if not args.run_dir:p.error('--run-dir required')
    result=evaluate_run(args.run_dir,args.output,Settings().root,args.review_dir)
print(json.dumps(result,ensure_ascii=False,indent=2))
