"""One-shot progress snapshot for the bounded diagnostic suite."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]/'outputs/round2_v2_diagnostics_20261018'


def main():
    status={'time_utc':datetime.now(timezone.utc).isoformat(),'training':{},'evaluation':{}}
    for name in ('raw_joint','raw_frozen','raw_frozen_v1recipe','raw_frozen_v1recipe_tol004'):
        path=ROOT/name/'history.jsonl'
        rows=path.read_text().splitlines() if path.exists() else []
        last=json.loads(rows[-1]) if rows else {}
        status['training'][name]={'step':last.get('step',0),'expected_steps':500,
            'elapsed_seconds':last.get('elapsed_seconds'), 'complete':(ROOT/name/'complete.json').exists()}
    for label,modes,expected in [('dev64',('parent','v1','raw','bias_only','layer_view_roll'),128),
        ('strong16',('parent','v1','raw'),32),
        ('pilot32',('baseline500','raw_joint','raw_frozen','raw_frozen_v1recipe','raw_frozen_v1recipe_tol004'),128)]:
        status['evaluation'][label]={m:{'records':len(list((ROOT/label/m/'records').glob('*.json'))),
                                        'expected':expected} for m in modes}
    status['evaluation']['mesh']={'records':len(list((ROOT/'mesh/records').glob('*.json'))),'expected':384}
    (ROOT/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))


if __name__=='__main__':main()
