import hashlib
from .config import OUT, STORE, PARAMS, registry, read, write, now
from .report import build


def main():
    summary=build()
    if not summary['complete']:raise RuntimeError('cohort incomplete')
    if not read(OUT/'figures/manifest.json')['complete']:raise RuntimeError('figures incomplete')
    checks=read(OUT/'validation/artifact_checks.json')
    if not checks['passed'] or len(checks['cases'])!=32:raise RuntimeError('artifact checks incomplete')
    audit=STORE/'审计.md';before=audit.read_bytes();marker='# 10.1 第十轮自然错误纠正：完整结果'
    if marker.encode() not in before:
        addition='\n\n'+marker+'\n\n'+(OUT/'REPORT.md').read_text()
        with audit.open('ab') as stream:stream.write(addition.encode())
    if not audit.read_bytes().startswith(before):raise RuntimeError('previous audit content changed')
    write(OUT/'closeout.json',dict(time=now(),complete=True,cases=32,objects=16,
          audit_prior_preserved=True,audit_prior_sha256=hashlib.sha256(before).hexdigest(),
          report=str(OUT/'REPORT.md'),external_baselines_evaluated=False,real_photos_evaluated=False))
    print(dict(complete=True,cases=32,report=str(OUT/'REPORT.md')),flush=True)


if __name__=='__main__':main()
