#!/usr/bin/env bash
set -euo pipefail
SCA_FULL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_FULL_ROOT"
mkdir -p .cache vendor
if [[ ! -d vendor/CodeOCR/.git ]]; then
  git clone --no-checkout https://github.com/YerbaPage/CodeOCR vendor/CodeOCR
  git -C vendor/CodeOCR checkout --detach 411fc815f97ed13ecafa246b6042bb17dbc40748
fi
.venv/bin/python - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
pins=json.loads(Path('CODEOCR_FULL_UPSTREAM.json').read_text())
commit=subprocess.check_output(['git','-C','vendor/CodeOCR','rev-parse','HEAD'],text=True).strip()
if commit!=pins['codeocr_commit']:
    raise SystemExit('CodeOCR checkout is not the pinned version')
for key in ['main_font','fallback_font']:
    item=pins[key];path=Path(item['path'])
    if not path.exists():
        subprocess.run(['curl','-fL','--retry','3',item['url'],'-o',str(path)],check=True)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise SystemExit(f'Font checksum mismatch: {path}')
data=Path('vendor/CodeOCR/reconstruction/dataset/dataset.json')
if hashlib.sha256(data.read_bytes()).hexdigest()!=pins['reconstruction_dataset_sha256']:
    raise SystemExit('Dataset checksum mismatch')
PY
if [[ ! -f datasets/generated/codeocr_full_v4/build_report.json ]]; then
  .venv/bin/python scripts/prepare_official.py codeocr --full --output datasets/generated/codeocr_full_v4
fi
