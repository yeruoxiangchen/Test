#!/usr/bin/env bash
set -euo pipefail
SCA_BENCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_BENCH_ROOT"
mkdir -p .cache .runtime datasets vendor

sca_pin_clone() {
  local sca_name="$1" sca_url="$2" sca_commit="$3"
  if [[ ! -d "vendor/$sca_name/.git" ]]; then
    git clone --depth 1 --no-checkout "$sca_url" "vendor/$sca_name"
    git -C "vendor/$sca_name" fetch --depth 1 origin "$sca_commit"
    git -C "vendor/$sca_name" checkout --detach "$sca_commit"
  fi
  [[ "$(git -C "vendor/$sca_name" rev-parse HEAD)" == "$sca_commit" ]] || {
    printf 'Unexpected revision in vendor/%s; inspect it before replacing this cache.\n' "$sca_name" >&2
    exit 1
  }
}
sca_pin_clone SWE-bench https://github.com/SWE-bench/SWE-bench 02e7a74ffd0b707aab73d203fe87bdc7c76afc8e
sca_pin_clone swe-bench-tasks https://github.com/SWE-bench/swe-bench-tasks 3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d
sca_pin_clone CodeOCR https://github.com/YerbaPage/CodeOCR 411fc815f97ed13ecafa246b6042bb17dbc40748

if [[ ! -f datasets/swebench_lite_test.parquet ]]; then
  curl -fL --retry 3 --retry-all-errors \
    'https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite/resolve/b0dde1093fe417d83b7184254edf8199c1f0dff5/data/test-00000-of-00001.parquet' \
    -o datasets/swebench_lite_test.parquet
fi
if [[ ! -f .cache/python39.tar.gz ]]; then
  curl -fL --retry 3 --retry-all-errors \
    'https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.9.20%2B20241016-x86_64-unknown-linux-gnu-install_only.tar.gz' \
    -o .cache/python39.tar.gz
fi
if [[ ! -f .cache/NotoSansMonoCJKsc-Regular.otf ]]; then
  curl -fL --retry 3 --retry-all-errors \
    'https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/Mono/NotoSansMonoCJKsc-Regular.otf' \
    -o .cache/NotoSansMonoCJKsc-Regular.otf
fi
.venv/bin/python - <<'PY'
import hashlib, json, tarfile
from pathlib import Path
pins=json.loads(Path('BENCHMARK_UPSTREAM.json').read_text())
for path,key in [('datasets/swebench_lite_test.parquet','dataset'),('.cache/python39.tar.gz','python'),
                 ('.cache/NotoSansMonoCJKsc-Regular.otf','font')]:
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=pins[key]['sha256']:
        raise SystemExit(f'Checksum mismatch: {path}')
prefix=Path('.runtime/python39')
if not (prefix/'python/bin/python3').exists():
    prefix.mkdir(parents=True,exist_ok=True)
    with tarfile.open('.cache/python39.tar.gz') as archive:
        archive.extractall(prefix,filter='data')
PY
.runtime/python39/python/bin/python3 -m pip install --cache-dir .cache/pip-task \
  -r configs/swebench-python39.lock.txt
.venv/bin/python scripts/prepare_official.py select
if [[ ! -f datasets/generated/codeocr_v3/images.jsonl ]]; then
  .venv/bin/python scripts/prepare_official.py codeocr
fi
if [[ ! -f datasets/generated/codeocr_v3_unicode/images.jsonl ]]; then
  .venv/bin/python scripts/audit_screenshot_glyphs.py codeocr
fi
printf 'Official sources and task runtime are ready. Build/validate tasks with scripts/prepare_official.py build.\n'
