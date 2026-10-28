"""Inspect download progress using metadata and file stats; never read model contents."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from huggingface_hub import get_hf_file_metadata, hf_hub_url

SOURCE = Path(__file__).resolve().parents[1]
LABELS = {
    'sd_xl_base_1.0.safetensors': 'SDXL 主模型',
    'wan2.2_ti2v_5B_fp16.safetensors': 'Wan 2.2 主模型',
    'umt5_xxl_fp8_e4m3fn_scaled.safetensors': 'Wan 文本编码器',
    'wan2.2_vae.safetensors': 'Wan VAE',
}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def load_metadata(root, catalog):
    path = root / 'runtime' / 'model-download-metadata.json'
    metadata = read_json(path)
    manifest = read_json(root / 'models' / 'manifest.json')
    revisions = {key: item['revision'] for key, item in manifest.items()}
    log_path = root / 'logs' / 'aigc-studio-download.log'
    if log_path.exists():
        with log_path.open('rb') as log:
            log.seek(max(0, log_path.stat().st_size - 65536))
            text = log.read().decode(errors='replace')
        revisions.update(re.findall(r'^Downloading (\S+) at ([0-9a-f]{40})$', text, re.MULTILINE))
    changed = False
    for model, files in catalog.items():
        for item in files:
            key = model + '/' + Path(item['file']).name
            old = metadata.get(key, {})
            revision = revisions.get(key, old.get('revision', 'main'))
            if old.get('revision') == revision and old.get('bytes'):
                continue
            try:
                info = get_hf_file_metadata(hf_hub_url(item['repo'], item['file'], revision=revision), timeout=10)
                metadata[key] = {'revision': info.commit_hash, 'etag': info.etag, 'bytes': info.size}
                changed = True
            except Exception as exc:
                print(f'暂时无法读取 {key} 的总大小：{type(exc).__name__}', flush=True)
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(path)
    return metadata


def snapshot(root, catalog, metadata):
    manifest = read_json(root / 'models' / 'manifest.json')
    rows = []
    for model, files in catalog.items():
        for item in files:
            name = Path(item['file']).name
            key = model + '/' + name
            local = root / 'models' / 'downloads' / model
            destination = local / item['file']
            info = metadata.get(key, {})
            expected = info.get('bytes')
            size, state = 0, '待下载'
            try:
                if destination.is_file():
                    size = destination.stat().st_size
                    state = '已完成' if key in manifest and size == manifest[key]['bytes'] else '已下载，等待登记'
                else:
                    cache = local / '.cache' / 'huggingface' / 'download' / Path(item['file']).parent
                    pattern = '*.' + info['etag'] + '.incomplete' if info.get('etag') else '*.incomplete'
                    for partial in cache.glob(pattern):
                        stat = partial.stat()
                        # Avoid counting unwritten sparse extents as transferred data.
                        size = max(size, min(stat.st_size, stat.st_blocks * 512))
                    if size:
                        state = '未完成'
            except FileNotFoundError:
                # The downloader can rename a partial file between stat calls.
                state = '状态更新中'
            rows.append({'name': LABELS.get(name, name), 'bytes': size, 'total': expected, 'state': state})
    return rows


def service_state():
    try:
        result = subprocess.run(['systemctl', '--user', 'show', 'aigc-studio-download.service', '--property=ActiveState', '--value'], capture_output=True, text=True, timeout=3)
        return result.stdout.strip() if result.returncode == 0 else '未知'
    except (OSError, subprocess.SubprocessError):
        return '未知'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='refresh every five seconds; Ctrl+C stops only this viewer')
    args = parser.parse_args()
    root = Path(os.environ['AIGC_DATA_ROOT'])
    catalog = read_json(SOURCE / 'config' / 'models.json')
    metadata = load_metadata(root, catalog)
    previous = None
    try:
        while True:
            rows = snapshot(root, catalog, metadata)
            service = service_state()
            transfer = read_json(root / 'models' / 'download-state.json')
            unfinished = any(row['state'] != '已完成' for row in rows)
            now = time.monotonic()
            size = sum(row['bytes'] for row in rows)
            if args.watch and os.isatty(1):
                print('\033[2J\033[H', end='')
            print(time.strftime('%Y-%m-%d %H:%M:%S %Z'), '| 下载服务：' + service)
            if unfinished and service in {'inactive', 'failed'}:
                print('下载服务已停止，当前没有继续下载。恢复命令：bash scripts/run.sh start-download')
            elif service == 'activating':
                print('下载服务正在启动或等待自动重启。')
            elif service == 'active' and transfer.get('status') == 'retrying':
                print(f"网络中断，约 {max(0, int(transfer['next_retry'] - time.time()))} 秒后自动续传；已下载数据保留。")
            elif service == 'active' and transfer.get('status') == 'downloading':
                print('当前文件：' + transfer.get('key', ''))
            if transfer.get('error'):
                print('最近错误：' + transfer['error'])
            for row in rows:
                total = row['total']
                fraction = min(100, 100 * row['bytes'] / total) if total else None
                percentage = f'{fraction:.1f}%' if fraction is not None else '未知'
                total_text = f'{total / 1e9:.2f}' if total else '?'
                print(f"{row['name']:<14} {percentage:>7}  {row['bytes'] / 1e9:.2f} / {total_text} GB  {row['state']}")
            if all(row['total'] for row in rows):
                total = sum(row['total'] for row in rows)
                print(f'整体约 {100 * min(size, total) / total:.1f}%  ({size / 1e9:.2f} / {total / 1e9:.2f} GB)')
            if previous:
                delta = max(0, size - previous[0])
                print(f'本次采样间隔的落盘增量：{delta / 1e6:.1f} MB，平均 {delta / (now - previous[1]) / 1e6:.2f} MB/s')
            print('进度根据临时文件落盘量估算，可能受缓冲影响；以“已完成”为准。', flush=True)
            if not args.watch:
                return
            previous = (size, now)
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n已退出进度查看，后台下载继续。')


if __name__ == '__main__':
    main()
