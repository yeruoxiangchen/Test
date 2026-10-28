"""Read-only Chinese mesh progress panel; Ctrl+C never stops the workers."""
import argparse
from pathlib import Path
import time

from .evaluate_mesh import DEFAULT_OUTPUT
from .prepare_official_experiment import read_json


def show(output):
    output = Path(output)
    status_path = output/'background_status.json'
    status = read_json(status_path) if status_path.exists() else {}
    records = [read_json(p) for p in sorted((output/'records').glob('*.json'))]
    successes = sum(r.get('success', False) for r in records)
    print(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
    print(f"Mesh 重建：{successes} / 256（64 对象 × 2 seeds × 父/候选）")
    print('后台状态：' + str(status.get('state', '尚未启动正式后台；可能正在 smoke')))
    for job in status.get('jobs', []):
        name = f"{status['attempt']}_gpu{job['gpu']}"
        path = output/'workers'/f'{name}_status.json'
        value = read_json(path) if path.exists() else {}
        print(f"GPU {job['gpu']}：{value.get('completed', 0)}/{len(job['indices'])*4}；"
              f"{value.get('state', '初始化/加载模型')}；对象 {value.get('current_index', '-')}；"
              f"{value.get('mode', '-')} seed {value.get('seed', '-')}；{value.get('phase', '-')}")
        if value.get('error'):
            print('  错误：' + value['error'])
    if status.get('error'):
        print('总错误：' + status['error'])
    result = output/'aggregate.json'
    if result.exists():
        report = read_json(result)
        p, c = report['summary']['parent'], report['summary']['candidate']
        print(f"Chamfer-L1 ↓：{p['chamfer_l1']:.6f} → {c['chamfer_l1']:.6f}")
        print(f"F-score@0.02 ↑：{p['fscore_0p02']:.6f} → {c['fscore_0p02']:.6f}")
        print(f"Normal consistency ↑：{p['normal_consistency']:.6f} → {c['normal_consistency']:.6f}")
        print('GT：官方 SLat 解码网格，并非原始未压缩 mesh。')
    print('只退出此监视器不会停止后台评估。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--watch', type=float, default=0.)
    args = parser.parse_args()
    try:
        while True:
            show(args.output_dir)
            if args.watch <= 0:
                break
            time.sleep(max(1., args.watch))
            print()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
