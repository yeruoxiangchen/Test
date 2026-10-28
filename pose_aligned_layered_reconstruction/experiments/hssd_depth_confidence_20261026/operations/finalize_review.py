"""Publish the completed CPU/visual review without changing experiment outputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_cross_review_common', HERE.parent / 'common.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


def main(write):
    folder = c.OUT / 'review_20261026'
    analysis = c.read(folder / 'analysis.json'); verified = c.read(folder / 'verification.json')
    summary = c.read(c.OUT / 'summary.json'); identity = c.digest(c.read(c.OUT / 'evaluation_identity.json'))
    if not (analysis['passed'] and verified['passed'] and analysis['identity'] == verified['identity'] == summary['identity'] == identity):
        raise ValueError('Missing or inconsistent numerical review')
    c.verify(verified['review_source'])
    notes = c.read(HERE / 'review_visual_notes.json')
    expected = {(u, s) for u in c.UIDS for s in c.SEEDS}
    if len(notes['sheets']) != 12 or {(r['uid'], r['seed']) for r in notes['sheets']} != expected:
        raise ValueError('Visual inspection must include every object and seed')
    figures = {(r['uid'], r['seed']): r for r in summary['figures']}
    if set(figures) != expected: raise ValueError('Incomplete figure set')
    for row in notes['sheets']:
        row['figure'] = figures[row['uid'], row['seed']]; c.verify(row['figure'])
    notes.update(time=c.now(), identity=identity, all_visuals_inspected=True,
                 inspection_source=c.binding(HERE / 'review_visual_notes.json'),
                 human_review_completed=False)
    source_report = HERE / 'REVIEW.md'
    appendix = (
        '\n## 机器验收结果\n\n'
        f"复核时间：{verified['time']}。{verified['cases']}个结果，"
        f"{verified['checks_passed']}项结构/来源/坐标检查通过，"
        f"{verified['scalar_or_exact_value_checks']}项数值或精确字段比较（训练字段{verified['training_field_checks']}项），"
        f"汇总最大误差{verified['max_abs_aggregation_error']:.3g}。"
        f"验证{verified['files_hashed']}个唯一文件哈希；{len(verified['large_bindings_not_rehashed'])}个大模型绑定未重复哈希。\n\n"
        f"24个新canonical/world mesh的{verified['new_vertices_checked']:,}个顶点全量核对，"
        f"world最大误差{verified['max_world_error_m']:.9g}米、canonical往返最大误差"
        f"{verified['max_roundtrip_error']:.9g}。面索引一致，无逐结果拟合旋转、平移、缩放。"
        f"{verified['render_records_checked']}个渲染记录/{verified['render_views_checked']}视图无裁剪，"
        f"12张比较图绑定通过。CPU审阅没有初始化CUDA。\n\n"
        '[独立统计](analysis.json) · [产物及坐标验收](verification.json) · '
        '[逐图审阅](visual_review.json) · [审阅回执](review.json)\n'
    )
    report = source_report.read_text() + appendix
    result = {'time': c.now(), 'identity': identity, 'review_completed': True,
              'artifacts_verified': True, 'all_visuals_inspected': True,
              'reviewer': 'Codex', 'human_review_completed': False,
              'training_evaluation_completed': True, 'numerical_runtime_modified': False,
              'new_gpu_jobs_launched': False, 'analysis': c.binding(folder / 'analysis.json'),
              'verification': c.binding(folder / 'verification.json'),
              'report_source': c.binding(source_report), 'visual_notes_source': c.binding(HERE / 'review_visual_notes.json'),
              'conclusion': 'Depth and confidence policy interact; unit confidence gains concentrate in chair, worsen macro CD and do not repair bench/table failures.',
              'next_step': 'End four-object tuning; freeze a scene/object-disjoint evaluation protocol with matched appearance baseline.',
              'provisional_rgb_main_configuration': 'geometry_observed',
              'unit_confidence_retained_as_ablation': True,
              'original_lari_long_training_restarted': False,
              'scope': 'Review of four previously trained objects, one scene, three sampling repeats; no held-out generalization claim.'}
    if write:
        c.write(folder / 'visual_review.json', notes)
        (folder / 'REPORT.md').write_text(report)
        result.update(report=c.binding(folder / 'REPORT.md'), visual_review=c.binding(folder / 'visual_review.json'))
        c.write(folder / 'review.json', result)
    print(json.dumps({k: v for k, v in result.items() if k in
                      ('time', 'review_completed', 'artifacts_verified', 'all_visuals_inspected', 'new_gpu_jobs_launched', 'report')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--write', action='store_true')
    main(parser.parse_args().write)
