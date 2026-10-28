"""Freeze official-scene-based splits before any reconstruction scores exist."""
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path

from . import common as c
from .catalog import safe, scene_dependencies


def family_table(metadata):
    table = {r['id']: r for r in metadata}
    def family(asset):
        seen = set()
        while asset in table and table[asset].get('decomposedFrom'):
            if asset in seen: raise ValueError('Cycle in asset decomposition metadata')
            seen.add(asset); asset = table[asset]['decomposedFrom']
        return asset
    return table, family


def split_scenes(official, available):
    # Preserve official train/val boundary. Test is a deterministic subdivision
    # of public official val, not a claim about a hidden official test server.
    if not isinstance(official, dict): raise ValueError('Invalid scene split file')
    values = {}
    for key in ('train', 'val'):
        raw = official.get(key)
        if raw is None: raise ValueError('Official split missing ' + key)
        values[key] = {Path(str(s)).name.split('.')[0] for s in raw}
    if values['train'] & values['val']: raise ValueError('Official scene overlap')
    train = sorted((values['train'] & available) - c.DEVELOPMENT_SCENES, key=c.rank)
    public_val = sorted((values['val'] & available) - c.DEVELOPMENT_SCENES, key=c.rank)
    if len(train) < 16 or len(public_val) < 8: raise ValueError('Insufficient public scenes')
    return {'train': train[:16], 'val': public_val[:4], 'test': public_val[4:8]}


def prepare():
    import yaml
    catalog = c.read(c.OUT / 'catalog.json')
    available = {Path(s).name.split('.')[0] for s in catalog['scenes']}
    official = yaml.safe_load(safe('scene_splits.yaml').read_text())
    split = split_scenes(official, available)
    metadata = list(csv.DictReader(safe('metadata/fpmodels-with-decomposed.csv').open()))
    table, family = family_table(metadata)
    scene_docs = {sid: c.read(safe(f'scenes/{sid}.scene_instance.json')) for sid in available}
    history = {family(r['template_name']) for sid in c.DEVELOPMENT_SCENES for r in scene_docs[sid]['object_instances']}
    # Metadata-only feasibility, before downloading/decoding any geometry or
    # observing a reconstruction score. Eight eventual targets need reserves.
    attempts=[];used={sid for ids in split.values() for sid in ids}
    pools={subset:sorted(({str(s) for s in official['train' if subset=='train' else 'val']} & available)
                         -c.DEVELOPMENT_SCENES,key=c.rank) for subset in split}
    for iteration in range(len(available)):
        presence=defaultdict(set)
        for subset,ids in split.items():
            for sid in ids:
                for obj in scene_docs[sid]['object_instances']:presence[family(obj['template_name'])].add(subset)
        weak=[]
        for subset,ids in split.items():
            for position,sid in enumerate(ids):
                groups={family(obj['template_name']) for obj in scene_docs[sid]['object_instances']
                        if table.get(obj['template_name'],{}).get('main_category') in c.CATEGORIES
                        and family(obj['template_name']) not in history and presence[family(obj['template_name'])]=={subset}}
                if len(groups)<16:weak.append((subset,position,sid,len(groups)))
        if not weak:break
        subset,position,sid,count=weak[0]
        replacement=next((s for s in pools[subset] if s not in used),None)
        if replacement is None:raise ValueError('Insufficient metadata-eligible disjoint scenes')
        split[subset][position]=replacement;used.add(replacement)
        attempts.append({'split':subset,'scene':sid,'eligible_families':count,'replacement':replacement,
                         'reason':'fewer_than_16_metadata_eligible_families_before_model_scores'})
    else:raise ValueError('Scene-capacity selection did not converge')
    selected_scenes = [sid for ids in split.values() for sid in ids]
    deps = scene_dependencies(selected_scenes)
    remote = {r['path']: r for r in deps['geometry_files']}
    scene_deps = {r['scene']: r for r in deps['scenes']}
    family_presence = defaultdict(set); content_presence = defaultdict(set)
    for subset, ids in split.items():
        for sid in ids:
            dep = scene_deps[sid]
            cfg_for = {Path(s).name.removesuffix('.object_config.json'): s for s in dep['configs']}
            for row in scene_docs[sid]['object_instances']:
                asset = row['template_name']; family_presence[family(asset)].add(subset)
                source = remote[dep['render_assets'][cfg_for[asset]]]
                content_presence[source.get('sha256') or source['git_oid']].add(subset)
    candidates = []; excluded = Counter(); scene_counts = {}
    for subset, ids in split.items():
        for sid in ids:
            dep = scene_deps[sid]
            cfg_for = {Path(s).name.removesuffix('.object_config.json'): s for s in dep['configs']}
            regions = c.read(safe(dep['semantic'])).get('region_annotations', [])
            usable_regions = [r for r in regions if all(k in r for k in ('min_bounds', 'max_bounds'))]
            usable_regions = sorted(usable_regions, key=lambda r: (
                -(r['max_bounds'][0]-r['min_bounds'][0])*(r['max_bounds'][2]-r['min_bounds'][2]), str(r.get('id', r.get('name')))))[:8]
            pool = []
            for index, row in enumerate(scene_docs[sid]['object_instances']):
                asset = row['template_name']; info = table.get(asset, {}); group = family(asset)
                raw = dep['render_assets'][cfg_for[asset]]; remote_file = remote[raw]; content = remote_file.get('sha256') or remote_file['git_oid']
                reason = None
                if info.get('main_category') not in c.CATEGORIES: reason = 'category_not_in_furniture_scope'
                elif group in history: reason = 'historical_development_family'
                elif len(family_presence[group]) != 1: reason = 'family_visible_in_multiple_splits'
                elif len(content_presence[content]) != 1: reason = 'same_source_bytes_visible_in_multiple_splits'
                elif any(abs(float(v)) < 1e-8 for v in row.get('non_uniform_scale', [1, 1, 1])): reason = 'singular_scale'
                else:
                    xyz = row.get('translation', [0, 0, 0])
                    if not any(r['min_bounds'][0] <= xyz[0] <= r['max_bounds'][0]
                               and r['min_bounds'][2] <= xyz[2] <= r['max_bounds'][2] for r in usable_regions):
                        reason = 'outside_fixed_capture_rooms'
                if reason: excluded[reason] += 1; continue
                pool.append({'uid': f'hssd_{sid}_i{index:03d}', 'scene': sid, 'split': subset, 'instance': index,
                             'asset_uid': asset, 'asset_family': group, 'category': info.get('main_category'),
                             'config': cfg_for[asset], 'source_asset': raw, 'source_content_id': content,
                             'instance_data': row})
            chosen = []; owners = set()
            for row in sorted(pool, key=lambda r: c.rank(r['uid'])):
                if row['asset_family'] in owners: excluded['same_family_same_scene'] += 1; continue
                chosen.append(row); owners.add(row['asset_family'])
                if len(chosen) == c.CANDIDATES_PER_SCENE: break
            if len(chosen) < c.TARGETS_PER_SCENE:
                raise ValueError(f'{sid}: only {len(chosen)} scene/asset-disjoint candidates; do not shrink the cohort silently')
            candidates.extend(chosen); scene_counts[sid] = len(chosen)
            dep['capture_regions'] = usable_regions
    values = {'format': 'hssd_disjoint192_data.v1', 'seed': c.SEED, 'repo': c.REPO, 'revision': c.REVISION,
              'scene_splits': split, 'target_counts': {'train': 128, 'val': 32, 'test': 32},
              'targets_per_scene': 8, 'candidates_per_scene_cap': 32, 'candidates': candidates,
              'dependencies': deps, 'excluded_counts': dict(excluded), 'candidate_counts_per_scene': scene_counts,
              'historical_development_scenes': sorted(c.DEVELOPMENT_SCENES), 'historical_families': sorted(history),
              'metadata_scene_capacity_attempts':attempts,'minimum_metadata_eligible_families_per_scene':16,
              'split_policy': 'Official train; deterministic disjoint public-val subdivisions for development val/test. Candidate target family and identical GLB bytes never appear in any other selected split, including context.',
              'selection_policy': 'Ranked candidates before model scores; one instance per family per scene candidate pool. Reserve pools do not preempt families in other same-split scenes; enforce global target-family uniqueness when selecting exactly8 per scene after shared-photo visibility checks, before latent decoding. No replacement based on latent/reconstruction quality.',
              'capture_policy': {'side': 2072, 'cycles_samples': 16, 'rooms_per_scene_cap': 8,
                                 'camera_positions_per_room': 4, 'yaw_degrees': [0, 90, 180, 270],
                                 'height_m': 1.45, 'fov_degrees': 70, 'clearance_m': .18,
                                 'input_views_min': 2, 'input_views_max': 4, 'visible_pixels_518_min': 128},
              'target_policy': {'views': 150, 'side': 512, 'cycles_samples': 128, 'ss_resolution': 64,
                                'ss_roundtrip_min_iou': .90, 'dino': 'x_prenorm, no feature layernorm',
                                'posterior': 'mean', 'no_icp_or_per_result_fitted_transform': True},
              'storage_policy': {'root': str(c.DATASET), 'outputs': str(c.OUT), 'raw_download_limit_gib': 40,
                                 'minimum_free_gib': 100, 'resident_objects_per_gpu_worker': 1},
              'catalog': c.binding(c.OUT / 'catalog.json'), 'official_split': c.binding(safe('scene_splits.yaml')),
              'metadata': c.binding(safe('metadata/fpmodels-with-decomposed.csv')),
              'categories': sorted(c.CATEGORIES), 'stage': 'data_qualification_before_formal_training'}
    result = {'identity': c.digest(values), 'plan': values}; c.locked(c.OUT / 'candidate_plan.json', result)
    c.event('candidate_plan_complete', scene_counts={k: len(v) for k,v in split.items()}, candidates=len(candidates),
            targets=192, geometry_bytes=deps['download_bytes'], identity=result['identity'])


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--refine-prelaunch',action='store_true');args=parser.parse_args()
    if args.refine_prelaunch and (c.OUT/'candidate_plan.json').exists():
        if (c.OUT/'runtime.json').exists() or (c.OUT/'download_complete.json').exists() or (c.DATASET/'targets.json').exists():
            raise ValueError('Cannot refine after queue freeze or data/model materialization')
        old=c.read(c.OUT/'candidate_plan.json');history=c.OUT/'planning_history'
        history.mkdir(parents=True,exist_ok=True)
        (c.OUT/'candidate_plan.json').replace(history/('candidate_pool_initial_'+old['identity']+'.json'))
        c.write(history/('candidate_pool_refinement_'+old['identity']+'.json'),{
            'time':c.now(),'previous_identity':old['identity'],
            'reason':'Prelaunch candidate feasibility: do not preempt same-split reserve families; require16 metadata-eligible families before freezing eight eventual targets per scene.',
            'previous_scene_ids':old['plan']['scene_splits'],'target_budgets_unchanged':old['plan']['target_counts'],
            'latent_or_model_scores_observed':False,'queue_not_frozen_yet':True})
    prepare()
