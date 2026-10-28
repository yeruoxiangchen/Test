"""Visibility-selected physical occluders on a preregistered held-out cohort."""
import os
import sys
os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
for k, v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','SPCONV_ALGO':'native',
             'CUDA_HOME':'/usr/local/cuda-12.6','TORCH_CUDA_ARCH_LIST':'8.6',
             'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
             'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}.items():
    os.environ.setdefault(k, v)
import argparse
import json
from pathlib import Path
import torch

from ...data_v2.build_legacy_occlusion import augment
from ...data_v2.build_feature_cache import cache_scene
from ...paired_encoder import build_frozen_dino_encoder
from ...frozen_slat import load_frozen_slat30k
from ...v2.dataset import SceneDataset, write_manifest
from ...prepare_official_experiment import atomic_json
from ...paired_dataset import sha256_file, canonical_json_sha256

PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE/'outputs/round2_v2_diagnostics_20261018/strong_data'
POLICY = {'max_attempts':64,'strong_threshold':.5,'min_strong_views':4,
          'min_mean_visible':.15,'min_nonempty_views':4}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, choices=range(8), default=7)
    p.add_argument('--prepare_only', action='store_true')
    a=p.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    torch.set_num_threads(2)
    manifest=PACKAGE/'outputs/round1_20260909/data_v2/dev64.json'
    dataset=SceneDataset(manifest)
    SceneDataset(PACKAGE/'outputs/round2_v2_overfit32_20261017/train96.json').assert_disjoint(dataset)
    plan={'manifest':str(manifest),'manifest_sha256':sha256_file(manifest),
          'indices':list(range(16)),'object_uids':[r['object_uid'] for r in dataset.rows[:16]],
          'box_count':8,'seed':20261018,'visibility_policy':POLICY,
          'selection':'first16 Dev64 objects, one accepted fixed-3D scene each; no performance-based filtering',
          'scope':'synthetic proxy-composite stress test, original poses; not natural occlusion'}
    plan_path=ROOT/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan: raise ValueError('stress-test plan changed')
    else: atomic_json(plan_path,plan)
    if a.prepare_only: return
    device=torch.device('cuda:0')
    runtime=load_frozen_slat30k(device,evaluation_uids=plan['object_uids'])
    # Rendering needs only the frozen decoder; free the unused diffusion flow.
    del runtime['model']; torch.cuda.empty_cache()
    encoder=build_frozen_dino_encoder(device=device)
    previous=PACKAGE/'outputs/mesh_comparison_20260910/dev64_v2'
    cache_plan=json.loads((previous/'plan.json').read_text())
    if canonical_json_sha256(cache_plan['plan']) != cache_plan['identity_hash']:
        raise ValueError('previous target mesh plan hash differs')
    rows=[]; coverage=[]
    for i in plan['indices']:
        sample=dataset[i]; folder=ROOT/f'object_{i:03d}'
        scene=folder/'scene.json'
        if not scene.exists():
            if folder.exists():
                if (folder/'rejected_candidates.json').exists():
                    raise RuntimeError('visibility quota exhausted; preserve rejected object instead of filtering it')
                import time
                incomplete=ROOT/'incomplete_attempts';incomplete.mkdir(exist_ok=True)
                folder.rename(incomplete/(folder.name+'_'+str(time.time_ns())))
            encoder.models['image_cond_model'].cpu()
            import gc
            gc.collect();torch.cuda.empty_cache()
            cache_meta=json.loads((previous/'targets'/f'{i:05d}.json').read_text())
            cache_row=cache_plan['plan']['rows'][i]
            if (cache_meta['uid'] != sample['object_uid'] or cache_row['uid'] != sample['object_uid']
                    or cache_meta['identity_hash'] != cache_plan['identity_hash']):
                raise ValueError('previous target cache UID/plan mismatch')
            cached_target={**cache_meta['mesh'],'source_slat_sha256':cache_row['official_slat']['sha256'],
                           'decoder':cache_row['expected_decoder'],'plan_hash':cache_plan['identity_hash']}
            scene=augment(sample,folder,runtime,plan['seed']+i*1009,plan['box_count'],5,POLICY,cached_target)
            torch.cuda.empty_cache()
            encoder.models['image_cond_model'].to(device)
        meta=json.loads(scene.read_text())
        if meta['recipe']['visibility_policy'] != POLICY or meta['object_uid'] != sample['object_uid']:
            raise ValueError('cached stress scene identity differs')
        rows.append(cache_scene(scene,sample['source_binding']['ss'],folder/'sample.pt',encoder,resume=True))
        coverage.append({'index':i,'object_uid':sample['object_uid'],'visible_fraction':meta['visible_fraction'],
                         'attempts':len(meta['recipe']['selection_attempts'])})
        atomic_json(ROOT/'progress.json',{'completed':len(rows),'expected':16,'coverage':coverage})
        print(json.dumps(coverage[-1]),flush=True)
    output=ROOT/'manifest.json'
    if not output.exists(): write_manifest(output,rows,{'plan_sha256':canonical_json_sha256(plan)},ROOT)
    atomic_json(ROOT/'complete.json',{'sample_count':16,'manifest_sha256':sha256_file(output),'coverage':coverage})


if __name__=='__main__': main()
