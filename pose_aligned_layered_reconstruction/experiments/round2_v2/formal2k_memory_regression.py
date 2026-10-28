"""One-load CUDA regression for the explicit mesh memory resource amendment.

Writes a fresh isolated directory only. It does not publish production samples,
change the cohort, or relax numerical equality after a mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('expose exactly the requested physical GPU')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from . import formal2k_data as data
    from . import memory_efficient_mesh as efficient
    from . import formal2k_data_memory as scheduling
    report = {'passed': False, 'state': 'starting', 'utc': data.utc(),
        'gpu': args.gpu, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'artificial_memory_cap': False, 'torch': torch.__version__,
        'torch_git_version': torch.version.git_version, 'cases': [], 'stages': [],
        'sources': {name: data.binding(path) for name, path in (
            ('regression', __file__), ('reduction', efficient.__file__),
            ('residency', scheduling.__file__))},
        'original_mesh_sources': efficient.original_source_bindings(),
        'original_data_sources': data.code_binding(), 'production_data_modified': False}
    report_path = output / 'report.json'
    started = time.monotonic()
    hooks, residency, utils, installed = [], None, None, None

    def publish():
        data.atomic_json(report_path, report)

    def stage(name, index=None):
        row = {'stage': name, 'index': index, 'utc': data.utc()}
        if torch.cuda.is_initialized():
            row.update(allocated_mib=torch.cuda.memory_allocated() / 2**20,
                reserved_mib=torch.cuda.memory_reserved() / 2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
            row['driver_free_mib'] = torch.cuda.mem_get_info()[0] / 2**20
        report['stages'].append(row)
        publish()
        print(json.dumps(row), flush=True)
        return row

    def tensor_hash(value):
        cpu = value.detach().contiguous().cpu()
        return hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy()).hexdigest()

    def equal_tensor(left, right, name):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError('tensor layout mismatch: ' + name)
        if left.is_floating_point() and (not torch.isfinite(left).all() or not torch.isfinite(right).all()):
            raise ValueError('nonfinite tensor: ' + name)
        exact = torch.equal(left, right)
        bitwise = torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                              right.contiguous().reshape(-1).view(torch.uint8))
        if not exact or not bitwise:
            delta = float((left.double() - right.double()).abs().max()) if left.numel() else 0.
            raise ValueError(f'bitwise mismatch {name}; max_abs={delta}')

    def tensor_fields(value, prefix=''):
        found = {}
        if isinstance(value, torch.Tensor):
            found[prefix] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                found.update(tensor_fields(child, prefix + '/' + str(key)))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                found.update(tensor_fields(child, prefix + '/' + str(index)))
        return found

    publish()
    try:
        plan, identity = data.load_plan(Path(args.data_dir).resolve())
        report['data_plan_identity'] = identity
        report['data_plan'] = data.binding(Path(args.data_dir) / 'plan.json')
        renders = data.load_render_audit(plan)
        data.verify(plan['dino_weights'])
        torch.set_num_threads(2)
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        from ...frozen_slat import load_frozen_slat30k
        from ...paired_encoder import build_frozen_dino_encoder
        from ...v2.dataset import SceneDataset
        from ...data_v2.build_legacy_occlusion import augment
        from ...data_v2.build_feature_cache import cache_scene
        runtime = load_frozen_slat30k(device)
        if runtime['binding']['decoder'] != plan['proxy_decoder_binding']:
            raise ValueError('wrong frozen proxy decoder')
        runtime['model'].cpu()
        torch.cuda.empty_cache()
        encoder = build_frozen_dino_encoder(device=device)
        report.update(runtime_binding=runtime['binding'], dino_weights=plan['dino_weights'],
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            gpu_name=torch.cuda.get_device_name(device))
        dataset = SceneDataset(plan['clean_source']['path'], require_ss_target=True)
        from trellis.modules import sparse as sp
        import trellis.representations.mesh.utils_cube as utils
        import trellis.representations.mesh.cube2mesh as cube2mesh
        original = utils.cubes_to_verts

        @torch.no_grad()
        def decode_fresh(sample):
            with np.load(sample['source_binding']['slat'], allow_pickle=False) as source:
                xyz = torch.from_numpy(np.asarray(source['coords'], np.int32)).to(device)
                feats = torch.from_numpy(np.asarray(source['feats'], np.float32)).to(device)
            sparse = sp.SparseTensor(feats=feats, coords=torch.cat((torch.zeros_like(xyz[:, :1]), xyz), 1))
            return runtime['decoder'](sparse)[0]

        sample35 = dataset[35]
        torch.cuda.reset_peak_memory_stats()
        stage('original_decoder35_started', 35)
        baseline = decode_fresh(sample35)
        mesh_keys = ('vertices', 'faces', 'vertex_attrs', 'face_normal', 'vertex_normal')
        baseline_cpu = {key: getattr(baseline, key).detach().cpu().clone() for key in mesh_keys}
        report['same_run_original_mesh35'] = {'vertices': len(baseline.vertices),
            'faces': len(baseline.faces), 'mesh_resolution': int(baseline.res),
            'mesh_success': bool(baseline.success)}
        if baseline.res != 256 or not baseline.success:
            raise ValueError('original decoder did not produce a full-resolution valid mesh')
        torch.save(baseline_cpu, output / 'original_mesh35.pt')
        report['original_mesh35'] = data.binding(output / 'original_mesh35.pt')
        report['original_decoder35_peak'] = stage('original_decoder35_completed', 35)
        del baseline, sample35
        residency = scheduling.DinoResidency()
        residency.attach_runtime(runtime)
        residency.attach_encoder(encoder, device)
        report['mesh_patch'] = efficient.install_memory_efficient_mesh(chunk_channels=1)
        installed = utils.cubes_to_verts

        for index in (35, 65, 1918):
            item = next(item for item in plan['occluded_rows'] if item['index'] == index and item['variant'] == 0)
            sample = dataset[index]
            case = {'index': index, 'passed': False, 'item': item, 'reduction_checks': [],
                'memory_peak_scope': 'optimized_full_pipeline' if index == 35 else 'optimized_full_pipeline_plus_original_two_channel_reference'}
            report['cases'].append(case)
            report['state'] = f'case_{index}'
            case_started = time.monotonic()

            def check_reduction(num_verts, cubes, value, reduce='mean'):
                before = (tensor_hash(cubes), tensor_hash(value))
                actual = installed(num_verts, cubes, value, reduce=reduce)
                for start in range(0, value.shape[-1], 2):
                    end = min(start + 2, value.shape[-1])
                    reference = original(num_verts, cubes, value[..., start:end], reduce=reduce)
                    equal_tensor(actual[:, start:end], reference, f'{index}/reduction_channels_{start}_{end}')
                    del reference
                if before != (tensor_hash(cubes), tensor_hash(value)):
                    raise ValueError('cube reduction modified its inputs')
                case['reduction_checks'].append({'rows': int(cubes.shape[0]), 'corners': 8,
                    'channels': int(value.shape[-1]), 'reference_channel_chunk': 2,
                    'bitwise_equal': True, 'finite': True, 'max_abs': 0., 'sdf_sign_flips': 0,
                    'inputs_unchanged': True, 'input_sha256': list(before)})
                publish()
                return actual

            if index != 35:
                utils.cubes_to_verts = check_reduction
                cube2mesh.cubes_to_verts = check_reduction

            def check_mesh(module, inputs, meshes):
                if len(meshes) != 1:
                    raise ValueError('unexpected mesh batch')
                mesh = meshes[0]
                case.update(vertices=len(mesh.vertices), faces=len(mesh.faces),
                            mesh_resolution=int(mesh.res), mesh_success=bool(mesh.success))
                expected = plan['clean_rows'][index]
                case['historical_mesh_counts'] = {'vertices': expected['target_mesh_vertices'],
                                                  'faces': expected['target_mesh_faces']}
                case['historical_mesh_count_delta'] = {
                    'vertices': len(mesh.vertices) - expected['target_mesh_vertices'],
                    'faces': len(mesh.faces) - expected['target_mesh_faces']}
                if mesh.res != 256 or not mesh.success or not len(mesh.vertices) or not len(mesh.faces):
                    raise ValueError('optimized decoder did not produce a full-resolution valid mesh')
                if index == 35:
                    case['same_run_original_mesh_counts'] = {
                        'vertices': len(baseline_cpu['vertices']), 'faces': len(baseline_cpu['faces'])}
                    for key in mesh_keys:
                        equal_tensor(getattr(mesh, key).detach().cpu(), baseline_cpu[key], 'mesh35/' + key)
                    case['mesh_tensor_fields_bitwise_equal'] = list(mesh_keys)
                publish()

            hook = runtime['decoder'].register_forward_hook(check_mesh)
            hooks.append(hook)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            stage('optimized_case_started', index)
            folder = output / f'object_{index:05d}_variant_00'
            scene_path = augment(sample, folder, runtime, item['seed'], plan['box_count'], plan['layers'])
            hook.remove()
            hooks.remove(hook)
            utils.cubes_to_verts = installed
            cube2mesh.cubes_to_verts = installed
            data.validate_scene(data.read(scene_path), item, sample['source_binding'], plan, renders[item['source_uid']])
            row = cache_scene(scene_path, sample['source_binding']['ss'], folder / 'sample.pt', encoder)
            cached = data.validate_cached_sample(row, item, sample)
            if index == 35:
                receipt_path = Path(args.data_dir) / 'receipts/00035_00.json'
                old_receipt = data.read(receipt_path)
                old_folder = Path(args.data_dir).resolve() / 'occlusion/object_00035_variant_00'
                old_row = data.verify_completed_receipt(old_receipt, item, identity, old_folder)
                old_cache = data.validate_cached_sample(old_row, item, sample)
                previous, current = tensor_fields(old_cache), tensor_fields(cached)
                if set(previous) != set(current):
                    raise ValueError('production cache tensor field set changed')
                for key in previous:
                    equal_tensor(current[key], previous[key], 'cache35' + key)
                for key in cached:
                    if key != 'source_files' and not tensor_fields(cached[key]):
                        if cached[key] != old_cache[key]:
                            raise ValueError('production cache metadata changed: ' + key)
                if data.read(scene_path) != data.read(old_folder / 'scene.json'):
                    raise ValueError('original recipe/scene metadata changed')
                for png in sorted(folder.glob('*.png')):
                    if data.sha256_file(png) != data.sha256_file(old_folder / png.name):
                        raise ValueError('original scene pixels changed: ' + png.name)
                for name in ('camera.npz', 'layers.npz'):
                    with np.load(folder / name) as new, np.load(old_folder / name) as old:
                        if set(new.files) != set(old.files):
                            raise ValueError('NPZ field set changed: ' + name)
                        for key in new.files:
                            equal_tensor(torch.from_numpy(new[key].copy()), torch.from_numpy(old[key].copy()), name + '/' + key)
                case.update(production_receipt=data.binding(receipt_path),
                    production_cache_all_tensor_fields_bitwise_equal=sorted(previous),
                    production_metadata_and_scene_outputs_equal=True)
                del old_cache, previous, current
            del cached
            torch.cuda.synchronize()
            case.update(peak=stage('optimized_case_completed', index), cache_row=row,
                scene=data.binding(scene_path), passed=True, elapsed_seconds=time.monotonic() - case_started)
            if index != 35 and len(case['reduction_checks']) != 1:
                raise ValueError('reference comparison did not cover exactly one full mesh reduction')
            publish()
            del sample

        for item in report['sources'].values():
            data.verify(item)
        if data.code_binding() != report['original_data_sources']:
            raise ValueError('frozen source changed during regression')
        efficient.original_source_bindings()
        report.update(passed=True, state='complete', elapsed_seconds=time.monotonic() - started,
                      residency=residency.summary(), completed_at_utc=data.utc())
        publish()
        print(json.dumps({'passed': True, 'report': str(report_path)}), flush=True)
    except BaseException as error:
        report.update(passed=False, state='failed', error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        try:
            report['failure_memory'] = stage('failed')
        except Exception as memory_error:
            report['failure_memory_error'] = str(memory_error)
        publish()
        raise
    finally:
        for hook in hooks:
            hook.remove()
        if utils is not None and installed is not None:
            utils.cubes_to_verts = installed
            cube2mesh.cubes_to_verts = installed
        if residency is not None:
            residency.close()


if __name__ == '__main__':
    main()
