import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from video_count_probe.core import atomic_json, digest, file_sha, writer_lock
from video_count_probe.wan22_action_probe import model, protocol, review, runtime


class ProtocolTests(unittest.TestCase):
    def test_six_uncounted_jobs_and_full_geometry_smoke(self):
        pilot, smoke = protocol.build_plan('pilot'), protocol.build_plan('smoke')
        self.assertEqual(len(pilot['jobs']), 6)
        self.assertEqual(len({j['id'] for j in pilot['jobs']}), 6)
        self.assertEqual({j['seed'] for j in pilot['jobs']}, {42, 123})
        self.assertEqual(len(smoke['jobs']), 1)
        for key in ('height', 'width', 'frames', 'fps'):
            self.assertEqual(smoke['config'][key], pilot['config'][key])
        self.assertEqual(pilot['config']['steps'], 50)
        self.assertEqual(smoke['config']['steps'], 2)
        self.assertFalse(smoke['scientific_evaluation'])
        self.assertEqual(pilot['reference']['plan_hash'], 'f756e474096557c4c52e8e4a90a01fdcd7bfbd87c731204629240746b5b8a02a')

    def test_hash_check_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'model.safetensors'
            p.write_bytes(b'abcd')
            expected = {'path': p.name, 'size': 4, 'sha256': hashlib.sha256(b'abcd').hexdigest()}
            model.verify_file(p, expected)
            p.write_bytes(b'abce')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                model.verify_file(p, expected)

    def test_writer_lock_blocks_second_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '.writer.lock'
            with writer_lock(path):
                with self.assertRaisesRegex(RuntimeError, 'Another writer'):
                    with writer_lock(path):
                        pass

    def setup_run(self, tmp):
        run = Path(tmp) / 'pilot'
        plan = protocol.build_plan('pilot')
        identity = {'plan_hash': plan['plan_hash']}
        atomic_json(run / 'plan.json', plan)
        atomic_json(run / 'identity.json', identity)
        job = plan['jobs'][0]
        video = run / 'videos' / (job['id'] + '.mp4')
        video.parent.mkdir()
        video.write_bytes(b'test video contents')
        record = {'status': 'ok', 'identity': identity, 'job': job, 'video': str(video.relative_to(run)),
                  'video_sha256': file_sha(video), 'peak_allocated_mib': 10, 'peak_reserved_mib': 12}
        atomic_json(run / 'records' / (job['id'] + '.json'), record)
        return run, plan, record, identity

    def test_status_preserves_ai_provenance_and_null_is_not_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, plan, record, _ = self.setup_run(tmp)
            data = {'plan_hash': plan['plan_hash'], 'rater_type': 'ai', 'rater': 'test reviewer', 'items': [
                {'job_id': record['job']['id'], 'video_sha256': record['video_sha256'], 'complete_cycle': None, 'notes': ''}]}
            atomic_json(run / 'annotations.json', data)
            result = review.report(tmp, 'pilot', write=False)
            self.assertEqual(result['annotated'], 0)
            self.assertFalse(result['assessment_complete'])
            data['items'][0].update(complete_cycle=True, notes='A complete cycle was observed.')
            atomic_json(run / 'annotations.json', data)
            result = review.report(tmp, 'pilot', write=False)
            self.assertEqual(result['complete_action_videos'], 1)
            self.assertEqual(result['review_provenance']['rater_type'], 'ai')
            self.assertFalse((run / 'summary.json').exists())
            data['items'][0]['video_sha256'] = 'bad'
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                review.annotation_scores(data, plan, {record['job']['id']: record})

    def test_finished_generation_does_not_load_gpu_or_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, plan, record, identity = self.setup_run(tmp)
            # A shortened fake plan is permitted only inside this test fixture.
            plan['jobs'] = plan['jobs'][:1]
            plan['plan_hash'] = digest({k: v for k, v in plan.items() if k != 'plan_hash'})
            identity.update(plan_hash=plan['plan_hash'], packages={'diffusers': '0.35.1'})
            record['identity'] = identity
            atomic_json(run / 'plan.json', plan)
            atomic_json(run / 'identity.json', identity)
            atomic_json(run / 'records' / (record['job']['id'] + '.json'), record)
            with patch.object(runtime, 'current_identity', return_value=identity), patch.object(runtime, 'check_gpu') as gpu, patch.object(runtime, 'text_identity') as text, patch.object(runtime, 'load_pipe') as load:
                runtime.generate(tmp, 'pilot', '/unused')
            gpu.assert_not_called()
            text.assert_not_called()
            load.assert_not_called()

    def test_sharing_requires_headroom_and_caps_only_visible_device(self):
        import torch
        with patch.object(runtime, 'check_gpu', return_value={'uuid': 'test'}) as check, \
             patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=24576 * 2**20)), \
             patch.object(torch.cuda, 'set_per_process_memory_fraction') as cap:
            result = runtime.gpu_preflight(17408)
            check.assert_called_once_with(allow_busy=True, min_free_mib=19456)
            cap.assert_called_once_with(17408 / 24576, device=0)
            self.assertEqual(result['allocator_budget_mib'], 17408)
        # A failed occupancy check must not allocate a CUDA context or set a cap.
        with patch.object(runtime, 'check_gpu', side_effect=RuntimeError('not enough free')), \
             patch.object(torch.cuda, 'get_device_properties') as properties:
            with self.assertRaisesRegex(RuntimeError, 'not enough free'):
                runtime.gpu_preflight(17408)
            properties.assert_not_called()
        with patch.object(runtime, 'check_gpu') as check:
            runtime.gpu_preflight()
            check.assert_called_once_with(min_free_mib=22500)
            with self.assertRaises(ValueError):
                runtime.gpu_preflight(0)

    def test_sharing_recheck_accounts_for_own_reserved_pool(self):
        import torch
        gpu = {'allocator_budget_mib': 17408, 'required_headroom_mib': 2048}
        with patch.object(torch.cuda, 'memory_reserved', return_value=12288 * 2**20), \
             patch.object(torch.cuda, 'mem_get_info', return_value=(8192 * 2**20, 24576 * 2**20)):
            runtime.ensure_shared_headroom(gpu)
        with patch.object(torch.cuda, 'memory_reserved', return_value=12288 * 2**20), \
             patch.object(torch.cuda, 'mem_get_info', return_value=(6144 * 2**20, 24576 * 2**20)):
            with self.assertRaisesRegex(RuntimeError, 'headroom shrank'):
                runtime.ensure_shared_headroom(gpu)


class RealLibraryTests(unittest.TestCase):
    def test_real_wan22_expanded_timestep_and_patch_vae_decode(self):
        import torch
        import numpy as np
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
        torch.set_num_threads(2)
        transformer = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=12, in_channels=4,
                                            out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48,
                                            num_layers=1, rope_max_seq_len=64).eval()
        vae = AutoencoderKLWan(base_dim=4, decoder_base_dim=4, z_dim=4, dim_mult=[1, 1, 1, 1],
                               num_res_blocks=1, patch_size=2, in_channels=12, out_channels=12, is_residual=True,
                               scale_factor_temporal=4, scale_factor_spatial=16,
                               latents_mean=[0.] * 4, latents_std=[1.] * 4).eval()
        pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=vae, transformer=transformer,
                           scheduler=UniPCMultistepScheduler(flow_shift=5, prediction_type='flow_prediction',
                                                           use_flow_sigmas=True), expand_timesteps=True)
        pipe.set_progress_bar_config(disable=True)
        seen = []
        hook = transformer.register_forward_pre_hook(lambda _m, _a, kw: seen.append(tuple(kw['timestep'].shape)), with_kwargs=True)
        try:
            result = pipe(prompt_embeds=torch.randn(1, 8, 16), negative_prompt_embeds=torch.randn(1, 8, 16),
                          height=64, width=64, num_frames=5, num_inference_steps=2, guidance_scale=5,
                          generator=torch.Generator().manual_seed(42), output_type='np').frames
        finally:
            hook.remove()
        self.assertEqual(tuple(result.shape), (1, 5, 64, 64, 3))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(len(seen), 4)  # conditional + unconditional, two steps
        self.assertTrue(all(shape == (1, 8) for shape in seen))

        from video_count_probe.wan22_action_probe.vae_compat import install
        install(vae)
        latent = torch.randn(1, 4, 2, 4, 4)
        with torch.no_grad():
            # A single tile must exactly reproduce the stock untiled residual decode.
            expected = vae.decode(latent).sample
            actual = vae.tiled_decode(latent).sample
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            # Force multiple overlapping tiles, including edge tiles, through decode().
            vae.enable_tiling(tile_sample_min_height=16, tile_sample_min_width=16,
                              tile_sample_stride_height=8, tile_sample_stride_width=8)
            actual = vae.decode(latent).sample
            self.assertEqual(tuple(actual.shape), (1, 3, 5, 64, 64))
            self.assertTrue(torch.isfinite(actual).all())

    def test_ffmpeg_24fps_review_and_wrong_metadata_rejection(self):
        import numpy as np
        from diffusers.utils import export_to_video
        config = {'frames': 7, 'height': 32, 'width': 64, 'fps': 24}
        with tempfile.TemporaryDirectory() as tmp:
            video, sheet = Path(tmp) / 'test.mp4', Path(tmp) / 'sheet.jpg'
            export_to_video([np.full((32, 64, 3), i * 30, dtype=np.uint8) for i in range(7)], str(video), fps=24)
            review.make_sheet(video, sheet, config, 'Test review')
            self.assertTrue(sheet.is_file())
            with self.assertRaisesRegex(ValueError, 'fps differs'):
                review.make_sheet(video, sheet, {**config, 'fps': 16}, 'Wrong fps')


if __name__ == '__main__':
    unittest.main()
